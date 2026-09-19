"""v1.py -- 실제로 동작하는 작은 V1 회로.

명세 [7]. **생물학적 층 이름과 이 구현의 계산 역할은 다르다.**
README_ko.md 의 대조표를 함께 읽을 것.

구성
----
* **L4** (격자점마다 ON/OFF 각 1개): 전처리 값을 tick 0 의 ``ExternalDrive`` 로
  받는 임계 발화 뉴런. threshold 기본 0.1. 관찰 tick 0.
* **INH 중계** (L4 마다 1개): L4→INH 는 weight=1 고정 흥분, delay 1.
  P=1, threshold=0.5. 관찰 tick 1. 실제 억제성 세포의 다양성을 재현한 것이
  아니라 "L4 발화를 한 tick 뒤에 전달"하는 단순화다.
* **L2** (격자점마다 방향 후보 0°/90°): 원본 좌표에서 가까운 L4 격자점
  k<=16 개의 ON/OFF 를 직접 흥분(delay 2)으로, 대응 INH 중계를
  억제(delay 1)로 받는다. 두 경로 모두 tick 2 에 도착한다. 관찰 tick 2.
  **학습 대상은 이 L2 입력 연결뿐이다.**
* **L3** (관찰용 출력): 같은 방향의 가까운 L2 k_pool 개를 1/k_pool 고정
  가중치로 모으고 threshold 0.5. delay 1 -> 관찰 tick 3.
* **L1**: 교정 신호가 들어오는 **논리적 인터페이스**다. 별도의 피라미드
  세포체 집단을 만들지 않는다 (``teacher.L1Relay`` 참조).
* L5/L6, V2/V4/IT 는 **구현하지 않았다** (명세 [1], ``protocols.py`` 의
  입출력 프로토콜 설명만 제공).

선호 방향(``preferred_orientation_deg``)은 메타데이터이며 임계값과 다른
변수다. 학생 가중치에 정답 방향 필터를 미리 복사하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .connections import ConnectionTable
from .engine import SimulationEngine
from .neuron import Population, ReceptiveField
from .retina import RetinaConfig, RetinaEncoder
from .spatial_mapping import (
    GridConfig,
    InputNormalizer,
    LogPolarGrid,
    LogPolarSampler,
    build_grid,
)

LAYER_DEPTH_Z = {"L4": 0.0, "INH": 0.5, "L2": 1.0, "L3": 2.0}
"""모형 깊이 배치 (설계값). 실제 조직 측정값이 아니다."""


@dataclass
class V1Circuit:
    """구축된 회로 + 전처리 파이프라인 묶음."""

    population: Population
    table: ConnectionTable
    engine: SimulationEngine
    grid: LogPolarGrid
    sampler: LogPolarSampler
    retina: RetinaEncoder
    normalizer: InputNormalizer
    config: dict[str, Any]

    # 인덱스 (모두 int64 배열)
    l4_on_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    l4_off_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    l4_drive_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    l2_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    l2_orientation: np.ndarray = field(default_factory=lambda: np.zeros(0))
    l2_point: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    l3_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    build_stats: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @staticmethod
    def build(config: dict[str, Any], seed: int, init_rng: np.random.Generator) -> "V1Circuit":
        """설정으로 회로를 만든다. ``init_rng`` 는 초기 가중치 전용 스트림."""
        img_cfg = config["image"]
        H, W = int(img_cfg["height"]), int(img_cfg["width"])
        gcfg = GridConfig(**config["grid"])
        grid = build_grid(H, W, gcfg)
        sampler = LogPolarSampler(grid, method=config["grid_sampling"]["method"])
        retina = RetinaEncoder(RetinaConfig(**config["retina"]))
        normalizer = InputNormalizer(config["normalizer"]["percentile"])

        ccfg = config["circuit"]
        orientations = [float(o) for o in ccfg["orientations_deg"]]
        k_near = int(ccfg["l2_input_k_nearest"])
        k_pool = int(ccfg["l3_pool_k"])
        P = grid.n_points

        pop = Population()

        # --- 뉴런 선언 (등록 순서는 결과에 영향을 주지 않아야 한다) ---------
        for p in range(P):
            k, j = int(grid.radial_bin[p]), int(grid.angular_bin[p])
            for pol in ("ON", "OFF"):
                pop.declare(
                    f"L4/{pol}/k{k}/j{j}", layer="L4", cell_type="excitatory",
                    threshold=float(ccfg["l4_threshold"]), transmission=1.0,
                    observation_tick=int(ccfg["observe_ticks"]["L4"]),
                )
                pop.declare(
                    f"INH/{pol}/k{k}/j{j}", layer="INH", cell_type="inhibitory",
                    threshold=float(ccfg["inh_relay_threshold"]),
                    transmission=float(ccfg["inh_relay_P"]),
                    observation_tick=int(ccfg["observe_ticks"]["INH"]),
                )
        for p in range(P):
            k, j = int(grid.radial_bin[p]), int(grid.angular_bin[p])
            for ori in orientations:
                pop.declare(
                    f"L2/ori{int(ori)}/k{k}/j{j}", layer="L2", cell_type="excitatory",
                    threshold=float(ccfg["l2_threshold"]), transmission=1.0,
                    observation_tick=int(ccfg["observe_ticks"]["L2"]),
                    preferred_orientation_deg=ori,
                )
                pop.declare(
                    f"L3/ori{int(ori)}/k{k}/j{j}", layer="L3", cell_type="excitatory",
                    threshold=float(ccfg["l3_threshold"]), transmission=1.0,
                    observation_tick=int(ccfg["observe_ticks"]["L3"]),
                    preferred_orientation_deg=ori,
                )
        pop.finalize()

        # --- 메타데이터: 수용장과 모형 피질 좌표 -----------------------------
        ang_span = gcfg.angular_bins
        for p in range(P):
            k, j = int(grid.radial_bin[p]), int(grid.angular_bin[p])
            rf = ReceptiveField(
                center_x=float(grid.x[p]),
                center_y=float(grid.y[p]),
                radius_px=float(grid.rf_radius[p]),
                sigma_pool_px=float(grid.sigma_pool[p]),
                eccentricity_px=float(grid.r_k[p]),
            )
            for prefix, layer in (("L4/ON", "L4"), ("L4/OFF", "L4"),
                                  ("INH/ON", "INH"), ("INH/OFF", "INH")):
                n = pop.by_name(f"{prefix}/k{k}/j{j}")
                n.receptive_field = ReceptiveField(**rf.to_dict())
                n.set_position(float(j) / ang_span * 10.0, float(k), LAYER_DEPTH_Z[layer])
            for ori in orientations:
                for layer in ("L2", "L3"):
                    n = pop.by_name(f"{layer}/ori{int(ori)}/k{k}/j{j}")
                    n.receptive_field = ReceptiveField(**rf.to_dict())
                    n.set_position(float(j) / ang_span * 10.0, float(k), LAYER_DEPTH_Z[layer])

        # --- 이웃 계산 (원본 영상 좌표 기준) ---------------------------------
        pts = np.stack([grid.x, grid.y], axis=1)               # (P,2)
        d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)  # (P,P)
        # 동률에서도 결정적이도록 (거리, 인덱스) 사전식 정렬
        order = np.lexsort((np.tile(np.arange(P), (P, 1)), d2), axis=1)
        neighbors = order[:, :k_near]                           # (P, k_near)

        # --- 연결 ------------------------------------------------------------
        table = ConnectionTable(len(pop), pop.names())
        w_lo, w_hi = float(ccfg["init_weight_low"]), float(ccfg["init_weight_high"])
        d_direct = int(ccfg["delay_l4_to_l2"])
        d_relay_in = int(ccfg["delay_l4_to_inh"])
        d_relay_out = int(ccfg["delay_inh_to_l2"])

        n_trainable = 0
        for p in range(P):
            k, j = int(grid.radial_bin[p]), int(grid.angular_bin[p])
            for pol in ("ON", "OFF"):
                l4 = pop.index(f"L4/{pol}/k{k}/j{j}")
                inh = pop.index(f"INH/{pol}/k{k}/j{j}")
                table.add(l4, inh, "excitatory", 1.0, d_relay_in, trainable=False)

        for p in range(P):
            kp, jp = int(grid.radial_bin[p]), int(grid.angular_bin[p])
            for ori in orientations:
                l2 = pop.index(f"L2/ori{int(ori)}/k{kp}/j{jp}")
                for q in neighbors[p]:
                    kq, jq = int(grid.radial_bin[q]), int(grid.angular_bin[q])
                    for pol in ("ON", "OFF"):
                        l4 = pop.index(f"L4/{pol}/k{kq}/j{jq}")
                        inh = pop.index(f"INH/{pol}/k{kq}/j{jq}")
                        w1 = float(init_rng.uniform(w_lo, w_hi))
                        w2 = float(init_rng.uniform(w_lo, w_hi))
                        table.add(l4, l2, "excitatory", w1, d_direct, trainable=True)
                        table.add(inh, l2, "inhibitory", w2, d_relay_out, trainable=True)
                        n_trainable += 2

        # L2 -> L3 (같은 방향의 가까운 L2 k_pool 개, 고정 1/k_pool)
        for p in range(P):
            kp, jp = int(grid.radial_bin[p]), int(grid.angular_bin[p])
            pool = neighbors[p][:k_pool]
            for ori in orientations:
                l3 = pop.index(f"L3/ori{int(ori)}/k{kp}/j{jp}")
                for q in pool:
                    kq, jq = int(grid.radial_bin[q]), int(grid.angular_bin[q])
                    l2 = pop.index(f"L2/ori{int(ori)}/k{kq}/j{jq}")
                    table.add(l2, l3, "excitatory", 1.0 / len(pool),
                              int(ccfg["delay_l2_to_l3"]), trainable=False)
        table.finalize()

        for cid in range(table.n_connections):
            pop.neurons[int(table.pre_id[cid])].outgoing_connection_ids.append(cid)
            pop.neurons[int(table.post_id[cid])].incoming_connection_ids.append(cid)

        engine = SimulationEngine(
            pop, table,
            n_ticks=int(ccfg["n_ticks"]),
            event_mode=config["engine"]["event_mode"],
        )

        l4_on = np.array([pop.index(f"L4/ON/k{int(grid.radial_bin[p])}/j{int(grid.angular_bin[p])}")
                          for p in range(P)], dtype=np.int64)
        l4_off = np.array([pop.index(f"L4/OFF/k{int(grid.radial_bin[p])}/j{int(grid.angular_bin[p])}")
                           for p in range(P)], dtype=np.int64)
        l2_ids, l2_ori, l2_pt = [], [], []
        l3_ids = []
        for p in range(P):
            k, j = int(grid.radial_bin[p]), int(grid.angular_bin[p])
            for ori in orientations:
                l2_ids.append(pop.index(f"L2/ori{int(ori)}/k{k}/j{j}"))
                l2_ori.append(ori)
                l2_pt.append(p)
                l3_ids.append(pop.index(f"L3/ori{int(ori)}/k{k}/j{j}"))

        circuit = V1Circuit(
            population=pop, table=table, engine=engine, grid=grid, sampler=sampler,
            retina=retina, normalizer=normalizer, config=config,
            l4_on_ids=l4_on, l4_off_ids=l4_off,
            l4_drive_ids=np.concatenate([l4_on, l4_off]),
            l2_ids=np.array(l2_ids, dtype=np.int64),
            l2_orientation=np.array(l2_ori, dtype=np.float64),
            l2_point=np.array(l2_pt, dtype=np.int64),
            l3_ids=np.array(l3_ids, dtype=np.int64),
            build_stats={
                "n_neurons": len(pop),
                "n_connections": table.n_connections,
                "n_trainable_connections": int(table.trainable.sum()),
                "n_grid_points": P,
                "l2_input_k_nearest": k_near,
                "l3_pool_k": k_pool,
                "orientations_deg": orientations,
                "layer_depth_z": LAYER_DEPTH_Z,
            },
        )
        return circuit

    # ------------------------------------------------------------------
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """이미지 -> **정규화 전** 격자 ON/OFF 표본, shape (2P,) [ON..., OFF...]."""
        ch = self.retina.encode(image, input_colorspace=self.config["image"]["colorspace"])
        s = self.sampler.sample(ch)
        return np.concatenate([s["on"], s["off"]])

    def drive_from_image(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """이미지 -> (L4 뉴런 id 배열, 정규화된 감각값 배열). 학습 라벨을 쓰지 않는다."""
        raw = self.preprocess(image)
        return self.l4_drive_ids, self.normalizer.transform(raw)

    def drive_from_raw(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.l4_drive_ids, self.normalizer.transform(raw)

    def trainable_mask(self) -> np.ndarray:
        return self.table.trainable
