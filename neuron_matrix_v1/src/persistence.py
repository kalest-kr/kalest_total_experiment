"""persistence.py -- 모델 저장·복원.

명세 [12] T9: 가중치뿐 아니라 **연결 구조, 뉴런 고정값, 전처리 계수,
수용장 설정**을 저장한다. 복원 후 교사 없는 예측이 일치해야 한다.

저장 형식: ``{prefix}.json`` (설정/메타데이터) + ``{prefix}.npz`` (배열).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import numpy as np

from .connections import ConnectionTable
from .neuron import ReceptiveField
from .spatial_mapping import InputNormalizer
from .v1 import V1Circuit
from .rngs import stream


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_model(prefix: str, circuit: V1Circuit, extra: dict[str, Any] | None = None) -> dict[str, str]:
    """회로 전체를 저장한다 (부작용: 파일 2개 생성).

    Returns
    -------
    dict : {"json": path, "npz": path, "json_sha256": ..., "npz_sha256": ...}
    """
    os.makedirs(os.path.dirname(os.path.abspath(prefix)) or ".", exist_ok=True)
    pop = circuit.population
    tb = circuit.table
    meta = {
        "config": circuit.config,
        "build_stats": circuit.build_stats,
        "grid": circuit.grid.to_dict(),
        "normalizer": circuit.normalizer.to_dict(),
        "weight_version": tb.weight_version,
        "neuron_meta": [
            {
                "neuron_id": n.neuron_id,
                "name": n.name,
                "area": n.area,
                "layer": n.layer,
                "cell_type": n.cell_type,
                "preferred_orientation_deg": n.preferred_orientation_deg,
                "receptive_field": n.receptive_field.to_dict() if n.receptive_field else None,
                "position_space": n.position_space,
                "observation_tick": n.observation_tick,
            }
            for n in pop.neurons
        ],
        "extra": extra or {},
    }
    json_path = f"{prefix}.json"
    npz_path = f"{prefix}.npz"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    np.savez_compressed(
        npz_path,
        states=pop.store.states,
        b_base=pop.store.b_base,
        observation_tick=pop.store.observation_tick,
        pre_id=tb.pre_id, post_id=tb.post_id, kind=tb.kind,
        weight=tb.weight, delay_ticks=tb.delay_ticks, trainable=tb.trainable,
        l4_on_ids=circuit.l4_on_ids, l4_off_ids=circuit.l4_off_ids,
        l2_ids=circuit.l2_ids, l2_orientation=circuit.l2_orientation,
        l2_point=circuit.l2_point, l3_ids=circuit.l3_ids,
    )
    return {
        "json": json_path, "npz": npz_path,
        "json_sha256": file_sha256(json_path),
        "npz_sha256": file_sha256(npz_path),
    }


def load_model(prefix: str) -> V1Circuit:
    """저장된 모델을 복원한다. 구조·가중치·전처리 계수·수용장 설정을 모두 되살린다."""
    with open(f"{prefix}.json", encoding="utf-8") as f:
        meta = json.load(f)
    data = np.load(f"{prefix}.npz", allow_pickle=False)
    cfg = meta["config"]
    # 구조를 동일 설정으로 재구성한 뒤 저장 배열로 덮어쓴다.
    circuit = V1Circuit.build(cfg, 0, stream(0, "init"))
    pop = circuit.population
    pop.store.states[...] = data["states"]
    pop.store.b_base[...] = data["b_base"]
    pop.store.observation_tick[...] = data["observation_tick"]

    tb = circuit.table
    if not np.array_equal(tb.pre_id, data["pre_id"]) or not np.array_equal(
        tb.post_id, data["post_id"]
    ) or not np.array_equal(tb.kind, data["kind"]) or not np.array_equal(
        tb.delay_ticks, data["delay_ticks"]
    ) or not np.array_equal(tb.trainable, data["trainable"]):
        raise ValueError("저장된 연결 구조가 설정으로 재구성한 구조와 다르다")
    tb.restore_weights(data["weight"], int(meta["weight_version"]))

    circuit.normalizer = InputNormalizer.from_dict(meta["normalizer"])
    for nm in meta["neuron_meta"]:
        n = pop.neurons[nm["neuron_id"]]
        if n.name != nm["name"]:
            raise ValueError("복원한 뉴런 이름 순서가 다르다")
        n.preferred_orientation_deg = nm["preferred_orientation_deg"]
        n.receptive_field = (
            ReceptiveField.from_dict(nm["receptive_field"]) if nm["receptive_field"] else None
        )
        n.area, n.layer, n.cell_type = nm["area"], nm["layer"], nm["cell_type"]
        n.position_space = nm["position_space"]
    # 엔진의 초기 스냅샷을 복원된 상태로 갱신한다.
    circuit.engine._initial_states = pop.store.states.copy()
    circuit.engine._initial_b_base = pop.store.b_base.copy()
    circuit.engine._initial_weights = tb.snapshot_weights()
    circuit.engine._initial_weight_version = tb.weight_version
    circuit.engine.reset_transient_state()
    return circuit


def hash_manifest(paths: list[str], root: str = ".") -> list[dict[str, Any]]:
    """코드·설정·결과 파일의 해시 목록 (명세 [16]-7)."""
    out = []
    for p in sorted(paths):
        if not os.path.isfile(p):
            continue
        out.append({
            "path": os.path.relpath(p, root),
            "bytes": os.path.getsize(p),
            "sha256": file_sha256(p),
        })
    return out
