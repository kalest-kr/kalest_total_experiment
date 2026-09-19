"""predictive_coding.py -- Rao & Ballard 계열 연속값 예측 부호화 **참조 모델**.

명세 8절. 이것은 스파이크/전도도 회로와 **다른 엔진**이며, 수학적 기준을
제공하기 위한 것이다. 두 모델을 같은 물리 모델이라고 주장하지 않는다.

목적함수 (여러 모듈이 상위 표현 r2 를 공유하는 구성)::

    E = sum_m 0.5/sigma^2   * ||I_m - U1_m r1_m||^2
      + sum_m 0.5/sigma_td^2* ||r1_m - U2_m r2||^2
      + 0.5*alpha * ( sum_m ||r1_m||^2 + ||r2||^2 )
      + 0.5*lambda* ( sum_m ||U1_m||^2 + sum_m ||U2_m||^2 )

여기서 유도되는 갱신 (모두 같은 전체 목적함수에서 나온다)::

    dr1_m = U1_m.T @ (I_m - U1_m r1_m)/sigma^2 + (U2_m r2 - r1_m)/sigma_td^2 - alpha*r1_m
    dr2   = sum_m U2_m.T @ (r1_m - U2_m r2)/sigma_td^2 - alpha*r2
    dU1_m = outer(I_m - U1_m r1_m, r1_m)/sigma^2      - lambda*U1_m
    dU2_m = outer(r1_m - U2_m r2,  r2)/sigma_td^2     - lambda*U2_m

모듈이 1개면 명세에 적힌 식과 정확히 같다 (``r_td = U2 r2``).

**정직한 표기**

* 이것은 국소 계산으로 표현한 **기울기하강 모델**이며 ``U`` 의 전치를 쓴다.
  "미분도 대칭 가중치도 없는 학습"이 아니다.
* 상위 예측 ``r_td`` 는 완벽한 정답이 아니다. 영역 간 표현 오차와 영상 재구성
  오차를 **다른 변수**로 보관한다 (:class:`RaoState.errors`).
* 정착(settling) 중에는 ``U`` 를 동결하고, 모든 모듈의 **이전 상태**로 오차를
  계산한 뒤 활동을 동기적으로 갱신한다.
* 두 분산 계수는 양수여야 한다 (생성자에서 검증).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class RaoState:
    """정착 결과와 오차 변수들."""

    r1: np.ndarray                     # (M, n1)
    r2: np.ndarray                     # (n2,)
    energy_trace: list[float] = field(default_factory=list)
    errors: dict[str, Any] = field(default_factory=dict)
    settled_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "r1_shape": list(self.r1.shape), "r2_shape": list(self.r2.shape),
            "energy_trace": [float(x) for x in self.energy_trace],
            "errors": self.errors, "settled_steps": self.settled_steps,
        }


class RaoModel:
    """연속값 예측 부호화 참조 엔진.

    Parameters
    ----------
    n_modules : int
        하위(level-1) 모듈 수. 모두 같은 상위 표현 ``r2`` 를 공유한다.
    input_dim : int
        모듈 하나가 받는 입력 차원.
    n1, n2 : int
        level-1 / level-2 표현 차원.
    sigma, sigma_td : float
        관측/상위예측 분산 계수. **양수여야 한다.**
    alpha, lam : float
        활동/가중치 정규화 계수 (0 이상).
    """

    def __init__(self, n_modules: int, input_dim: int, n1: int, n2: int,
                 sigma: float, sigma_td: float, alpha: float, lam: float,
                 rng: np.random.Generator) -> None:
        if not (sigma > 0.0):
            raise ValueError(f"sigma 는 양수여야 한다: {sigma}")
        if not (sigma_td > 0.0):
            raise ValueError(f"sigma_td 는 양수여야 한다: {sigma_td}")
        if alpha < 0.0 or lam < 0.0:
            raise ValueError("alpha 와 lambda 는 0 이상이어야 한다")
        self.M = int(n_modules)
        self.input_dim = int(input_dim)
        self.n1 = int(n1)
        self.n2 = int(n2)
        self.sigma2 = float(sigma) ** 2
        self.sigma_td2 = float(sigma_td) ** 2
        self.alpha = float(alpha)
        self.lam = float(lam)
        scale = 1.0 / np.sqrt(max(1, self.n1))
        self.U1 = rng.normal(0.0, scale, size=(self.M, self.input_dim, self.n1))
        self.U2 = rng.normal(0.0, 1.0 / np.sqrt(max(1, self.n2)),
                             size=(self.M, self.n1, self.n2))

    # ------------------------------------------------------------------
    def energy(self, I: np.ndarray, r1: np.ndarray, r2: np.ndarray) -> float:
        """전체 목적함수 E 의 값 (스칼라)."""
        I = np.asarray(I, dtype=np.float64)
        rec = I - np.einsum("mij,mj->mi", self.U1, r1)
        td = r1 - np.einsum("mij,j->mi", self.U2, r2)
        return float(
            0.5 / self.sigma2 * np.sum(rec ** 2)
            + 0.5 / self.sigma_td2 * np.sum(td ** 2)
            + 0.5 * self.alpha * (np.sum(r1 ** 2) + np.sum(r2 ** 2))
            + 0.5 * self.lam * (np.sum(self.U1 ** 2) + np.sum(self.U2 ** 2))
        )

    def gradients_r(self, I: np.ndarray, r1: np.ndarray,
                    r2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``dr1, dr2`` (= -dE/dr). 모든 오차는 **이전 상태**로 계산한다."""
        rec = np.asarray(I, dtype=np.float64) - np.einsum("mij,mj->mi", self.U1, r1)
        pred = np.einsum("mij,j->mi", self.U2, r2)
        td = r1 - pred
        dr1 = (np.einsum("mij,mi->mj", self.U1, rec) / self.sigma2
               - td / self.sigma_td2 - self.alpha * r1)
        dr2 = (np.einsum("mij,mi->j", self.U2, td) / self.sigma_td2
               - self.alpha * r2)
        return dr1, dr2

    def gradients_U(self, I: np.ndarray, r1: np.ndarray,
                    r2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``dU1, dU2`` (= -dE/dU)."""
        rec = np.asarray(I, dtype=np.float64) - np.einsum("mij,mj->mi", self.U1, r1)
        td = r1 - np.einsum("mij,j->mi", self.U2, r2)
        dU1 = np.einsum("mi,mj->mij", rec, r1) / self.sigma2 - self.lam * self.U1
        dU2 = np.einsum("mi,j->mij", td, r2) / self.sigma_td2 - self.lam * self.U2
        return dU1, dU2

    # ------------------------------------------------------------------
    def settle(self, I: np.ndarray, *, steps: int, r_step: float,
               r1_init: np.ndarray | None = None, r2_init: np.ndarray | None = None,
               freeze_U: bool = True,
               on_step: Callable[[int, float], None] | None = None) -> RaoState:
        """활동을 정착시킨다. ``freeze_U=True`` 면 U 를 동결한다.

        갱신은 **동기적**이다: 모든 모듈의 이전 상태로 오차를 계산한 뒤 한 번에
        r1, r2 를 갱신한다.
        """
        I = np.asarray(I, dtype=np.float64)
        if I.shape != (self.M, self.input_dim):
            raise ValueError(f"입력 shape 는 ({self.M},{self.input_dim}) 여야 한다: {I.shape}")
        r1 = np.zeros((self.M, self.n1)) if r1_init is None else np.array(r1_init, float)
        r2 = np.zeros(self.n2) if r2_init is None else np.array(r2_init, float)
        if not freeze_U:
            raise ValueError(
                "정착 중 U 는 동결해야 한다 (명세 8절). U 갱신은 learn_step 에서 한다."
            )
        trace: list[float] = [self.energy(I, r1, r2)]
        for s in range(int(steps)):
            dr1, dr2 = self.gradients_r(I, r1, r2)
            r1 = r1 + float(r_step) * dr1
            r2 = r2 + float(r_step) * dr2
            e = self.energy(I, r1, r2)
            trace.append(e)
            if on_step is not None:
                on_step(s, e)

        rec = I - np.einsum("mij,mj->mi", self.U1, r1)
        td = r1 - np.einsum("mij,j->mi", self.U2, r2)
        return RaoState(
            r1=r1, r2=r2, energy_trace=trace, settled_steps=int(steps),
            errors={
                # 영상 재구성 오차와 영역 간 표현 오차를 **다른 변수**로 둔다
                "image_reconstruction_error_l2": float(np.linalg.norm(rec)),
                "image_reconstruction_error_per_module":
                    [float(np.linalg.norm(rec[m])) for m in range(self.M)],
                "interarea_representation_error_l2": float(np.linalg.norm(td)),
                "interarea_representation_error_per_module":
                    [float(np.linalg.norm(td[m])) for m in range(self.M)],
                "energy_first": float(trace[0]), "energy_last": float(trace[-1]),
                "energy_decreased": bool(trace[-1] <= trace[0]),
                "note_ko": "상위 예측 r_td 는 완벽한 정답이 아니다.",
            },
        )

    def learn_step(self, I: np.ndarray, state: RaoState, u_step: float) -> dict[str, Any]:
        """정착된 활동에서 U 를 한 스텝 갱신한다 (부작용: U 변경)."""
        dU1, dU2 = self.gradients_U(I, state.r1, state.r2)
        before = float(np.sum(self.U1 ** 2) + np.sum(self.U2 ** 2))
        self.U1 += float(u_step) * dU1
        self.U2 += float(u_step) * dU2
        return {
            "u_step": float(u_step),
            "dU1_abs_sum": float(np.abs(dU1).sum()),
            "dU2_abs_sum": float(np.abs(dU2).sum()),
            "U_sq_before": before,
            "U_sq_after": float(np.sum(self.U1 ** 2) + np.sum(self.U2 ** 2)),
        }

    # ------------------------------------------------------------------
    def finite_difference_check(self, I: np.ndarray, r1: np.ndarray, r2: np.ndarray,
                                eps: float = 1e-6, n_probe: int = 8,
                                rng: np.random.Generator | None = None
                                ) -> dict[str, Any]:
        """해석적 갱신 방향과 유한차분 기울기를 비교한다 (검증 12번).

        ``dr = -dE/dr`` 이므로 ``dr + numeric_grad ~ 0`` 이어야 한다.
        스파이크 모델의 참 기울기를 유한차분으로 검증했다고 주장하지 않는다.
        이 검사는 **연속값 Rao 모델에만** 해당한다.
        """
        rng = rng or np.random.default_rng(0)
        I = np.asarray(I, dtype=np.float64)
        dr1, dr2 = self.gradients_r(I, r1, r2)
        dU1, dU2 = self.gradients_U(I, r1, r2)

        def probe(shape: tuple[int, ...], n: int) -> list[tuple[int, ...]]:
            total = int(np.prod(shape))
            n = min(n, total)
            flat = rng.choice(total, size=n, replace=False)
            return [tuple(np.unravel_index(int(f), shape)) for f in flat]

        results: dict[str, list[dict[str, float]]] = {"r1": [], "r2": [], "U1": [], "U2": []}
        for idx in probe(r1.shape, n_probe):
            rp, rm = r1.copy(), r1.copy()
            rp[idx] += eps
            rm[idx] -= eps
            num = (self.energy(I, rp, r2) - self.energy(I, rm, r2)) / (2 * eps)
            results["r1"].append({"analytic_minus_grad": float(dr1[idx]),
                                  "numeric_grad": float(num),
                                  "residual": float(dr1[idx] + num)})
        for idx in probe(r2.shape, n_probe):
            rp, rm = r2.copy(), r2.copy()
            rp[idx] += eps
            rm[idx] -= eps
            num = (self.energy(I, r1, rp) - self.energy(I, r1, rm)) / (2 * eps)
            results["r2"].append({"analytic_minus_grad": float(dr2[idx]),
                                  "numeric_grad": float(num),
                                  "residual": float(dr2[idx] + num)})
        for name, arr, grad in (("U1", self.U1, dU1), ("U2", self.U2, dU2)):
            for idx in probe(arr.shape, n_probe):
                orig = arr[idx]
                arr[idx] = orig + eps
                ep = self.energy(I, r1, r2)
                arr[idx] = orig - eps
                em = self.energy(I, r1, r2)
                arr[idx] = orig
                num = (ep - em) / (2 * eps)
                results[name].append({"analytic_minus_grad": float(grad[idx]),
                                      "numeric_grad": float(num),
                                      "residual": float(grad[idx] + num)})

        max_res = max((abs(x["residual"]) for vals in results.values() for x in vals),
                      default=0.0)
        scale = max((abs(x["numeric_grad"]) for vals in results.values() for x in vals),
                    default=1.0)
        return {
            "eps": float(eps), "n_probe_per_tensor": int(n_probe),
            "max_abs_residual": float(max_res),
            "max_abs_numeric_grad": float(scale),
            "max_relative_residual": float(max_res / max(scale, 1e-30)),
            "details": results,
            "scope_note_ko": ("연속값 Rao 모델의 목적함수 미분만 검증한다. "
                              "불연속 스파이크의 참 기울기를 유한차분이나 surrogate 로 "
                              "검증했다고 주장하지 않는다."),
        }

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"U1": self.U1.copy(), "U2": self.U2.copy()}

    def load_state_dict(self, d: dict[str, np.ndarray]) -> None:
        self.U1 = np.asarray(d["U1"], dtype=np.float64)
        self.U2 = np.asarray(d["U2"], dtype=np.float64)

    def summary(self) -> dict[str, Any]:
        return {
            "engine": "rao_reference_continuous",
            "n_modules": self.M, "input_dim": self.input_dim,
            "n1": self.n1, "n2": self.n2,
            "sigma2": self.sigma2, "sigma_td2": self.sigma_td2,
            "alpha": self.alpha, "lambda": self.lam,
            "note_ko": ("전도도 LIF 회로와 같은 물리 모델이 아니다. "
                        "동일 아키텍처 대조군인 것처럼 순위를 매기지 않는다."),
        }


def apical_error_coupling(errors: np.ndarray, cfg: dict[str, Any]
                          ) -> dict[str, Any]:
    """Rao 오차를 apical 전류로 바꾸는 **선택 기능**의 가정 명세.

    기본값은 꺼져 있다. 켜는 경우 아래 가정을 반드시 함께 보고해야 한다.

    * 단위 변환: 무차원 오차 -> pA. 계수 ``gain_pA_per_unit`` 은 **모형 파라미터**.
    * 양/음 오차 부호화: ``split_sign`` 이면 양수부와 음수부를 서로 다른 뉴런
      집단에 넣는다 (전류 부호를 그대로 쓰지 않는다).
    * 좌표 대응: Rao 모듈 인덱스 -> 피질 뉴런 인덱스 사상을 명시해야 한다.
    * 연결 학습: 이 경로의 시냅스가 학습되는지 고정인지 명시해야 한다.

    임의 전류 주입이 Rao 식을 재현한다고 가정하지 않는다.
    """
    c = cfg["learning"]["apical_error_coupling"]
    if not c["enabled"]:
        return {"enabled": False,
                "note_ko": "apical 오차 결합은 꺼져 있다 (기본값)."}
    e = np.asarray(errors, dtype=np.float64)
    gain = float(c["gain_pA_per_unit"])
    if c["split_sign"]:
        pos, neg = np.maximum(e, 0.0) * gain, np.maximum(-e, 0.0) * gain
    else:
        pos, neg = e * gain, np.zeros_like(e)
    return {
        "enabled": True, "gain_pA_per_unit": gain, "split_sign": bool(c["split_sign"]),
        "current_positive_pA": pos, "current_negative_pA": neg,
        "assumptions_ko": [
            "무차원 Rao 오차를 pA 로 바꾸는 계수는 모형 파라미터이며 측정값이 아니다.",
            "양/음 오차를 별도 집단으로 부호화했다 (전류 부호를 그대로 쓰지 않음).",
            "Rao 모듈 <-> 피질 뉴런 좌표 대응은 실험 설정에서 명시해야 한다.",
            "이 전류 주입이 Rao 식을 재현한다고 가정하지 않는다.",
        ],
    }


def make(cfg: dict[str, Any], input_dim: int, rng: np.random.Generator) -> RaoModel:
    r = cfg["learning"]["rao"]
    sizes = list(r["level_sizes"])
    n1 = int(sizes[0])
    n2 = int(sizes[1]) if len(sizes) > 1 else int(sizes[0])
    return RaoModel(n_modules=1, input_dim=int(input_dim), n1=n1, n2=n2,
                    sigma=float(r["sigma"]), sigma_td=float(r["sigma_td"]),
                    alpha=float(r["alpha"]), lam=float(r["lambda_u"]), rng=rng)


__all__ = ["RaoState", "RaoModel", "apical_error_coupling", "make"]
