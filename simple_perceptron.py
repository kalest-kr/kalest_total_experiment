#!/usr/bin/env python3
"""simple_perceptron.py -- 입력 2개, 출력 1개짜리 단순 퍼셉트론.

순전파, 역전파, 가중치 갱신에서 일어나는 **모든 연산**을 한 줄씩 출력한다.
입력 노드 두 개의 값(x1, x2)과 정답 레이블(t)은 사용자가 지정한다.

구조::

    x1 --(w1)--+
               +--> z = w1*x1 + w2*x2 + b --> y = sigmoid(z) --> E = 1/2 * (t - y)^2
    x2 --(w2)--+          (b: 편향)

    역전파(연쇄법칙):  dE/dw = dE/dy * dy/dz * dz/dw
    가중치 갱신:       w <- w - lr * dE/dw

사용법::

    # 1) 대화형: 실행하면 x1, x2, 정답 레이블을 차례로 물어본다
    python simple_perceptron.py

    # 2) 명령행: --sample x1 x2 정답  (여러 번 쓰면 여러 샘플)
    python simple_perceptron.py --sample 1 0 1
    python simple_perceptron.py --sample 0 0 0 --sample 0 1 0 \\
        --sample 1 0 0 --sample 1 1 1 --epochs 2 --lr 0.5

    # 초기 가중치/편향, 출력 자릿수도 지정할 수 있다
    python simple_perceptron.py --sample 1 1 1 --w1 0.5 --w2 -0.5 --bias 0 --digits 4

정답 레이블은 sigmoid 출력 범위와 같은 0 ~ 1 사이 값이어야 한다.
외부 라이브러리 없이 표준 라이브러리(argparse, math, os, sys)만 사용한다.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

DEFAULT_W1 = 0.3
DEFAULT_W2 = -0.2
DEFAULT_BIAS = 0.1
DEFAULT_LR = 0.5
DEFAULT_EPOCHS = 1
DEFAULT_DIGITS = 6
THRESHOLD = 0.5  # y >= 0.5 이면 클래스 1 로 판정

LINE = "=" * 72
SUBLINE = "-" * 72


def fmt(v: float, digits: int) -> str:
    """고정 소수점 문자열. 반올림 결과가 0 이면 '-0.000' 같은 부호를 떼어 낸다."""
    s = f"{v:.{digits}f}"
    if float(s) == 0.0:
        s = s.lstrip("-")
    return s


class SimplePerceptron:
    """입력 2개 -> 출력 1개. 활성화 sigmoid, 오차 E = 1/2 * (t - y)^2."""

    def __init__(self, w1: float, w2: float, b: float, lr: float, digits: int):
        self.w1 = w1
        self.w2 = w2
        self.b = b
        self.lr = lr
        self.digits = digits

    # ---- 출력 도우미 -------------------------------------------------
    def v(self, x: float) -> str:
        """값 그대로."""
        return fmt(x, self.digits)

    def p(self, x: float) -> str:
        """식 안에 넣을 값. 음수면 괄호로 감싼다."""
        s = fmt(x, self.digits)
        return f"({s})" if s.startswith("-") else s

    def params(self) -> str:
        return f"w1 = {self.v(self.w1)}, w2 = {self.v(self.w2)}, b = {self.v(self.b)}"

    # ---- 순전파 ------------------------------------------------------
    def forward(self, x1: float, x2: float, t: float) -> tuple[float, float, float]:
        """순전파. 모든 중간 연산을 출력하고 (z, y, E) 를 돌려준다."""
        v, p = self.v, self.p
        print("  [순전파]")

        print("   (1) 가중합  z = w1*x1 + w2*x2 + b")
        w1x1 = self.w1 * x1
        print(f"       w1*x1 = {p(self.w1)} * {p(x1)} = {v(w1x1)}")
        w2x2 = self.w2 * x2
        print(f"       w2*x2 = {p(self.w2)} * {p(x2)} = {v(w2x2)}")
        z = w1x1 + w2x2 + self.b
        print(f"       z = {p(w1x1)} + {p(w2x2)} + {p(self.b)} = {v(z)}")

        print("   (2) 활성화  y = sigmoid(z) = 1 / (1 + e^(-z))")
        if z >= -700.0:
            neg_z = -z
            print(f"       -z = {v(neg_z)}")
            e = math.exp(neg_z)
            print(f"       e^(-z) = e^({v(neg_z)}) = {v(e)}")
            denom = 1.0 + e
            print(f"       1 + e^(-z) = 1 + {p(e)} = {v(denom)}")
            y = 1.0 / denom
            print(f"       y = 1 / {p(denom)} = {v(y)}")
        else:
            # e^(-z) 가 float 범위를 넘으므로 같은 값인 e^z / (1 + e^z) 로 계산한다.
            print("       (z 가 매우 작아 e^(-z) 가 넘치므로 같은 식 y = e^z / (1 + e^z) 로 계산)")
            e = math.exp(z)
            print(f"       e^z = e^({v(z)}) = {v(e)}")
            denom = 1.0 + e
            print(f"       1 + e^z = 1 + {p(e)} = {v(denom)}")
            y = e / denom
            print(f"       y = {p(e)} / {p(denom)} = {v(y)}")

        print("   (3) 오차  E = 1/2 * (t - y)^2")
        diff = t - y
        print(f"       t - y = {p(t)} - {p(y)} = {v(diff)}")
        sq = diff * diff
        print(f"       (t - y)^2 = {p(diff)} * {p(diff)} = {v(sq)}")
        err = 0.5 * sq
        print(f"       E = 0.5 * {p(sq)} = {v(err)}")

        pred = 1 if y >= THRESHOLD else 0
        sign = ">=" if pred == 1 else "<"
        verdict = ""
        if t in (0.0, 1.0):
            verdict = f"  (정답 {int(t)} -> {'맞음' if pred == int(t) else '틀림'})"
        print(f"   (4) 판정  y = {v(y)} {sign} {THRESHOLD} -> 예측 클래스 {pred}{verdict}")
        return z, y, err

    # ---- 역전파 ------------------------------------------------------
    def backward(self, x1: float, x2: float, t: float, y: float) -> tuple[float, float, float]:
        """역전파. 연쇄법칙의 각 항을 출력하고 (dE/dw1, dE/dw2, dE/db) 를 돌려준다."""
        v, p = self.v, self.p
        print("  [역전파]  연쇄법칙: dE/dw = dE/dy * dy/dz * dz/dw")

        print("   (1) dE/dy = -(t - y) = y - t")
        dE_dy = y - t
        print(f"       dE/dy = {p(y)} - {p(t)} = {v(dE_dy)}")

        print("   (2) dy/dz = sigmoid'(z) = y * (1 - y)")
        one_minus_y = 1.0 - y
        print(f"       1 - y = 1 - {p(y)} = {v(one_minus_y)}")
        dy_dz = y * one_minus_y
        print(f"       y * (1 - y) = {p(y)} * {p(one_minus_y)} = {v(dy_dz)}")

        print("   (3) delta = dE/dz = dE/dy * dy/dz")
        delta = dE_dy * dy_dz
        print(f"       delta = {p(dE_dy)} * {p(dy_dz)} = {v(delta)}")

        print("   (4) 파라미터별 기울기  (dz/dw1 = x1, dz/dw2 = x2, dz/db = 1)")
        g_w1 = delta * x1
        print(f"       dE/dw1 = delta * x1 = {p(delta)} * {p(x1)} = {v(g_w1)}")
        g_w2 = delta * x2
        print(f"       dE/dw2 = delta * x2 = {p(delta)} * {p(x2)} = {v(g_w2)}")
        g_b = delta * 1.0
        print(f"       dE/db  = delta * 1  = {p(delta)} * 1 = {v(g_b)}")
        return g_w1, g_w2, g_b

    # ---- 가중치 갱신 -------------------------------------------------
    def update(self, g_w1: float, g_w2: float, g_b: float) -> None:
        """경사하강법 w <- w - lr * dE/dw. 모든 연산을 출력한다."""
        print(f"  [가중치 갱신]  w <- w - lr * dE/dw   (lr = {self.v(self.lr)})")
        self.w1 = self._step("w1", self.w1, g_w1)
        self.w2 = self._step("w2", self.w2, g_w2)
        self.b = self._step("b", self.b, g_b)

    def _step(self, name: str, w: float, g: float) -> float:
        v, p = self.v, self.p
        step = self.lr * g
        new = w - step
        label = name.ljust(2)
        print(f"       {label}: lr * dE/d{label} = {p(self.lr)} * {p(g)} = {v(step)}")
        print(f"           {label} = {p(w)} - {p(step)} = {v(new)}")
        return new


# ---- 학습 / 평가 -----------------------------------------------------
def train(model: SimplePerceptron, samples: list[tuple[float, float, float]], epochs: int) -> None:
    """온라인 학습: 샘플 하나마다 순전파 -> 역전파 -> 갱신."""
    v = model.v
    n = len(samples)
    for epoch in range(1, epochs + 1):
        print(LINE)
        print(f"에포크 {epoch}/{epochs}")
        losses = []
        for i, (x1, x2, t) in enumerate(samples, 1):
            print(SUBLINE)
            print(f" 에포크 {epoch}/{epochs} | 샘플 {i}/{n}:  x1 = {v(x1)}, x2 = {v(x2)}, 정답 t = {v(t)}")
            print(f"  현재 파라미터: {model.params()}")
            _, y, err = model.forward(x1, x2, t)
            grads = model.backward(x1, x2, t, y)
            model.update(*grads)
            print(f"  갱신 후 파라미터: {model.params()}")
            losses.append(err)

        total = sum(losses)
        print(SUBLINE)
        print(f" 에포크 {epoch} 평균 오차 (각 샘플 갱신 직전의 E 평균)")
        print(f"   = ({' + '.join(v(e) for e in losses)}) / {n}")
        print(f"   = {v(total)} / {n} = {v(total / n)}")


def evaluate(model: SimplePerceptron, samples: list[tuple[float, float, float]]) -> None:
    """학습이 끝난 가중치로 순전파만 수행하고 결과 표를 출력한다."""
    v = model.v
    n = len(samples)
    print(LINE)
    print("학습 후 평가 (갱신된 가중치로 순전파만 수행, 가중치는 바꾸지 않음)")
    print(f" 최종 파라미터: {model.params()}")
    rows = []
    for i, (x1, x2, t) in enumerate(samples, 1):
        print(SUBLINE)
        print(f" 샘플 {i}/{n}:  x1 = {v(x1)}, x2 = {v(x2)}, 정답 t = {v(t)}")
        _, y, err = model.forward(x1, x2, t)
        rows.append((i, x1, x2, t, y, err, 1 if y >= THRESHOLD else 0))

    width = model.digits + 6
    print(LINE)
    print("결과 요약")
    print("  #  " + "".join(h.rjust(width) for h in ("x1", "x2", "t", "y", "E")) + "  예측")
    for i, x1, x2, t, y, err, pred in rows:
        cells = "".join(v(c).rjust(width) for c in (x1, x2, t, y, err))
        print(f" {i:>2}  {cells}  {pred}")
    total = sum(r[5] for r in rows)
    print(f" 평균 오차 = ({' + '.join(v(r[5]) for r in rows)}) / {n} = {v(total / n)}")


# ---- 입력 처리 -------------------------------------------------------
def check_finite(x: float) -> str | None:
    return None if math.isfinite(x) else "유한한 숫자를 입력하세요."


def check_label(t: float) -> str | None:
    if not math.isfinite(t) or not 0.0 <= t <= 1.0:
        return "정답 레이블은 0 ~ 1 사이여야 합니다 (sigmoid 출력 범위)."
    return None


def check_lr(x: float) -> str | None:
    return None if math.isfinite(x) and x > 0.0 else "학습률은 0 보다 큰 숫자여야 합니다."


def ask_float(label: str, default: float | None = None, check=check_finite) -> float:
    suffix = f" [기본값 {default:g}]" if default is not None else ""
    while True:
        raw = input(f"{label}{suffix}: ").strip()
        if raw == "" and default is not None:
            return default
        try:
            x = float(raw)
        except ValueError:
            print("   숫자를 입력하세요.")
            continue
        problem = check(x)
        if problem:
            print(f"   {problem}")
            continue
        return x


def ask_int(label: str, default: int, minimum: int) -> int:
    while True:
        raw = input(f"{label} [기본값 {default}]: ").strip()
        if raw == "":
            return default
        try:
            x = int(raw)
        except ValueError:
            print("   정수를 입력하세요.")
            continue
        if x < minimum:
            print(f"   {minimum} 이상이어야 합니다.")
            continue
        return x


def prompt_samples() -> list[tuple[float, float, float]]:
    print("입력 노드 x1, x2 에 넣을 값과 정답 레이블 t 를 지정하세요.")
    n = ask_int("샘플 개수", default=1, minimum=1)
    samples = []
    for i in range(1, n + 1):
        print(f"[샘플 {i}/{n}]")
        x1 = ask_float("  x1 값")
        x2 = ask_float("  x2 값")
        t = ask_float("  정답 레이블 t (0 ~ 1)", check=check_label)
        samples.append((x1, x2, t))
    return samples


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="입력 2개 단순 퍼셉트론: 순전파/역전파의 모든 연산을 출력한다.",
        epilog="--sample 을 주지 않으면 입력 값과 정답 레이블을 대화형으로 묻는다.",
    )
    ap.add_argument("--sample", nargs=3, type=float, action="append",
                    metavar=("X1", "X2", "T"),
                    help="입력 x1, x2 와 정답 레이블 t (0~1). 여러 번 지정 가능")
    ap.add_argument("--lr", type=float, help=f"학습률 (기본 {DEFAULT_LR})")
    ap.add_argument("--epochs", type=int, help=f"에포크 수 (기본 {DEFAULT_EPOCHS})")
    ap.add_argument("--w1", type=float, help=f"초기 가중치 w1 (기본 {DEFAULT_W1})")
    ap.add_argument("--w2", type=float, help=f"초기 가중치 w2 (기본 {DEFAULT_W2})")
    ap.add_argument("--bias", type=float, help=f"초기 편향 b (기본 {DEFAULT_BIAS})")
    ap.add_argument("--digits", type=int, default=DEFAULT_DIGITS,
                    help=f"출력 소수점 자릿수 (기본 {DEFAULT_DIGITS})")
    args = ap.parse_args(argv)

    for x1, x2, t in args.sample or []:
        for x in (x1, x2):
            if check_finite(x):
                ap.error(f"입력 값이 유한한 숫자가 아닙니다: {x}")
        if check_label(t):
            ap.error(f"정답 레이블 {t:g}: {check_label(t)}")
    if args.lr is not None and check_lr(args.lr):
        ap.error(check_lr(args.lr))
    if args.epochs is not None and args.epochs < 1:
        ap.error("에포크 수는 1 이상이어야 합니다.")
    for name in ("w1", "w2", "bias"):
        x = getattr(args, name)
        if x is not None and check_finite(x):
            ap.error(f"--{name} 값이 유한한 숫자가 아닙니다: {x}")
    if not 0 <= args.digits <= 15:
        ap.error("--digits 는 0 ~ 15 사이여야 합니다.")
    return args


def main(argv: list[str] | None = None) -> int:
    # 콘솔 인코딩이 표현하지 못하는 문자가 있어도 멈추지 않게 한다.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    args = parse_args(argv)
    samples = args.sample
    lr, epochs = args.lr, args.epochs
    w1, w2, b = args.w1, args.w2, args.bias

    if samples is None:
        # 대화형: 샘플과, 명령행에서 지정하지 않은 설정만 묻는다.
        try:
            samples = prompt_samples()
            print("학습 설정 (그냥 Enter 를 누르면 기본값 사용)")
            if lr is None:
                lr = ask_float("  학습률 lr", DEFAULT_LR, check=check_lr)
            if epochs is None:
                epochs = ask_int("  에포크 수", DEFAULT_EPOCHS, minimum=1)
            if w1 is None:
                w1 = ask_float("  초기 가중치 w1", DEFAULT_W1)
            if w2 is None:
                w2 = ask_float("  초기 가중치 w2", DEFAULT_W2)
            if b is None:
                b = ask_float("  초기 편향 b", DEFAULT_BIAS)
        except (EOFError, KeyboardInterrupt):
            print("\n입력이 중단되어 종료합니다.")
            return 1

    lr = DEFAULT_LR if lr is None else lr
    epochs = DEFAULT_EPOCHS if epochs is None else epochs
    w1 = DEFAULT_W1 if w1 is None else w1
    w2 = DEFAULT_W2 if w2 is None else w2
    b = DEFAULT_BIAS if b is None else b

    model = SimplePerceptron(w1, w2, b, lr, args.digits)
    v = model.v
    print(LINE)
    print("단순 퍼셉트론  (입력 2개 -> 출력 1개, 활성화 sigmoid, 오차 E = 1/2 * (t - y)^2)")
    print(f" 초기 파라미터: {model.params()}")
    print(f" 학습률 lr = {v(lr)}, 에포크 = {epochs}, 샘플 수 = {len(samples)}")
    print(" 샘플 목록:")
    for i, (x1, x2, t) in enumerate(samples, 1):
        print(f"   {i}: x1 = {v(x1)}, x2 = {v(x2)}, 정답 t = {v(t)}")

    train(model, samples, epochs)
    evaluate(model, samples)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        # `| head`, `| less` 로 긴 출력을 중간에 끊어도 traceback 을 내지 않는다.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(1)
