# equations.md — 실제 구현된 수식과 이산화

이 문서는 **코드에 실제로 구현된 식**만 적는다. 구현하지 않은 식은 적지 않는다.
기호 옆의 단위는 `cortex/units.py` 의 규약을 따른다.

## 0. 단위와 자료형

| 기호 | 단위 | 자료형 | 배열 |
|---|---|---|---|
| `V` 막전위 | mV | float64 | `NeuronArrays.V_mV` shape `(N, 3)` |
| `g` 전도도 | nS | float64 | `NeuronArrays.g_nS` shape `(N, 3, R)` |
| `C` 막 용량 | pF | float64 | `NeuronArrays.C_pF` shape `(N, 3)` |
| `gL` 누설 전도도 | nS | float64 | `NeuronArrays.gL_nS` |
| `I` 외부 전류 | pA | float64 | `NeuronArrays.Iext_pA` |
| `t`, `dt`, `tau` | ms | float64 | — |
| 발화율 | Hz | float64 | `rate_estimate_hz` |
| 피질 좌표 | mm | float64 | `position_mm` shape `(N, 3)` |
| 시야 좌표 | deg | float64 | `visual_field_xy_deg` shape `(N, 2)` |

`[nS]·[mV] = [pA]` 이고 `[pF]·[mV]/[ms] = [pA]` 이므로 이 조합에서는 환산
계수가 필요 없다. 코드 어디에서도 SI 기본 단위로 되돌리지 않는다.

구획 인덱스는 `0=soma, 1=basal, 2=apical` 이다 (`cortex.ids.COMPARTMENT_NAMES`).

---

## 1. 모드 A: `sum_threshold` (무차원 합산 모델)

한 처리 구간(길이 `engine.sum_threshold_interval_steps` 스텝) 안에서:

```
s_j = Σ_k  sign(k) · w_k · P_src(k)          (구간 안에 도착한 모든 사건 k)
q_j = 1  if  s_j ≥ θ_j  else 0
```

* `sign(k) = +1` (흥분성 발신 뉴런) / `−1` (억제성 발신 뉴런). 가중치 `w_k` 자체는
  비음수 크기다.
* `P_src(k)` 는 발신 뉴런의 출력 이득이다 (3×3 `[1][2]`).
* `engine.weight_application = "emit"` 이면 `w_k`, `P` 는 **발신 시각의 스냅샷**이고,
  `"arrival"` 이면 도착 시각의 현재 값이다. 실제 적용한 값이 이벤트에 저장된다.
* 구간의 입력을 **모두 모은 뒤** 한 번에 판정하므로 입력 목록 순서가 결과를 바꾸지
  않는다. 누적 전에 `synapse_id` 로 정렬한다.
* 판정 후 작업 버퍼 `interval_activation` 만 0 으로 비운다. **장기 입력 로그는
  삭제하지 않는다.**
* 경계 조건: 입력이 없어 `s_j = 0` 이어도 `θ_j ≤ 0` 이면 발화한다.

이 모드는 **무차원**이다. `s_j` 를 막전위(mV)로 해석하지 않는다.

---

## 2. 모드 B: `conductance_lif` (구획 전도도 LIF)

### 2.1 연속 시간 식

구획 `c ∈ {soma, basal, apical}` 에 대해

```
C_c dV_c/dt = gL_c (EL_c − V_c)
            + Σ_d g_cd (V_d − V_c)
            + Σ_r g_{c,r} B_r(V_c) (E_r − V_c)
            + I_ext,c
```

* 구획 결합은 soma↔basal (`g_sb`), soma↔apical (`g_sa`) 두 개만 있다.
  basal 과 apical 은 직접 결합하지 않는다.
* `r` 은 수용체 종류다. 구현된 것은 `AMPA`, `NMDA`, `GABA_A` 뿐이다.
* `B_r(V)` 는 NMDA 의 Mg²⁺ 차단 계수다. AMPA/GABA_A 는 `B ≡ 1`.

```
B_NMDA(V) = 1 / (1 + exp(−0.062·V) · [Mg]/3.57),   [Mg] = 1 mM
```

계수는 `cortex/units.py` 의 `MG_BLOCK_*` 에 있으며 **모형 파라미터**다.

### 2.2 시냅스 전도도의 이산화

도착 사건은 스텝 시작에 **한 번만** 점프로 반영된다:

```
g⁺_{c,r} = g_{c,r}(t) + Σ_(도착 사건) Δg      (Δg ≥ 0)
```

억제성 시냅스도 `Δg ≥ 0` 이다. 전류의 방향은 `E_r` 과 `V` 가 결정한다.

스텝 `[t, t+dt]` 동안의 **시간 평균** 전도도와 스텝 끝 값:

```
ḡ_{c,r} = g⁺_{c,r} · (τ_r/dt) · (1 − exp(−dt/τ_r))
g_{c,r}(t+dt) = g⁺_{c,r} · exp(−dt/τ_r)
```

지수 감쇠의 정확한 시간 평균을 쓰므로 1차 근사보다 정확하다.

### 2.3 막전위의 이산화 (후향 오일러 + 구획 결합 직접 해)

`ḡ` 와 `B_r(V(t))` (스텝 시작 전압에서 평가 — 문서화된 선형화) 로

```
g^eff_{c,r} = ḡ_{c,r} · B_r(V_c(t))
G_c = Σ_r g^eff_{c,r}
E_c = Σ_r g^eff_{c,r} · E_r
```

후향 오일러를 적용하면 각 뉴런마다 3×3 선형계가 된다:

```
a_s = C_s/dt + gL_s + G_s + g_sb + g_sa
a_b = C_b/dt + gL_b + G_b + g_sb
a_a = C_a/dt + gL_a + G_a + g_sa

rhs_c = C_c/dt · V_c(t) + gL_c·EL_c + E_c + I_ext,c

a_s·V_s − g_sb·V_b − g_sa·V_a = rhs_s
−g_sb·V_s + a_b·V_b          = rhs_b
−g_sa·V_s          + a_a·V_a = rhs_a
```

basal 과 apical 이 서로 결합하지 않으므로 **닫힌 해**가 있다:

```
V_s(t+dt) = [ rhs_s + g_sb·rhs_b/a_b + g_sa·rhs_a/a_a ]
            / [ a_s − g_sb²/a_b − g_sa²/a_a ]
V_b(t+dt) = (rhs_b + g_sb·V_s(t+dt)) / a_b
V_a(t+dt) = (rhs_a + g_sa·V_s(t+dt)) / a_a
```

**분모의 양수성**: `a_b ≥ C_b/dt + gL_b + g_sb > g_sb` 이므로 `g_sb²/a_b < g_sb`,
같은 이유로 `g_sa²/a_a < g_sa`. 따라서

```
a_s − g_sb²/a_b − g_sa²/a_a  >  C_s/dt + gL_s + G_s  >  0
```

분모가 항상 양수이므로 0 으로 나누는 일이 없고, 후향 오일러이므로 무조건
안정이다. **발산을 가리기 위한 임의 전압 클리핑을 넣지 않았다.**
비유한 값이 생기면 `StepReport.n_nonfinite` 에 그대로 기록된다.

존재하지 않는 구획(`has_compartment` 가 False)은 계산 후 `V_c = EL_c` 로 되돌려
결과에 영향을 주지 않게 한다 (해당 구획의 `g_couple = 0`).

### 2.4 발화·리셋·불응기

```
불응기:  t < refractory_until_i  →  V_soma,i ← V_reset,i  (soma 만 고정)
발화:    (V_soma,i ≥ θ_i) 이고 (불응기가 아님)
리셋:    V_soma,i ← V_reset,i,  refractory_until_i ← t + t_ref,i
```

* 불응기 동안에도 basal/apical 은 계속 적분되고 시냅스 전도도도 계속 갱신된다.
* `cell_types.*.refractory_input_policy = "discard"` 인 세포 유형은 불응기 동안
  **soma 표적 도착**만 전도도에 반영하지 않는다 (basal/apical 은 계속 받는다).
  기본값은 `"accumulate"` 다.

### 2.5 방출

```
emitted_amount_k = w_k · P_src(k)
arrival_step_k   = current_step + effective_delay_steps_k
```

---

## 3. 지연

```
delay_ms      = synaptic_delay_ms + ‖pos_src − pos_dst‖₂ / conduction_velocity
delay_steps   = max(min_delay_steps, ceil(delay_ms / dt))        (기본 ceil)
```

* `‖·‖₂` 는 3D 피질 좌표(mm)의 직선 거리다. 축삭 길이를 직선 거리로 근사한 것이며
  `use_straight_line_distance` 로 표시된다.
* `min_delay_steps ≥ 1` 이 강제되므로 **0 지연 재귀로 인한 무한 루프가 구조적으로
  불가능**하다.
* 연속 지연 `delay_ms` 와 양자화 `delay_steps` 를 **둘 다** 저장한다.

### 3.1 `local_radius` 의 거리 공간

후보를 고르는 거리는 규칙의 `radius_space` 가 정한다.

```
cortical_3d : d = ‖(x,y,z)_src − (x,y,z)_dst‖₂        깊이 포함 (같은 층 수평 연결)
surface     : d = ‖(u,v)_src − (u,v)_dst‖₂            표면 접선 거리 (층간 투사)
```

층을 가로지르는 투사는 같은 기둥 안에서 깊이를 따라 내려가므로 표면 거리가
맞다. 깊이를 포함한 3D 거리로 재면 층 간격보다 작은 반경에서는 후보가 하나도
나오지 않아 **이름만 있는 경로**가 된다 (필수 검증 9 가 이 경우를 잡는다).

**지연은 어느 경우에도 3D 직선 거리를 쓴다.** 축삭이 실제로 지나는 길이는 깊이를
포함하기 때문이다.

### 3.2 전달 여유 (실행 전 진단)

정상상태 근사로 "이 배선이 표적을 임계까지 올릴 수 있는가" 를 본다.

```
g_need   = gL · (V_th − E_L) / (E_rev − V_th)                      [nS]
g(R)     = deg · w · (τ/1000) · R                                  [nS]
R_need   = g_need / (deg · w · τ/1000)                             [Hz]
R_max    = 1000 / t_ref        (피질)
         = baseline + gain · max_rate_hz   (망막; 외부 구동이 상한)
```

`deg` 는 표적 1개당 평균 시냅스 수, `w` 는 평균 가중치, `τ` 는 수용체 시상수다.
`R_need > R_max` 면 그 단계는 **어떤 입력에도 전달되지 않는다.** 단일 구획
정상상태 근사이므로 정확한 예측이 아니라 자릿수 점검이다
(`manifest.json` 의 `transmission_headroom`).

`sum_threshold` 모드의 망막 구동에는 같은 뜻의 더 단순한 조건을 쓴다.

```
한 구간 최대 기여 = (baseline + gain · max_rate_hz) · (dt/1000)
                   · sum_mode_scale · sum_threshold_interval_steps
                   ≥ 망막 세포의 θ
```

---

## 4. 망막 전처리

```
x01     = uint8/255  또는  [0,1] float (범위 검사)
lin     = srgb_to_linear(x01)                       (input_colorspace="srgb")
XYZ     = lin @ RGB_TO_XYZ.T
LMS     = XYZ @ XYZ_TO_LMS.T                         (근사)
opp     = OPPONENT_MATRIX @ LMS
DoG(p)  = G(σ_c) * p − G(σ_s) * p                    (각 커널 합 = 1)
ON      = max(DoG, 0),   OFF = max(−DoG, 0)
lowpass = max(G(σ_lp) * LMS, 0)                      (선택 경로)
```

* `srgb_to_linear(x) = x/12.92` (x ≤ 0.04045), `((x+0.055)/1.055)^2.4` (그 외)
* `OPPONENT_MATRIX` 는 `[[0.6,0.4,0],[1,−1,0],[−0.5,−0.5,1]]` 이며 **모형 선택**이다.
* 두 가우시안 커널이 각각 합 1 이므로 균일 입력의 DoG 응답은 이론상 0 이고,
  실제로는 경계 처리에서만 잔여 응답이 남는다 (`uniform_response_check` 로 측정).

발화율과 사건 수:

```
rate_hz   = max(0, baseline + gain · max_rate · v̂)      (v̂ ∈ [0,1] 정규화 채널값)
λ         = rate_hz · dt_ms / 1000                       (dt 동안의 기대 사건 수)
poisson:  n ~ Poisson(λ)
rate:     I_ext = rate_hz · current_per_hz_pA            (conductance 모드)
          Δs    = rate_hz · dt_ms/1000 · sum_mode_scale  (sum_threshold 모드)
```

Hz 와 ms 를 섞지 않는다. 변환은 항상 `/1000` 을 거친다.

---

## 5. 로그-극좌표 사상

```
rho   = log(1 + ecc/e0)          ecc = e0 · (exp(rho) − 1)
theta = atan2(y, x)  (mod 2π)
x_deg = ecc·cos(theta),  y_deg = ecc·sin(theta)
col   = cx + x_deg · px_per_deg,   row = cy + y_deg · px_per_deg
px_per_deg = max(H, W) / fov_deg
```

* 로그를 두 번 적용하지 않는다.
* 반경 bin `i` 의 셀 폭과 저역통과 폭:

```
Δecc_i  = ecc(rho_{i+1}) − ecc(rho_i)
arc_i   = ecc_center,i · 2π / n_angular
σ_deg,i = sigma_scale · max(Δecc_i, arc_i)
σ_px,i  = max(σ_deg,i · px_per_deg, min_sigma_px)
cell_area_i = π(ecc_{i+1}² − ecc_i²) / n_angular          [deg²]
```

* 반경(deg), 면적(deg²), 셀당 픽셀 수 추정(`area · px_per_deg²`)을 **따로** 기록한다.
* 중심 특이점은 반경 `fovea_patch.radius_deg` 안쪽을 `grid × grid` Cartesian 격자로
  대체한다 (채택한 근사).
* 샘플 위치를 원본 영상 좌표로 계산해 **원본 영상에서** 값을 읽는다. 왜곡된
  로그-극좌표 배열 위에 직선 필터를 적용하지 않는다.

---

## 6. Gabor 계수 (초기 배선 / 고정 대조 경로)

원본 시야 Cartesian 좌표에서 평가한다:

```
x_r =  Δx·cosθ + Δy·sinθ
y_r = −Δx·sinθ + Δy·cosθ
gabor = exp(−½[(x_r/σ)² + (y_r/(σ·aspect))²]) · cos(2π·f·x_r + φ)
```

**음의 계수를 음의 전도도로 만들지 않는다**:

```
src 가 ON 극성  →  w = max(gabor, 0)
src 가 OFF 극성 →  w = max(−gabor, 0)
극성이 없으면   →  w = |gabor|   (부호는 Dale 유형이 만든다)
```

고정 Gabor 대조 경로의 참조 에너지:

```
energy = sqrt(r0² + rq² + eps²) − eps
```

`r0`, `rq` 는 위상 `0`, `−π/2` 필터의 선형 응답이다. 유한 필터·경계·공간 왜곡
조건에서 완벽한 위상 불변성을 **보장하지 않는다**.

---

## 7. STDP (pair-based, 흔적 방식)

스텝마다:

```
x_pre  ← x_pre  · exp(−dt/τ₊)
x_post ← x_post · exp(−dt/τ₋)
```

발화 시 (흔적 증가는 **가중치 갱신 뒤**에 한다):

```
post j 발화 →  Δw_(i→j) += A₊ · x_pre[i]        (LTP)
pre  i 발화 →  Δw_(i→j) −= A₋ · x_post[j]       (LTD)
w ← clip(w + Δw, weight_min ≥ 0, weight_max)
```

* `simultaneous_policy`
  * `both` — 같은 스텝의 전/후 발화 모두에 대해 두 항을 적용하되, 두 항 모두
    **그 스텝의 흔적 증가 이전 값**을 쓴다.
  * `pre_first` — 전 흔적을 먼저 증가시킨 뒤 LTP 를 계산한다 (동시 발화 시 LTP 우세).
  * `post_first` — 후 흔적을 먼저 증가시킨 뒤 LTD 를 계산한다 (동시 발화 시 LTD 우세).
* `update_order` 는 LTP/LTD 중 어느 쪽을 먼저 누적할지 정한다. clip 은 한 번만 한다.
* `apply_to_inhibitory=false` (기본) 면 억제성 발신 시냅스는 학습하지 않는다.
* clip 하한이 0 이상이므로 **흥분 연결이 억제 연결로 뒤집히지 않는다**.

선택적 가중치 감쇠: `Δw −= weight_decay_per_ms · dt · w` (기본 0).

이것은 "동시 발화 곱" 이 아니다. 전/후 사건의 **시간 차**가 흔적을 통해 들어간다.

---

## 8. 항상성 임계 적응

```
inst_hz[i] = spike[i] / (dt/1000)
α          = dt / window_ms
r[i]     ← r[i] + α·(inst_hz[i] − r[i])                       [Hz]
θ[i]     ← clip(θ[i] + η_θ·(r[i] − target[i])·α, θ_lo, θ_hi)
```

* 누적 시간은 `window_ms` 다. 목표 활동률 `target[i]` 는 세포 유형별로 다를 수 있다.
* 상하한은 모드에 따라 mV 또는 무차원 단위를 쓴다.
* 이웃 임계값 평균화는 **기본값이 아니며** 소거 실험 옵션이다
  (`learning.neighbor_theta_averaging`). 이것을 "정답 교정" 이라고 부르지 않는다.

---

## 9. Rao 참조 모델 (연속값, 별도 엔진)

모듈 `m = 1..M` 이 하나의 상위 표현 `r₂` 를 공유한다.

```
E = Σ_m ½/σ²  ‖I_m − U1_m r1_m‖²
  + Σ_m ½/σ_td² ‖r1_m − U2_m r2‖²
  + ½α ( Σ_m ‖r1_m‖² + ‖r2‖² )
  + ½λ ( Σ_m ‖U1_m‖² + Σ_m ‖U2_m‖² )
```

같은 전체 목적함수에서 유도한 갱신:

```
dr1_m = U1_mᵀ(I_m − U1_m r1_m)/σ² + (U2_m r2 − r1_m)/σ_td² − α r1_m
dr2   = Σ_m U2_mᵀ(r1_m − U2_m r2)/σ_td² − α r2
dU1_m = outer(I_m − U1_m r1_m, r1_m)/σ²   − λ U1_m
dU2_m = outer(r1_m − U2_m r2,  r2)/σ_td²  − λ U2_m
```

`M = 1` 이면 `r_td = U2 r2` 로 두었을 때 명세의 식과 정확히 같다.

* 정착은 **동기적**이다: 모든 모듈의 이전 상태로 오차를 계산한 뒤 `r` 을 한 번에
  갱신한다. 정착 중 `U` 는 동결된다 (`freeze_U=False` 는 예외를 낸다).
* `σ, σ_td > 0` 을 생성자에서 검증한다.
* `dr = −∂E/∂r` 이므로 유한차분 검사는 `dr + numeric_grad ≈ 0` 을 본다.
* **이것은 `U` 의 전치를 쓰는 기울기하강 모델이다.** "미분도 대칭 가중치도 없는
  학습" 이 아니다. 전도도 LIF 회로와 같은 물리 모델도 아니다.
* 영상 재구성 오차 `‖I − U1 r1‖` 와 영역 간 표현 오차 `‖r1 − U2 r2‖` 를 서로 다른
  변수로 보관한다.

---

## 10. 한 스텝의 실행 순서 (`Engine.step`)

1. 현재 스텝의 예약 사건을 큐에서 꺼낸다 (꺼낸 버킷은 비워진다) + 외부 자극 수집
2. 도착 사건을 `EventLog` 에 기록하고 `last_consumed_event` 를 갱신
3. 도착값을 작업 버퍼(전도도 또는 구간 누적)에 **한 번만** 더한다
4. §2.2–2.3 (또는 §1) 에 따라 상태를 `dt` 만큼 갱신
5. 발화·불응기 판정과 리셋
6. 발화 뉴런의 출력을 `delay_steps` 에 따라 미래 큐에 예약 (발신 시점 스냅샷)
7. 가소성 흔적과 학습 규칙 갱신 (§7, §8)
8. 선택된 상태와 실제 변화량 기록
9. `step_index += 1`, `time_ms += dt`

한 스텝의 모든 발화는 **같은 가중치 상태**로 판정·예약되고, 학습 갱신은 7단계에서
한꺼번에 적용된다. 수정된 가중치를 같은 스텝의 다른 뉴런만 먼저 쓰는 숨은 순서
의존성이 없다.
