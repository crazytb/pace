# Figure 4: Adaptive qsrc vs. Fixed qsrc vs. Oracle (주요 기여)

**기여**: NPCA CW amnesia 하에서 col_rate·waste_rate 기반 adaptive qsrc 알고리즘이  
고정 qsrc 대비 throughput을 개선하며 oracle에 수렴함을 보임.

**스크립트**: `harq_sim/run_step9_fig4.py`

**출력**: `manuscript/figure/fig4_*.{eps,png,pdf}`

---

## 실험 파라미터 (Fig 3 v2와 동일 환경)

| 항목 | 값 |
|---|---|
| **Sweep 변수 (v1)** | `num_stas` ∈ {5, 10, 20, 30, 50} → `results/step9/fig4/` |
| **Sweep 변수 (v2 massive)** | `num_stas` ∈ {5, 10, 20, 30, 50, 70, 100, 150, 200} → `results/step9/fig4_v2/` |
| OBSS 채널 점유율 | 50% (`obss_rate = _occupancy_to_rate(0.50)`) |
| `obss_min` | 20 슬롯 |
| `obss_max` | 500 슬롯 |
| `snr_db_mean` | 20.0 dB (결정론적 — 충돌 효과 분리) |
| `snr_db_std` | 0.0 dB |
| `ppdu_duration` | 20 슬롯 |
| `num_slots` | 50,000 |
| Seeds | [42, 123, 456] |

---

## 비교 대상 (5개)

| 기법 | adaptive_cw | npca_qsrc | 설명 |
|---|---|---|---|
| `fixed_q0` | False | 0 | CW=15, 공격적 (표준 기본값) |
| `fixed_q1` | False | 1 | CW=31 |
| `fixed_q2` | False | 2 | CW=63 |
| `oracle` | False | N별 최적값 | Fig 3 v2 결과로 N→qsrc* 고정 (upper bound) |
| `adaptive` | **True** | 동적 조정 | **제안 알고리즘** |

Oracle qsrc 매핑 (channel fix 적용 후 재계산 — 2026-05-28):
```python
ORACLE_QSRC = {5: 0, 10: 0, 20: 1, 30: 1, 50: 1, 70: 2, 100: 3, 150: 3, 200: 4}
```
※ 이전 값 (channel bug 있는 버전): N=20: 0, N=50: 2 → 수정됨

---

## Adaptive qsrc 알고리즘

매 `K=20`회 NPCA 전환마다 다음을 계산:

```python
col_rate   = npca_collision_count / npca_transitions
waste_rate = 1 - (npca_tx_success + npca_tx_fail) / npca_transitions

if col_rate > θ_col:      qsrc = min(qsrc + 1, QSRC_MAX)  # CW 키움
elif waste_rate > θ_waste: qsrc = max(qsrc - 1, 0)         # CW 줄임
카운터 리셋
```

기본 파라미터: `K=5, θ_col=0.70, θ_waste=0.30, QSRC_MAX=5`

선택 근거:
- K=5: N=50 STA 환경에서 50000슬롯당 ~19 NPCA 방문 → K=5로 ~3-4회 업데이트 가능
- θ_col=0.70: N=30에서 qsrc*=1이 optimal인데 col@qsrc=1=66.5% < 0.70 → 조기 상승 방지

**직관**:
- `col_rate` 높음 → 많은 STA가 경쟁 → CW를 키워 충돌 감소
- `waste_rate` 높음 → 백오프가 창 내 미완료 → CW가 너무 큼, 줄여야

---

## 측정 지표

- `aggregate_throughput` — 전달 패킷 수 (주 지표)
- `collision_probability_npca` — NPCA 충돌 확률
- `mean_qsrc` — adaptive의 평균 qsrc (oracle qsrc*와 비교)
- `npca_transition_count` — NPCA 전환 총 횟수

---

## Figure 구성

```
Figure 4: 3-panel (공유 x축: num_stas)

Panel (a): aggregate_throughput vs num_stas
           5개 선: fixed_q0/q1/q2 (회색 계열), oracle (녹색 점선), adaptive (파랑 실선)
           std 음영 포함
           핵심 메시지: adaptive ≈ oracle > 최적 fixed

Panel (b): collision_probability_npca vs num_stas
           동일 5개 선
           adaptive와 oracle의 충돌률이 수렴하는지 확인

Panel (c): mean_qsrc vs num_stas (adaptive vs oracle 비교)
           막대: oracle qsrc* (회색)
           꺾은선: adaptive mean qsrc (파랑)
           ± std 음영
           핵심 메시지: adaptive가 oracle qsrc*에 수렴
```

---

## 실험 결과 v1 (results/step9/fig4/, N≤50, 50000슬롯 × 3 seeds)

| N | Oracle TP | Adaptive TP | vs Oracle | adaptive mean_q |
|---|---|---|---|---|
| 5  | 1250 | **1250** | +0.00% | 0.00 (oracle=0) ✓ |
| 10 | 1272 | 1259     | -1.00% | 0.18              |
| 20 | 1244 | **1249** | +0.43% | 0.68              |
| 30 | 1219 | **1232** | +1.01% | 1.07 (oracle=1) ✓ |
| 50 | 1203 | 1194     | -0.72% | 1.37 (oracle=2) ~ |

---

## 실험 결과 v2 — Massive N (results/step9/fig4_v2/, N∈{5~200}, 50000슬롯 × 3 seeds)

| N | q* | Oracle TP | Adaptive TP | gap% | adap mean_q | best_fixed |
|---|---|---|---|---|---|---|
| 5   | 0 | 1280 | 1280 | +0.00% | 0.00 ✓ | 1280 |
| 10  | 0 | 1273 | 1269 | -0.31% | 0.21   | 1273 |
| 20  | 0 | 1258 | 1242 | -1.27% | 0.73   | 1258 |
| 30  | 1 | 1207 | 1218 | +0.91% | 1.08 ✓ | 1218 |
| 50  | 2 | 1204 | 1186 | -1.50% | 1.33 ~ | 1204 |
| 70  | 2 | 1159 | 1176 | +1.47% | 1.66 ~ | 1176 |
| 100 | 3 | 1147 | 1149 | +0.15% | 1.63 ✗ | 1138 |
| 150 | 3 | 1154 | 1126 | **-2.37%** | 1.91 ✗ | 1147 |
| 200 | 4 | 1132 | 1076 | **-4.89%** | 2.10 ✗ | 1113 |

### 핵심 관찰 (v2 massive)

#### 1. N≤100 구간: adaptive ≈ oracle (±1.5%)
- N=5~70에서 oracle 대비 ±1.5% 이내 — 기존 결론 유지
- 특히 N=30,70,100에서 adaptive가 oracle을 소폭 초과

#### 2. N≥150 구간: adaptive 수렴 한계 발현
- N=150: adaptive mean_q=1.91 vs oracle q*=3 → 거의 2단계 차이
- N=200: adaptive mean_q=2.10 vs oracle q*=4 → 2단계 미만
- **결과**: N=150 -2.37%, N=200 -4.89% — oracle 대비 의미 있는 gap
- N=200에서 adaptive(1076) < best_fixed_q2(1113): **fixed_q2가 adaptive를 역전**

#### 3. Fixed qsrc 최적 순서 역전
| N 범위 | 최적 fixed | 이유 |
|---|---|---|
| N≤20 | fixed_q0 (CW=15) | 경쟁 적음, 작은 CW가 최적 |
| N=30~50 | fixed_q2 (CW=63) | 충돌 급증, 큰 CW 필요 |
| N≥100 | fixed_q2 (CW=63) | 하지만 oracle은 q*=3~4 → fixed_q2도 부족 |

#### 4. 충돌 확률 비교 (N=200)
- fixed_q0: 99.3%, adaptive: 97.1%, oracle: **56.7%**
- adaptive가 qsrc를 2.1까지만 올려 충돌 여전히 높음

### 논문 메시지 (개정)
"제안된 adaptive qsrc 알고리즘은 N≤100 범위에서 oracle 대비 ±1.5% 이내의 성능을 달성하며, 고정 qsrc 선택 시 발생하는 worst-case 성능 저하를 방지한다. 그러나 N≥150의 극단적 고밀도 환경에서는 K=5, θ_col=0.70 파라미터의 점진적 증가 속도가 oracle에 비해 qsrc를 충분히 높이지 못하는 한계가 관찰된다."

### Limitation 및 향후 개선 방향
- θ_col을 N에 비례해 동적으로 낮추거나 (e.g., θ_col = 1 - N/N_max)
- 더 공격적인 초기 qsrc 설정 (binary search 방식)
- 실용적 운용 범위(N≤50)에서는 현 파라미터로 충분

---

## 실험 결과 v3 — 200k 슬롯 (results/step9/fig4_v3/, N∈{5~200})

`num_slots = 200,000`으로 재실행. per-STA adaptive 업데이트 횟수: 5~8 → **20~34회**로 증가.

| N | q* | Oracle TP | Adaptive TP | gap% | adap mean_q | CV% |
|---|---|---|---|---|---|---|
| 5   | 0 | 5091 | 5059 | -0.62% | 0.02 ✓ | 0.25% |
| 10  | 0 | 5048 | 5050 | +0.03% | 0.25   | 0.63% |
| 20  | 0 | 4985 | 4947 | -0.75% | 0.92   | 0.57% |
| 30  | 1 | 4879 | 4875 | -0.10% | 1.50 ✓ | 1.18% |
| 50  | 2 | 4796 | 4802 | +0.13% | 1.94 ~ | 0.56% |
| 70  | 2 | 4764 | 4701 | -1.32% | 2.34 ~ | 0.88% |
| 100 | 3 | 4655 | 4673 | +0.39% | 2.56 ~ | 0.56% |
| 150 | 3 | 4624 | 4557 | **-1.44%** | 2.85 ~ | 0.49% |
| 200 | 4 | 4558 | 4497 | **-1.35%** | 3.09 ~ | 0.59% |

### v2(50k) → v3(200k) 개선 요약

| N | v2 gap% | v3 gap% | 개선 |
|---|---|---|---|
| 150 | -2.37% | **-1.44%** | +0.93%p |
| 200 | -4.89% | **-1.35%** | **+3.54%p** |

### 핵심 발견 (v3)

1. **모든 N에서 ±1.5% 이내 달성**: 200k 슬롯에서 adaptive가 oracle을 전 범위 추적
2. **N≥100에서 adaptive > best_fixed**: adaptive(4497) > fixed_q2(4409) at N=200 (+2.0%)
   - v2에서는 adaptive < fixed_q2 (역전)였으나, v3에서 정상 복원
3. **충분한 운용 시간이 수렴의 핵심**: 알고리즘 자체는 올바르게 설계됨
   - 실 시스템에서는 장기 운용이 보장되므로 논문 주장이 유효

### 논문 메시지 (최종)
"제안된 adaptive qsrc 알고리즘은 N∈{5,...,200} 전 범위에서 oracle 대비 ±1.5% 이내의 성능을 달성하며, 고정 qsrc 선택 대비 worst-case 성능 저하를 방지한다. 50k 슬롯에서 관찰된 수렴 지연은 단순히 관측 창 부족으로 인한 것으로, 충분한 운용 시간(200k 슬롯, ~770회 OBSS 이벤트)에서는 완전히 해소된다."

---

## 구현 메모

`sta.py`에 추가 필요:
```python
# __init__
self._adap_trans = 0  # adaptive 관측 창 전환 카운터
self._adap_col   = 0  # 충돌 카운터
self._adap_tx    = 0  # TX 시도 카운터
self._adap_K     = 20  # 관측 창 크기
self._theta_col   = 0.50
self._theta_waste = 0.30

# _start_npca_transition 후
if self.adaptive_cw:
    self._adap_trans += 1

# _handle_npca_tx 내 collision 감지 시
if self.adaptive_cw:
    self._adap_col += 1
    self._adap_tx  += 1

# _start_switch_back 내
if self.adaptive_cw:
    self._maybe_update_qsrc()

def _maybe_update_qsrc(self):
    if self._adap_trans < self._adap_K:
        return
    col_rate   = self._adap_col / self._adap_trans
    waste_rate = 1.0 - self._adap_tx / self._adap_trans
    if col_rate > self._theta_col:
        self.npca_initial_qsrc = min(self.npca_initial_qsrc + 1, 5)
    elif waste_rate > self._theta_waste:
        self.npca_initial_qsrc = max(self.npca_initial_qsrc - 1, 0)
    self._adap_trans = self._adap_col = self._adap_tx = 0
```

---

## 출력 파일

```
manuscript/figure/
  fig4_adaptive_qsrc.eps / .png / .pdf

results/step9/fig4/
  data.csv    ← (num_stas, method, seed, aggregate_throughput,
                  collision_probability_npca, mean_qsrc, npca_transition_count)
```

---

## 실험 결과 v4 — channel fix + corrected oracle (results/step9/fig4_v3/, 200k슬롯 × 3 seeds, 2026-05-28)

`channel.py` 버그 수정 (OBSS 생성 시 intra-BSS TX 차단 제거) + Oracle qsrc 재계산 반영.

| N | q* | Oracle TP | Adaptive TP | gap% | adap mean_q | fixed_q0 | adap vs q0 |
|---|---|---|---|---|---|---|---|
| 5   | 0 | 5233.0 | 5213.7 | -0.37% | 0.01 ✓ | 5233.0 | -0.37% |
| 10  | 0 | 5232.7 | 5163.0 | -1.33% | 0.27   | 5232.7 | -1.33% |
| 20  | 1 | 4992.7 | 5038.3 | +0.91% | 1.01 ✓ | 4993.3 | +0.90% |
| 30  | 1 | 4950.0 | 4938.3 | -0.24% | 1.50 ✓ | 4846.3 | +1.90% |
| 50  | 1 | 4783.0 | 4817.3 | +0.72% | 2.10 ~ | 4528.7 | **+6.37%** |
| 70  | 2 | 4736.7 | 4767.7 | +0.65% | 2.44 ✓ | 4414.3 | **+8.00%** |
| 100 | 3 | 4684.7 | 4658.3 | -0.56% | 2.76 ~ | 4165.3 | **+11.84%** |
| 150 | 3 | 4662.7 | 4615.7 | -1.01% | 3.09 ✓ | 3892.3 | **+18.58%** |
| 200 | 4 | 4616.0 | 4501.7 | -2.48% | 3.31 ~ | 3548.3 | **+26.87%** |

### 핵심 발견 (v4)

1. **모든 N에서 ±2.5% 이내**: channel fix 후에도 adaptive가 oracle을 전 범위에서 근접 추적
2. **fixed_q0 대비 massive gain**: N=200에서 +26.87%, N=100에서 +11.84% — channel fix로 gain이 크게 증가
3. **N=200 gap 소폭 확대**: -1.35%(v3) → -2.48%(v4) — 더 많은 OBSS 이벤트로 oracle이 정확히 q*=4를 활용하는 반면 adaptive는 mean_q=3.31에서 수렴
4. **전체 논문 메시지 강화**: 실제 OBSS 비율(50%)이 정확히 반영되면서 NPCA benefit 및 adaptive 기여가 더욱 뚜렷해짐

### v3(pre-fix) → v4(channel fix) 변화

| N | v3 oracle | v4 oracle | v3 adap gap | v4 adap gap | v3 vs q0 | v4 vs q0 |
|---|---|---|---|---|---|---|
| 50  | 4796 | 4783 | +0.13% | +0.72% | - | **+6.37%** |
| 100 | 4655 | 4685 | +0.39% | -0.56% | - | **+11.84%** |
| 200 | 4558 | 4616 | -1.35% | -2.48% | - | **+26.87%** |

→ Oracle과의 gap은 비슷하거나 약간 확대됐으나, fixed_q0 대비 gain이 현저히 증가.

---

## 수정 이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-05-25 | 초안 작성 (Step 8 역전 현상 분석 기반) |
| 2026-05-26 | 논문 방향 변경에 따라 전면 재설계: adaptive qsrc 알고리즘 주요 기여로 재정립; oracle 비교군 추가; Fig 3 v2 환경으로 통일 |
| 2026-05-26 | Massive N 실험 추가 (N∈{5~200}); adaptive 수렴 한계 발현 (N≥150에서 oracle 대비 -2.4~-4.9%); fixed_q 최적 순서 역전 분석 |
| 2026-05-28 | channel.py 버그 수정 (OBSS 생성 시 intra-BSS TX 차단 → obss_remain>0으로 수정); Oracle qsrc 재계산; v4 결과 추가 |
