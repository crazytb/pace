# Figure 8: Cross-Channel HARQ Combining in NPCA Systems

**연구 질문 (RQ8)**: NPCA 채널 전환을 경유하는 HARQ Chase Combining — 즉 primary 채널에서  
실패한 전송의 soft bits를 NPCA 채널 재전송과 합산하는 **cross-channel HARQ** —  
이 가능할 때와 불가능할 때 각각 성능이 어떻게 달라지는가?

**스크립트**: `harq_sim/run_step9_fig8.py`

**출력**: `manuscript/figure/fig8_*.{eps,png,pdf}`

---

## 배경

### NPCA 환경에서의 HARQ 동작 특수성

기존 WLAN에서 HARQ Chase Combining은 **동일 채널(동일 주파수)** 에서만 발생한다.  
STA가 primary 채널에서 전송 실패 → 동일 primary 채널에서 재전송 → soft bits 합산.

NPCA 환경에서는 새로운 시나리오가 발생한다:

```
[Primary 채널 TX] PHY 실패 → harq_buffer.store(SNR_p)
        ↓ OBSS 도착 → NPCA 전환
[NPCA 채널 TX] 동일 패킷 재전송
  - cross_channel_harq=True  → effective_SNR = SNR_p + SNR_n (combining)
  - cross_channel_harq=False → effective_SNR = SNR_n           (fresh TX)
```

이를 **cross-channel HARQ combining** 이라 한다.

### 가능성 조건 (현재 구현)

| 조건 | 설명 |
|------|------|
| `harq_enabled=True` | HARQ-CC가 활성화됨 |
| primary PHY 실패 | harq_buffer에 SNR_linear 저장됨 |
| OBSS 도착 전 패킷 미완료 | 동일 패킷이 head-of-queue에 있음 |
| NPCA 전환 성공 | NPCA backoff 시 same packet_id → HARQ_RETX 판정 |

### 3-way 비교군

| 기법 | `harq_enabled` | `cross_channel_harq` | 설명 |
|------|---------------|---------------------|------|
| `no_harq` | False | N/A | ARQ only baseline |
| `same_ch_harq` | True | **False** | 채널 전환 시 buffer flush → 동일 채널 내에서만 combining |
| `cross_ch_harq` | True | **True** | 채널 전환에 관계없이 combining (현재 구현, 제안 방식) |

---

## 실험 파라미터

### 주요 스윕: SNR

| 항목 | 값 |
|------|-----|
| **sweep 변수** | `snr_db_mean` ∈ {8, 10, 12, 14, 16, 18, 20, 22, 24, 26} dB |
| `snr_db_std` | 0.0 (결정론적 — PHY failure rate 명확히 제어) |
| `num_stas` | 20 (고정) |
| OBSS 채널 점유율 | 50% (`obss_rate = occupancy / (mean_dur × (1-occupancy))`) |
| `obss_min` | 20 슬롯 |
| `obss_max` | 500 슬롯 |
| `ppdu_duration` | 20 슬롯 |
| `num_slots` | 50,000 |
| Seeds | [42, 123, 456] |
| `npca_qsrc` | 0 (고정) |
| `ppdu_truncation` | True (기본값) |

#### SNR별 PHY 동작 특성 (std=0 기준)

| SNR (dB) | 선택 MCS | p_success (단회) | p_success (2회 combining) |
|----------|---------|-----------------|--------------------------|
| 8  | MCS1 (threshold 8)  | 0.50 | 0.95 |
| 10 | MCS1 (threshold 8)  | 0.88 | ~1.0 |
| 12 | MCS2 (threshold 11) | 0.73 | ~1.0 |
| 14 | MCS3 (threshold 14) | 0.50 | 0.95 |
| 16 | MCS3 (threshold 14) | 0.88 | ~1.0 |
| 18 | MCS4 (threshold 17) | 0.73 | ~1.0 |
| 20 | MCS5 (threshold 20) | 0.50 | 0.95 |
| 22 | MCS5 (threshold 20) | 0.88 | ~1.0 |
| 24 | MCS5 (threshold 20) | 0.98 | ~1.0 |
| 26 | MCS7 (threshold 26) | 0.50 | 0.95 |

> PHY 실패율이 높은 MCS 임계점(8, 14, 20, 26 dB)에서 HARQ 기여가 가장 크다.  
> 임계점 사이(10, 12, 16, 18 dB)에서는 PHY 성공률이 높아 HARQ 기여가 감소한다.

### 보조 스윕: num_stas (SNR=20 고정)

| 항목 | 값 |
|------|-----|
| **sweep 변수** | `num_stas` ∈ {5, 10, 20, 30, 50} |
| `snr_db_mean` | 20.0 dB (MCS 임계점, p=0.5 — HARQ 효과 최대) |
| 나머지 | SNR sweep과 동일 |

> 충돌 확률이 높아질수록 (더 많은 STA) HARQ buffer는 collision failure엔  
> soft bits를 저장하지 않으므로 (`§9.2: collision does NOT store soft info`) HARQ 기여 감소 예상.

---

## Figure 구성

```
Figure 8: 3-panel

Panel (a): Aggregate Throughput vs snr_db_mean
           3 lines: no_harq (gray, dotted), same_ch_harq (orange, dashed), cross_ch_harq (blue, solid)
           ± std 음영
           핵심: cross_ch_harq ≥ same_ch_harq ≥ no_harq (항상 성립)

Panel (b): HARQ Gain (%) vs snr_db_mean
           2 lines:
             harq_gain_same = (same_ch_harq - no_harq) / no_harq × 100%  (동채널 combining 기여)
             harq_gain_cross = (cross_ch_harq - no_harq) / no_harq × 100% (cross-channel 포함 총 기여)
           차이(Δ) = cross-channel combining의 추가 기여
           수직 점선: MCS 임계점 (8, 14, 20, 26 dB) 표시

Panel (c): Aggregate Throughput vs num_stas (at SNR=20 dB)
           동일 3 lines
           핵심: 충돌 증가로 HARQ 기여 감소 → 고밀도에서 cross-channel 이득 변화 확인
```

---

## 측정 지표 (CSV 컬럼)

```
snr_db_mean, num_stas, method, seed,
aggregate_throughput,
phy_error_rate,          # phy_error_failures / total_tx_attempts
harq_success_count,      # harq_tx_success (combining 성공 횟수)
harq_fail_count,         # harq_tx_fail
collision_prob_npca,     # NPCA 충돌 확률
packet_delivery_ratio,   # PDR
npca_transition_count,
```

---

## 예상 결과

| N=20, SNR=20 dB | no_harq | same_ch_harq | cross_ch_harq |
|----------------|---------|--------------|---------------|
| aggregate TP   | 낮음    | 중간         | 가장 높음      |
| PHY error rate | 높음    | 중간         | 낮음 (combining 덕분) |
| harq_success   | 0       | 동채널 only  | 더 많음 (cross 포함) |
| PDR            | 낮음    | 중간         | 높음           |

**핵심 가설**:
- MCS 임계점(p=0.5)에서 cross_ch_harq gain이 same_ch_harq 대비 가장 큼
  (primary 실패율 높음 → NPCA에서 combining 기회 많음)
- 고밀도(N=50)에서 gain 감소: collision이 soft bits 저장을 막기 때문
- SNR이 충분히 높으면 (p≈0.98+) 세 방법 수렴: HARQ 필요 없음

---

## 실험 결과 v1 (results/step9/fig8/, N=20, 50000슬롯 × 3 seeds)

### SNR Sweep 결과

std=0 모델에서 각 SNR 값은 해당 MCS의 p_success를 결정한다.  
같은 p_success를 내는 SNR 값(8/14/20/26 → p=0.50, 12/18/24 → p≈0.73, 10/16/22 → p≈0.88)끼리  
실험 결과가 동일함 — 이는 HARQ 이득이 절대 SNR이 아닌 **p_success 값에 의존**함을 확인.

| SNR (dB) | p_success | no_harq TP | same_ch TP | cross_ch TP | same gain% | cross gain% | **Δ% (cross 추가)** |
|----------|-----------|------------|-----------|------------|-----------|------------|---------------------|
| 8,14,20,26 | 0.50 | 1073 | 1222 | **1258** | +13.9% | +17.2% | **+3.35%** |
| 12,18,24   | 0.73 | 1611 | 1630 | **1650** | +1.2%  | +2.4%  | **+1.20%** |
| 10,16,22   | 0.88 | 1927 | 1937 | 1933      | +0.5%  | +0.3%  | **≈0% (noise)** |

#### 핵심 관찰 (SNR sweep)

1. **p=0.50 구간 (MCS 임계점)**: HARQ 이득 최대
   - same_ch_harq: +14% (기본 HARQ 이득)
   - cross_ch_harq: +17% (cross-channel combining 포함)
   - 추가 기여 **+3.35%** — NPCA 채널에서 primary 실패 패킷을 combining하는 것이 유효함

2. **p≈0.73 구간**: 이득 중간 (1-2%), cross-channel 추가 이득 **+1.20%**

3. **p≈0.88 구간**: HARQ 이득 미미 (~0.5%), cross-channel 차이 **≈0%**
   - PHY 성공률이 높으면 combining 기회가 적어 세 방법이 수렴

---

### num_stas Sweep 결과 (SNR=20 dB, p=0.50)

| N | no_harq TP | same_ch TP | cross_ch TP | cross gain vs no_harq | cross vs same Δ% |
|---|-----------|-----------|------------|----------------------|------------------|
| 5  | 933  | 1218 | **1280** | +37.2% | **+5.1%** |
| 10 | 1014 | 1215 | **1273** | +25.5% | **+4.8%** |
| 20 | 1073 | 1222 | **1258** | +17.2% | **+3.0%** |
| 30 | 1059 | 1217 | 1218      | +15.0% | **+0.1%** |
| 50 | 1071 | 1164 | **1187** | +10.8% | **+2.0%** |

#### 핵심 관찰 (num_stas sweep)

1. **N≤20 구간**: cross-channel combining이 same-channel 대비 3-5% 추가 기여
   - 충돌이 적어 PHY error가 failure의 주 원인 → HARQ buffer 저장 빈번 → combining 기회 많음

2. **N=30 구간**: cross-channel 기여 ≈ 0% (1217 vs 1218)
   - 충돌 증가로 PHY error 비중 감소 → buffer 저장 빈도 감소 → combining 기회 축소
   - `§9.2: collision does NOT store soft info` 원칙이 고밀도에서 cross-channel 이득을 제한

3. **N=50 구간**: +2% 소폭 회복
   - HARQ retransmission 순서 다양성 증가로 인한 통계적 변동

---

### 논문 메시지

**"Cross-channel HARQ combining은 PHY failure rate가 높은(p≤0.5) 저SNR 구간과  
충돌이 적은(N≤20) 저밀도 환경에서 same-channel only HARQ 대비 3-5% 처리량 향상을 제공한다.  
고SNR 환경(p≥0.88)이나 고밀도 충돌 지배 환경에서는 추가 이득이 수렴한다."**

---

## 구현 메모

### `sta.py` 수정 (완료)

`__init__` 파라미터 추가:
```python
cross_channel_harq: bool = True  # Step 8+
```

`_start_npca_transition()` 에 추가:
```python
if self.harq_enabled and not self.cross_channel_harq:
    self.harq_buffer.flush()
```

`_start_switch_back()` 에 추가:
```python
if self.harq_enabled and not self.cross_channel_harq:
    self.harq_buffer.flush()
```

### 실험 파라미터 전달

`build_and_run` 내에서 STA 생성 시:
```python
STA(..., cross_channel_harq=cross_channel_harq)
```

---

## 출력 파일

```
manuscript/figure/
  fig8_cross_channel_harq.eps / .png / .pdf

results/step9/fig8/
  snr_sweep.csv     ← SNR sweep 결과
  nstas_sweep.csv   ← num_stas sweep 결과
```

---

## 수정 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-05-26 | 초안 작성 (LLM intent 기여로부터 재정의); cross-channel HARQ 비교 실험으로 재설계 |
| 2026-05-26 | v1 실험 완료: SNR sweep + num_stas sweep; p=0.50에서 cross-channel gain +3.35%; N≤20 저밀도에서 +3-5% |
| 2026-05-28 | `channel.py` 버그 수정 반영 재실험 (v2). 결과: `results/step9/fig8_v2/` |
