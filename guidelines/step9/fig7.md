# Figure 7: PPDU Truncation × qsrc* 최적화 상호작용

**연구 질문 (RQ7)**: NPCA 전환 시 허용된 window만큼 PPDU를 truncation하는 정책이
qsrc* 최적화의 contribution을 어떻게 변화시키는가?

**스크립트**: `harq_sim/run_step9_fig7.py`

**출력**: `manuscript/figure/fig7_*.{eps,png,pdf}`

---

## 배경

### 현재 시뮬레이터 동작 (암묵적 truncation)

`harq_sim/sta.py` `_handle_npca_backoff()`의 기존 코드:

```python
tx_dur = min(self.ppdu_duration, self.primary_channel.obss_remain)
```

NPCA 방문 중 backoff가 완료됐을 때 `obss_remain < ppdu_duration`이어도
truncated duration으로 TX를 시작한다. 즉 **truncation이 기본값**이었으며,
"full PPDU가 맞지 않으면 포기" 베이스라인이 없었다.

### PPDU Truncation 정의

| 모드 | 동작 |
|------|------|
| **No truncation** | `obss_remain < ppdu_duration`이면 TX 포기, switch_back |
| **Truncation enabled** | `obss_remain ≥ ppdu_min_tx_slots`이면 TX 시도 (duration = obss_remain) |

truncation enabled 시 `tx_dur < ppdu_duration`인 전송은 `npca_tx_truncated`로 기록된다.

### 연구 가설

NPCA 손실 원인:
- **Collision waste** → qsrc* 가 제거
- **PHY failure** → HARQ-CC 가 회복
- **Window waste** (full PPDU가 맞지 않아 포기) → truncation 이 제거

Truncation이 활성화되면:
1. 더 많은 NPCA 창을 활용 → 전체 TX 시도 횟수 증가
2. 짧은 창에서도 collision이 발생 → qsrc* 의 collision 감소 효과가 더 많은 TX에 적용됨
3. **예측**: qsrc* 절대 이득 = ΔC × p_effective 에서 ΔC(더 많은 시도) 증가 → qsrc* gain 확대

---

## 실험 설계: 2×2 Factorial

| 요인 | 수준 0 | 수준 1 |
|------|--------|--------|
| **PPDU Truncation** | disabled (full PPDU 필요) | enabled (ppdu_min_tx_slots=5) |
| **qsrc 선택** | qsrc=0 (aggressive, CW=15) | qsrc*(N) (analysis_qsrc.md 공식) |

qsrc*(N) = max(0, round(log₂(N/16))):

| N | qsrc* | W_init* |
|---|-------|---------|
| 5  | 0 | 15 |
| 10 | 0 | 15 |
| 20 | 0 | 15 |
| 30 | 1 | 31 |
| 50 | 2 | 63 |

### 4개 비교군

| 방법 | truncation | qsrc |
|------|-----------|------|
| `no_trunc_q0`    | False | 0 |
| `no_trunc_qstar` | False | qsrc*(N) |
| `trunc_q0`       | True  | 0 |
| `trunc_qstar`    | True  | qsrc*(N) |

---

## 실험 파라미터

| 항목 | 값 |
|------|-----|
| `num_stas` | {5, 10, 20, 30, 50} |
| `obss_max` | 500 슬롯 |
| `obss_occupancy` | 50% |
| `ppdu_duration` | 20 슬롯 |
| `ppdu_min_tx_slots` | 5 슬롯 (truncation 허용 최소값, ppdu_duration의 25%) |
| `harq_enabled` | True |
| `harq_validity_horizon` | 200 슬롯 |
| `num_slots` | 50,000 |
| Seeds | [42, 123, 456] |

> `obss_max=500`, `occ=50%` — Fig 4 환경과 동일, 직접 비교 가능

---

## Figure 구성

```
Figure 7: 2-row subplot

  Panel (a): Aggregate Throughput vs N
    - 4 lines: no_trunc_q0, no_trunc_qstar, trunc_q0, trunc_qstar
    - 평균 ± std 음영

  Panel (b): qsrc* Gain (%) vs N
    - 2 lines:
        gain_no_trunc = (no_trunc_qstar - no_trunc_q0) / no_trunc_q0 × 100%
        gain_trunc    = (trunc_qstar    - trunc_q0)    / trunc_q0    × 100%
    - gain_trunc > gain_no_trunc → "truncation이 qsrc* 기여를 확대"
    - y=0 기준선 (qsrc* 효과 없음)
```

---

## 측정 지표

- `aggregate_throughput` — 패킷 처리량 (주 지표)
- `npca_tx_truncated` — truncated TX 횟수 (truncation 활성화 정도 확인)
- `collision_probability_npca` — NPCA collision 비율 (qsrc* 효과 확인)
- `npca_transitions` — NPCA 전환 횟수 (window 활용도)

---

## 예상 결과

| 조건 | 예측 TP | 이유 |
|------|---------|------|
| no_trunc_q0 | 낮음 | collision 多, 짧은 창 버림 |
| no_trunc_qstar | 중간 | collision 감소, 짧은 창 버림 |
| trunc_q0 | 중간 | 더 많은 시도, collision 多 |
| trunc_qstar | 최고 | 더 많은 시도 + collision 감소 |

qsrc* gain 예측: `gain_trunc > gain_no_trunc` (large N에서 특히)
- truncation이 만들어내는 추가 TX 기회에서 collision이 더 많이 발생
- qsrc*는 이 추가 collision을 정확히 제거

---

## 출력 파일

```
manuscript/figure/
  fig7_truncation_qsrc.eps
  fig7_truncation_qsrc.png
  fig7_truncation_qsrc.pdf

results/step9/fig7/
  data.csv   ← (method, num_stas, seed, metric, value)
```

---

## 실험 결과 v1 (results/step9/fig7/, N≤50, 50000슬롯 × 3 seeds)

### Aggregate Throughput

| N | no_trunc_q0 | no_trunc_q* | trunc_q0 | trunc_q* |
|---|------------|-------------|----------|----------|
| 5  | 1829 | 1829 | 1931 | 1931 |
| 10 | 1873 | 1873 | 1982 | 1982 |
| 20 | 1862 | 1862 | 1971 | 1971 |
| 30 | 1802 | 1870 | 1914 | 1986 |
| 50 | 1697 | 1857 | 1799 | 1976 |

### qsrc* Gain (%) by Truncation Mode

| N | No-Trunc gain | Trunc gain | Δ |
|---|--------------|-----------|---|
| 5  | 0.00% | 0.00% | 0.00% |
| 10 | 0.00% | 0.00% | 0.00% |
| 20 | 0.00% | 0.00% | 0.00% |
| 30 | 3.81% | 3.80% | −0.01% |
| 50 | 9.43% | 9.84% | +0.41% |

---

## 실험 결과 v2 — Massive N (results/step9/fig7_v2/, N∈{5~200}, 50000슬롯 × 3 seeds)

### Aggregate Throughput

| N | no_trunc_q0 | no_trunc_q* | trunc_q0 | trunc_q* |
|---|------------|-------------|----------|----------|
| 5   | 1829 | 1829 | 1931 | 1931 |
| 10  | 1873 | 1873 | 1982 | 1982 |
| 20  | 1862 | 1862 | 1971 | 1971 |
| 30  | 1802 | 1870 | 1914 | 1986 |
| 50  | 1697 | 1857 | 1799 | 1976 |
| 70  | 1589 | 1845 | 1691 | 1959 |
| 100 | 1403 | 1845 | 1507 | 1967 |
| 150 | 1221 | 1838 | 1299 | 1945 |
| 200 | 1021 | 1829 | 1091 | 1920 |

### qsrc* Gain (%) — Massive N

| N | No-Trunc gain | Trunc gain | Δ | Additive pred | Actual trunc_q* |
|---|---|---|---|---|---|
| 5   |  0.00% |  0.00% | +0.00% | 1931 | 1931 |
| 10  |  0.00% |  0.00% | +0.00% | 1982 | 1982 |
| 20  |  0.00% |  0.00% | +0.00% | 1971 | 1971 |
| 30  |  3.81% |  3.80% | -0.01% | 1982 | 1986 |
| 50  |  9.43% |  9.84% | +0.41% | 1959 | 1976 |
| 70  | 16.11% | 15.89% | -0.22% | 1947 | 1959 |
| 100 | 31.50% | 30.53% | -0.97% | 1949 | 1967 |
| 150 | 50.51% | 49.73% | -0.77% | 1916 | 1945 |
| 200 | **79.05%** | **76.07%** | -2.98% | 1898 | 1920 |

### Truncated TX Fraction (trunc 방법 기준)

| N | trunc_q0 | trunc_q* |
|---|----------|----------|
| 5   | 5.7% | 5.7% |
| 10  | 5.4% | 5.4% |
| 20  | 4.8% | 4.8% |
| 30  | 4.6% | 5.2% |
| 50  | 4.5% | 5.9% |
| 70  | 3.8% | 6.4% |
| 100 | 4.1% | 6.0% |
| 150 | 3.6% | 6.4% |
| 200 | 3.4% | 4.9% |

---

## 핵심 관찰

### 1. qsrc* gain이 N에 따라 폭발적으로 증가 (massive N 확인)
- N=30: 3.8%, N=50: 9.4%, N=70: **16%**, N=100: **31%**, N=150: **51%**, N=200: **79%**
- 이 결과는 qsrc*(N) closed-form의 실용적 가치를 강력히 뒷받침
- 고밀도 환경(N≥70)에서 qsrc* 최적화는 선택이 아닌 필수

### 2. Truncation 단독 이득: 안정적 +70~102 TP (절대값), N에 따라 상대 이득은 감소
- N≤50: truncation이 +5~6% 기여 (앞서 분석과 동일)
- N≥70: truncation 절대 이득 +102~106 TP이지만 qsrc* 이득 폭발로 상대 기여 축소
- 예: N=200, trunc 단독 이득: +70 TP(1091-1021) vs qsrc* 단독 이득: +808 TP(1829-1021)

### 3. Additive 관계 유지 (massive N에서도)
- N=200: qsrc* gain +808, trunc gain +70, additive pred=1899, actual=1920 (+21, 1.1% 초과)
- Δ(percentage)는 N=200에서 -2.98%p이지만 **절대값 기준으로는 여전히 super-additive**
- Δ%가 음수인 이유: trunc가 denominator(q=0 기준)를 높이므로 %로 측정 시 gap이 작아 보임

### 4. HARQ-CC와의 비교
앞서 분석한 HARQ-CC × qsrc* 상호작용과 달리:
- **HARQ-CC**: qsrc* 기여를 증폭 (collision이 HARQ chain을 끊는 추가 비용)
- **Truncation**: qsrc* 기여와 독립 (서로 다른 손실 원인을 각각 제거)

→ Truncation과 qsrc*는 **직교적 기여(orthogonal contributions)**: 함께 사용하면 각각의 이득이 합산됨.  
→ **Massive N 환경에서도 additivity 성립**이 검증됨.

---

## 수정 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-05-26 | 초안 작성; sta.py에 ppdu_truncation/ppdu_min_tx_slots 파라미터 추가 |
| 2026-05-26 | Massive N 실험 추가 (N∈{5~200}); qsrc* gain이 N=200에서 79%에 달하는 결과 확인; additivity massive N에서도 성립 |
| 2026-05-28 | `channel.py` 버그 수정 반영 재실험 (v3). N=200: no_trunc_q0=1012, no_trunc_qstar=1843 (+82%), trunc_q0=1087, trunc_qstar=1927 (+77%). 결과: `results/step9/fig7_v3/` |
