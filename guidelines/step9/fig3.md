# Figure 3: Fixed CW_npca_init(qsrc) → NPCA Collision Burst

**연구 질문 (RQ3)**: fixed small CW_npca_init은 NPCA collision burst를 유발하는가?

**스크립트**: `harq_sim/run_step9_fig3.py`

**출력**: `manuscript/figure/fig3_*.{eps,png,pdf}`

---

## 실험 파라미터

| 항목 | 값 |
|---|---|
| **Sweep 변수** | `npca_qsrc` ∈ {0, 1, 2, 3, 4, 5} × `num_stas` ∈ {2, 5, 10, 15, 20} |
| OBSS 채널 점유율 | 30% (`obss_rate = _occupancy_to_rate(0.30) ≈ 0.0039`) |
| `snr_db_mean` | 20.0 dB |
| `snr_db_std` | 0.0 dB |
| `obss_min` | 20 슬롯 |
| `obss_max` | 200 슬롯 |
| `num_slots` | 50,000 |
| Seeds | [42, 123, 456] |

> **주의**: `snr_db=20dB` (고 SNR, 결정론적) 사용 이유 —  
> PHY 실패를 최소화하여 충돌 효과를 분리 관찰.  
> PHY 실패와 충돌이 혼재하면 qsrc 효과 해석이 어려움.

## 비교 대상 (1개 기법, qsrc × num_stas 2D sweep)

| 기법 | npca_enabled | harq_enabled | adaptive_cw | npca_qsrc |
|---|---|---|---|---|
| `fixed_cw_npca_harq` | True | True | False | sweep (0~5) |

## CW 변환 공식

```
npca_cw = 2^qsrc × (CW_MIN + 1) − 1 = 2^qsrc × 16 − 1
qsrc=0 → CW=15,  qsrc=1 → CW=31,  qsrc=2 → CW=63
qsrc=3 → CW=127, qsrc=4 → CW=255, qsrc=5 → CW=511
```

## 측정 지표

- `collision_probability_npca` — NPCA 채널 충돌 확률
- `aggregate_throughput` — 전달 패킷 수
- `mean_access_delay` — 평균 지연
- `npca_transition_count` — 전환 횟수

## Figure 구성

```
Figure 3: 3-row subplot (x축: npca_qsrc for (a)(b), num_stas for (c))
  Panel (a): Throughput vs qsrc, 선 per num_stas (2/5/10/15/20)
             각 선의 최대값에 ★ 마커 = qsrc*(n)
  Panel (b): Collision Prob vs qsrc, 선 per num_stas
  Panel (c): 막대: 최적 qsrc*(n) vs num_stas
             점선: qsrc=0 대비 throughput gain (%) — 이중 y축
```

## 핵심 관찰 (실험 결과)

### v1 결과 (obss_max=200, N≤20, occupancy=30%)
- Collision prob: qsrc 증가에 따라 단조 감소 (num_stas 클수록 qsrc=0에서 높음)
- Throughput: 단조 감소 (예상한 역U자 미출현)
  - qsrc=0이 최고: 충돌률 높아도 전환 횟수 많아 총 TX 우위
  - qsrc=5: 충돌 0%지만 NPCA timer 만료 전 backoff 못 끝남 → transition count 급감
- 최적 qsrc*: num_stas에 무관하게 대부분 qsrc=0 (small CW 환경에서 aggressive 전략이 유리)

### v2 결과 (obss_max=500, N∈{5,10,20,30,50}, occupancy=50%) → `results/step9/fig3_v2/`
**qsrc* 이동 확인됨** (평균 NPCA 창 ~260슬롯):

| num_stas | qsrc* | CW* | TP gain vs qsrc=0 | col@qsrc=0 | col@qsrc* |
|---|---|---|---|---|---|
| 5  | 0 | 15  | 0.00% | 26.5% | 26.5% |
| 10 | 0 | 15  | 0.00% | 49.2% | 49.2% |
| 20 | 0 | 15  | 0.00% | 72.1% | 72.1% |
| 30 | **1** | **31** | +0.47% | 82.6% | 66.5% |
| 50 | **2** | **63** | +1.01% | 92.6% | 59.4% |

- N=50 상세: qsrc=2에서 transition count 최고(1558) — 충돌 감소 + 완료 가능한 backoff의 균형점
- qsrc* 이동 원인: 창이 길어지면(260슬롯) CW=31~63의 backoff가 완료될 공간이 생겨 충돌 감소 이득이 전환 횟수 감소 손실을 초과
- **⚠️ 모델 결함**: OBSS 생성이 intra-BSS TX 중 차단되어 실제 OBSS 점유율이 N=50에서 11%로 급감함
  - 원인: `channel.py generate_obss()`의 `if self.is_busy()` 조건이 intra-BSS TX도 차단
  - 결과: N이 클수록 경쟁이 심해 OBSS 이벤트가 줄어 NPCA 기회가 감소하고 qsrc 최적화 효과가 희석됨

### v3 결과 (channel.py 수정 후, OBSS 독립 생성) → `results/step9/fig3_v3/`

**수정**: `generate_obss()`의 `if self.is_busy()` → `if self.obss_remain > 0`
- 물리적 의미: OBSS는 외부 BSS의 hidden terminal로부터 도착하므로 우리 BSS의 intra-TX와 무관하게 생성
- 결과: N에 무관하게 OBSS 이벤트 수 ~96으로 균등 (vs 수정 전 N=50에서 22개)

| num_stas | qsrc* | CW* | TP gain vs qsrc=0 | col@qsrc=0 | col@qsrc* |
|---|---|---|---|---|---|
| 5  | 0 | 15  | 0.00% | 27.5% | 27.5% |
| 10 | 0 | 15  | 0.00% | 47.6% | 47.6% |
| 20 | **1** | **31** | **+1.52%** | 72.2% | 53.7% |
| 30 | **1** | **31** | **+3.31%** | 83.2% | 66.9% |
| 50 | **1** | **31** | **+6.11%** | 92.2% | 83.0% |

- N=20부터 qsrc*=1이 최적, N≥30에서 gain이 뚜렷하게 증가 (3~6%)
- N=50: qsrc=0 대비 +6.11% — 이전 v2의 +1.01% 대비 6배 개선
- transition count: N=50에서 ~5000 (v2: ~1100) — OBSS 기회 4.5배 증가

## 출력 파일

```
manuscript/figure/
  fig3_qsrc_sweep.eps / .png / .pdf   ← v3 결과 (최신, channel fix 적용)

results/step9/fig3/
  data.csv            ← v1 (obss_max=200, N≤20)
results/step9/fig3_v2/
  data.csv            ← v2 (obss_max=500, N≤50, channel bug 있음)
results/step9/fig3_v3/
  data.csv            ← v3 (channel fix 후)  ★ 현재 논문용
```

## 수정 이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-05-25 | 초안 작성 |
| 2026-05-26 | `harq_sim/run_step9_fig3.py` 구현 완료; obss_rate를 점유율 30% 기준으로 변환 적용 |
| 2026-05-26 | Fig 3을 `num_stas × qsrc` 2D 스윕으로 확장; 3-panel 구성 (throughput/collision per qsrc + optimal qsrc* per num_stas) |
| 2026-05-26 | 더 massive한 환경(obss_max=500, N≤50, occ=50%)으로 재실험; qsrc* 이동(N≥30에서 qsrc*=1→2) 확인; 결과 `results/step9/fig3_v2/` |
| 2026-05-28 | `channel.py` 수정: OBSS 생성 시 intra-BSS TX 중 차단 버그 수정 (`is_busy()` → `obss_remain > 0`); v3 재실험; N=50 gain +1.01% → +6.11%; 결과 `results/step9/fig3_v3/` |
