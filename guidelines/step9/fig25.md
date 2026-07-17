# Figure 25: Collision-Cost Sensitivity and Mandatory RTS/CTS — Mixed NPCA Channel (표준 단위 v2)

**연구 질문 (RQ25)**:
(A) 충돌 비용이 커지면 PACE 성능이 줄어드는가? conventional NPCA가 역전하는가?
(B) RTS/CTS **의무화**(성공 전송에도 핸드셰이크 OH 부과, 충돌은 RTS 충돌 비용만) 시 영향은?

> v1(추상 슬롯, OH=2/3 임의값)은 폐기. v2 = **모든 시간량을 IEEE 802.11
> 표준 파라미터에서 유도** (비율 논증 제거).

**스크립트**: `harq_sim/run_step9_fig25.py`
**출력**: `manuscript/figure/fig25_collision_cost.{eps,png,pdf}`
**데이터**: `results/step9/fig25/data.csv`

---

## 1. 표준 타이밍 (OFDM PHY, 5 GHz; 시뮬 슬롯 ≡ aSlotTime σ=9µs)

| 항목 | 유도 | 값 | σ 슬롯 |
|---|---|---|---|
| aSlotTime | 표준 | 9 µs | 1 |
| SIFS / DIFS | 표준 / SIFS+2σ | 16 / 34 µs | — |
| aPHY-RX-START-Delay | 표준 | 25 µs | — |
| RTS=CTS @24Mbps | 20µs 프리앰블+2심볼 | 28 µs | — |
| RTS=CTS @6Mbps | 20µs 프리앰블+8심볼 | 52 µs | — |
| **성공 OH @24M** | RTS+SIFS+CTS+SIFS | **88 µs** | **10** |
| 성공 OH @6M | 〃 | 136 µs | 15 |
| **RTS 충돌 @24M** | RTS+CTS_Timeout(SIFS+σ+RX-START=50µs) | **78 µs** | **9** |
| RTS 충돌 @6M | 〃 | 102 µs | 11 |

프레임/창 (σ 단위 현실화): visitor PPDU U[25,100]σ = **225–900µs**
(1500B@65Mbps ≈ 225µs ~ 소형 A-MPDU/저MCS), E[L]=62.5σ=562µs;
native 50σ=450µs; W_eff=420σ=**3.78ms** (TXOP급 NPCA 창, W/E[L]≈6.7로
fig24 regime 유지). DCF CW_min=16 (aCWmin=15).

## 2. 결과 (full: 5 seeds × 20 reps × 50 transitions, 2026-07-17)

### Sweep A — 충돌 비용 C 스윕 (basic access, 성공=ppdu)

| C (σ / µs) | dcf succ_v | pace succ_v | 비율 | dcf tot | pace tot |
|---|---|---|---|---|---|
| 9 / 81 | 0.479 | 0.739 | **1.54** | 0.805 | 0.804 |
| 31 / 279 | 0.377 | 0.544 | 1.44 | 0.621 | 0.593 |
| 63 / 567 ≈ E[L] | 0.293 | 0.400 | 1.36 | 0.475 | 0.438 |
| 125 / 1125 | 0.211 | 0.254 | **1.21** | 0.342 | 0.282 |
| nocd (max Lᵢ) | 0.284 | 0.356 | 1.25 | 0.463 | 0.393 |

- **H1 PASS**: PACE 절대 성능 단조 감소 (0.74→0.25).
- **H2 PASS**: visitor 우위 ×1.54→×1.21로 축소, **전 구간 역전 없음**.
- **H3 수정 (v1 결론 정정)**: 표준 단위에선 **basic access 전 구간에서 채널
  효율은 dcf ≥ pace** (C=9 동률 → C↑ 시 격차 확대, 최대 -0.06). v1의
  "C≈E[L] 교차" 주장은 현실에 없는 초저가 충돌(C≪RTS 충돌 비용) 구간에
  의존한 것이었음. 실현 가능한 최저 충돌 비용(RTS 78µs)에서도 동률이 한계.

### Sweep B — 의무 RTS/CTS

| 설정 | 방법 | visitor | Δ | 채널 | Δ |
|---|---|---|---|---|---|
| basic | dcf | 0.284 | — | 0.463 | — |
| basic | pace | 0.356 | — | 0.393 | — |
| **@24Mbps** | dcf | 0.418 | +47% | 0.691 | +49% |
| **@24Mbps** | **pace** | **0.640** | **+80%** | **0.697** | +78% |
| @6Mbps | dcf | 0.384 | +35% | 0.635 | +37% |
| @6Mbps | pace | 0.572 | +61% | 0.622 | +58% |

- **H4 PASS (비대칭)**: PACE 이득(+80%)이 dcf(+47%)의 ~1.7배. 단 v1과 달리
  **dcf도 크게 이득** — basic 충돌 = max Lᵢ ≈ 700µs로 워낙 비싸서 RTS/CTS
  절감이 dcf에도 큼.
- **채널 효율: 열세 → 동률 회복**: basic에서 pace 0.393 < dcf 0.463이던 것이
  @24Mbps에서 0.697 ≈ 0.691. 의무 RTS/CTS가 PACE의 유일한 채널 열세를 지움.
- visitor 우위는 유지·확대: ×1.25(basic) → ×1.53(@24M).
- OH 세금 = 창의 ~9%(@24M)/13%(@6M). 제어율 낮아도 순이득 유지 (양쪽 다).

## 3. 논문 함의

1. **PACE = visitor-side 프로토콜**: visitor 효율 우위는 충돌 가격 전 구간
   생존(+21~54%), 채널 효율은 basic access에서 소폭 열세 — 정직하게 명시.
2. **의무 RTS/CTS = PACE의 채널 열세 해소 장치**: 충돌을 78µs로 캡하면
   PACE 채널 효율이 dcf와 동률, visitor 우위 ×1.53. "PACE의 낭비(충돌)는
   RTS/CTS로 고칠 수 있고, conventional의 낭비(frozen backoff)는 불치"
   구도는 유지되나 v1보다 온건한 표현 필요.
3. 한계: 전 STA RTS/CTS 가정, fairness(native 억압) 축 불변, ACK/재전송
   미모델.
