# NPCA-HARQ 시뮬레이션 핵심 개요 (§1–3)

항상 로드: 프로젝트 목적과 시스템 구조를 이해하기 위한 최소 컨텍스트.

---

## 1. 시뮬레이션의 목적

본 시뮬레이션은 IEEE 802.11bn NPCA 환경에서 **Hybrid ARQ 기반 재전송 제어**가 channel access delay, throughput, packet delivery ratio, collision probability, fairness에 미치는 영향을 분석하기 위한 것이다.

기존 NPCA 시뮬레이터는 다음 동작을 포함한다고 가정한다.

- BSS primary channel에서 EDCA/CSMA/CA 기반 backoff 수행
- BSS primary channel이 inter-BSS PPDU 등에 의해 busy할 때 NPCA transition 수행
- NPCA primary channel에서 별도의 EDCA backoff 수행
- NPCA TXOP 획득 후 data transmission 수행
- NPCA 종료 후 BSS primary channel로 switch-back
- primary channel의 기존 EDCA state 저장 및 복원

본 확장에서는 여기에 다음 기능을 추가한다.

- MPDU/MSDU transmission failure 발생 시 ARQ 또는 HARQ 재전송 선택
- HARQ soft combining gain 모델링
- HARQ buffer 유지/flush
- primary channel 또는 NPCA primary channel 중 재전송 위치 선택
- adaptive `CW_npca_init` 또는 `Initial_NPCA_QSRC` 제어
- LLM-designed reward 또는 intent-based reward profile을 사용할 수 있는 RL interface 제공

---

## 2. 핵심 개념

### 2.1 기존 ARQ

ARQ에서는 전송 실패 시 동일 packet을 재전송하지만, receiver는 이전 실패 전송에서 얻은 soft information을 사용하지 않는다고 가정한다.

```text
Transmission attempt 1: fail
Transmission attempt 2: independent retry
```

성공 확률은 매 attempt마다 현재 channel condition, MCS, collision 여부에 의해 결정된다.

### 2.2 Hybrid ARQ

HARQ에서는 이전 실패 전송에서 얻은 soft information을 receiver buffer에 저장하고, 다음 재전송과 combining하여 decoding success probability를 높인다.

본 시뮬레이션에서는 우선 **HARQ-CC Chase Combining**을 기본으로 한다.

```text
Transmission attempt 1: fail, soft information stored
Transmission attempt 2: same packet retransmitted
Receiver combines attempt 1 + attempt 2
Decoding success probability increases
```

초기 구현에서는 다음과 같이 단순화한다.

- HARQ-CC 사용
- 동일 packet에 대한 retransmission count가 증가할수록 effective SNR 또는 decoding success probability 증가
- HARQ combining은 같은 receiver가 soft buffer를 유지하고 있을 때만 가능
- HARQ buffer lifetime 또는 validity horizon이 지나면 buffer flush

---

## 3. 전체 시스템 구조

### 3.1 주요 객체

기존 코드에 다음 객체 또는 속성을 추가한다.

```text
Environment
 ├── Channel primary_channel
 ├── Channel npca_channel
 ├── STA list
 ├── AP
 ├── event scheduler or slot loop
 ├── logger
 └── reward/evaluation module

STA
 ├── primary EDCA state
 ├── NPCA EDCA state
 ├── queue
 ├── HARQ buffer
 ├── current mode
 ├── transition timers
 ├── retry counters
 └── decision policy

HARQBuffer
 ├── packet_id
 ├── receiver_id
 ├── original_mcs
 ├── combining_count
 ├── accumulated_snr or accumulated_reliability
 ├── first_tx_time
 ├── last_tx_time
 ├── validity_deadline
 └── active flag

Packet
 ├── packet_id
 ├── arrival_time
 ├── size_bits
 ├── traffic_class
 ├── latency_deadline
 ├── retry_count
 ├── harq_count
 ├── current_mcs
 ├── status
 └── transmission_history
```

---

## 모듈 구조 요약 (harq_sim/)

```
harq_sim/
├── __init__.py          ← 모듈 export
├── enums.py             ← STAMode, ChannelType, TxType, FailureReason, TrafficClass, PacketStatus, Action, NPCA_ACTIONS
├── channel.py           ← Channel 클래스 (OBSS/intra-BSS, obss_remain = NPCA_PPDU_REM_DUR)
├── packet.py            ← Packet, TransmissionAttempt 데이터 클래스
├── phy.py               ← PHY layer: logistic PER 모델, MCS 선택, HARQ-CC SNR 변환 (Step 2+)
├── harq_buffer.py       ← HARQBuffer 클래스 — Chase Combining soft buffer (Step 3+)
├── policy.py            ← NPCAHARQPolicy — primary/NPCA delay 비교 기반 action 선택 (Step 4+)
├── sta.py               ← STA 상태 머신 (이중 EDCA state, NPCA_TIMER, ARQ, HARQ-CC, policy)
├── simulator.py         ← Slot-based 이벤트 루프, 충돌 해결, CSV 출력
├── configs.py           ← CW, 슬롯 시간, OBSS, 에너지 상수
├── run_step1.py ~ run_step4.py  ← CLI 실행 스크립트

tests/
├── test_step1_npca.py   ← 7개 테스트 (all pass)
├── test_step2_arq.py    ← 8개 테스트 (all pass)
├── test_step3_harq.py   ← 9개 테스트 (all pass)
└── test_step4_policy.py ← 8개 테스트 (all pass)
```
