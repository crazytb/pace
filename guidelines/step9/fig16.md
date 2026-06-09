# Figure 16: Topology-Aware MFG — Adjacency Graph 기반 NPCA 공간 재사용

**연구 질문 (RQ16)**: NPCA 경쟁을 전체 N STA 간 충돌(완전 그래프)이 아니라
실제 carrier sense 범위 기반 adjacency 그래프로 모델링하면,
MFG-adaptive 프로토콜의 공간 재사용(spatial reuse)이 얼마나 향상되는가?
그리고 그 이득은 그래프 연결도(connectivity)와 W_eff/N 비율에 따라 어떻게 달라지는가?

**핵심 전제 변경**:
- OBSS 트리거: 전체 BSS에 동시 영향 (글로벌, Fig 15와 동일)
- NPCA 충돌: carrier sense 범위 내 이웃 STA끼리만 발생 (로컬)
- 비인접 STA: 같은 슬롯에 동시 전송 성공 가능 → **공간 재사용**

**스크립트**: `harq_sim/run_step9_fig16.py` (미구현)

**출력**: `manuscript/figure/fig16_topology_mfg.{eps,png,pdf}`

---

## 1. 시스템 모델

### 그래프 정의

```
G = (V, E)
  V = {1, ..., N}     STA 집합
  (i, j) ∈ E  ⟺  STA i와 j가 서로 carrier sense 가능 (충돌 도메인 공유)

conflict neighborhood:
  N_i = { j ∈ V : (i,j) ∈ E }   (STA i의 이웃 집합)
```

### 슬롯 결과 정의

슬롯 t에서 전송 집합 T(t) ⊆ V:

```
STA i ∈ T(t) 의 전송 결과:
  SUCCESS   ⟺   T(t) ∩ (N_i ∪ {i}) = {i}    (이웃 중 자기만 전송)
  COLLISION ⟺   ∃ j ∈ N_i : j ∈ T(t)         (이웃 중 누군가도 전송)

한 슬롯에서 동시 성공 가능:
  {i, k} ⊆ T(t) 이고 (i,k) ∉ E 이면 i와 k 동시 성공 가능
  → 최대 동시 성공 수 = maximum independent set α(G)
```

### 처리량 상한

```
완전 그래프 K_N:   슬롯당 최대 성공 = 1    (α(K_N) = 1)
체인 P_N:          슬롯당 최대 성공 = ⌊N/2⌋  (α(P_N) = ⌈N/2⌉)
2D 격자 √N × √N:   슬롯당 최대 성공 = N/2   (체커보드 패턴)
독립 집합(no edge): 슬롯당 최대 성공 = N    (모두 동시 성공)
```

---

## 2. 알고리즘: Topology-Aware MFG Adaptive

### 핵심 변경: τ가 STA별로 달라짐

```
Fig 15 (완전 그래프):
  n(t) = global remaining
  τ*(t) = 1 / n(t)         ← 모든 STA 동일

Fig 16 (임의 토폴로지):
  n_i(t) = |active neighbors of i| + 1   ← STA별 로컬 카운트
  τ_i*(t) = 1 / n_i(t)                   ← STA별 상이
```

### 의사코드

```
Algorithm: Topology-Aware MFG-Adaptive
Input:  G = (V, E), W_eff, 정보 모델 (perfect | carrier_sense)
Output: success_count

─────────────────────────────────────────────────────
INITIALIZATION
  active = V                // 전체 STA
  success = 0
  t = 0

─────────────────────────────────────────────────────
MAIN LOOP (t = 0, 1, ..., W_eff - 1)

  IF active == ∅: BREAK

  STEP 1. 각 STA i ∈ active 의 n_i(t) 계산
    [Perfect local info]:
      n_i(t) = |N_i ∩ active| + 1        // 정확한 active 이웃 수

    [Carrier sense only]:
      n_i(t) = |N_i| + 1 − (감지된 성공 수)  // 이전 슬롯 busy 관찰 기반 추정

  STEP 2. 전송 확률 계산 및 Bernoulli 시행
    For each i ∈ active:
      τ_i = 1 / n_i(t)
      transmit[i] = Bernoulli(τ_i)

  STEP 3. 충돌 판정 (그래프 기반)
    For each i ∈ active where transmit[i] = 1:
      collision[i] = ∃ j ∈ N_i ∩ active : transmit[j] = 1

    // 충돌 없는 STA → 성공
    For each i where transmit[i] = 1 AND collision[i] = False:
      success += 1
      active.remove(i)

  t += 1
```

### 플로우차트

```
┌──────────────────────────────────────────────────────┐
│  NPCA Visit 시작                                       │
│  active = V = {1,...,N}  success = 0  t = 0           │
└────────────────────────────┬─────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │  t < W_eff AND active ≠ ∅ ?  │
              └──────┬───────────────┬────────┘
                     │ YES           │ NO → return success
                     ▼
     ┌───────────────────────────────────────────┐
     │  각 STA i ∈ active:                        │
     │    n_i = |N_i ∩ active| + 1               │
     │    τ_i = 1 / n_i                          │
     │    transmit[i] ~ Bernoulli(τ_i)            │
     └───────────────────────┬───────────────────┘
                             │
                             ▼
     ┌───────────────────────────────────────────┐
     │  충돌 판정 (그래프 기반):                    │
     │  For each i with transmit[i]=1:            │
     │    IF ∃ j∈N_i: transmit[j]=1              │
     │      → COLLISION (i 재시도)                │
     │    ELSE                                    │
     │      → SUCCESS (active에서 제거)           │
     └───────────────────────┬───────────────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │  t += 1                  │
              └──────────────────────────┘
```

---

## 3. 토폴로지 종류

### Sweep 대상 5종

```python
TOPOLOGY_TYPES = {
    "complete":  complete_graph(N),               # K_N (Fig 15 baseline)
    "chain":     path_graph(N),                   # 1D P_N
    "grid":      grid_2d_graph(N),                # √N × √N (N must be perfect square)
    "rgg":       random_geometric_graph(N, r),    # 반경 r의 임의 기하 그래프
    "erdos":     erdos_renyi_graph(N, p),          # G(N, p)
}
```

### 각 토폴로지 특성

| 토폴로지 | 평균 degree | α(G)/N | 공간 재사용 잠재력 | 802.11 대응 |
|---------|------------|--------|----------------|-----------|
| Complete K_N | N−1 | 1/N | 없음 | 단일 AP BSS |
| Chain P_N | 2 | ~1/2 | 높음 | 선형 복도 배치 |
| 2D Grid | 4 | ~1/2 | 높음 | 격자형 실내 |
| RGG(r) | ~πr²N | 가변 | 가변 | 현실적 실내 |
| Erdos G(N,p) | pN | 가변 | 가변 | 랜덤 배치 |

### RGG 파라미터 (가장 현실적)

```python
# N STAs를 [0,1]² 공간에 균등 배치, 반경 r 이내 STA끼리 edge
RGG_RADII = {
    "dense":  r = sqrt(log(N) / (π*N)),   # connectivity threshold (connected with high prob)
    "medium": r = 2 * threshold,
    "sparse": r = 1.5 * threshold,
}
```

---

## 4. 비교 프로토콜

| 프로토콜 | n_i(t) 계산 | 토폴로지 인식 | 비고 |
|---------|------------|-------------|------|
| `dcf_complete`        | N (고정)           | 없음 | Fig 15 dcf_qsrc_star 동일 |
| `dcf_topology`        | N (고정), 충돌은 로컬 | 물리적으로만 | 현실적 DCF |
| `mfg_full_N`          | N (고정), τ=1/N   | 없음 | Fig 15 adaptive를 그대로 적용 |
| `mfg_perfect_local`   | 정확한 n_i(t)      | 완전 | 정보 완전 oracle |
| `mfg_carrier_sense`   | 캐리어 센스 추정     | 부분 | 현실적 분산 구현 |
| `oracle_topology`     | 최적 CW_0 탐색     | 물리적으로만 | DCF 상한 |

**핵심 비교**:
- `mfg_full_N` vs `mfg_perfect_local`: 토폴로지 정보의 가치
- `mfg_perfect_local` vs `mfg_carrier_sense`: 정보 불완전성 비용
- `dcf_topology` vs `mfg_carrier_sense`: 현실적 구현에서 MFG의 이득

---

## 5. 정보 모델 상세

### Perfect Local Info

```python
# STA i는 매 슬롯 active ∩ N_i를 정확히 앎
n_i_t = len([j for j in neighbors[i] if j in active]) + 1
tau_i = 1.0 / n_i_t
```

언제 성립: AP가 active STA 목록을 매 슬롯 브로드캐스트하거나,
이전 성공 TX를 모두 청취 가능할 때.

### Carrier Sense Only (현실적)

```python
# STA i는 이웃의 TX 시도(busy)는 감지하지만
# 성공/충돌 여부는 모름 (ACK 없으면 불명확)
# → 추정: busy 감지된 슬롯 수 기반으로 n_i 하향 조정

class STALocalState:
    def __init__(self, degree):
        self.n_hat = degree + 1    # 초기 추정 = 초기 이웃 수 + 자기
    
    def observe(self, channel_busy: bool):
        if channel_busy:
            # busy = 이웃 중 누군가 아직 active → 정보 없음
            pass
        else:
            # idle = 이웃 모두 backoff 중 또는 이미 성공
            # EWMA로 n_hat 감소 (보수적)
            self.n_hat = max(1, self.n_hat * 0.95)
    
    @property
    def tau(self):
        return 1.0 / self.n_hat
```

한계: busy 슬롯이 "이웃이 많음"인지 "이웃이 적지만 충돌"인지 구분 불가.

---

## 6. 실험 파라미터

### Sweep 설정

```python
N_LIST      = [9, 16, 25, 36, 49]   # 완전 제곱수 (grid 구성용)
WEFF_LIST   = [20, 50, 100, 200, 500]
SEEDS       = [42, 123, 456, 789, 1234]   # 5 seeds (RGG 랜덤성 평균화)
N_VISITS    = 500                    # visits per config

# RGG 반경 (connectivity threshold 배수)
RGG_R_MULT  = [1.0, 1.5, 2.0, 3.0]  # threshold 배수
```

### 총 실험 횟수

```
5 N × 5 W_eff × 6 프로토콜 × 5 seeds × 4 토폴로지 = 3,000 configs
+ RGG 반경 sweep: 5 N × 5 W_eff × 6 프로토콜 × 5 seeds × 4 반경 = 3,000 configs
총 ≈ 6,000 runs
```

---

## 7. 측정 지표

| 지표 | 설명 | 핵심 비교 |
|------|------|-----------|
| `total_success` | visit당 총 성공 TX 수 | 처리량 |
| `spatial_reuse` | 동시 성공 발생 슬롯 비율 | 공간 재사용 실현 여부 |
| `efficiency` | `total_success / min(N, W_eff)` | Fig 15 대비 스케일 비교 |
| `topology_gain` | `(mfg_topo − mfg_full_N) / mfg_full_N` | 토폴로지 인식의 순수 이득 |
| `fairness` | Jain's index per STA success 분포 | 외곽 vs 중앙 STA 공정성 |
| `alpha_efficiency` | `total_success / (α(G) × W_eff)` | 이론 상한 대비 효율 |

### 보조 지표 (RGG 분석용)

```python
degree_stats = {
    "mean_degree": avg |N_i|,
    "degree_cv":   std/mean (불균등도),
    "alpha_ratio": α(G)/N,
}
```

---

## 8. Figure 구성

```
Figure 16: 4-panel

Panel (a): 토폴로지별 total_success vs N
           x: N ∈ {9,16,25,36,49}
           lines: dcf_complete / dcf_topology / mfg_full_N /
                  mfg_perfect_local / mfg_carrier_sense
           sub-plots: complete / chain / grid / rgg(medium) — 4열
           핵심: sparse 토폴로지에서 mfg_perfect_local이 mfg_full_N 크게 초과

Panel (b): topology_gain vs 평균 degree / N (연결도 정규화)
           x: mean_degree / (N−1)  ∈ [0, 1]  (0=완전 분리, 1=완전 그래프)
           y: (mfg_perfect_local − mfg_full_N) / mfg_full_N (%)
           lines: W_eff/N ∈ {0.5, 1, 2, 5, 10}
           핵심: 연결도 낮을수록 + W_eff/N 낮을수록 토폴로지 이득 증가

Panel (c): spatial_reuse vs W_eff/N ratio
           x: W_eff/N (log scale)
           y: 동시 성공 발생 비율
           lines: chain / grid / rgg(medium) (complete는 항상 0)
           핵심: W_eff < N 구간에서 spatial reuse가 mfg_adaptive 이득을 증폭

Panel (d): Information model 비교 — mfg_perfect_local vs mfg_carrier_sense
           x: N
           y: topology_gain (%)
           lines: perfect / carrier_sense (정보 불완전성 비용 시각화)
           sub-plots: chain / grid / rgg (3열)
           핵심: carrier sense 추정이 perfect에 얼마나 근접하는가
```

---

## 9. 예상 결과

### Panel (a) 예측

```
Complete graph (K_N):
  모든 프로토콜 동일 (Fig 15 결과 재현)
  dcf_complete = dcf_topology, mfg_full_N ≈ mfg_perfect_local

Chain (P_N), N=25, W_eff=25:
  dcf_complete:      ≈ 8   (완전 그래프 가정, 낭비)
  dcf_topology:      ≈ 10  (로컬 충돌, 약간 더 나음)
  mfg_full_N:        ≈ 8   (τ=1/N, 공간 재사용 못 활용)
  mfg_perfect_local: ≈ 15  (τ_i=1/2, 비인접 동시 성공 활용)
  이론 상한 α(P_25)=13: min(13, W_eff=25) → ≈13
```

### Panel (b) 예측

```
연결도 → 0 (sparse):
  n_i → 2 (이웃 1~2개) → τ_i → 1/2 → 공격적
  비인접 STA 동시 성공 ↑ → topology_gain 최대

연결도 → 1 (complete):
  n_i → N → τ_i → 1/N (Fig 15와 동일)
  topology_gain = 0%

예상: topology_gain ≈ f(1 − mean_degree/N) × g(N/W_eff)
```

### Panel (c) 예측

```
W_eff < N (tight window):
  공간 재사용 없이도 MFG adaptive가 이미 +70% 이득 (Fig 15)
  토폴로지 공간 재사용으로 추가 이득

W_eff >> N:
  모든 STA가 순차 성공 가능 → 공간 재사용 기회 적음
  → spatial_reuse 낮음
```

### Panel (d) 예측

```
Chain topology: carrier sense이 perfect에 가까울 것
  (degree=2 → 이웃 2개만 추적, 추정 오차 작음)

Grid topology: degree=4 → 약간 더 어려운 추정

RGG (high degree): carrier sense 추정 오차 증가
  → mfg_carrier_sense가 mfg_perfect_local보다 5~15% 낮을 것
```

---

## 10. 구현 설계

### 핵심 데이터 구조

```python
import numpy as np
from scipy.sparse import csr_matrix

class TopologyVisitSim:
    """
    Adjacency 토폴로지 기반 NPCA visit 시뮬레이터.
    Non-adjacent STAs can succeed simultaneously.
    """
    def __init__(self, adj: np.ndarray):
        """
        adj: (N, N) boolean adjacency matrix (symmetric, no self-loops)
        """
        self.adj = adj  # adj[i,j] = True iff i and j are neighbors
        self.N = adj.shape[0]
    
    def sim_mfg_perfect(self, W_eff: int, n_visits: int, rng) -> np.ndarray:
        """τ_i*(t) = 1/n_i(t) with perfect local knowledge."""
        success = np.zeros(n_visits, dtype=np.int32)
        
        for v in range(n_visits):
            active = np.ones(self.N, dtype=bool)
            
            for t in range(W_eff):
                if not active.any():
                    break
                
                # n_i(t) = active neighbors + self
                active_neighbors = self.adj & active[np.newaxis, :]  # (N, N)
                n_i = active_neighbors.sum(axis=1) + 1               # (N,)
                n_i = np.where(active, n_i, 1)
                
                tau_i = np.where(active, 1.0 / n_i, 0.0)
                tx = rng.random(self.N) < tau_i                      # (N,)
                
                # 충돌 판정: 이웃 중 tx한 STA 존재?
                neighbor_tx = (self.adj & tx[np.newaxis, :]).any(axis=1)  # (N,)
                solo = tx & ~neighbor_tx & active
                
                success[v] += solo.sum()
                active &= ~solo
        
        return success
    
    def sim_dcf(self, W_eff: int, CW0: int, n_visits: int, rng) -> np.ndarray:
        """Standard DCF with local collision detection."""
        ...
```

### 그래프 생성 유틸리티

```python
def make_topology(topo_type: str, N: int, **kwargs) -> np.ndarray:
    """Returns (N, N) adjacency matrix."""
    if topo_type == "complete":
        adj = np.ones((N, N), dtype=bool)
        np.fill_diagonal(adj, False)
    
    elif topo_type == "chain":
        adj = np.zeros((N, N), dtype=bool)
        for i in range(N-1):
            adj[i, i+1] = adj[i+1, i] = True
    
    elif topo_type == "grid":
        side = int(np.round(np.sqrt(N)))
        assert side * side == N, "N must be perfect square for grid"
        adj = np.zeros((N, N), dtype=bool)
        for i in range(N):
            r, c = divmod(i, side)
            for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < side and 0 <= nc < side:
                    j = nr * side + nc
                    adj[i, j] = True
    
    elif topo_type == "rgg":
        # Random geometric graph in [0,1]^2
        r = kwargs.get("radius", np.sqrt(np.log(N) / (np.pi * N)) * 2)
        rng = np.random.default_rng(kwargs.get("seed", 42))
        pos = rng.random((N, 2))
        dist = np.linalg.norm(pos[:, np.newaxis] - pos[np.newaxis, :], axis=2)
        adj = (dist < r) & ~np.eye(N, dtype=bool)
    
    elif topo_type == "erdos":
        p = kwargs.get("p", 4 * np.log(N) / N)   # connectivity threshold
        rng = np.random.default_rng(kwargs.get("seed", 42))
        adj = rng.random((N, N)) < p
        adj = adj & adj.T & ~np.eye(N, dtype=bool)
    
    return adj
```

---

## 11. Fig 15와의 연결

Fig 16은 Fig 15의 **완전 그래프 특수 케이스를 일반화**:

```
Fig 15 결과 재현 조건:
  topo = "complete" → n_i(t) = remaining(t) → τ_i*(t) = 1/remaining
  → mfg_perfect_local ≡ mfg_adaptive (Fig 15)

Fig 16 신규 기여:
  sparse 토폴로지 → 공간 재사용 + 개인화된 τ_i
  → 처리량 상한이 1 → α(G)로 증가
  → W_eff < α(G) 구간에서 추가 이득
```

### 논문에서의 위치

```
§ Extension 4 : Fig 16 (topology-aware MFG)
  "When carrier-sense range limits collision domain to N_i ⊂ V,
   the MFG fixed-point condition per STA becomes τ_i*(t) = 1/n_i(t),
   enabling spatial reuse. Across graph topologies, the throughput
   gain over full-connectivity DCF scales with (1 − mean_degree/N)
   and is amplified in the tight-window regime (W_eff/N ≤ 4)."
```

---

## 12. 출력 파일

```
manuscript/figure/
  fig16_topology_mfg.eps / .png / .pdf

results/step9/fig16/
  data.csv    ← (topology, N, W_eff, protocol, seed,
                  mean_success, std_success, spatial_reuse,
                  mean_degree, alpha_G, efficiency)
  fig16_topology_mfg_preview.png
```

---

## 실험 결과 (results/step9/fig16/, 200 visits × 5 seeds)

### 핵심 발견: topology_gain (mfg_perfect_local vs mfg_full_N)

| topology | N  | W_eff=20 | W_eff=50 | W_eff=100 |
|----------|---:|--------:|---------:|----------:|
| Chain    |  9 |   +0.0% |    +0.0% |     +0.0% |
| Chain    | 25 |  +35.5% |    +0.0% |     +0.0% |
| Chain    | 49 | +156.7% |    +7.0% |     +0.0% |
| Grid     |  9 |   +0.1% |    +0.0% |     +0.0% |
| Grid     | 25 |  +41.7% |    +0.0% |     +0.0% |
| Grid     | 49 | +164.4% |    +8.8% |     +0.0% |
| Complete | 25 |    0.0% |    +0.0% |     +0.0% |  ← 이론 검증 ✅

**Gain 조건**: W_eff/N < ~1.5 구간에서만 발생. W_eff ≥ 2N이면 모든 STA가 순차 성공 가능 → spatial reuse 불필요.

**논문 메시지**: "With sparse topology (chain/grid), MFG τᵢ=1/nᵢ(t) achieves up to +157% throughput over τ=1/N baseline at W_eff/N=0.41, because non-adjacent STAs succeed simultaneously and τ_i adapts to local neighborhood size."

### Spatial Reuse (chain N=25, mfg_perfect_local)

| W_eff | W_eff/N | Spatial reuse rate |
|------:|--------:|-------------------:|
|    20 |    0.80 |              0.295 |
|    50 |    2.00 |              0.118 |
|   100 |    4.00 |              0.059 |
|   200 |    8.00 |              0.030 |
|   500 |   20.00 |              0.012 |

Spatial reuse rate ∝ 1/W_eff — tight window에서 매 슬롯 공간 재사용이 빈번.

### IEEE CW_min=15 vs MFG (dcf_ieee 추가 비교)

#### Complete K_N, W_eff=50

| N  | dcf_ieee (CW=15) | dcf_topology (CW=2N) | mfg_perfect_local | oracle |
|---:|-----------------:|---------------------:|------------------:|-------:|
|  9 |             8.23 |                 8.06 |              9.00 |   8.49 |
| 25 |             9.94 |                11.41 |             18.95 |  11.80 |
| 49 |         **6.05** |                11.14 |             18.47 |  11.81 |

→ N=49: IEEE CW=15가 최악. N이 커질수록 충돌 폭발 (N >> CW_min+1).

#### Chain P_N, W_eff=20

| N  | dcf_ieee | mfg_perfect_local | oracle |
|---:|---------:|------------------:|-------:|
|  9 |     8.27 |          **9.00** |   8.89 |
| 16 |    14.74 |         **16.00** |  15.57 |
| 25 |    23.00 |         **25.00** |  24.26 |
| 36 |    33.03 |         **36.00** |  33.52 |
| 49 |    44.86 |         **49.00** |  44.98 |

→ Chain에서 mfg_perfect_local은 **항상 N 달성** (모든 STA 성공, 공간 재사용).
→ dcf_ieee는 chain에서 oracle과 유사 — sparse 토폴로지에서는 CW=15도 준-최적.

**핵심 교차 인사이트**: IEEE CW_min=15는 K_N 가정(802.11 원래 설계) 하에서는 N>~15이면 급격히 열화. 반면 실제 sparse 토폴로지(chain/grid)에서는 CW=15도 괜찮지만 MFG local이 모든 경우에서 이론 상한 달성.

---

## 수정 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-06-03 | 초안 작성 — Fig 15 완전 그래프 → adjacency 토폴로지 일반화 설계 |
| 2026-06-03 | 구현 완료 (`run_step9_fig16.py`) 및 실험 결과 추가 — Chain/Grid N=49 W_eff=20에서 +157/+164% |
| 2026-06-03 | dcf_ieee (IEEE CW_min=15 고정) 프로토콜 추가 — Complete에서 열화, Chain에서 oracle 근접 확인 |
