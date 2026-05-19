# npca_semi_mdp_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, Optional
import random

# Import from your random_access.py
from drl_framework.random_access import STA, Channel, Simulator, STAState, OccupyRequest
from drl_framework.configs import (
    RANDOM_OBSS_DURATION_RANGE, 
    RANDOM_PPDU_VARIANTS, 
    RANDOM_OBSS_GENERATION_RATE_RANGE,
    PPDU_DURATION_VARIANTS
)

class NPCASemiMDPEnv(gym.Env):
    """
    Semi-MDP Environment for NPCA decision making
    Decision points occur when STA is in PRIMARY_BACKOFF and detects OBSS
    """
    def __init__(self, 
                 num_stas: int = 2,
                 num_slots: int = 1000,
                 obss_generation_rate: float = 0.1,
                 npca_enabled: bool = True,
                 # Throughput + Latency reward weights
                 throughput_weight: float = 10.0,  # 처리량 보상 강화
                 latency_penalty_weight: float = 0.1,  # 지연 패널티
                 # State space enhancement
                 history_length: int = 10,  # Channel history length
                 # Random environment parameters
                 random_env: bool = False,  # Enable random OBSS/PPDU variations
                 obss_duration_range: Tuple[int, int] = None,  # Custom OBSS duration range
                 ppdu_variants: list = None):  # Custom PPDU variants list
        super().__init__()
        
        self.num_stas = num_stas
        self.num_slots = num_slots
        self.obss_generation_rate = obss_generation_rate
        self.npca_enabled = npca_enabled
        
        # Reward components
        self.throughput_weight = throughput_weight
        self.latency_penalty_weight = latency_penalty_weight
        
        # State enhancement
        self.history_length = history_length
        self.channel_history = []  # [primary_busy, obss_busy, npca_busy] for last N slots
        
        # Random environment setup
        self.random_env = random_env
        self.obss_duration_range = obss_duration_range or RANDOM_OBSS_DURATION_RANGE
        self.ppdu_variants = ppdu_variants or RANDOM_PPDU_VARIANTS
        
        # Current episode parameters (will be randomized if random_env=True)
        self.current_obss_duration = None
        self.current_ppdu_variant = None
        self.current_ppdu_duration = None
        
        # Action Space: 0 = stay in PRIMARY (frozen), 1 = switch to NPCA
        self.action_space = spaces.Discrete(2)
        
        # Enhanced observation space with channel history and environment parameters
        self.observation_space = spaces.Dict({
            'current_slot': spaces.Discrete(num_slots),
            'backoff_counter': spaces.Discrete(1024),  # Max CW
            'cw_index': spaces.Discrete(7),  # CW stages
            'obss_remaining': spaces.Discrete(500),  # Max OBSS duration (increased for random env)
            'channel_busy_intra': spaces.Discrete(2),  # Boolean
            'channel_busy_obss': spaces.Discrete(2),   # Boolean
            'npca_channel_busy': spaces.Discrete(2),   # Boolean
            # Channel history features
            'primary_busy_history': spaces.Box(low=0, high=1, shape=(history_length,), dtype=np.float32),
            'obss_busy_history': spaces.Box(low=0, high=1, shape=(history_length,), dtype=np.float32),
            'npca_busy_history': spaces.Box(low=0, high=1, shape=(history_length,), dtype=np.float32),
            # Aggregate statistics
            'obss_frequency': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'avg_obss_duration': spaces.Box(low=0, high=500, shape=(1,), dtype=np.float32),
            # Environment parameters (for model awareness)
            'current_obss_duration': spaces.Discrete(500),  # Current episode OBSS duration
            'current_ppdu_duration': spaces.Discrete(100),  # Current episode PPDU duration
        })
        
        self.reset()
    
    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        
        # Simplified random environment: only PPDU duration varies, OBSS fixed at 100
        if self.random_env:
            self.current_obss_duration = 100  # Fixed OBSS duration
            # Random PPDU variant only
            variant_options = ['short', 'medium', 'long', 'extra_long']
            self.current_ppdu_variant = random.choice(variant_options)
            self.current_ppdu_duration = PPDU_DURATION_VARIANTS[self.current_ppdu_variant]
            current_obss_gen_rate = self.obss_generation_rate  # Fixed OBSS rate
        else:
            # Use fixed values (backward compatibility)
            self.current_obss_duration = 100  # Default fixed value
            self.current_ppdu_variant = 'medium'
            self.current_ppdu_duration = PPDU_DURATION_VARIANTS['medium']
            current_obss_gen_rate = self.obss_generation_rate
        
        # Initialize channels with current episode parameters
        self.primary_channel = Channel(0, current_obss_gen_rate, 
                                     obss_duration_range=(self.current_obss_duration, self.current_obss_duration))
        self.npca_channel = Channel(1, 0.0) if self.npca_enabled else None
        self.channels = [self.primary_channel, self.npca_channel] if self.npca_channel else [self.primary_channel]
        
        # Initialize STAs with current PPDU duration
        self.stas = []
        for i in range(self.num_stas):
            sta = STA(
                sta_id=i,
                channel_id=0,  # Primary channel
                primary_channel=self.primary_channel,
                npca_channel=self.npca_channel,
                npca_enabled=self.npca_enabled,
                ppdu_duration=self.current_ppdu_duration
            )
            self.stas.append(sta)
        
        # Current decision-making STA (we'll focus on STA 0 for simplicity)
        self.decision_sta = self.stas[0]
        
        # Simulator state
        self.current_slot = 0
        self.episode_reward = 0
        self.decision_count = 0
        
        # Initialize channel history
        self.channel_history = []
        self.obss_events = []  # Track OBSS events for statistics
        
        # Find first decision point
        self._advance_to_next_decision()
        
        return self._get_observation(), {}
    
    def _is_decision_point(self, sta: STA, slot: int) -> bool:
        """
        Check if current state requires decision making
        Decision needed when: PRIMARY_BACKOFF + OBSS detected
        (regardless of backoff counter value)
        """
        return (sta.state == STAState.PRIMARY_BACKOFF and 
                sta.primary_channel.is_busy_by_obss(slot))
    
    def _update_channel_history(self):
        """Update channel history with current slot information"""
        sta = self.decision_sta
        
        # Current channel status
        primary_intra_busy = int(sta.primary_channel.is_busy_by_intra_bss(self.current_slot))
        obss_busy = int(sta.primary_channel.is_busy_by_obss(self.current_slot))
        npca_busy = int(sta.npca_channel.is_busy(self.current_slot)) if sta.npca_channel else 0
        
        # Add to history
        self.channel_history.append([primary_intra_busy, obss_busy, npca_busy])
        
        # Track OBSS events for statistics
        if obss_busy and (not self.channel_history or 
                         len(self.channel_history) < 2 or 
                         self.channel_history[-2][1] == 0):
            # New OBSS event started
            self.obss_events.append({'start': self.current_slot, 'duration': 1})
        elif obss_busy and len(self.obss_events) > 0:
            # Continue existing OBSS
            self.obss_events[-1]['duration'] = self.current_slot - self.obss_events[-1]['start'] + 1
        
        # Keep only recent history
        if len(self.channel_history) > self.history_length:
            self.channel_history = self.channel_history[-self.history_length:]
    
    def _advance_to_next_decision(self) -> bool:
        """
        Advance simulation until next decision point or episode end
        Returns True if decision point found, False if episode ended
        """
        max_advance = 1000  # Prevent infinite loops
        advance_count = 0
        
        while (self.current_slot < self.num_slots and 
               advance_count < max_advance and
               not self._is_decision_point(self.decision_sta, self.current_slot)):
            
            # Update channels
            for ch in self.channels:
                ch.update(self.current_slot)
                ch.generate_obss(self.current_slot)
            
            # Update channel history
            self._update_channel_history()
            
            # Update all STAs (passive simulation)
            for sta in self.stas:
                sta.occupy_request = None
                sta.step(self.current_slot)
                sta.state = sta.next_state
            
            self.current_slot += 1
            advance_count += 1
        
        return self.current_slot < self.num_slots and advance_count < max_advance
    
    def _get_observation(self) -> Dict:
        """Get enhanced observation for the decision-making STA"""
        sta = self.decision_sta
        
        # Pad history if not enough data
        if len(self.channel_history) < self.history_length:
            padding_size = self.history_length - len(self.channel_history)
            padded_history = [[0, 0, 0]] * padding_size + self.channel_history
        else:
            padded_history = self.channel_history[-self.history_length:]
        
        # Extract history arrays
        primary_history = np.array([h[0] for h in padded_history], dtype=np.float32)
        obss_history = np.array([h[1] for h in padded_history], dtype=np.float32)
        npca_history = np.array([h[2] for h in padded_history], dtype=np.float32)
        
        # Calculate aggregate statistics
        total_slots = len(self.channel_history)
        obss_frequency = np.mean(obss_history) if len(obss_history) > 0 else 0.0
        
        if self.obss_events:
            avg_obss_duration = np.mean([event['duration'] for event in self.obss_events])
        else:
            avg_obss_duration = 0.0
        
        return {
            'current_slot': self.current_slot,
            'backoff_counter': sta.backoff,
            'cw_index': sta.cw_index,
            'obss_remaining': sta.primary_channel.obss_remain,
            'channel_busy_intra': int(sta.primary_channel.is_busy_by_intra_bss(self.current_slot)),
            'channel_busy_obss': int(sta.primary_channel.is_busy_by_obss(self.current_slot)),
            'npca_channel_busy': int(sta.npca_channel.is_busy(self.current_slot)) if sta.npca_channel else 0,
            # Enhanced features
            'primary_busy_history': primary_history,
            'obss_busy_history': obss_history,
            'npca_busy_history': npca_history,
            'obss_frequency': np.array([obss_frequency], dtype=np.float32),
            'avg_obss_duration': np.array([avg_obss_duration], dtype=np.float32),
            # Environment parameters (for model awareness)
            'current_obss_duration': self.current_obss_duration,
            'current_ppdu_duration': self.current_ppdu_duration,
        }
    
    def _update_random_ppdu_duration(self):
        """매 결정 시점에서 PPDU duration을 랜덤하게 업데이트"""
        from drl_framework.configs import PPDU_DURATION_VARIANTS
        
        # PPDU 변형 중 랜덤 선택
        variant_options = ['short', 'medium', 'long', 'extra_long']
        selected_variant = np.random.choice(variant_options)
        new_ppdu_duration = PPDU_DURATION_VARIANTS[selected_variant]
        
        # 현재 PPDU duration 업데이트
        self.current_ppdu_variant = selected_variant
        self.current_ppdu_duration = new_ppdu_duration
        
        # 모든 STA의 PPDU duration 업데이트
        for sta in self.stas:
            sta.ppdu_duration = new_ppdu_duration
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute action and advance to next decision point
        action: 0 = stay in PRIMARY (frozen), 1 = switch to NPCA
        """
        if not self._is_decision_point(self.decision_sta, self.current_slot):
            raise ValueError("step() called when not at decision point!")
        
        # Randomize PPDU duration at each decision point (if random_env enabled)
        if self.random_env:
            self._update_random_ppdu_duration()
        
        sta = self.decision_sta
        start_slot = self.current_slot
        
        # Apply action - decision between staying primary vs switching to NPCA
        if action == 0:
            # Stay in primary channel -> PRIMARY_FROZEN (wait for OBSS to end)
            sta.next_state = STAState.PRIMARY_FROZEN
        else:
            # Switch to NPCA channel
            if sta.npca_enabled and sta.npca_channel:
                # Reset CW and backoff for NPCA
                sta.cw_index = 0
                sta.backoff = sta.generate_backoff()
                
                # Check NPCA channel status
                if sta.npca_channel.is_busy_by_intra_bss(self.current_slot):
                    sta.next_state = STAState.NPCA_FROZEN
                else:
                    sta.next_state = STAState.NPCA_BACKOFF
            else:
                # Fallback to primary frozen if NPCA not available
                sta.next_state = STAState.PRIMARY_FROZEN
        
        sta.state = sta.next_state
        
        # Simulate until next decision point - Throughput + Latency 기반 보상
        duration = 0
        waiting_slots = 0  # 실제 대기한 슬롯 수 (frozen/backoff 상태)
        
        # 옵션 시작 시점의 channel_occupancy_time 기록
        initial_occupancy = sta.channel_occupancy_time
        
        while (self.current_slot < self.num_slots and 
               duration < 500 and  # Max duration limit
               not self._is_decision_point(sta, self.current_slot)):
            
            # Update channels
            for ch in self.channels:
                ch.update(self.current_slot)
                ch.generate_obss(self.current_slot)
            
            # Update channel history
            self._update_channel_history()
            
            # Count waiting slots (when STA is not transmitting)
            if (sta.state in [STAState.PRIMARY_FROZEN, STAState.NPCA_FROZEN, 
                             STAState.PRIMARY_BACKOFF, STAState.NPCA_BACKOFF]):
                waiting_slots += 1
            
            # Update all STAs
            for s in self.stas:
                s.occupy_request = None
                s.step(self.current_slot)
                s.state = s.next_state
            
            self.current_slot += 1
            duration += 1
        
        # Original reward calculation (ppdu_* 모델과 동일)
        successful_transmission_slots = sta.channel_occupancy_time - initial_occupancy
        
        # 1. Base throughput reward (전송 성공한 슬롯 수에 비례)
        throughput_reward = self.throughput_weight * successful_transmission_slots
        
        # 2. Original latency penalty (ppdu_* 모델과 완전히 동일)
        latency_penalty = self.latency_penalty_weight * duration
        
        # 총 보상 = 처리량 보상 - 지연 패널티 (원래 공식)
        cumulative_reward = throughput_reward - latency_penalty
        
        # Check if episode is done
        done = (self.current_slot >= self.num_slots)
        
        # Get next observation
        next_obs = self._get_observation() if not done else {}
        
        info = {
            'duration': duration,
            'start_slot': start_slot,
            'end_slot': self.current_slot,
            'decision_count': self.decision_count,
            'successful_transmission_slots': successful_transmission_slots,
            'waiting_slots': waiting_slots,
            'transmission_efficiency': successful_transmission_slots / max(duration, 1),
            'action_taken': "Stay PRIMARY" if action == 0 else "Switch to NPCA",
            # Compatible reward components
            'throughput_reward': throughput_reward,
            'latency_penalty': latency_penalty,
            'total_reward': cumulative_reward
        }
        
        self.decision_count += 1
        
        return next_obs, cumulative_reward, done, False, info
    
    def _calculate_slot_reward(self, sta: STA) -> float:
        """지연된 보상 구조: 슬롯별 즉시 보상 제거"""
        # 슬롯별 즉시 보상 제거 - 모든 보상은 에피소드 종료 시에만 계산
        # 실제 보상은 에피소드 종료 시 성공적으로 전송한 슬롯 수로 계산됨
        return 0.0
    
    def render(self, mode='human'):
        """Render current state (optional)"""
        if mode == 'human':
            print(f"Slot: {self.current_slot}, STA State: {self.decision_sta.state.name}, "
                  f"Backoff: {self.decision_sta.backoff}, Decisions: {self.decision_count}")


# Test the environment
if __name__ == "__main__":
    env = NPCASemiMDPEnv(num_stas=2, num_slots=1000)
    
    print("Testing Semi-MDP NPCA Environment...")
    obs, _ = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Decision point - Primary backoff with OBSS detected!")
    
    for step_count in range(10):  # Test 10 decisions
        # Random action for testing
        action = random.choice([0, 1])
        action_name = "Stay PRIMARY" if action == 0 else "Switch to NPCA"
        print(f"\nStep {step_count + 1}: Taking action {action} ({action_name})")
        
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"Reward: {reward:.3f}")
        print(f"Duration: {info['duration']} slots")
        print(f"Next observation: {obs}")
        
        if done:
            print("Episode finished!")
            break
    
    print(f"\nTotal decisions made: {env.decision_count}")