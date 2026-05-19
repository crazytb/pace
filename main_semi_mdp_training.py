#!/usr/bin/env python3
"""
Semi-MDP 기반 NPCA STA 학습을 위한 메인 실행 파일

이 파일은 Semi-MDP를 사용하여 STA의 NPCA 결정을 학습시킵니다.
STA는 Primary 채널이 OBSS로 점유된 상황에서 다음 두 액션 중 선택합니다:
- Action 0: StayPrimary (Primary 채널에서 대기)  
- Action 1: GoNPCA (NPCA 채널로 이동하여 전송)

사용법:
    python main_semi_mdp_training.py
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from drl_framework.random_access import Channel
from drl_framework.train import train_semi_mdp, train_semi_mdp_with_env
from drl_framework.configs import (
    PPDU_DURATION, 
    PPDU_DURATION_VARIANTS,
    RADIO_TRANSITION_TIME, 
    OBSS_GENERATION_RATE, 
    OBSS_DURATION_RANGE,
    DEFAULT_NUM_EPISODES,
    DEFAULT_NUM_SLOTS_PER_EPISODE,
    DEFAULT_NUM_STAS_CH0,
    DEFAULT_NUM_STAS_CH1,
)

def create_training_config(obss_duration=None, ppdu_variant='medium', random_env=False):
    """학습을 위한 설정 생성"""
    
    if random_env:
        # Random environment mode - channel and STA configs will be handled by environment
        from drl_framework.configs import RANDOM_OBSS_DURATION_RANGE, RANDOM_PPDU_VARIANTS
        print(f"Random environment enabled:")
        print(f"  OBSS duration range: {RANDOM_OBSS_DURATION_RANGE}")
        print(f"  PPDU variants: {RANDOM_PPDU_VARIANTS}")
        
        # Return None for channels - environment will handle randomization
        return None, None
    else:
        # Fixed environment mode (backward compatibility)
        obss_duration = obss_duration or 50  # Default value
        
        # PPDU duration 설정
        ppdu_duration = PPDU_DURATION_VARIANTS.get(ppdu_variant, PPDU_DURATION)
        
        # 채널 설정 - centralized configs 사용
        channels = [
            Channel(channel_id=0, obss_generation_rate=OBSS_GENERATION_RATE['secondary']),  # Secondary/NPCA channel (no OBSS)
            Channel(channel_id=1, obss_generation_rate=OBSS_GENERATION_RATE['primary'], obss_duration_range=(obss_duration, obss_duration))  # Primary channel (with OBSS)
        ]
        
        # STA 설정 - PPDU duration 변형 적용
        stas_config = []

        # Secondary 채널의 STA들 (기존 방식) - DEFAULT_NUM_STAS_CH0개
        for i in range(DEFAULT_NUM_STAS_CH0):
            stas_config.append({
                "sta_id": i,
                "channel_id": 0,  # Secondary channel
                "npca_enabled": False,
                "ppdu_duration": ppdu_duration,
                "radio_transition_time": RADIO_TRANSITION_TIME
            })
        
        # Primary 채널의 STA들 (NPCA 지원) - DEFAULT_NUM_STAS_CH1개
        for i in range(DEFAULT_NUM_STAS_CH1):
            stas_config.append({
                "sta_id": i,
                "channel_id": 1,  # Primary channel
                "npca_enabled": True,
                "ppdu_duration": ppdu_duration,
                "radio_transition_time": RADIO_TRANSITION_TIME
            })

        return channels, stas_config

def calculate_running_average(data, window_size):
    """주어진 데이터에 대한 이동 평균을 계산합니다."""
    if len(data) < window_size:
        return data
    
    running_avg = []
    for i in range(len(data)):
        if i < window_size - 1:
            # 초기 구간에서는 사용 가능한 모든 데이터 평균 사용
            running_avg.append(np.mean(data[:i+1]))
        else:
            # 윈도우 크기만큼의 구간 평균 사용
            running_avg.append(np.mean(data[i-window_size+1:i+1]))
    
    return running_avg

def plot_training_results(episode_rewards, episode_losses, save_dir="./results", reward_window=100, loss_window=50, obss_duration=None):
    """학습 결과를 플롯으로 저장 (러닝 평균 오버레이 포함)"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 보상 러닝 평균 계산
    reward_running_avg = calculate_running_average(episode_rewards, reward_window)
    
    # 보상 그래프
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    # 원본 데이터 (반투명)
    plt.plot(episode_rewards, alpha=0.3, color='lightblue', label='Raw Rewards')
    # 러닝 평균 (진한 색상)
    plt.plot(reward_running_avg, color='darkblue', linewidth=2, label=f'Running Avg (window={reward_window})')
    title = 'Episode Rewards with Running Average'
    if obss_duration:
        title += f' (OBSS Duration: {obss_duration})'
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 손실 그래프
    plt.subplot(1, 2, 2)
    if episode_losses:
        # 손실 러닝 평균 계산
        loss_running_avg = calculate_running_average(episode_losses, loss_window)
        
        # 원본 데이터 (반투명)
        plt.plot(episode_losses, alpha=0.3, color='lightcoral', label='Raw Loss')
        # 러닝 평균 (진한 색상)
        plt.plot(loss_running_avg, color='darkred', linewidth=2, label=f'Running Avg (window={loss_window})')
        title = 'Training Loss with Running Average'
        if obss_duration:
            title += f' (OBSS Duration: {obss_duration})'
        plt.title(title)
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_results.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Results saved to {save_dir}/training_results.png")
    print(f"Running averages: Rewards (window={reward_window}), Loss (window={loss_window})")

def run_experiment(obss_duration=None, experiment_name="experiment", 
                   ppdu_variant='medium', random_env=False, random_ppdu=False):
    """실험 실행 - 고정 환경 또는 랜덤 환경 지원"""
    
    if random_env:
        print(f"실험 시작: {experiment_name} (Random Environment)")
        print("Random OBSS/PPDU variations enabled")
    else:
        ppdu_duration = PPDU_DURATION_VARIANTS.get(ppdu_variant, PPDU_DURATION)
        print(f"실험 시작: {experiment_name}")
        print(f"OBSS Duration: {obss_duration}, PPDU Duration: {ppdu_duration} ({ppdu_variant})")
        print(f"STA Density: CH0={DEFAULT_NUM_STAS_CH0} STAs, CH1={DEFAULT_NUM_STAS_CH1} STAs")
        if random_ppdu:
            print("PPDU Duration will be randomized during training.")
    print("-" * 50)
    
    # 설정 생성
    channels, stas_config = create_training_config(obss_duration, ppdu_variant, random_env)
    
    # 학습 파라미터 - centralized configs 사용
    num_episodes = DEFAULT_NUM_EPISODES
    num_slots_per_episode = DEFAULT_NUM_SLOTS_PER_EPISODE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if random_env:
        # # Semi-MDP 환경 직접 사용 (random_env=True)
        # from npca_semi_mdp_env import NPCASemiMDPEnv
        
        # # Environment 기반 학습 - 새로운 학습 함수 필요
        # episode_rewards, episode_losses, learner = train_semi_mdp_with_env(
        #     num_episodes=num_episodes,
        #     num_slots_per_episode=num_slots_per_episode,
        #     device=device,
        #     random_env=True
        # )
        episode_rewards, episode_losses, learner = train_semi_mdp(
            channels=channels,
            stas_config=stas_config, 
            num_episodes=num_episodes,
            num_slots_per_episode=num_slots_per_episode,
            device=device,
            random_ppdu=True
        )
    else:
        # 기존 방식 - 채널/STA 설정 기반
        episode_rewards, episode_losses, learner = train_semi_mdp(
            channels=channels,
            stas_config=stas_config, 
            num_episodes=num_episodes,
            num_slots_per_episode=num_slots_per_episode,
            device=device,
            random_ppdu=False
        )
    
    # 결과 저장
    results_dir = f"./density_comparison_results/{experiment_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    # 학습된 모델 저장
    torch.save({
        'policy_net_state_dict': learner.policy_net.state_dict(),
        'target_net_state_dict': learner.target_net.state_dict(),
        'optimizer_state_dict': learner.optimizer.state_dict(),
        'episode_rewards': episode_rewards,
        'episode_losses': episode_losses,
        'steps_done': learner.steps_done,
        'obss_duration': obss_duration if not random_env else "random",
        'ppdu_variant': ppdu_variant if not random_env else "random",
        'ppdu_duration': PPDU_DURATION_VARIANTS.get(ppdu_variant, PPDU_DURATION) if not random_env else "random",
        'random_env': random_env
    }, f"{results_dir}/model.pth")
    
    # 그래프 생성
    plot_training_results(episode_rewards, episode_losses, results_dir, obss_duration=obss_duration)
    
    # 실험 통계
    final_avg_reward = sum(episode_rewards[-50:]) / min(50, len(episode_rewards))
    max_reward = max(episode_rewards)
    
    print(f"실험 완료: {experiment_name}")
    print(f"평균 보상 (최근 50 에피소드): {final_avg_reward:.2f}")
    print(f"최대 보상: {max_reward:.2f}")
    print(f"총 학습 스텝: {learner.steps_done}")
    print("-" * 50)
    print()
    
    return {
        'experiment_name': experiment_name,
        'obss_duration': obss_duration,
        'episode_rewards': episode_rewards,
        'episode_losses': episode_losses,
        'final_avg_reward': final_avg_reward,
        'max_reward': max_reward,
        'steps_done': learner.steps_done
    }

def plot_comparison_results(experiment_results):
    """모든 실험 결과를 비교하는 그래프 생성"""
    plt.figure(figsize=(20, 12))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # 1. 에피소드별 보상 비교 (러닝 평균)
    plt.subplot(2, 3, 1)
    for i, result in enumerate(experiment_results):
        rewards = result['episode_rewards']
        running_avg = calculate_running_average(rewards, 50)
        plt.plot(running_avg, color=colors[i], linewidth=2, 
                label=f"OBSS Duration: {result['obss_duration']}")
    plt.title('Episode Rewards Comparison (Running Average)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 최종 성능 비교 (최근 50 에피소드 평균)
    plt.subplot(2, 3, 2)
    durations = [result['obss_duration'] for result in experiment_results]
    final_rewards = [result['final_avg_reward'] for result in experiment_results]
    plt.bar(range(len(durations)), final_rewards, color=colors[:len(durations)])
    plt.title('Final Performance Comparison')
    plt.xlabel('OBSS Duration')
    plt.ylabel('Average Reward (Last 50 Episodes)')
    plt.xticks(range(len(durations)), durations)
    for i, v in enumerate(final_rewards):
        plt.text(i, v + 0.1, f'{v:.2f}', ha='center')
    plt.grid(True, alpha=0.3)
    
    # 3. 최대 보상 비교
    plt.subplot(2, 3, 3)
    max_rewards = [result['max_reward'] for result in experiment_results]
    plt.bar(range(len(durations)), max_rewards, color=colors[:len(durations)])
    plt.title('Maximum Reward Comparison')
    plt.xlabel('OBSS Duration')
    plt.ylabel('Maximum Reward')
    plt.xticks(range(len(durations)), durations)
    for i, v in enumerate(max_rewards):
        plt.text(i, v + 0.1, f'{v:.2f}', ha='center')
    plt.grid(True, alpha=0.3)
    
    # 4. 학습 진행 곡선 비교 (처음 200 에피소드)
    plt.subplot(2, 3, 4)
    for i, result in enumerate(experiment_results):
        rewards = result['episode_rewards'][:200]
        running_avg = calculate_running_average(rewards, 20)
        plt.plot(running_avg, color=colors[i], linewidth=2, 
                label=f"OBSS Duration: {result['obss_duration']}")
    plt.title('Early Learning Progress (First 200 Episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. 수렴 속도 비교 (보상이 특정 임계값을 넘는 에피소드)
    plt.subplot(2, 3, 5)
    convergence_episodes = []
    threshold = 10.0  # 임계 보상값
    
    for result in experiment_results:
        rewards = result['episode_rewards']
        running_avg = calculate_running_average(rewards, 20)
        convergence_ep = next((i for i, r in enumerate(running_avg) if r >= threshold), len(rewards))
        convergence_episodes.append(convergence_ep)
    
    plt.bar(range(len(durations)), convergence_episodes, color=colors[:len(durations)])
    plt.title(f'Convergence Speed (Episodes to reach {threshold} reward)')
    plt.xlabel('OBSS Duration')
    plt.ylabel('Episodes to Convergence')
    plt.xticks(range(len(durations)), durations)
    for i, v in enumerate(convergence_episodes):
        plt.text(i, v + 5, f'{v}' if v < len(experiment_results[i]['episode_rewards']) else 'N/A', ha='center')
    plt.grid(True, alpha=0.3)
    
    # 6. 학습 안정성 비교 (최근 100 에피소드 표준편차)
    plt.subplot(2, 3, 6)
    stability = []
    for result in experiment_results:
        recent_rewards = result['episode_rewards'][-100:]
        stability.append(np.std(recent_rewards))
    
    plt.bar(range(len(durations)), stability, color=colors[:len(durations)])
    plt.title('Learning Stability (Std of Last 100 Episodes)')
    plt.xlabel('OBSS Duration')
    plt.ylabel('Standard Deviation')
    plt.xticks(range(len(durations)), durations)
    for i, v in enumerate(stability):
        plt.text(i, v + 0.1, f'{v:.2f}', ha='center')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 저장
    comparison_dir = "./density_comparison_results"
    plt.savefig(f"{comparison_dir}/obss_duration_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"비교 결과 저장됨: {comparison_dir}/obss_duration_comparison.png")

def run_ppdu_experiments(obss_duration=100):
    """여러 PPDU duration 변형에 대해 실험 실행"""
    print("="*70)
    print("PPDU Duration 변인통제 실험")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"OBSS Duration: {obss_duration} slots")
    print()
    
    results = []
    
    # 각 PPDU duration 변형에 대해 실험
    for variant, duration in PPDU_DURATION_VARIANTS.items():
        print(f"\n{'='*50}")
        print(f"PPDU Duration 실험: {variant} ({duration} slots)")
        print(f"{'='*50}")
        
        experiment_name = f"ppdu_{variant}_obss_{obss_duration}"
        result = run_experiment(obss_duration, experiment_name, ppdu_variant=variant)
        result['ppdu_variant'] = variant
        results.append(result)
    
    return results

def main():
    """메인 학습 함수"""
    print("="*60)
    print("Semi-MDP DRL 모델 훈련")
    print("="*60)
    
    # 커맨드라인 인자 처리
    parser = argparse.ArgumentParser(description='Semi-MDP 기반 NPCA STA 학습 - STA Density 실험')
    
    # Boolean flags - 현재 사용하지 않는 옵션들 (고정 설정)
    # parser.add_argument('--random-ppdu', action='store_true', default=True,
    #                     help='PPDU duration을 랜덤화 (기본값: True)')
    # parser.add_argument('--no-random-ppdu', dest='random_ppdu', action='store_false',
    #                     help='PPDU duration 랜덤화 비활성화')
    # parser.add_argument('--random-env', action='store_true', default=False,
    #                     help='랜덤 환경 모드 활성화')
    
    # Value arguments - 현재 사용하지 않는 옵션들 (고정 설정)
    # parser.add_argument('--obss-duration', type=int, default=100,
    #                     help='OBSS duration (slots, 기본값: 100)')
    # parser.add_argument('--ppdu-variant', choices=list(PPDU_DURATION_VARIANTS.keys()), 
    #                     default='medium', help='PPDU duration 변형 (기본값: medium)')
    # parser.add_argument('--mode', choices=['single', 'ppdu_experiments'], default='single',
    #                     help='실행 모드 (기본값: single)')
    
    # args = parser.parse_args()
    
    # 고정 설정값들 (STA density 실험용)
    random_ppdu_flag = True      # 항상 랜덤 PPDU 사용
    random_env = False           # 고정 환경 사용
    obss_duration = 100          # OBSS duration 고정
    ppdu_variant = 'medium'      # PPDU variant 고정 (랜덤이므로 실제로는 사용되지 않음)
    run_mode = 'single'          # 단일 실험 모드
    
    if random_ppdu_flag:
        print("Random PPDU duration enabled.")
    if random_env:
        print("Random environment mode enabled")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # PPDU experiments mode는 현재 사용하지 않음 (STA density 실험에 집중)
    # if run_mode == 'ppdu_experiments':
    #     # PPDU duration 변인통제 실험
    #     results = run_ppdu_experiments(obss_duration)
    #     
    #     print("\n" + "="*70)
    #     print("PPDU Duration 실험 결과 요약")
    #     print("="*70)
    #     for result in results:
    #         variant = result['ppdu_variant']
    #         ppdu_dur = PPDU_DURATION_VARIANTS[variant]
    #         print(f"{variant:12} ({ppdu_dur:2d} slots): 평균 보상 {result['final_avg_reward']:6.2f}, "
    #               f"최대 보상 {result['max_reward']:6.2f}")
    #     print("="*70)
    #     
    # else:
    
    # STA Density 실험 실행
    if random_env:
        experiment_name = "random_env_robust_model"
        print("Random environment training enabled")
    else:
        print(f"OBSS Duration: {obss_duration} slots")
        if random_ppdu_flag:
            ppdu_naming = "random"
        else:
            ppdu_naming = ppdu_variant
        print(f"PPDU Variant: {ppdu_naming}")
        print(f"STA Configuration: CH0={DEFAULT_NUM_STAS_CH0}, CH1={DEFAULT_NUM_STAS_CH1}")
        experiment_name = f"ch0_{DEFAULT_NUM_STAS_CH0}_ch1_{DEFAULT_NUM_STAS_CH1}"
    print()
    
    result = run_experiment(obss_duration, experiment_name, ppdu_variant, random_env, random_ppdu=random_ppdu_flag)
    
    # 결과 출력
    print("\n" + "="*60)
    print("DRL 모델 훈련 완료!")
    print("="*60)
    print("훈련 결과:")
    print("-" * 40)
    print(f"OBSS Duration: {result['obss_duration']} slots")
    print(f"PPDU Variant: {ppdu_variant}")
    print(f"STA Density: CH0={DEFAULT_NUM_STAS_CH0}, CH1={DEFAULT_NUM_STAS_CH1}")
    print(f"최종 평균 보상 (최근 50 에피소드): {result['final_avg_reward']:.2f}")
    print(f"최대 보상: {result['max_reward']:.2f}")
    print(f"총 학습 스텝: {result['steps_done']}")
    
    # 저장 위치 안내
    results_dir = f"./density_comparison_results/{experiment_name}"
    print(f"\n모델 저장 위치: {results_dir}/model.pth")
    print(f"훈련 그래프: {results_dir}/training_results.png")
    
    print("\n다음 단계:")
    print("python comparison_test.py 를 실행하여 훈련된 모델과 베이스라인을 비교하세요.")
    print("="*60)


if __name__ == "__main__":
    main()
