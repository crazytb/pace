# Env parameters
MAX_COMP_UNITS = 100
MAX_TERMINALS = 10
MAX_EPOCH_SIZE = 20
MAX_QUEUE_SIZE = 20 # MAX_EPOCH_SIZE and MAX_QUEUE_SIZE should be the same
REWARD_WEIGHTS = 0.1

# DQN parameters
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000  # Random environment를 위한 더 긴 탐험 기간
TAU = 0.005
LR = 1e-4  # 학습률 증가로 학습 속도 개선