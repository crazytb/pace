import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.random import default_rng
import math
import random

# https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
# https://www.youtube.com/@cartoonsondemand

class CustomEnv(gym.Env):
    def __init__(self, max_comp_units, max_terminals, max_epoch_size, max_queue_size, reward_weights):
        super(CustomEnv, self).__init__()
        # self.max_comp_units = max_comp_units
        self.max_terminals = max_terminals
        self.max_epoch_size = max_epoch_size
        self.max_queue_size = max_queue_size

        self.reward_weight = reward_weights
        self.max_available_computation_units = max_comp_units
        self.max_number_of_associated_terminals = max_terminals
        self.max_channel_quality = 2
        self.max_remain_epochs = max_epoch_size
        self.max_comp_units = np.array([max_comp_units] * max_queue_size)
        # self.max_proc_times = np.array([max_epoch_size] * max_queue_size)
        self.max_proc_times = np.array([int(np.ceil(max_epoch_size/2))] * max_queue_size)

        # 0: process, 1: offload
        self.action_space = spaces.Discrete(2)

        self.reward = 0

        self.observation_space = spaces.Dict({
            "available_computation_units": spaces.Discrete(self.max_available_computation_units),
            "number_of_associated_terminals": spaces.Discrete(self.max_number_of_associated_terminals),
            "channel_quality": spaces.Discrete(self.max_channel_quality),
            "remain_epochs": spaces.Discrete(self.max_remain_epochs),
            "mec_comp_units": spaces.MultiDiscrete([max_comp_units] * max_queue_size),
            "mec_proc_times": spaces.MultiDiscrete([max_epoch_size] * max_queue_size),
            "queue_comp_units": spaces.MultiDiscrete([max_comp_units] * max_queue_size),
            "queue_proc_times": spaces.MultiDiscrete([max_epoch_size] * max_queue_size),
        })
        self.rng = default_rng()
        self.current_obs = None
       
    def get_obs(self):
        return {"available_computation_units": self.available_computation_units,
                "number_of_associated_terminals": self.number_of_associated_terminals,
                "channel_quality": self.channel_quality,
                "remain_epochs": self.remain_epochs,
                "mec_comp_units": self.mec_comp_units,
                "mec_proc_times": self.mec_proc_times,
                "queue_comp_units": self.queue_comp_units,
                "queue_proc_times": self.queue_proc_times}
    
    def stepfunc(self, thres, x):
        if x > thres:
            return 1
        else:
            return 0
    
    def change_channel_quality(self):
        # State settings
        velocity = 10   # km/h
        snr_thr = 15
        snr_ave = snr_thr + 10
        f_0 = 5.9e9 # Carrier freq = 5.9GHz, IEEE 802.11bd
        speedoflight = 300000   # km/sec
        f_d = velocity/(3600*speedoflight)*f_0  # Hz
        packettime = 300    # us
        fdtp = f_d*packettime/1e6
        TRAN_01 = (fdtp*math.sqrt(2*math.pi*snr_thr/snr_ave))/(np.exp(snr_thr/snr_ave)-1)
        TRAN_00 = 1 - TRAN_01
        # TRAN_11 = fdtp*math.sqrt((2*math.pi*snr_thr)/snr_ave)
        TRAN_10 = fdtp*math.sqrt((2*math.pi*snr_thr)/snr_ave)
        TRAN_11 = 1 - TRAN_10

        if self.channel_quality == 0:  # Bad state
            if self.stepfunc(TRAN_00, random.random()) == 0: # 0 to 0
                channel_quality = 0
            else:   # 0 to 1
                channel_quality = 1
        else:   # Good state
            if self.stepfunc(TRAN_11, random.random()) == 0: # 1 to 1
                channel_quality = 1
            else:   # 1 to 0
                channel_quality = 0
    
        return channel_quality
    
    def fill_first_zero(self, arr, value):
        for i in range(len(arr)):
            if arr[i] == 0:
                arr[i] = value
                break
        return arr
    
    def flatten_dict_values(self, dict):
        flattened = np.array([])
        for v in list(dict.values()):
            if isinstance(v, np.ndarray):
                flattened = np.concatenate([flattened, v])
            else:
                flattened = np.concatenate([flattened, np.array([v])])
        return flattened

    def reset(self, seed=None, options=None):
        """
        Returns: observation
        """
        super().reset(seed=seed)

        self.available_computation_units = self.max_available_computation_units
        self.number_of_associated_terminals = self.rng.integers(1, self.max_number_of_associated_terminals + 1, size=1)
        self.channel_quality = self.rng.integers(0, self.max_channel_quality)
        self.remain_epochs = self.max_remain_epochs
        self.remain_processing = 0
        self.mec_comp_units = np.zeros(self.max_queue_size, dtype=int)
        self.mec_proc_times = np.zeros(self.max_queue_size, dtype=int)
        self.queue_comp_units = self.rng.integers(1, self.max_comp_units + 1, size=self.max_queue_size)
        self.queue_proc_times = self.rng.integers(1, self.max_proc_times + 1, size=self.max_queue_size)

        self.reward = 0

        observation = self.get_obs()
        
        return observation, {}
    
    def step(self, action):
        """
        Returns: observation, reward, terminated, truncated, info
        """
        self.reward = 0
        # forwarding phase
        # 0: local process, 1: offload
        if action == 0:
            # if available computation units are enough to process the first queue task and mec_comp_unit has empty slot and queue_comp_units has nonzero value.
            case_action = ((self.available_computation_units >= self.queue_comp_units[0]) and 
                           (self.mec_comp_units[self.mec_comp_units == 0].size > 0) and
                           (self.queue_comp_units[0] > 0))
            if case_action:
                self.available_computation_units -= self.queue_comp_units[0]
                self.mec_comp_units = self.fill_first_zero(self.mec_comp_units, self.queue_comp_units[0])
                self.mec_proc_times = self.fill_first_zero(self.mec_proc_times, self.queue_proc_times[0])
                self.queue_comp_units = np.concatenate([self.queue_comp_units[1:], np.array([0])])
                self.queue_proc_times = np.concatenate([self.queue_proc_times[1:], np.array([0])])
            else:
                pass
                # self.reward = -1 * self.queue_comp_units[0] # penalty

            self.channel_quality = self.change_channel_quality()
            self.remain_epochs = self.remain_epochs - 1
        else:
            self.reward = self.reward_weight * self.queue_comp_units[0] if self.channel_quality == 1 else 0
            self.channel_quality = self.change_channel_quality()
            self.remain_epochs = self.remain_epochs - 1
            # shift left-wise queue information with 1 and pad 0
            self.queue_comp_units = np.concatenate([self.queue_comp_units[1:], np.array([0])])
            self.queue_proc_times = np.concatenate([self.queue_proc_times[1:], np.array([0])])

        # processing phase
        zeroed_index = (self.mec_proc_times == 1)
        if zeroed_index.any():
            self.available_computation_units += self.mec_comp_units[zeroed_index].sum()
            self.reward = self.mec_comp_units[zeroed_index].sum()
            self.mec_proc_times = np.concatenate([self.mec_proc_times[zeroed_index == False], np.zeros(zeroed_index.sum(), dtype=int)])
            self.mec_comp_units = np.concatenate([self.mec_comp_units[zeroed_index == False], np.zeros(zeroed_index.sum(), dtype=int)])
        self.mec_proc_times = np.clip(self.mec_proc_times - 1, 0, self.max_proc_times)

        next_obs = self.get_obs()

        return next_obs, self.reward, self.remain_epochs == 0, False, {}


    def render(self):
        """
        Returns: None
        """
        pass

    def close(self):
        """
        Returns: None
        """
        pass