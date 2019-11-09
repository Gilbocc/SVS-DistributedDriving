import numpy as np
import math
import os

class CollisionRewarder:

    def __init__(self, data_dir):
        self.__data_dir = data_dir

    # Computes the reward functinon based on the car position.
    def compute_reward(self, has_collided, car_state):
    
        # If the car has collided, the reward is always zero
        if (has_collided):
            return 0.0, True
        
        # If the car is stopped, the reward is always zero
        if (car_state.speed < 2):
            return 0.0, True

        return 1.0, False

    def discount_rewards(self, rewards):
        i = 4
        for x in range(2, 6):
            if len(rewards) - x >= 0:
                rewards[len(rewards) - x] = rewards[len(rewards) - x] - (0.2 * i)
                print('Modified reward {0}: {1}'.format(len(rewards) - x, rewards[len(rewards) - x]))
                i = i - 1