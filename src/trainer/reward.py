import numpy as np
import math
import os

class Rewarder:

    def __init__(self, data_dir):
        self.__data_dir = data_dir
        self.__init_reward_points()

    # Computes the reward functinon based on the car position.
    def compute_reward(self, has_collided, car_state):
        #Define some constant parameters for the reward function
        THRESH_DIST = 3.5                # The maximum distance from the center of the road to compute the reward function
        DISTANCE_DECAY_RATE = 1.2        # The rate at which the reward decays for the distance function
        CENTER_SPEED_MULTIPLIER = 2.0    # The ratio at which we prefer the distance reward to the speed reward

        # If the car has collided, the reward is always zero
        if (has_collided):
            return 0.0, True
        
        # If the car is stopped, the reward is always zero
        if (car_state.speed < 2):
            return 0.0, True

        car_point = np.array([car_state.position.x_val, car_state.position.y_val, 0])
        
        # Distance component is exponential distance to nearest line
        distance = 999
        
        #Compute the distance to the nearest center line
        for line in self.__reward_points:
            local_distance = 0
            length_squared = ((line[0][0]-line[1][0])**2) + ((line[0][1]-line[1][1])**2)
            if (length_squared != 0):
                t = max(0, min(1, np.dot(car_point-line[0], line[1]-line[0]) / length_squared))
                proj = line[0] + (t * (line[1]-line[0]))
                local_distance = np.linalg.norm(proj - car_point)
            
            distance = min(local_distance, distance)
            
        distance_reward = math.exp(-(distance * DISTANCE_DECAY_RATE))
        
        return distance_reward, distance > THRESH_DIST

    def discount_rewards(self, rewards):
        print('No discount configured')
        return rewards

    # Initializes the points used for determining the optimal position of the vehicle during the reward function
    def __init_reward_points(self):
        self.__reward_points = []
        with open(os.path.join(self.__data_dir, 'reward_points.txt'), 'r') as f:
            for line in f:
                point_values = line.split('\t')
                first_point = np.array([float(point_values[0]), float(point_values[1]), 0])
                second_point = np.array([float(point_values[2]), float(point_values[3]), 0])
                self.__reward_points.append(tuple((first_point, second_point)))