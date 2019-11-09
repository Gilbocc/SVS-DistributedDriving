from airsim_client import *
from car_connector import CarConnector, ConnectorException, Vector3, CarInternalState
from utils import *
import datetime
import os
import json
import copy
import numpy as np
import msgpackrpc
import time

class AirSimCarConnector(CarConnector):

    def __init__(self, airsim_path, simulation_name, data_dir):
        self.__airsim_path = airsim_path
        self.__simulation_name = simulation_name
        self.__data_dir = data_dir
        self.__car_client = None
        self.__car_controls = None
        self.__car_client = None
        self.__angle_values = [-1, -0.5, 0, 0.5, 1]

        self.__init_road_points()

        os.system('START "" powershell.exe {0} {1} -windowed'.format(
            os.path.join(self.__airsim_path, 'AD_Cookbook_Start_AirSim.ps1'), self.__simulation_name))


    # Connects to the AirSim Exe.
    # Assume that it is already running. After 10 successive attempts, attempt to restart the executable.
    def connect(self):
        attempt_count = 0
        while True:
            try:
                print('Attempting to connect to AirSim (attempt {0})'.format(attempt_count), flush=True)
                self.__car_client = CarClient(ip="127.0.0.1")
                self.__car_client.confirmConnection()
                self.__car_client.enableApiControl(True)
                self.__car_controls = CarControls()
                print('Connected!', flush=True)
                return
            except:
                print('Failed to connect.', flush=True)
                attempt_count += 1
                if (attempt_count % 10 == 0):
                    print('10 consecutive failures to connect. Attempting to start AirSim on my own.', flush=True)
                    os.system('START "" powershell.exe {0} {1} -windowed'.format(
                        os.path.join(self.__airsim_path, 'AD_Cookbook_Start_AirSim.ps1'), self.__simulation_name))
                print('Waiting a few seconds.', flush=True)
                time.sleep(10)

    # Prepare car state fo next iteration
    def reset_car(self):
        try:
            # Pick a random starting point on the roads
            starting_points, starting_direction = self.__get_starting_point()

            print('Getting Pose', flush=True)
            # self.__car_client.simSetPose(Pose(Vector3r(starting_points[0], starting_points[1], starting_points[2]), AirSimClientBase.toQuaternion(starting_direction[0], starting_direction[1], starting_direction[2])), True)
            self.__car_client.reset()

            # Currently, simSetPose does not allow us to set the velocity. 
            # So, if we crash and call simSetPose, the car will be still moving at its previous velocity.
            # We need the car to stop moving, so push the brake and wait for a few seconds.
            print('Waiting for momentum to die', flush=True)
            self.__car_controls.steering = 0
            self.__car_controls.throttle = 0
            self.__car_controls.brake = 1
            self.__car_client.setCarControls(self.__car_controls)
            time.sleep(4)
            
            print('Resetting', flush=True)
            # self.__car_client.simSetPose(Pose(Vector3r(starting_points[0], starting_points[1], starting_points[2]), AirSimClientBase.toQuaternion(starting_direction[0], starting_direction[1], starting_direction[2])), True)
            self.__car_client.reset()

            #Start the car rolling so it doesn't get stuck
            print('Running car for a few seconds...', flush=True)
            self.__car_controls.steering = 0
            self.__car_controls.throttle = 1
            self.__car_controls.brake = 0
            self.__car_client.setCarControls(self.__car_controls)
        except msgpackrpc.error.TimeoutError:
            raise ConnectorException

    def car_state(self):
        try:
            car_state = self.__car_client.getCarState()
            try:
                position_key = bytes('position', encoding='utf8')
                x_val_key = bytes('x_val', encoding='utf8')
                y_val_key = bytes('y_val', encoding='utf8')
                car_point = Vector3r(car_state.kinematics_true[position_key][x_val_key], car_state.kinematics_true[position_key][y_val_key])
            except:
                car_point = np.array(car_state.position[x_val_key], car_state.position[y_val_key])
            
            return CarInternalState(car_state.speed, car_point)
        except msgpackrpc.error.TimeoutError:
            raise ConnectorException

    def has_collided(self):
        try: 
            return self.__car_client.getCollisionInfo().has_collided
        except msgpackrpc.error.TimeoutError:
            raise ConnectorException

    def execute_action(self, action):
        try:
            # Convert the selected state to a control signal
            next_control_signals = self.__state_to_control_signals(action, self.car_state())
            # Take the action
            self.__car_controls.steering = next_control_signals[0]
            self.__car_controls.throttle = next_control_signals[1]
            self.__car_controls.brake = next_control_signals[2]
            self.__car_client.setCarControls(self.__car_controls)
        except msgpackrpc.error.TimeoutError:
            raise ConnectorException

    # Gets an image from AirSim
    def get_image(self):
        try:
            image_response = self.__car_client.simGetImages([ImageRequest(0, AirSimImageType.Scene, False, False)])[0]
            image1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
            image_rgba = image1d.reshape(image_response.height, image_response.width, 4)

            return image_rgba[76:135,0:255,0:3].astype(float)
        except msgpackrpc.error.TimeoutError:
            raise ConnectorException
        

    # Randomly selects a starting point on the road
    # Used for initializing an iteration of data generation from AirSim
    def __get_starting_point(self):

        # Pick a random road.
        random_line_index = np.random.randint(0, high=len(self.__road_points))
        
        # Pick a random position on the road. 
        # Do not start too close to either end, as the car may crash during the initial run.
        random_interp = (np.random.random_sample() * 0.4) + 0.3
        
        # Pick a random direction to face
        random_direction_interp = np.random.random_sample()

        # Compute the starting point of the car
        random_line = self.__road_points[random_line_index]
        random_start_point = list(random_line[0])
        random_start_point[0] += (random_line[1][0] - random_line[0][0])*random_interp
        random_start_point[1] += (random_line[1][1] - random_line[0][1])*random_interp

        # Compute the direction that the vehicle will face
        # Vertical line
        if (np.isclose(random_line[0][1], random_line[1][1])):
            if (random_direction_interp > 0.5):
                random_direction = (0,0,0)
            else:
                random_direction = (0, 0, math.pi)
        # Horizontal line
        elif (np.isclose(random_line[0][0], random_line[1][0])):
            if (random_direction_interp > 0.5):
                random_direction = (0,0,math.pi/2)
            else:
                random_direction = (0,0,-1.0 * math.pi/2)
        else:
            random_direction = (0,0,0)

        # The z coordinate is always zero
        random_start_point[2] = -0
        
        # random_start_point = (random_line[1][0], random_line[1][1], 0)
        print('Start point ' + json.dumps(random_start_point), flush=True)
        return (random_start_point, random_direction)

    # Initializes the points used for determining the starting point of the vehicle
    def __init_road_points(self):
        self.__road_points = []
        car_start_coords = [12961.722656, 6660.329102, 0]
        with open(os.path.join(self.__data_dir, 'road_lines.txt'), 'r') as f:
            for line in f:
                points = line.split('\t')
                first_point = np.array([float(p) for p in points[0].split(',')] + [0])
                second_point = np.array([float(p) for p in points[1].split(',')] + [0])
                self.__road_points.append(tuple((first_point, second_point)))

        # Points in road_points.txt are in unreal coordinates
        # But car start coordinates are not the same as unreal coordinates
        for point_pair in self.__road_points:
            for point in point_pair:
                point[0] -= car_start_coords[0]
                point[1] -= car_start_coords[1]
                point[0] /= 100
                point[1] /= 100

    # Convert the current state to control signals to drive the car.
    # As we are only predicting steering angle, we will use a simple controller to keep the car at a constant speed
    def __state_to_control_signals(self, state, car_state):
        if car_state.speed > 9:
            return (self.__angle_values[state], 0, 1)
        else:
            return (self.__angle_values[state], 1, 0)