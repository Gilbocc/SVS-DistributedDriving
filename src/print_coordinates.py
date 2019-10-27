from trainer.airsim_client import *
from trainer.rl_model import RlModel
import numpy as np
import time
import sys
import json
import PIL
import PIL.ImageFilter
import datetime

# Helper function to connect to AirSim
def __get_car_client():
    print('Connecting to AirSim...')
    car_client = CarClient(ip = "192.168.1.6")
    car_client.confirmConnection()
    # car_client.enableApiControl(True)
    # car_controls = CarControls()
    print('Connected!')
    return car_client

def __start_evaluation():

    car_client = __get_car_client()
    while(True):
        state = car_client.getCarState()
        print(state.kinematics_true[b'position'][b'x_val'], state.kinematics_true[b'position'][b'y_val'])
        # print(state.position[b'x_val'], state.position[b'y_val'])
        # print(state.collision[b'has_collided'])
        time.sleep(10)

if __name__ == "__main__":
    __start_evaluation()
