from trainer.airsim_client import *
from trainer.rl_model import RlModel
import numpy as np
import time
import sys
import json
import PIL
import PIL.ImageFilter
import datetime

# Helper function to add an item to a given ring buffer
def __append_to_ring_buffer(item, buffer, buffer_size):
    if (len(buffer) >= buffer_size):
        buffer = buffer[1:]
    buffer.append(item)

# Helper function to connect to AirSim
def __get_car_client():
    print('Connecting to AirSim...')
    car_client = CarClient(ip="192.168.1.6")
    car_client.confirmConnection()
    car_client.enableApiControl(True)
    print('Connected!')
    return car_client

# Helper function to obtain images from the simulator
def __get_image(car_client):
    image_response = car_client.simGetImages([ImageRequest(0, AirSimImageType.Scene, False, False)])[0]
    image1d = np.frombuffer(image_response.image_data_uint8, dtype=np.uint8)
    image_rgba = image1d.reshape(image_response.height, image_response.width, 4)
    return image_rgba[76:135,0:255,0:3].astype(float)

# Helper function to load model from disk
def __get_model(model_path):
    model = RlModel(model_path, False)
    # model = RlModel(None, False)
    # with open(model_path, 'r') as f:
    #     checkpoint_data = json.loads(f.read())
    #     model.from_packet(checkpoint_data['model'])
    return model

# Helper function to initialize the content of the car state buffer
def __initialize_state_buffer(car_client, state_buffer, state_buffer_len):
    print('Running car for a few seconds...')
    car_controls = CarControls()
    car_controls.steering = 0
    car_controls.throttle = 1
    car_controls.brake = 0
    car_client.setCarControls(car_controls)
    stop_run_time = datetime.datetime.now() + datetime.timedelta(seconds=2)
    while(datetime.datetime.now() < stop_run_time):
        time.sleep(0.01)
        __append_to_ring_buffer(__get_image(car_client), state_buffer, state_buffer_len)

# Helper function to evaluate the given model sing the AirSim simulator
def __start_evaluation(model_path):

    state_buffer = []
    state_buffer_len = 4
    car_client = __get_car_client()
    model = __get_model(model_path)
    __initialize_state_buffer(car_client, state_buffer, state_buffer_len)

    print('Running model')
    while(True):
        __append_to_ring_buffer(__get_image(car_client), state_buffer, state_buffer_len)
        next_state, dummy = model.predict_state(state_buffer)
        next_control_signal = model.state_to_control_signals(next_state, car_client.getCarState())

        car_controls = CarControls()
        car_controls.steering = next_control_signal[0]
        car_controls.throttle = next_control_signal[1]
        car_controls.brake = next_control_signal[2]

        print('State = {0}, steering = {1}, throttle = {2}, brake = {3}'.format(
            next_state, car_controls.steering, car_controls.throttle, car_controls.brake))

        car_client.setCarControls(car_controls)
        time.sleep(0.1)

if __name__ == "__main__":
    __start_evaluation(sys.argv[1])
