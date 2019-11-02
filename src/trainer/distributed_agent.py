from airsim_car_connector import AirSimCarConnector, ConnectorException
from rl_model import RlModel
from reward import Rewarder
from utils import *
import time
import numpy as np
import json
import os
import datetime
import sys
import requests
import copy
import datetime
import json

# A class that represents the agent that will drive the vehicle, train the model, and send the gradient updates to the trainer.
class DistributedAgent():
    def __init__(self, parameters):
        
        required_parameters = ['data_dir', 'max_epoch_runtime_sec', 'replay_memory_size', 'batch_size', 'min_epsilon', 'per_iter_epsilon_reduction', 'experiment_name', 'train_conv_layers', 'airsim_path', 'airsim_simulation_name', 'coordinator_address']
        for required_parameter in required_parameters:
            if required_parameter not in parameters:
                raise ValueError('Missing required parameter {0}'.format(required_parameter))

        print('Starting time: {0}'.format(datetime.datetime.utcnow()), file=sys.stderr)

        self.__data_dir = parameters['data_dir']
        self.__max_epoch_runtime_sec = float(parameters['max_epoch_runtime_sec'])
        self.__per_iter_epsilon_reduction = float(parameters['per_iter_epsilon_reduction'])
        self.__min_epsilon = float(parameters['min_epsilon'])
        self.__replay_memory_size = int(parameters['replay_memory_size'])
        self.__batch_size = int(parameters['batch_size'])
        self.__experiment_name = parameters['experiment_name']
        self.__weights_path = parameters['weights_path'] if 'weights_path' in parameters else None
        self.__train_conv_layers = bool((parameters['train_conv_layers'].lower().strip() == 'true'))
        self.__epsilon = 1
        self.__experiences = {}
        
        self.__possible_ip_addresses = [parameters['coordinator_address']]
        self.__trainer_ip_address = None
            
        self.__model = RlModel(parameters['weights_path'] if 'weights_path' in parameters else None, self.__train_conv_layers)
        self.__car_connector = AirSimCarConnector(parameters['airsim_path'], parameters['airsim_simulation_name'], self.__data_dir)
        self.__rewarder = Rewarder(self.__data_dir)

    # Starts the agent
    def start(self):
        self.__run_function()

    # The function that will be run during training.
    # It will initialize the connection to the trainer, start AirSim, and continuously run training iterations.
    def __run_function(self):
        def train():
            try:
                self.__experiences = {}
                self.__model = RlModel(self.__weights_path, self.__train_conv_layers)
                self.__fill_replay_memory()
                self.__get_latest_model()    
                while True:
                    self.__train_model()
            except Exception as e:
                print(e)
                print('Error during training - reinitialization')
                return

        self.__ping_coordinator()
        self.__get_latest_model()
        self.__car_connector.connect()
        while True:
            train()

    # We have the IP address for the trainer. Attempt to ping the trainer.
    def __ping_coordinator(self):
        ping_idx = -1
        while True:
            ping_idx += 1
            print('Attempting to ping trainer...')
            try:
                print('\tPinging {0}...'.format(self.__possible_ip_addresses[ping_idx % len(self.__possible_ip_addresses)]))
                response = requests.get('http://{0}/ping'.format(self.__possible_ip_addresses[ping_idx % len(self.__possible_ip_addresses)])).json()
                if response['message'] != 'pong':
                    raise ValueError('Received unexpected message: {0}'.format(response))
                print('Success!')
                self.__trainer_ip_address = self.__possible_ip_addresses[ping_idx % len(self.__possible_ip_addresses)]
                return
            except Exception as e:
                print('Could not get response. Message is {0}'.format(e))
                if (ping_idx % len(self.__possible_ip_addresses) == 0):
                    print('Waiting 5 seconds and trying again...')
                    time.sleep(5)

    # Fill the replay memory by driving randomly.
    def __fill_replay_memory(self):
        print('Filling replay memory...')
        while True:
            print('Running Airsim Epoch.')
            try:
                self.__run_training_epoch(True)
                percent_full = 100.0 * len(self.__experiences['actions']) / self.__replay_memory_size
                print('Replay memory now contains {0} members. ({1}% full)'.format(len(self.__experiences['actions']), percent_full))

                if (percent_full >= 100.0):
                    return
            except ConnectorException:
                print('Lost connection to car while fillling replay memory. Attempting to reconnect...')
                self.__car_connector.connect()

    # Training step
    def __train_model(self):
        try:
            #Generate a series of training examples by driving the vehicle in AirSim
            print('Running Airsim Epoch.')
            experiences, frame_count = self.__run_training_epoch(False)

            # If we didn't immediately crash, train on the gathered experiences
            if (frame_count > 0):
                print('Generating {0} minibatches...'.format(frame_count))
                print('Sampling Experiences.')
                # Sample experiences from the replay memory
                sampled_experiences = self.__sample_experiences(experiences, frame_count, True)

                # If we successfully sampled, train on the collected minibatches and send the gradients to the trainer node
                if (len(sampled_experiences) > 0):
                    print('Publishing training epoch results...')
                    self.__publish_batch_and_update_model(sampled_experiences, frame_count)

        # Occasionally, the AirSim exe will stop working.
        # For example, if a user connects to the node to visualize progress.
        # In that case, attempt to reconnect.
        except ConnectorException:
            print('Lost connection to car while training. Attempting to reconnect...')
            self.__car_connector.connect()

    # Runs an interation of data generation from AirSim.
    # Data will be saved in the replay memory.
    def __run_training_epoch(self, always_random):
        print('Training epoch.')
        self.__car_connector.reset_car()

        # Initialize the state buffer.
        # For now, save 4 images at 0.01 second intervals.
        state_buffer_len = 4
        state_buffer = []
        wait_delta_sec = 0.01
        
        # Start initializing the state buffer
        stop_run_time = datetime.datetime.now() + datetime.timedelta(seconds=2)
        while(datetime.datetime.now() < stop_run_time):
            time.sleep(wait_delta_sec)
            state_buffer = append_to_ring_buffer(self.__car_connector.get_image(), state_buffer, state_buffer_len)
        
        done = False
        actions = [] 
        pre_states = []
        post_states = []
        rewards = []
        predicted_rewards = []
        # car_state = self.__car_connector.car_state()

        start_time = datetime.datetime.utcnow()
        end_time = start_time + datetime.timedelta(seconds=self.__max_epoch_runtime_sec)
        
        num_random = 0
        far_off = False
        
        # Main data collection loop
        # Check for terminal conditions:
        # 1) Car has collided
        # 2) Car is stopped (or car_state.speed < 2) DELETED
        # 3) The run has been running for longer than max_epoch_runtime_sec. 
        #       This constraint is so the model doesn't end up having to churn through huge chunks of data, slowing down training
        while not (self.__car_connector.has_collided() or datetime.datetime.utcnow() > end_time or far_off):
            
            # The Agent should occasionally pick random action instead of best action
            do_greedy = np.random.random_sample()
            pre_state = copy.deepcopy(state_buffer)
            if (do_greedy < self.__epsilon or always_random):
                num_random += 1
                next_state = self.__model.get_random_state()
                predicted_reward = 0
            else:
                next_state, predicted_reward = self.__model.predict_state(pre_state)
                print('Model predicts {0}'.format(next_state))

            self.__car_connector.execute_action(next_state)
            
            # Wait for a short period of time to see outcome
            time.sleep(wait_delta_sec)
            
            # Observe outcome and compute reward from action
            state_buffer = append_to_ring_buffer(self.__car_connector.get_image(), state_buffer, state_buffer_len)
            car_state = self.__car_connector.car_state()
            collision_info = self.__car_connector.has_collided()
            reward, far_off = self.__rewarder.compute_reward(collision_info, car_state)
            
            # Add the experience to the set of examples from this iteration
            pre_states.append(pre_state)
            post_states.append(state_buffer)
            rewards.append(reward)
            predicted_rewards.append(predicted_reward)
            actions.append(next_state)

        print('Start time: {0}, end time: {1}'.format(start_time, datetime.datetime.utcnow()), file=sys.stderr)
        if (datetime.datetime.utcnow() > end_time):
            print('timed out.')
            print('Full autonomous run finished at {0}'.format(datetime.datetime.utcnow()), file=sys.stderr)
        sys.stderr.flush()

        # Only the last state is a terminal state.
        is_not_terminal = [1 for i in range(0, len(actions) - 1, 1)]
        is_not_terminal.append(0)
        
        # Add all of the states from this iteration to the replay memory
        self.__add_to_replay_memory('pre_states', pre_states)
        self.__add_to_replay_memory('post_states', post_states)
        self.__add_to_replay_memory('actions', actions)
        self.__add_to_replay_memory('rewards', rewards)
        self.__add_to_replay_memory('predicted_rewards', predicted_rewards)
        self.__add_to_replay_memory('is_not_terminal', is_not_terminal)

        print('Percent random actions: {0}'.format(num_random / max(1, len(actions))))
        print('Num total actions: {0}'.format(len(actions)))
        
        # If we are in the main loop, reduce the epsilon parameter so that the model will be called more often
        # Note: this will be overwritten by the trainer's epsilon if running in distributed mode
        if not always_random:
            self.__epsilon -= self.__per_iter_epsilon_reduction
            self.__epsilon = max(self.__epsilon, self.__min_epsilon)
        
        return self.__experiences, len(actions)
    
    # Adds a set of examples to the replay memory
    def __add_to_replay_memory(self, field_name, data):
        if field_name not in self.__experiences:
            self.__experiences[field_name] = data
        else:
            self.__experiences[field_name] += data
            start_index = max(0, len(self.__experiences[field_name]) - self.__replay_memory_size)
            self.__experiences[field_name] = self.__experiences[field_name][start_index:]

    # Sample experiences from the replay memory
    def __sample_experiences(self, experiences, frame_count, sample_randomly):
        sampled_experiences = {}
        sampled_experiences['pre_states'] = []
        sampled_experiences['post_states'] = []
        sampled_experiences['actions'] = []
        sampled_experiences['rewards'] = []
        sampled_experiences['predicted_rewards'] = []
        sampled_experiences['is_not_terminal'] = []

        # Compute the surprise factor, which is the difference between the predicted an the actual Q value for each state.
        # We can use that to weight examples so that we are more likely to train on examples that the model got wrong.
        suprise_factor = np.abs(np.array(experiences['rewards'], dtype=np.dtype(float)) - np.array(experiences['predicted_rewards'], dtype=np.dtype(float)))
        suprise_factor_normalizer = np.sum(suprise_factor)
        suprise_factor /= float(suprise_factor_normalizer)

        # Generate one minibatch for each frame of the run
        for _ in range(0, frame_count, 1):
            if sample_randomly:
                idx_set = set(np.random.choice(list(range(0, suprise_factor.shape[0], 1)), size=(self.__batch_size), replace=False))
            else:
                idx_set = set(np.random.choice(list(range(0, suprise_factor.shape[0], 1)), size=(self.__batch_size), replace=False, p=suprise_factor))

            sampled_experiences['pre_states'] += [experiences['pre_states'][i] for i in idx_set]
            sampled_experiences['post_states'] += [experiences['post_states'][i] for i in idx_set]
            sampled_experiences['actions'] += [experiences['actions'][i] for i in idx_set]
            sampled_experiences['rewards'] += [experiences['rewards'][i] for i in idx_set]
            sampled_experiences['predicted_rewards'] += [experiences['predicted_rewards'][i] for i in idx_set]
            sampled_experiences['is_not_terminal'] += [experiences['is_not_terminal'][i] for i in idx_set]
            
        return sampled_experiences
        
     
    # Train the model on minibatches and post to the trainer node.
    # The trainer node will respond with the latest version of the model that will be used in further data generation iterations.
    def __publish_batch_and_update_model(self, batches, batches_count):
        # Train and get the gradients
        print('Publishing epoch data and getting latest model from parameter server...')
        gradients = self.__model.get_gradient_update_from_batches(batches)
        if gradients is None:
            return
        
        # Post the data to the trainer node
        post_data = {}
        post_data['gradients'] = gradients
        post_data['batch_count'] = batches_count
        
        response = requests.post('http://{0}/gradient_update'.format(self.__trainer_ip_address), json=post_data)
        print('Response:')
        print(response)

        new_model_parameters = response.json()
        
        # Update the existing model with the new parameters
        self.__model.from_packet(new_model_parameters)
        
        #If the trainer sends us a epsilon, allow it to override our local value
        if ('epsilon' in new_model_parameters):
            new_epsilon = float(new_model_parameters['epsilon'])
            print('Overriding local epsilon with {0}, which was sent from trainer'.format(new_epsilon))
            self.__epsilon = new_epsilon
                
    # Gets the latest model from the trainer node
    def __get_latest_model(self):
        print('Getting latest model from parameter server...')
        response = requests.get('http://{0}/latest'.format(self.__trainer_ip_address)).json()
        self.__model.from_packet(response)


# Parse the command line parameters
parameters = {}
for arg in sys.argv:
    if '=' in arg:
        args = arg.split('=')
        print('0: {0}, 1: {1}'.format(args[0], args[1]))
        parameters[args[0].replace('--', '')] = args[1]

#Make the debug statements easier to read
np.set_printoptions(threshold=sys.maxsize, suppress=True)

# Set up the logging to the file share if not running locally.
setup_logs(parameters['data_dir'], parameters['experiment_name'])

# Start the training
agent = DistributedAgent(parameters)
agent.start()
