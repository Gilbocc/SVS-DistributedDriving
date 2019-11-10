## DISTRIBUTED DEEP REINFORCEMENT LEARNING FOR AUTONOMOUS DRIVING

This work aims to explore the potentialities of distributed reinforcement learning algorithms in the autonomous driving field

## INFO

This work is based on the following research:

    http://www.mitchellspryn.com/content/Autonomous-Driving-With-Deep-Reinforcement-Learning/DistributedRlForAd.pdf

A reference implementation (utilized as a starting point for this work) is available at the following address:

    https://github.com/microsoft/AutonomousDrivingCookbook/tree/master/DistributedRL

## ENVIRONMENT PREPARATION

0 - Install Anaconda

    Fix SSL: https://github.com/conda/conda/issues/8273

2 - Create the virtual environment
    
    conda create --prefix=./envs python=3.6

3 - Activete the newly created environment
    
    conda activate ./envs

4 - Install the dependencies
    
    python ./install_dependencies.py

5 - Disable the virtual environment
    
    conda deactivate

6 - To delete the envireonment use

    conda env remove -p ./envs

## SIMULATOR PREPARATION

1 - Download the simulator

    azcopy copy 'https://airsimtutorialdataset.blob.core.windows.net/e2edl/AD_Cookbook_AirSim.7z' './'

2 - Start the simulator

    .\AD_Cookbook_Start_AirSim.ps1 neighborhood -window

## COORDINATOR NODE

To start the coordinator agent use the following command (parameters must be replaced)

    python src\manage.py runserver ip:port data_dir={0} experiment_name={1} batch_update_frequency={2} weights_path={3} train_conv_layers={4} per_iter_epsilon_reduction={5} min_epsilon={6}

Example:

    python .\manage.py runserver 0.0.0.0:7777 data_dir='C:\\Users\\peppe_000\\Documents\\MyProjects\\SmartVehicularSystems\\DistributedRL\\data' experiment_name='experiment_refactored_1' batch_update_frequency=200 weights_path='C:\\Users\\peppe_000\\Documents\\MyProjects\\SmartVehicularSystems\\DistributedRL\\data\\pretrain_model_weights.h5' train_conv_layers='false' per_iter_epsilon_reduction=0.003 min_epsilon=0.1

## WORKER NODE

To start a node agent use the following command (parameters must be replaced)

    python src\app\distributed_agent.py data_dir={0} max_epoch_runtime_sec={1} batch_size={2} replay_memory_size={3} experiment_name={4} weights_path={5} train_conv_layers={6} 

Example:

    python .\distributed_agent.py data_dir='C:\\Users\\peppe_000\\Documents\\MyProjects\\SmartVehicularSystems\\DistributedRL\\data' max_epoch_runtime_sec=30 batch_size=32 replay_memory_size=1500 experiment_name='experiment_refactored_1' weights_path='C:\\Users\\peppe_000\\Documents\\MyProjects\\SmartVehicularSystems\\DistributedRL\\data\\pretrain_model_weights.h5' train_conv_layers='false' airsim_path='D:\\AirSim\\AD_Cookbook_AirSim' airsim_simulation_name='neighborhood' coordinator_address='192.168.1.6:7777'

## PARAMETERS

**batch_update_frequency**: This is how often the weights from the actively trained network get copied to the target network. It is also how often the model gets saved to disk. For more details on how this works, check out the Deep Q-learning paper.

**max_epoch_runtime_sec**: This is the maximum runtime for each epoch. If the car has not reached a terminal state after this many seconds, the epoch will be terminated and training will begin.

**per_iter_epsilon_reduction**: The agent uses an epsilon greedy linear annealing strategy while training. This is the amount by which epsilon is reduced each iteration.

**min_epsilon**: The minimum value for epsilon. Once reached, the epsilon value will not decrease any further.
batch_size: The minibatch size to use for training.

**replay_memory_size**: The number of examples to keep in the replay memory. The replay memory is a FIFO buffer used to reduce the effects of nearby states being correlated. Minibatches are generated from randomly selecting examples from the replay memory.

**weights_path**: If we are doing transfer learning and using pretrained weights for the model, they will be loaded from this path.

**train_conv_layers**: If we are using pretrained weights, we may prefer to freeze the convolutional layers to speed up training.

**airsim_path**: Location of the AirSim executable (AD_Cookbook_Start_AirSim.ps1)

**airsim_simulation_name**: Simulation scenario. The default AirSim deistribution contains the following configurations: 'city', 'landscape', 'neighborhood', 'coastline', 'hawaii'

**coordinator_address**: The address of the master node in the form 'IP:PORT'

Example parameters:

    batch_update_frequency = 300
    max_epoch_runtime_sec = 30
    per_iter_epsilon_reduction = 0.003
    min_epsilon = 0.1
    batch_size = 32
    replay_memory_size = 2000
    weights_path = 'Z:\\data\\pretrain_model_weights.h5'
    train_conv_layers = 'false'
    airsim_path = 'Z:\\AirSim'
    airsim_simulation_name = 'neighborhood'
    coordinator_address = '192.169.1.5:7777'

## TEST

To run the simulator using a given model use the following command:

    python .\tester.py 'model_path' 'isH5file'

**model_path**: the path from where the model will be loaded.

**isH5file**: True/False. True if the model to load is an H5 file

Example:

    python .\tester.py 'C:\\Users\\peppe_000\\Documents\\MyProjects\\SmartVehicularSystems\\DistributedRL\\data\\checkpoint\\experiment_refactored_1\\3444.json' 'False'

The simulator must be launched manually
