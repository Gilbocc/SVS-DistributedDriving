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
    
    conda create --prefix=./envs python=3.7

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

    .\AD_Cookbook_Start_AirSim.ps1 neighborhood

## COORDINATOR

To start the coordinator agent use the following command (parameters must be replaced)

    python src\manage.py runserver 0.0.0.0:80 data_dir={-1} role=trainer experiment_name={0} batch_update_frequency={1} weights_path={2} train_conv_layers={3} per_iter_epsilon_reduction={4} min_epsilon={5}

Example:

    python .\manage.py runserver 0.0.0.0:7777 data_dir='C:\\Users\\peppe_000\\Documents\\MyProjects\\SmartVehicularSystems\\DistributedRL\\data' role=trainer experiment_name='experiment_1' batch_update_frequency=1 weights_path='C:\\Users\\peppe_000\\Documents\\MyProjects\\SmartVehicularSystems\\DistributedRL\\data\\pretrain_model_weights.h5' train_conv_layers='false' per_iter_epsilon_reduction=0.003 min_epsilon=0.1

## WORKER NODE

To start a node agent use the following command (parameters must be replaced)

    python src\app\distributed_agent.py data_dir={-1} role=agent max_epoch_runtime_sec={0} per_iter_epsilon_reduction={1:f} min_epsilon={2:f} batch_size={3} replay_memory_size={4} experiment_name={5} weights_path={6} train_conv_layers={7}

Example:

    python .\distributed_agent.py data_dir='C:\\Users\\peppe_000\\Documents\\MyProjects\\SmartVehicularSystems\\DistributedRL\\data' role=agent max_epoch_runtime_sec=30 per_iter_epsilon_reduction=0.003 min_epsilon=0.1 batch_size=32 replay_memory_size=32 experiment_name='experiment_1' weights_path='C:\\Users\\peppe_000\\Documents\\MyProjects\\SmartVehicularSystems\\DistributedRL\\data\\pretrain_model_weights.h5' train_conv_layers='false' airsim_path='D:\\AirSim\\AD_Cookbook_AirSim' airsim_simulation_name='neighborhood'

## PARAMETERS

**batch_update_frequency**: This is how often the weights from the actively trained network get copied to the target network. It is also how often the model gets saved to disk. For more details on how this works, check out the Deep Q-learning paper.

**max_epoch_runtime_sec**: This is the maximum runtime for each epoch. If the car has not reached a terminal state after this many seconds, the epoch will be terminated and training will begin.

**per_iter_epsilon_reduction**: The agent uses an epsilon greedy linear annealing strategy while training. This is the amount by which epsilon is reduced each iteration.

**min_epsilon**: The minimum value for epsilon. Once reached, the epsilon value will not decrease any further.
batch_size: The minibatch size to use for training.

**replay_memory_size**: The number of examples to keep in the replay memory. The replay memory is a FIFO buffer used to reduce the effects of nearby states being correlated. Minibatches are generated from randomly selecting examples from the replay memory.

**weights_path**: If we are doing transfer learning and using pretrained weights for the model, they will be loaded from this path.
train_conv_layers: If we are using pretrained weights, we may prefer to freeze the convolutional layers to speed up training.

**airsim_path**: Location of the AirSim executable (AD_Cookbook_Start_AirSim.ps1)

**airsim_simulation_name**: Simulation scenario. The default AirSim deistribution contains the following configurations: 'city', 'landscape', 'neighborhood', 'coastline', 'hawaii'

Example parameters:

    batch_update_frequency = 300
    max_epoch_runtime_sec = 30
    per_iter_epsilon_reduction=0.003
    min_epsilon = 0.1
    batch_size = 32
    replay_memory_size = 2000
    weights_path = 'Z:\\data\\pretrain_model_weights.h5'
    train_conv_layers = 'false'

## SINGLE NODE EXECUTION

Start the agent

    src\app\distributed_agent.py

using the following parameters

    batch_update_frequency=10
    max_epoch_runtime_sec=30
    per_iter_epsilon_reduction=0.003
    min_epsilon=0.1
    batch_size=32
    replay_memory_size=50
    weights_path=os.path.join(os.getcwd(), 'Share\\data\\pretrain_model_weights.h5')
    train_conv_layers='false'
    airsim_path='E:\\AD_Cookbook_AirSim\\'
    data_dir=os.path.join(os.getcwd(), 'Share')
    experiment_name='local_run'
    local_run=true