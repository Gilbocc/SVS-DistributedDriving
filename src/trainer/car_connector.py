import abc
import numpy as np

class ConnectorException(Error):
   pass

class Vector3(MsgpackMixin):
    x_val = np.float32(0)
    y_val = np.float32(0)
    z_val = np.float32(0)

    def __init__(self, x_val = np.float32(0), y_val = np.float32(0), z_val = np.float32(0)):
        self.x_val = x_val
        self.y_val = y_val
        self.z_val = z_val

class CarInternalState:
    speed = np.float32(0)
    position = Vector3()

    def __init__(self, speed, position):
        self.speed = speed
        self.position = position

class CarConnector(abc.ABC):

    @abc.abstractmethod
    def connect(self):
        pass

    @abc.abstractmethod
    def reset_car(self):
        pass

    @abc.abstractmethod
    def get_image(self):
        pass

    @abc.abstractmethod
    def car_state(self):
        pass

    @abc.abstractmethod
    def has_collided(self):
        pass

    @abc.abstractmethod
    def execute_action(self, action):
        pass
