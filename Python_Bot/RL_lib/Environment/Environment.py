from abc import ABC, abstractmethod
import numpy as np

class Environment(ABC):
    '''Abstract class to create environment compatible with Reinforcement Learning Agent'''

    state_shape :np.ndarray
    action_shape :np.ndarray

    state: np.ndarray
    next_state: np.ndarray
    reward: float
    done: bool

    @abstractmethod
    def step(self, action: np.ndarray) -> np.ndarray:
        '''Update environment based on chosen action'''

    @abstractmethod
    def reset(self) -> np.ndarray:
        '''Reset environment'''

    def get_state_shape(self):
        return self.state_shape

    def get_action_shape(self):
        return self.action_shape
