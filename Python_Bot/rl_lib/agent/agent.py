import numpy as np
from abc import ABC, abstractmethod

from rl_lib.memory.memory import Experience

class Agent(ABC):

    epsilon: float # Ratio between exploration vs exploitation
    action_shape: np.ndarray
    state_shape: np.ndarray

    def get_action_shape(self):
        return self.action_shape

    def get_state_shape(self):
        return self.state_shape

    @abstractmethod
    def get_action(state: np.ndarray) -> np.ndarray:
        '''Determine best action based on state'''

    @abstractmethod
    def get_action_training(state: np.ndarray) -> np.ndarray:
        '''Determine action with random samples'''

    @abstractmethod
    def fit():
        '''Fit the model based on experiences (training)'''
    
    @abstractmethod
    def memorize(experience: Experience) -> float:
        '''Insert new experience into memory'''

class DefaultAgent(Agent):

    def __init__(self, action_shape: np.ndarray, state_shape: np.ndarray):
        self.action_shape=action_shape
        self.state_shape=state_shape

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return np.any(state!=0)*np.ones(self.action_shape)

    def get_action_training(self, state: np.ndarray) -> np.ndarray:
        return np.any(state!=0)*np.random.rand(*self.action_shape)

    def fit(self):
        '''Do nothing'''
    
    def memorize(experience: Experience) -> float:
        '''Do nothing'''
        return super().memorize()
        

