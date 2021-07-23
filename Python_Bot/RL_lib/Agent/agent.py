import numpy as np
from abc import ABC, abstractmethod

from RL_lib.Memory.Memory import Experience

class Agent(ABC):

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