import numpy as np
from abc import ABC, abstractmethod
import keras

from rl_lib.memory.memory import Experience

class Agent(ABC):

    epsilon: float # Ratio between exploration vs exploitation
    
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

    def save_weights(self, filepath, overwrite=False):
        keras.models.save_model(self.model, filepath, overwrite=overwrite)
