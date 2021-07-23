from RL_lib.Environment import Environment
import numpy as np

class Crypto_Choice_Environment(Environment):
    '''Environment to determine the best choice between 2 cryptos'''

    def __init__(self, state_shape):
        self.state_shape = state_shape
        self.action_shape = np.array((2,))
        
    def step(self, action: np.ndarray) -> tuple:
        '''Update environment based on chosen action'''

    def reset(self) -> np.ndarray:
        '''Reset environment'''
