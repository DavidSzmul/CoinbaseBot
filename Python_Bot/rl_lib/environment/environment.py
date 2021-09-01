from abc import ABC, abstractmethod
import numpy as np

class Environment(ABC):
    '''Abstract class to create environment compatible with Reinforcement Learning Agent'''

    state_shape :np.ndarray
    action_shape :np.ndarray

    state: np.ndarray
    reward: float
    done: bool

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple:
        '''Update environment based on chosen action'''

    @abstractmethod
    def reset(self) -> np.ndarray:
        '''Reset environment'''

    def verify_action_shape(self, action: np.ndarray):
        if np.shape(action) != self.action_shape:
            raise ValueError("Size of action is not corrsponding with environment")

    def verify_state_shape(self, state: np.ndarray):
        if np.shape(state) != self.state_shape:
            raise ValueError("Size of state is not corrsponding with environment")

    def get_state_shape(self) -> np.ndarray:
        return self.state_shape

    def get_action_shape(self) -> np.ndarray:
        return self.action_shape

    def get_state(self) -> np.ndarray:
        return self.state

class Default_Env(Environment):

    def __init__(self) -> None:
        super().__init__()
        self.state_shape = np.array((4,))
        self.action_shape = np.array((2,))

    def step(self, action: np.ndarray):
        '''Basic Environment'''
        if np.shape(action) != self.action_shape:
            raise ValueError("Size of action is not corrsponding with environment")

        self.state = np.ones(self.state_shape)
        reward = max(np.sum(action), 10)
        done = False
        info = None
        return list((self.state, reward, done, info))

    def reset(self):
        '''Reset environment'''
        self.state = np.zeros(self.state_shape)
        return self.state

if __name__=="__main__":
    env = Default_Env()
    state = env.reset()
    action = np.ones(env.action_shape)
    next_state, reward, done, info = env.step(action)
    print(f"state: {state}, next_state: {next_state}, reward: {reward}, done: {done}, info: {info}")
