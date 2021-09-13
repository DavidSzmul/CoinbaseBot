from collections import deque
import itertools
from dataclasses import dataclass
from typing import Any, List, Tuple
import numpy as np

from rl_lib.memory.memory import Experience
from rl_lib.agent.agent import Agent
from rl_lib.environment.environment import Environment

@dataclass
class RL_Train_Perfs:
    total_reward: float
    total_loss: float
    epsilon: float

class RL_Train_Perfs_Historic:
    '''Dataclass Constaining all info about Reinforcement Learning training'''
    max_deque: int          # Max size of dequeue
    nb_window: int          # Window size for min/max/avg criteria

    t: deque
    total_rewards: deque    # Total reward during last cycles of training
    total_losses: deque     # Total loss during last cycles of training
    epsilons: deque         # Current ratio between exploration vs exploitation
    
    total_rewards_avg: deque
    total_rewards_std: deque
    envelopes_plus: deque
    envelopes_minus: deque

    def __init__(self, max_deque: int=10000, nb_window=None):
        self.max_deque = max_deque
        self.nb_window = nb_window
        self.reset()

    def is_avg_criteria(self):
        return (self.nb_window is not None)

    def reset(self):

        self.t = deque(maxlen=self.max_deque)
        self.total_rewards = deque(maxlen=self.max_deque)
        self.total_losses = deque(maxlen=self.max_deque)
        self.epsilons = deque(maxlen=self.max_deque)

        if self.is_avg_criteria():
            self.total_rewards_avg = deque(maxlen=self.max_deque)
            self.total_rewards_std = deque(maxlen=self.max_deque)
            self.envelopes_plus = deque(maxlen=self.max_deque)
            self.envelopes_minus = deque(maxlen=self.max_deque)

    def add(self, perfs: RL_Train_Perfs):
        if self.t:
            self.t.append(self.t[-1]+1)
        else:
            self.t.append(0)
        self.total_rewards.append(perfs.total_reward)
        self.total_losses.append(perfs.total_loss)
        self.epsilons.append(perfs.epsilon)

        if self.is_avg_criteria():
            NB_SIGMA = 3

            reward_window = list(itertools.islice(self.total_rewards, max(len(self.total_rewards)-self.nb_window, 0), len(self.total_rewards)))
            avg = np.mean(reward_window)
            std = np.std(reward_window)
            self.total_rewards_avg.append(avg)
            self.total_rewards_std.append(std)
            self.envelopes_plus.append(avg + NB_SIGMA*std)
            self.envelopes_minus.append(avg - NB_SIGMA*std)
            

class Agent_Environment_Manager:

    agent: Agent
    env: Environment
    flag_return_train_perfs: bool

    def __init__(self, agent: Agent, env: Environment, flag_return_train_perfs: bool=False):

        # Verify that agent is compatible with environment
        self._verify_compatibility(agent, env)
        self.agent = agent
        self.env = env
        self.flag_return_train_perfs = flag_return_train_perfs
        
    def _verify_compatibility(self, agent: Agent, env: Environment):
        flag_compatible = (np.all(agent.action_shape == env.action_shape) 
                        and np.all(agent.state_shape == env.state_shape))
        if not flag_compatible:
            raise AssertionError('Agent and Environment are not compatible based on Action/State shapes')
 
    def loop_episode_train(self, nb_cycle_train: int=1) -> Tuple[Any, RL_Train_Perfs]:
        '''Loop on Agent+Environment until environment is done'''
        if nb_cycle_train<1:
            raise ValueError('nb_cycle_train shall be strictly superior to 1')

        # Initialization
        state = self.env.reset()
        done = False
        ctr_cycle_train = 0
        perfs=None

        if self.flag_return_train_perfs:
            total_reward=0
            total_loss=0

        while not done:
            # Choose action depending on exploration
            action = self.agent.get_action_training(state)
            new_state, reward, done, info = self.env.step(action)
            experience = Experience(state, action, new_state, reward, done)
            loss = self.agent.memorize(experience)

            # Train Agent
            ctr_cycle_train = (ctr_cycle_train+1)%nb_cycle_train
            if (ctr_cycle_train==0):
                self.agent.fit()
            
            if self.flag_return_train_perfs:
                total_reward += reward
                total_loss += loss
            
            # New cycle
            state = new_state
        # End loop
        if self.flag_return_train_perfs:
            perfs=RL_Train_Perfs(total_reward, total_loss, self.agent.epsilon)
            
        return info, perfs

    def loop_episode(self) -> Any:
        '''Loop on Agent+Environment until environment is done (without training)'''
        # Initialization
        state = self.env.reset()
        done = False
        while not done:
            # Choose action (no randomness due to exploration)
            action = self.agent.get_action(state)
            state, _, done, info = self.env.step(action)
        return info
        

if __name__=="__main__":
    pass