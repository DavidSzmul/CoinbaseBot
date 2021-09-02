from dataclasses import dataclass
from typing import Any, List
import numpy as np

from rl_lib.memory.memory import Experience
from rl_lib.agent.agent import Agent
from rl_lib.environment.environment import Environment

@dataclass
class Train_perfs():
    total_reward: List[float]=[]   # Total reward during last cycles of training
    total_loss: List[float]=[]     # Total loss during last cycles of training
    epsilon: List[float]=[]        # Current ratio between exploration vs exploitation

    #TODO: Convert Train_Perfs into displayer

class Agent_Environment_Manager:

    agent: Agent
    env: Environment

    def __init__(self, agent: Agent, env: Environment):

        # Verify that agent is compatible with environment
        self._verify_compatibility(agent, env)
        self.agent = agent
        self.env = env
        
    def _verify_compatibility(agent: Agent, env: Environment):
        flag_compatible = (np.all(agent.action_shape == env.action_shape) 
                        and np.all(agent.state_shape == env.state_shape))
        if not flag_compatible:
            raise AssertionError('Agent and Environment are not compatible based on Action/State shapes')
 
    def loop_episode_train(self, nb_cycle_train: int=1, train_perfs: Train_perfs=None) -> Any:
        '''Loop on Agent+Environment until environment is done'''
        if nb_cycle_train<1:
            raise ValueError('nb_cycle_train shall be strictly superior to 1')

        # Initialization
        state = self.env.reset()
        done = False
        ctr_cycle_train = 0

        if train_perfs:
            train_perfs.total_reward.append(0)
            train_perfs.total_loss.append(0)
            train_perfs.epsilon.append(0)

        while not done:
            # Choose action (random or )
            action = self.agent.get_action_training(state)
            new_state, reward, done, info = self.env.step(action)
            experience = Experience(state, action, new_state, reward, done)
            loss = self.agent.memorize(experience)

            # Train Agent
            ctr_cycle_train = (ctr_cycle_train+1)%nb_cycle_train
            if (ctr_cycle_train==0):
                self.agent.fit()
            
            if train_perfs:
                train_perfs.total_reward[-1] += reward
                train_perfs.total_loss[-1] += loss
            
            # New cycle
            state = new_state
        # End loop
        if train_perfs:
            train_perfs.epsilon[-1] = self.agent.epsilon
        return info

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