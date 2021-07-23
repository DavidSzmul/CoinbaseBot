import numpy as np
from dataclasses import dataclass
from typing import List, Callable
import random
import keras

from RL_lib.Memory.Memory import Experience, DataMemoryUpdate, SimpleMemory, PER
from RL_lib.Network.NeuralNetwork import NetworkGenerator
from RL_lib.Agent.agent import Agent

@dataclass
class DQN_parameters:
    '''Class containing all parameters intrinsect to DQN Agent'''
    gamma: float=0.99               # Discount factor related to possible future reward
    epsilon_min: float=0.01         # Minimum random action ratio (only for train)
    epsilon_decay: float=0.9995     # Decay for random action (only for train)
    learning_rate: float=1e-3       # learning rate of NN weights
    tau: float=1e-2                 # 1rst order pole to update target weights 
    use_double_dqn: bool=True       # Double DQN Activation
    use_soft_update: bool=True      # Soft-update Activation (Hard if false)
    use_PER: bool=True              # PER for Memory
    memory_size: int=50000          # Size of Memory         
    batch_size: int=32              # Batch size of Memory

    epsilon_err = 1e-5              # Epsilon to clip reward stricly inferior to [-1,1]

class DQN_Agent(Agent):
    """Agent using Deep Q Learning"""

    update_model :Callable
    update_target :Callable

    def __init__(self, state_shape: np.ndarray, action_shape: np.ndarray,
            automatic_model: bool= True, layers_model: List[int]= [32, 32],      # In case of auto-generated model
            loading_model: bool= False, name_model: str ='', model=None,        # In case of loaded model (or model directly)
            params: DQN_parameters = DQN_parameters()):

        # Global Parameters
        self.state_shape = state_shape
        self.action_shape = action_shape

        # DQN Parameters
        self.params = params

        # Memory
        if self.params.use_PER:
            self.memory = PER(params.memory_size)
        else:
            self.memory = SimpleMemory(params.memory_size)

        # Update Function (depending if DQN or DDQN)
        if self.params.use_double_dqn:
            self.update_model = self.Double_DQN_update
        else:
            self.update_model = self.DQN_update

        # Hard or soft update target
        if self.params.use_soft_update:
            self.update_target = self.soft_update_target
        else:
            self.update_target = self.hard_update_target
        
        # Variables
        self.epsilon = 1.0  # exploration rate (variable)
        self.target_update_ctr = 0 # Used for hard target update

        # Model for DQN
        if loading_model: # Load Existing Model
            self.model = keras.models.load_model(name_model) 
            print('DQN Model LOADED')
        elif automatic_model: # Generate Automatic Model
            self.model = self._build_model(layers_model)
            print('DQN Model BUILDED')
        else: # Or directly in input
            assert model is not None, 'No model set in input'
            self.model = model

        # Generate target
        self.target_model=keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights()) 

        
    def _build_model(self, layers_model: List[int]):
        # Neural Net for Deep-Q learning Model
        return NetworkGenerator().create_DQN_Model(self.state_shape, self.action_shape, 
                                layers=layers_model, learning_rate=self.params.learning_rate)

    def _clip_reward(self, reward: float) -> float:
        #Clip of reward
        return np.clip(reward, -1+self.params.epsilon_err, 1-self.params.epsilon_err)

    def memorize(self, experience: Experience):
        state = experience.state
        action = experience.action
        reward = experience.reward
        next_state = experience.next_state 
        done = experience.done

        #Clip of reward
        reward=self._clip_reward(reward)

        Q_model = self.model.predict(np.array([state]))[0]
        Q_next_target = self.target_model.predict(np.array([next_state]))[0] #Target model
        old_Q = np.copy(Q_model)

        ### Adapt Q_model depending of reward of next state
        if done:
            Q_model[action] = reward
        else:
            Q_model[action] = reward + self.params.gamma * max(Q_next_target)   
        error = sum(pow(Q_model - old_Q, 2))/len(Q_model) 

        # Add error to experience
        experience.error = error
        self.memory.add(experience)
        return error

    # Trains main network every step during episode
    def fit(self):

        # Wait for warmup
        if len(self.memory) < self.batch_size:
            return

        # Sample experiences + train model
        mini_batch, idx_memory = self.memory.sample(self.batch_size)
        states, Q_model, errors =self.update_model(mini_batch)
        self.model.fit(states, Q_model, batch_size=self.batch_size, epochs=1, verbose=0)

        # Update Memory based on training
        data_update = DataMemoryUpdate(idx_memory, errors)
        self.memory.update(data_update)            

        # Update Target from Model (Hard or Soft)
        self.update_target()

        # Update Exploration epsilon number
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min) 
    

    def DQN_update(self, mini_batch: List[Experience]):
        states = np.array([experience.state for experience in mini_batch])
        action = np.array([experience.action for experience in mini_batch])
        reward = np.array([experience.reward for experience in mini_batch])
        next_states = np.array([experience.next_state for experience in mini_batch])
        done = np.array([experience.done for experience in mini_batch])
        errors = np.zeros(len(mini_batch))

        #Clip of reward
        reward=self._clip_reward(reward)

        Q_model = self.model.predict(states)
        Q_next_target = self.target_model.predict(next_states) #Target model
        old_Q = np.copy(Q_model)

        #DEBUG
        if np.any(Q_model > 1/(1-self.gamma)):
            print('DEBUG: Q value is diverging')

        ### Adapt Q_model depending of reward of next state
        for i in range(self.batch_size):
            if done[i]:
                Q_model[i][action[i]] = reward[i]
            else:
                Q_model[i][action[i]] = reward[i] + self.gamma * max(Q_next_target[i])
            
            # Error is MSE error (but only change on 1 with DQN)
            errors[i] = sum(pow(Q_model[i] - old_Q[i], 2))/len(Q_model[i])
        return (states, Q_model, errors)

    
    def Double_DQN_update(self, mini_batch):
        states = np.array([experience.state for experience in mini_batch])
        action = np.array([experience.action for experience in mini_batch])
        reward = np.array([experience.reward for experience in mini_batch])
        next_states = np.array([experience.next_state for experience in mini_batch])
        done = np.array([experience.done for experience in mini_batch])
        errors = np.zeros(len(mini_batch))

        #Clip of reward
        reward=self._clip_reward(reward)

        Q_model = self.model.predict(states)
        #DEBUG
        if np.any(Q_model > 1/(1-self.gamma)):
            print('DEBUG: Q value is diverging')

        Q_next_model = self.model.predict(next_states) #DQN
        Q_next_target = self.target_model.predict(next_states) #Target model
        old_Q = np.copy(Q_model)
        ### Adapt Q_model depending of reward of next state
        for i in range(self.batch_size):
            if done[i]:
                Q_model[i][action[i]] = reward[i]
            else:
                a = np.argmax(Q_next_model[i])
                Q_model[i][action[i]] = reward[i] + self.gamma * (Q_next_target[i][a])

            # Error is MSE error (but only change on 1 with DQN)
            errors[i] = sum(pow(Q_model[i] - old_Q[i], 2))/len(Q_model[i])
        return (states, Q_model, errors)

    def hard_update_target(self):
        self.target_update_ctr+=1
        if self.target_update_ctr%int(1/self.tau)==0:
            self.target_model.set_weights(self.model.get_weights()) 
            self.target_update_ctr=0

    def soft_update_target(self):
        for t, m in zip(self.target_model.trainable_variables, 
                        self.model.trainable_variables):
                        t.assign(t * (1 - self.tau) + m * self.tau)
            

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(state.reshape(-1, len(state)))[0]

    def get_action(self, state):
        # Get best action from Q depending on model
        return np.argmax(self.get_qs(state))

    def get_action_training(self, state):
        # Exploration during training
        if np.random.random() > self.epsilon:
            return self.get_action(state)
        else:
            # Get random action
            # ONLY FOR discrete events (as DQN)
            return random.choice(list(range(self.action_shape)))

    def save_weights(self, filepath, overwrite=False):
        keras.models.save_model(self.model, filepath, overwrite=overwrite)


if __name__ == '__main__':
    import os, sys, time
    import matplotlib.pyplot as plt
    from RL_lib.Environment.Environment import Default_Env

    # Display Results
    verbose = 1

    ### INITIALIZATION
    #TODO
    env = Default_Env()
    
    USE_SOFT_UPDATE=False
    USE_DOUBLE_DQN=True
    USE_PER=True

    flag_load_model = True
    path_best_Model = 'models/Best_Models/best_model.model'
    if flag_load_model:
        agent = DQN_Agent(env.state_shape, env.action_shape, loading_model=True, use_soft_update=USE_SOFT_UPDATE, use_double_dqn=USE_DOUBLE_DQN, use_PER=USE_PER, name_model=path_best_Model)
    else:
        agent = DQN_Agent(env.state_shape, env.action_shape, layers_model=[24,24],use_soft_update=USE_SOFT_UPDATE, use_double_dqn=USE_DOUBLE_DQN, use_PER=USE_PER)