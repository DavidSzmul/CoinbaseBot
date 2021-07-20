import numpy as np
from dataclasses import dataclass
from keras.models import Sequential, load_model, save_model

from RL_lib.Memory.Memory import SimpleMemory, PER
from RL_Lib.Network.NeuralNetwork import NetworkGenerator

@dataclass
class DQN_parameters:
    '''Class containing all parameters intrinsect to DQN Agent'''
    gamma: float=0.99               # Discount factor related to possible future reward
    epsilon_min: float= 0.01
    epsilon_decay: float= 0.9995
    learning_rate: float=1e-3
    tau: float = 1e-2
    use_double_dqn: bool= True
    use_soft_update: bool=False
    use_PER: bool=True
    memory_size: int=50000,
    batch_size:int=32

class DQN_Agent:
    """Agent using Deep Q Learning"""
    def __init__(self, state_shape: np.ndarray, action_size: np.ndarray,
    automatic_model: bool= True, layers_model: List[int]= [32, 32],      # In case of auto-generated model
    loading_model: bool= False, name_model: str ='', model=None,   # In case of loaded model (or model directly)
    ):

        # DQN Parameters
        self.gamma = gamma  # discount rate
        self.epsilon_min = epsilon_min # Minimum Exploration
        self.epsilon_decay = epsilon_decay # Exploration decay
        self.learning_rate = learning_rate # learning rate of objective
        self.tau = tau 

        self.use_soft_update=use_soft_update
        self.use_double_dqn = use_double_dqn
        self.use_PER = use_PER
        
        # Global Parameters
        self.state_size = state_size
        self.action_size = action_size

        if self.use_PER:
            self.memory = PER(memory_size)
        else:
            self.memory = SimpleMemory(memory_size)

        self.batch_size = batch_size

        # Variables
        self.epsilon = 1.0  # exploration rate (variable)
        self.target_update_ctr = 0 # Used to count when to update target network with main network's weights

        # Model for DQN
        if loading_model: # Load Existing Model
            self.model = load_model(name_model) 
            print('DQN Model LOADED')
        elif automatic_model: # Generate Automatic Model
            self.model = self._build_model(layers_model)
            print('DQN Model BUILDED')
        else: # Or directly in input
            assert model is not None, 'No model set in input'
            self.model = model
        # Generate target
        self.target_model=tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights()) 

        
    def _build_model(self, layers_model):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(layers_model[0], activation='relu', input_shape=self.state_size))
        for layer in layers_model[1:]:
            # model.add(BatchNormalization()) # Batch Normalization is source of divergence for Reinforcement Learning
            model.add(Dense(layer, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss="mse", 
                      optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        return model

    def memorize(self, experience):
        state = experience[0]
        action = experience[1]
        reward = experience[2]
        next_state = experience[3] 
        done = experience[4]

        #Clip of reward
        reward=np.clip(reward, -1, 1)

        Q_model = self.model.predict(np.array([state]))[0]
        Q_next_target = self.target_model.predict(np.array([next_state]))[0] #Target model
        old_Q = np.copy(Q_model)

        ### Adapt Q_model depending of reward of next state
        if done:
            Q_model[action] = reward
        else:
            Q_model[action] = reward + self.gamma * max(Q_next_target)   
        error = sum(pow(Q_model - old_Q, 2))/len(Q_model) 

        if not self.use_PER:
            self.memory.add(experience)
        else: 
            self.memory.add(experience, error)
        return error

    # Trains main network every step during episode
    def fit(self):

        # Start training only if certain number of samples is already saved
        if len(self.memory) < self.batch_size:
            return

        if self.use_PER:   
            mini_batch, idx_memory = self.memory.sample(self.batch_size)
        else:
            mini_batch = self.memory.sample(self.batch_size)

        if self.use_double_dqn:
            states, Q_model, errors =self.Double_DQN_update(mini_batch)
        else:
            states, Q_model, errors =self.DQN_update(mini_batch)
        self.model.fit(states, Q_model, batch_size=self.batch_size, epochs=1, verbose=0)

        if self.use_PER:
            # update priority
            for i in range(self.batch_size):
                self.memory.update(idx_memory[i], errors[i])

        # Update Target from Model Parameters
        self.update_target()
        # Update Exploration epsilon number
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min) 
    
    def DQN_update(self, mini_batch):
        states = np.array([experience[0] for experience in mini_batch])
        action = np.array([experience[1] for experience in mini_batch])
        reward = np.array([experience[2] for experience in mini_batch])
        next_states = np.array([experience[3] for experience in mini_batch])
        done = np.array([experience[4] for experience in mini_batch])
        errors = np.zeros(len(mini_batch))

        #Clip of reward
        epsilon_err = 1e-4 # To avoid divergence due to precision of float
        reward=np.clip(reward, -1+epsilon_err, 1-epsilon_err)
        
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
        states = np.array([transition[0] for transition in mini_batch])
        action = np.array([transition[1] for transition in mini_batch])
        reward = np.array([transition[2] for transition in mini_batch])
        next_states = np.array([transition[3] for transition in mini_batch])
        done = np.array([transition[4] for transition in mini_batch])
        errors = np.zeros(len(mini_batch))

        #Clip of reward
        epsilon_err = 1e-4 # To avoid divergence due to precision of float
        reward=np.clip(reward, -1+epsilon_err, 1-epsilon_err)

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

    def update_target(self):
        if self.use_soft_update:
            for t, m in zip(self.target_model.trainable_variables, 
                            self.model.trainable_variables):
                            t.assign(t * (1 - self.tau) + m * self.tau)
        else: #Hard update
            self.target_update_ctr+=1
            if self.target_update_ctr%int(1/self.tau)==0:
                self.target_model.set_weights(self.model.get_weights()) 
                self.target_update_ctr=0

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
            # return self.env.action_space.sample() # ONLY if environment linked to class
            # ONLY FOR discrete events for the moment
            return random.choice(list(range(self.action_size)))

    def save_weights(self, filepath, overwrite=False):
        save_model(self.model, filepath, overwrite=overwrite)


if __name__ == '__main__':
    import os, sys, time
    import matplotlib.pyplot as plt

    # ### Use of GPU ?
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # flag_use_GPU = True
    # os.environ["CUDA_VISIBLE_DEVICES"]=["-1", "0"][flag_use_GPU]

    # Environment settings
    MODEL_NAME = 'CartePole'    

    # Display Results
    verbose = 1

    ### INITIALIZATION
    env = Environment("gym", 'CartPole-v0')
    env = env.getEnv()
    
    USE_SOFT_UPDATE=False
    USE_DOUBLE_DQN=True
    USE_PER=True

    flag_load_model = True
    path_best_Model = 'models/Best_Models/best_model.model'
    if flag_load_model:
        agent = DQN_Agent(env, loading_model=True, use_soft_update=USE_SOFT_UPDATE, use_double_dqn=USE_DOUBLE_DQN, use_PER=USE_PER, name_model=path_best_Model)
    else:
        agent = DQN_Agent(env, layers_model=[24,24],use_soft_update=USE_SOFT_UPDATE, use_double_dqn=USE_DOUBLE_DQN, use_PER=USE_PER)
    
    ### TRAINING/EXPLORATION
    STEPS_MAX = 1e4
    DELTA_SAVE = 0 
    agent.train(nb_steps=STEPS_MAX,delta_save=DELTA_SAVE, verbose=verbose, name_model = path_best_Model)
    
