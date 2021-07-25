### Algorithm of choice of crypto using DQN Reinforcement Learning Algo
### Model can be continuously improved by generating augmented database
from Algorithms.rl_algo import Environment_Crypto
from dataclasses import dataclass
from enum import Enum
from collections import deque
from typing import Callable
import pandas as pd
import numpy as np
import json
from matplotlib import pyplot as plt
from sklearn.preprocessing import Normalizer
from scipy import signal
import asyncio
import time
import random
import itertools

import config

# from Algorithms.portfolio import Portfolio
from RL_lib.Environment.Environment import Environment
from RL_lib.Agent.agent import Agent
from RL_lib.Agent.executor import Executor



class Mode_Algo(Enum):
    train = 1
    test = 2
    real_time = 3

class Historic_Executor(Executor):
    '''Class communicating with Environment to automatically give the right input to environment
    according to the mode'''

    historic: np.ndarray
    mode: Mode_Algo
    nb_trade: int
    ctr_trade: int
    loop: Callable

    def __init__(self):
        super().__init__(len_loss=1)

    def reset_mode(self, mode: Mode_Algo, current_trade: int, historic: np.ndarray=None):
        '''Set new historic in order to update environment'''
        self.mode = mode
        self.current_trade = current_trade
        self.historic = historic
        self.nb_time = np.shape(self.historic)[0]
        self.nb_trade = np.shape(self.historic)[1]
        self.loss_queue = deque(maxlen=self.nb_trade) # Used for training

        
    def start(self, agent: Agent, env: Environment_Compare_Trading):
        switcher = {
            Mode_Algo.train: self.start_train,
            Mode_Algo.test: self.start_test,
            Mode_Algo.real_time: self.start_real_time,
        }
        self.loop = switcher[self.mode]
        self.loop(env) # Run appropriate mode

    def update_environment(self, 
        env: Exchanged_Var_Environment_Trading, 
        historic, current_trade: int, idx_time: int):
        '''Update Environment based on specific historic'''

        excanged_var = Exchanged_Var_Environment_Trading(
            historic_2_trades = historic[idx_time]['state'],
            next_values_2_trades = historic[idx_time]['next_state'],
            evolution_2_trades = historic[idx_time]['evolution'],
            current_trade = current_trade,
            )
        env.set_exchanged_var(excanged_var)

    def update_environment_realtime(self, 
        env: Exchanged_Var_Environment_Trading,
        current_trade: int,
        ):
        '''TODO'''
        pass

    def start_train(self, agent: Agent, env: Environment_Compare_Trading):
        
        for i_time in range(self.nb_time): # For each Time
            ### Select random current_trade
            self.current_trade = random.choice(list(range(self.nb_trade)))
            ### Update Environment to set new states
            self.update_environment(env, self.historic, self.current_trade, i_time)
            for j in range(self.nb_trade):
                ### Train Agent
                self.train_cycle(agent, env)

    def start_test(self, agent: Agent, env: Environment_Compare_Trading):
        if self.current_trade is None:
            raise ValueError('"current_trade" can''t be None during test phase')

        for i_time in range(self.nb_time): # For each Time
            ### Update Environment to set new states
            self.update_environment(env, self.historic, self.current_trade, i_time)
            for j in range(self.nb_trade):
                ### Execute Agent
                self.execute_cycle(agent, env)
            ### Determine new current trade
            self.current_trade = env.get_current_trade

    def start_real_time(self, agent: Agent, env: Environment_Compare_Trading):
        if self.current_trade is None:
            raise ValueError('"current_trade" can''t be None during real-time phase')

        ### Can only be finished by user
        finished = False
        while (not finished): # For each Time
            
            ### Get real-time values of trades (wait include)
            ### Update Environment to set new states
            self.update_environment_realtime(env, self.current_trade)

            for j in range(self.nb_trade):
                ### Test Agent
                self.execute_cycle(agent, env)
            ### Determine new current trade
            self.current_trade = env.get_current_trade

            ### Check if user finished the process
            # TODO
            finished = False

        def execute_cycle(self, agent: Agent, env: Environment_Compare_Trading):
            '''Polymorphism: Adaptation of execute in order to know the actions done by the 
            Agent based on the Environment'''
            state = env.get_state()
            action = agent.get_action(state)
            _, _, done, _ = env.step(action)
            return done


# Environment interacting with Coinbase interface to train/test based on historic of cryptos
# The environment is made to sequentially compare cryptos one by one (including taxes when necessary)
# The specificity of this environment is that it is automatically done after all cryptos has been studied
class Environment_Crypto(object):

    def __init__(self, duration_historic=120, prc_taxes=0.01,
                    duration_future = 60, mode=None):
        ## Model Properties 
        self.duration_historic = duration_historic
        self.duration_future = duration_future
        self.prc_taxes = prc_taxes
        self.cryptos_name = []

        ## Experiences
        self.train_experience = []
        self.test_experience = []
        self.curent_experiences = None
        self.normalizer = None
        self._mode = None

        ## State of Environment
        self.current_crypto = 0     # Index of chosen crypto
        self.has_taxes = True       # State that taxes are to include during comparison of 2 cryptos
        self.order_comparison = []  # Order to compare crypto by crypto
        self.nb_cryptos = 0
        self.state = None
        self.next_state = None
        self.reward = None
        self.done = False

        ## Reinitialization Mode
        self.reset_mode(mode)

    def reset_mode(self, mode):
        previous_mode = self._mode
        # Depending on the chosen mode of environment: the step will act differently
        possible_modes = ['train', 'test', 'real-time', None]
        mode_needing_dtb = ['train', 'test']

        if mode not in possible_modes:
            raise ValueError('mode needs to be contained in' + str.join(possible_modes))        
        
        # Check if a new train/test database needs to be regenerated
        flag_regenerate = (mode in mode_needing_dtb) and (previous_mode not in mode_needing_dtb)
        self.generate_train_test_environment(flag_regenerate=flag_regenerate)
        self._mode = mode
        
        # Set current experiences to appropriate mode (no memory added because it acts as a pointer)
        if mode == 'train':
            self.curent_experiences = self.train_experience
        elif mode == 'test':
            self.curent_experiences = self.test_experience
        else:
            self.curent_experiences = None
        self._ctr = 0
        self.last_experience = {'state': None, 'next_state':None, 'evolution': None}

    #TODO
    def _fit_normalizer(self, df_historic):
        
        # Need to normalize data in order to effectively train.
        # But this transform has to be done before running in real time
        self.normalizer = Normalizer()
        self.normalizer.fit(df_historic)
        # TODO
        # Or maybe use diff_prc, already normalized

    #TODO
    def _normalize(self, states):
        # TODO
        # Normalization of state (used for train/test/real-time state)
        return self.normalizer.transform(states.to_numpy())

    def generate_train_test_environment(self, flag_regenerate=True,
                                        ratio_unsynchrnous_time = 0.66, # 2/3 of of training is unsychronous to augment database
                                        ratio_train_test = 0.8, verbose=1): 

        ### LOAD HISTORIC + STUDIED CRYPTO
        if verbose:
            print('Loading of Historic of cryptos...')
        df_historic = config.load_df_historic('min')
        self.cryptos_name = list(df_historic.columns)  
        self.nb_cryptos = len(self.cryptos_name)

        ### PREPARE NORMALIZATION
        if verbose:
            print('Normalization of database...')
        self._fit_normalizer(df_historic)
        if not flag_regenerate: # No need to create new train/test env
            return

        ### CUT TRAIN/TEST DTB
        if verbose:
            print('Generation of train/test database...')
        df_arr_normalized = self._normalize(df_historic)
        size_dtb = len(df_historic.index)
        idx_cut_train_test = int(ratio_train_test*size_dtb)
        train_arr = df_arr_normalized[:idx_cut_train_test]
        test_arr = df_arr_normalized[idx_cut_train_test:]

        ### DETERMINE FUTURE EVOLUTION OF CRYPTOS (only for train)
        def get_evolution(historic_array):
            # For each crypto (column), determine the best/worst evolution
            # in the future
            # TODO
            evolution_array = np.zeros(np.shape(historic_array))
            return evolution_array

        ### DEFINITION OF EXPERIENCES
        def get_synchronous_experiences(array, evolution=None): # For train + test
            
            experiences = []
            evolution_predict = None
            for idx_start_time in range(np.shape(array)[0]-self.duration_historic-self.duration_future):
                
                idx_present = idx_start_time+self.duration_historic
                state = array[idx_start_time:idx_present, :]    # State
                next_state = array[idx_present, :]              # Next iteration of state

                if evolution is not None: # In train mode
                    evolution_predict = evolution[idx_present-1,:]
                experiences.append({'state': state, 'next_state':next_state, 'evolution': evolution_predict})

            return experiences
                
        def get_unsynchronous_experiences(array, nb_experience, evolution=None): # Only for train
            
            if nb_experience == 0:
                return None
            if evolution is None:
                raise ValueError('evolution shall not be null for unsynchronous experiences')

            experiences = []
            evolution_predict = None
            Range_t = list(range(0, np.shape(array)[0]-self.duration_historic-self.duration_future))
            nb_crypto = np.shape(array)[1]

            for i in range(nb_experience):
                ### Choose random timing for each crypto
                idx_time = random.sample(Range_t, nb_crypto)
                state = np.zeros((self.duration_historic, nb_crypto))
                next_state = np.zeros((nb_crypto, ))
                evolution_predict = np.zeros((nb_crypto, ))

                for j in range(nb_crypto):
                    idx_present_crypto = idx_time[j] + self.duration_historic
                    state[:,j] = array[idx_time[j]:idx_present_crypto, j]                               # State
                    next_state[j] = array[idx_time[j] + self.duration_historic, j]                      # Next iteration of state
                    evolution_predict[j] = evolution[idx_time[j] + self.duration_historic-1,j]   # Evolution related to reward

                experiences.append({'state': state, 'next_state':next_state, 'evolution': evolution_predict})

            return experiences

        ### GENERATION OF TRAIN
        #### Synchronous
        evolution_train = get_evolution(train_arr) # Only train, not necessary for test
        exp_sync_train = get_synchronous_experiences(train_arr, evolution=evolution_train)

        #### Unsynchronous
        len_synchronized_train = len(exp_sync_train)
        nb_train_asynchronous = int(len_synchronized_train/ratio_unsynchrnous_time)
        exp_unsync_train = get_unsynchronous_experiences(train_arr, nb_train_asynchronous, evolution=evolution_train)

        experiences_train =  random.shuffle(exp_sync_train + exp_unsync_train)

        ### GENERATION OF TEST
        experiences_test = get_synchronous_experiences(test_arr, evolution=None)

        self.train_experience = experiences_train
        self.test_experience = experiences_test
        if verbose:
            print('Train/test database generated')


if __name__ == '__main__':

    env = Environment_Crypto()
    env.generate_train_test_environment()

    # Ptf = Portfolio()
    # Ptf['USDC-USD']['last-price'] = 1
    # Ptf.add_money(50, need_confirmation=False)
    # Algo = Simple_Algo()

    ##################################
    ### Test on database
    # Algo.run(Ptf, df)
    # Algo.test(Ptf, df, verbose=True)
    ##################################
    ### Loop in real time
    # Algo.loop_RealTime(Ptf)
    ##################################
    
