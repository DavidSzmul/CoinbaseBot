### Algorithm of choice of crypto using DQN Reinforcement Learning Algo
### Model can be continuously improved by generating augmented database
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # To use only CPU (for debugging)
print('Use of CPU')

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
from Algorithms.portfolio import Portfolio
from Database.Get_RT_crypto_dtb import Scrapping_RT_crypto
from Coinbase_API.Scrapping_transfer_v3_Selenium import AutoSelector
from RL_lib.Agent.dqn import DQN_Agent

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
        self.train_experience = []
        self.test_experience = []
        self.curent_experiences = None
        self.current_crypto = 0 # Index of chosen crypto
        self.normalizer = None
        self._mode = None
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

    def _fit_normalizer(self, df_historic):
        
        # Need to normalize data in order to effectively train.
        # But this transform has to be done before running in real time
        self.normalizer = Normalizer()
        self.normalizer.fit(df_historic)
        # TODO
        # Or maybe use diff_prc, already normalized

    def generate_train_test_environment(self, flag_regenerate=True
                                        ratio_unsynchrnous_time = 0.66, # 2/3 of of training is unsychronous to augment database
                                        ratio_train_test = 0.8, verbose=1): 

        ### LOAD HISTORIC + STUDIED CRYPTO
        if verbose:
            print('Loading of Historic of cryptos...')
        df_historic = config.load_df_historic('min')
        
        ### NORMALIZE
        if verbose:
            print('Normalization of database...')
        self._fit_normalizer(df_historic)
        if not flag_regenerate:
            return

        ### CUT TRAIN/TEST DTB
        if verbose:
            print('Generation of train/test database...')
        cryptos = list(df_historic.columns)  
        df_arr_normalized = self.normalizer.transform(df_historic.to_numpy())
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

    def update_real_time_state(self, real_time):
        # TODO: Include normalization
        self.real_time_state = state

    
    def reset(self): #Reset state standardized as Gym
        if self._mode == 'train':
            # Need to get randomly a new experience
            self.last_experience = random.shuffle(self.curent_experiences)
            self._ctr+=1
            self.current_crypto = random.choice(list(range(self)))

        elif self._mode == 'test':
            self.last_experience = self.curent_experiences[self._ctr]
            self._ctr+=1

        # In real-time mode: last-experiene is updated by the user       
        return self.last_experience['state']

    def step(self, action):
        # Depending on action of Agent, return new state + reward
        info = None
        done = False
        reward = None

        new_state = self.last_experience['next_state']
        if (self.curent_experiences is not None) and (self._ctr>=len(self.curent_experiences)): # Train or test is finished
            done=1

        def get_reward(action, evolution):
            # The action to switch (action=1) is relevant only if evolution of change is better
            # than the taxes caused by switching
            # TODO: Maybe need to take into account if taxe exists or not (depending if switching is already made ?)
            return evolution - (action>0)*self.prc_taxes

        if self._mode == 'train':
            reward = get_reward(action, self.last_experience['evolution'])
        
        return new_state, reward, done, info


class DQN_Algo(object):

    def __init__(self, df_historic=None,
                    duration_historic=120, prc_taxes=0.01,
                    duration_future = 60,
                    loading_model=True, mode = None):
        ### Algo Properties 
        self.duration_historic = duration_historic
        self.duration_future= duration_future
        self.prc_taxes = prc_taxes

        ### Model Agent
        self.NAME_MODEL = 'data/model/DQN_Coinbase'
        LAYERS_MODEL = [128,64] # Only if new model
        USE_DOUBLE_DQN = True
        USE_SOFT_UPDATE = True
        USE_PER = True

        state_size = (self.duration_historic, 2)
        action_size = (2,)
        self.agent = DQN_Agent(state_size, action_size, loading_model=loading_model, name_model=path_best_Model, # In case of loaded model
                                    layers_model = LAYERS_MODEL, # In case of new model
                                        use_PER=USE_PER, use_double_dqn=USE_DOUBLE_DQN, use_soft_update=USE_SOFT_UPDATE)

        possible_modes = ['train_test', 'real-time']
        if mode not in possible_modes:
            raise ValueError(mode+' is not a valid mode for DQN_Algo')    
        self._mode = mode
        self.env = Environment_Crypto() #Can be reset to test

        ## Variables
        self._reset()
    
    def _reset(self):
        self.save = None
        self.transaction = None
        self.last_quality_factor = None

    def train(self, verbose=0):
        # Set environment to Train mode + Generate Train/Test Database
        self.env.reset_mode('train')

        # LOOP Episode
        nb_episodes = len(self.train_experience)
        nb_cryptos = nb.shape(self.train_experience('state'))[1]

        for ctr_episode in range(nb_episodes):
            
            ###########################################
            ### RESET OF EPISODE
            ###########################################
            part_state = self.env.reset() # Reset environment to new state
            current_crypto = random.choice(list(range(nb_cryptos))) # Choose randomly the currently used crypto

            state = part_state + [current_crypto]
            
            ###########################################
            ### DETERMINE BEST ACTION for multiple cryptos
            ###########################################
            order_comparison = random.shuffle([c for c in range(nb_cryptos) if c != current_crypto])

            has_taxes = True
            for idx_crypto_compared in order_comparison: # Study each cryptos to determine the best one
                action = self.agent.get_action_training(state)
                new_state, reward, done, _ = self.env.step(action)
                transition = (state, action, reward, new_state, done)
                state = new_state
                loss = self.agent.memorize(transition)
                episode_reward += reward
                episode_loss += loss

            if step>nb_steps_varmup and step%delta_train==0:
                self.agent.fit()
            if done:
                # print the score and break out of the loop
                print("episode: {}, Prc_training:, {:.2f}%, score: {}"
                    .format(episode, step/nb_steps*100, episode_reward))


            episode+=1

            # Restarting episode - Environment
            episode_reward = 0
            episode_loss = 0
            state = self.env.reset()
            done = False
            
            # LOOP Step
            while not done:       
                step+=1
                # Determine action to do     
                
                
                
                

                
            
            # END Episode   
            # DISPLAY HISTORIC
            if verbose==1:
                Disp.display_historic(self.agent.epsilon, episode_reward, episode_loss) 

            # SAVE MODEL
            if delta_save>0 and not step % delta_save: # and min_reward >= MIN_REWARD:
                self.agent.save_weights(name_model, overwrite=True)
            
        ### END OF EPISODES/TRAINING
        self.env.close()
        # np.save('historic_reward', ep_rewards)
        self.agent.save_weights(name_model,overwrite=True)

        # Training algorithm by simulating evolution of crypto
        

        pass

    def test(self, portfolio, df_historic, verbose=0):
        
        ### Initialization
        # Generate database needed for running
        if len(df_historic)<self.duration_historic:
            raise ValueError('Historic size is too low to test the algorithm')
        if verbose: # Init displaying
            portfolio_historic = {}
        
        ### Loop
        for i in range(self.duration_historic, len(df_historic)):
            df_cut = df_historic.iloc[i-self.duration_historic:i, :]
            self.run(portfolio, df_cut, time=df_historic.index[i])
            if verbose:
                self.append_portfolio_historic(portfolio_historic, df_historic.index[i], portfolio, self.transaction)

        ### Display results
        if verbose:
            self.display(portfolio_historic)
            self.save = portfolio_historic
        pass

    def loop_RealTime(self, portfolio, delta_t=60, verbose=0):

        # TODO Enabling the possibility to implement on local server the visualization
        # Create LocalSever
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.step_RealTime(portfolio, delta_t=delta_t))
    
    async def step_RealTime(self, portfolio, delta_t=60):
        
        ## Initializtion
        srap_RT_crypto = Scrapping_RT_crypto(max_len=self.duration_historic)
        scrap_transaction = AutoSelector()

        ## Loop at each new call
        while True:
            t_sart = time.time()

            ######################################
            ### Main Task
            # Update Crypto prices
            srap_RT_crypto.refresh_crypto_value()
            
            # Decision based on Algorithm
            self.run(portfolio, srap_RT_crypto.crypto_historic, time=int(t_sart))

            # Realize transaction if exists
            if self.transaction is not None:
                scrap_transaction.convert(self.transaction['from'], self.transaction['to'], self.transaction['value'])
            ######################################
            # Sleep current period of time
            t_sleep = delta_t-time.time()+t_sart
            if t_sleep<0:
                print(f'WARNING: Task takes more than {delta_t}s to execute')
            await asyncio.sleep(max(t_sleep,0))

    def run(self, portfolio, historic, time=None, debug_mode=False):
        
        #Initialization
        self.transaction = None
        matrix_in = historic.to_numpy()
        matrix_denoised = np.zeros(np.shape(matrix_in))
        matrix_denoised_normalized  = np.zeros(np.shape(matrix_in))
        matrix_percentage = np.zeros(np.shape(matrix_in))
        quality_factor = np.zeros((np.shape(matrix_in)[1],))

        keys_historic = list(historic.keys())
        keys_portfolio = list(portfolio.keys())

        ### Update prices of portfolio based on last time in historic
        portfolio.update_last_prices(historic.iloc[-1, :])

        ### Parameters
        Fs = 1/60 #Corresponding to 1min
        Fc = Fs/20
        Order = 2
        b,a = signal.butter(Order, Fc, 'low', fs=Fs, output='ba')
        pole = np.exp(-1/20)

        ### Determine the current crypto where all money is put      
        idx_from = np.argmax([portfolio[c]['value'] for c in keys_portfolio])
        from_ = keys_portfolio[idx_from]
        idx_historic_from = keys_historic.index(from_)

        ### PROCESSING
        for i in range(np.shape(matrix_in)[1]):
            ### Denoise to remove high variation
            matrix_denoised[:,i] = signal.filtfilt(b,a, matrix_in[:,i])

            ### Normalization in order to stay equivalent from one crypto to another
            matrix_denoised_normalized[:,i] = Normalizer().fit_transform([matrix_denoised[:,i]])[0]
        if np.any(matrix_denoised_normalized==0):
            raise ValueError('Null value that shouldn''t')

        ### Evolution of cryptos based on the current chosen one
        ### What is the most interesting is the derivative of evolution compared to the current crypto used
        ### Need also a middle term derivative in order verify if a local decrease is only temporary 
        NB_DELAY = 10 # In min
        diff_from = deriv_filter(matrix_denoised_normalized[:,idx_historic_from], NB_DELAY)
        for i in range(np.shape(matrix_in)[1]):
            # matrix_percentage[:,i] = matrix_denoised[:,i]/np.min(matrix_denoised[:,i]) -1 
            # matrix_percentage[:,i] = matrix_denoised[:,i]/np.average(matrix_denoised[:,i]) -1 
            # matrix_percentage[:,i] = matrix_denoised_normalized[:,i]/(matrix_denoised_normalized[:,idx_historic_from]) -1 
            matrix_percentage[:,i] = deriv_filter(matrix_denoised_normalized[:,i], NB_DELAY)

            ### Proposition of quality factor based on percentage
            in_factor = matrix_percentage[-np.shape(matrix_in)[0]//2:,i]
            ratio = np.ones(np.shape(in_factor))
            for j in (range(len(in_factor)-1)[::-1]):
                ratio[j] = pole*ratio[j+1]
            quality_factor[i] = sum(ratio*in_factor)/sum(ratio)*10000

        # Save quality for display
        self.last_quality_factor = {}
        ctr = 0
        for k in keys_historic:
            self.last_quality_factor[k] = quality_factor[ctr]
            ctr+=1
        
        #############################################################
        #### For the moment to simplify, only one crypto is chosen
        #############################################################

        value = portfolio[from_]['value']
        idx_to = np.argmax(quality_factor) # Index where to put all money
        to_ = keys_historic[idx_to]
        quality_to = quality_factor[idx_to]
        quality_from = quality_factor[idx_historic_from]

        stable_coin = 'USDC-USD'
        THR_ALLOW_CONVERSION = 40
        THR_STABLE = 10

        if (quality_to - quality_from>THR_ALLOW_CONVERSION) and (from_ != to_): #Allow transaction
            portfolio.convert_money(from_, to_, value, prc_taxes=self.prc_taxes)
            self.transaction = {'from': from_, 'to': to_, 'value': value, 'time': time}    
            print(self.transaction)

        elif (from_ !=stable_coin and quality_to-quality_factor[keys_historic.index(stable_coin)] < THR_STABLE): 
            portfolio.convert_money(from_, stable_coin, value, prc_taxes=self.prc_taxes)
            self.transaction = {'from': from_, 'to': stable_coin, 'value': value, 'time': time}    
            print(self.transaction)

        if debug_mode:
            t=np.linspace(0, np.shape(matrix_in)[0]-1, np.shape(matrix_in)[0])

            NB_SP = 4
            fig, sp = plt.subplots(NB_SP,1)
            for i in range(1,NB_SP-1):
                sp[0].get_shared_x_axes().join(sp[0], sp[i])

            matric_in_normalized = matrix_in.copy()
            for col in range(np.shape(matric_in_normalized)[1]):
                matric_in_normalized[:,col] = Normalizer().fit_transform([matric_in_normalized[:,col]])[0]
            
            sp[0].plot(t, matric_in_normalized, label=keys_historic)
            sp[1].plot(t, matrix_denoised_normalized, label=keys_historic)
            sp[2].plot(t, matrix_percentage, label=keys_historic)

            y_pos = np.arange(len(keys_historic))
            hbars = sp[3].barh(y_pos, quality_factor, align='center')
            sp[3].set_yticks(y_pos)
            sp[3].set_yticklabels(keys_historic)
            sp[3].invert_yaxis()  # labels read top-to-bottom
            sp[3].bar_label(hbars, fmt='%.2f')            
            sp[3].set_xlim(left=-1, right=1)  # adjust xlim to fit labels

            sp[0].set_title('Normalized Historic')
            sp[1].set_title('Normalized denoised Historic')
            sp[2].set_title('Percentage Evolution %')
            sp[2].set_xlabel('Time (min)')  
            sp[3].set_title('Quality factor')
            
            for i in range(3):
                sp[i].legend(loc="upper right")
            plt.show()  



    def append_portfolio_historic(self, historic, time, portfolio, transaction):

        #Initialization of dict
        if len(historic)==0:
            for k in portfolio.keys():
                historic[k] = {}
                for kk in portfolio[k].keys():
                    historic[k][kk] = []
            historic['time'] = []
            historic['transaction'] = []
            historic['quality_factor'] = pd.DataFrame()

        for k in portfolio.keys():
                for kk in portfolio[k].keys():
                    historic[k][kk].append(portfolio[k][kk])

        historic['time'].append(time)
        historic['transaction'].append((transaction is not None))
        historic['quality_factor']=historic['quality_factor'].append(self.last_quality_factor, ignore_index=True)

    def display(self, historic):
        NB_SP = 4
        fig, sp = plt.subplots(NB_SP, 1, sharex=True)

        ### Do copy of historic to remove useless cases
        hist = historic.copy()
        t = np.array(hist.pop('time'))
        t = np.floor(t/60)
        # t = (t-t[0])/60

        transaction_done = hist.pop('transaction')
        quality_factor = hist.pop('quality_factor')
        total_value = np.zeros(np.shape(t))

        for k in hist.keys():
            # # Verify that displaying crypto is relevant
            # if np.all(np.array(hist[k]['ammount'])==0):
            #     continue
            
            priceEvolution = np.array(hist[k]['last-price'])
            #########################
            ### Normalized Evolution
            priceEvolution_norm = Normalizer().fit_transform([priceEvolution])[0]
            sp[0].plot(t, priceEvolution_norm, '-*', label=k)
            #########################

            ### Values
            total_value += np.array(hist[k]['value'])
            sp[1].plot(t, hist[k]['value'], '-*', label=k)

            ### Quality factors
            sp[2].plot(t, quality_factor[k], '-*', label=k)
        ### Add total value
        sp[1].plot(t, total_value, '-*', label='Total')

        ### Confirmation of transaction done by algo
        sp[3].step(t, transaction_done, '-*', label='Transaction done')

        sp[0].set_ylabel('Normalized Price')
        sp[1].set_ylabel('Value Portfolio')
        sp[2].set_ylabel('Quality Factor')
        sp[-1].set_xlabel('Time (min)')

        for i in range(NB_SP):
            sp[i].legend()
        plt.tight_layout()
        plt.show()

def deriv_filter(x, nb_delay):
    b = [0]*(nb_delay+1)
    b[0] = 1
    b[-1] = -1
    a = [1]

    x_with_delay = np.concatenate((x[0]*np.ones((nb_delay,)), x), axis=0)
    diff_x = signal.lfilter(b,a, x_with_delay)
    return diff_x[nb_delay:]

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
    
