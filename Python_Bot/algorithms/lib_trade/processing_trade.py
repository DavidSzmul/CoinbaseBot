from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import random
from typing import List
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
    
class Scaler_Trade:

    default_scaler: StandardScaler
    nb_min_historic: List[int]          # Number of minutes between 2 values
    nb_iteration_historic: List[int]    # Number of iteration of min diff

    def __init__(self, nb_min_historic: List[int], nb_iteration_historic: List[int]):
        '''Initialization of Scaler specific to trades based on historical value'''

        self.default_scaler = StandardScaler(with_mean=False) # Supposed to be centered at 0
        self.nb_min_historic = nb_min_historic
        self.nb_iteration_historic = nb_iteration_historic

    def get_idx_window_historic(self) -> List[int]:
        '''Function generating the window of indexes to obtain an historic state of trades'''
        if not self.nb_iteration_historic and not self.nb_min_historic:
            return None
        if len(self.nb_iteration_historic) != len(self.nb_min_historic):
            raise ValueError('nb_iteration_historic and nb_min_historic parameters must have equivalent length')
        
        idx_step = []
        for nb_iter, nb_min in zip(self.nb_iteration_historic, self.nb_min_historic):
            idx_step = idx_step + [-nb_min for _ in range(nb_iter)]
        idx_window = list(np.cumsum(idx_step) - idx_step[0])
        idx_window.reverse()
        return idx_window


    def _get_std_list_normalize(self, list_std: List)-> List: 
        list_normalizer_std = []
        for std, nb_iter in zip(list_std, self.nb_iteration_historic):
            list_normalizer_std += [std]*nb_iter
        list_normalizer_std.reverse()
        return list_normalizer_std

    def reset_scaler(self, list_std: List):
        '''Change the form of scaler depending of number of trades
        (based on previously saved std)'''
        std_state = np.array(list_std)
        # Calibration
        states_calibration = np.array([
            0*std_state,
            2*std_state
        ])
        self.default_scaler.fit(states_calibration)

    def fit(self, historic_trades: pd.DataFrame):
        '''Fit normalization based on historic data'''

        list_std = []
        arr_trades = historic_trades.to_numpy()

        # Get pct change for all trades depending on number of minutes
        for nb_min in self.nb_min_historic:
            buff_trades = arr_trades[::nb_min,:]
            pct_change = self.get_pct_change(buff_trades)
            list_std.append(np.std(pct_change))

        # Get all std of pct change (depending on minute) and send it to scaler
        list_normalizer_std = self._get_std_list_normalize(list_std)
        self.reset_scaler(list_normalizer_std)

    def get_pct_change(self, raw_historic: np.ndarray) -> np.ndarray:
        '''From historic get percentage difference of a list'''
        pct = np.diff(raw_historic, axis=0) / raw_historic[:-1, :]
        pct = np.concatenate((np.zeros((1, np.shape(pct)[1])), pct), axis=0)
        return pct

    def transform_prc_change(self, prc_change: np.ndarray)-> np.ndarray:
        '''Normalize prc_change (transpose due to shape of state)'''
        return self.default_scaler.transform(prc_change.T).T 

    def transform(self, state: np.ndarray) -> np.ndarray:
        '''Normalize state after moving to prc_change'''
        prc_change = self.get_pct_change(state)
        normalized_prc = self.transform_prc_change(prc_change)
        return normalized_prc, prc_change

class Evolution_Trade(ABC):
    @abstractmethod
    def get_evolution(self, historic: pd.DataFrame):
        '''Method to get evolution of trade, used to estimate reward for Reinforcement Learning'''

    @abstractmethod
    def get_time_anticipation(self) ->int:
        '''Method to get the maximum time needed to calculate evolution'''

class Evolution_Trade_Median(Evolution_Trade):
    '''Class estimating evolution of trades based on median'''

    _start_check_future: int # Start of anticipation
    _end_check_future: int   # End   of anticipation
    def __init__(self, start_check_future, end_check_future):
        if not start_check_future or not end_check_future:
            raise ValueError('Parameters shall not be None')
        if start_check_future>end_check_future:
            raise ValueError('Start has to be inferior compared to end')
        if start_check_future<=0:
            raise ValueError('Start has to be strictly superior to zero')
        
        self._start_check_future = start_check_future
        self._end_check_future = end_check_future

    def get_evolution(self, historic: np.ndarray) -> np.ndarray:
        '''Evolution based on median'''
        evolution = np.zeros(historic.shape)

        for i in range(historic.shape[0] - self._end_check_future +1):
            array_window = historic[i + np.arange(self._start_check_future, self._end_check_future), :]
            evolution[i,:] = np.median(array_window, axis=0)/historic[i,:] - 1
        return evolution
    
    def get_time_anticipation(self) ->int:
        '''Max timing to get anticipation'''
        return self._end_check_future

    
            
@dataclass
class Experience_Trade:
    '''Correspond to one timing containing all informations based on trades to use RL
    state           -> normalized historic of trades
    evolution       -> future evolution of trades (used for reward)
    current_trades   -> Current value of trades
    '''
    state: np.ndarray
    evolution: np.ndarray
    current_trades: pd.DataFrame    

class Mode_Algo(Enum):
    train = 1
    test = 2
    real_time = 3

class Generator_Trade:
    '''Class that enables to generate Train/Test database especially for trades
    This generator is like an iterator'''

    mode: Mode_Algo=None
    scaler: Scaler_Trade
    ctr_train_test: int
    end_of_mode: bool

    def __init__(self, mode: Mode_Algo, scaler: Scaler_Trade):
        self.set_mode(mode)
        self.scaler = scaler

    def set_mode(self, mode: Mode_Algo):
        self.mode = mode
        self.end_of_mode = False
        self.ctr_train_test = 0

    def get_size_historic(self):
        return len(self.scaler.get_idx_window_historic())

    def get_new_experience(self):
        if self.mode == Mode_Algo.train: # TRAIN
            experience = self.experiences_train[self.ctr_train_test]
            self.ctr_train_test+=1
            self.end_of_mode = self.ctr_train_test>=len(self.experiences_train)

        elif self.mode == Mode_Algo.test:  # TEST
            experience = self.experiences_test[self.ctr_train_test]
            self.ctr_train_test+=1
            self.end_of_mode = self.ctr_train_test>=len(self.experiences_test)

        elif self.mode == Mode_Algo.real_time: # REAL-TIME
            # TODO: Call method of scrapping to get trades
            experience = None
            self.end_of_mode = False

        # Normalize state
        experience.state, _ = self.scaler.transform(experience.state)
        return experience

    def is_generator_finished(self):
        return self.end_of_mode

    def _get_synchronous_experiences(self, historic_trades: pd.DataFrame,
                                evolution_meth: Evolution_Trade=None) -> List[Experience_Trade]:
        '''Generate synchronous experiences based on duration of past and future used for input/evolution'''

        # Initialization
        experiences = []
        idx_window = self.scaler.get_idx_window_historic()
        min_index_research = -idx_window[0]+1
        arr_trades = historic_trades.to_numpy()

        # Extract evolution based on classmethod
        arr_evolution = None
        idx_anticipation = 0
        if evolution_meth: # If method defined
            arr_evolution = evolution_meth.get_evolution(arr_trades)
            idx_anticipation = evolution_meth.get_time_anticipation()

        # Loop over present
        for idx_present in range(min_index_research, np.shape(arr_trades)[0]-idx_anticipation):
            
            # Extract states based on present
            idx_current_window = np.array(idx_window) + idx_present
            state = arr_trades[idx_current_window, :]          
            
            # Evolution based on classmethod 
            evolution = None
            if arr_evolution is not None: # If evolution defined
                evolution = arr_evolution[idx_present,:]

            # Share real trade value over present
            current_trade = historic_trades.iloc[[idx_present]]
            experiences.append(Experience_Trade(state, evolution, current_trade))

        return experiences

    def _get_unsynchronous_experiences(self, historic_trades: np.ndarray, nb_experience: int,
                                        evolution_meth: Evolution_Trade=None) -> List[Experience_Trade]:
        '''Generate unsynchronous experiences (different trades on different timings to augment database. Only for train)'''
        
        # Initialization
        if nb_experience == 0:
            return None
        experiences = []
        idx_window = self.scaler.get_idx_window_historic()
        min_index_research = -idx_window[0]+1
        
        arr_trades = historic_trades.to_numpy()
        nb_crypto = np.shape(arr_trades)[1]
        
        # Extract evolution based on classmethod
        arr_evolution = None
        idx_anticipation = 0
        if evolution_meth: # If method defined
            arr_evolution = evolution_meth.get_evolution(arr_trades)
            idx_anticipation = evolution_meth.get_time_anticipation()
        
        # Loop
        Range_t = list(range(min_index_research, np.shape(arr_trades)[0]- idx_anticipation))
        for _ in range(nb_experience):

            ### Choose random present for each crypto
            idx_time = random.sample(Range_t, nb_crypto)
            state = np.zeros((len(idx_window), nb_crypto))

            for j in range(nb_crypto): # Over each crypto
                # Extract states based on timing
                idx_current_window = np.array(idx_window) + idx_time[j]
                state[:,j] = arr_trades[idx_current_window, j]         

            # Evolution based on classmethod 
            evolution = None
            if arr_evolution is not None: # If evolution defined
                evolution = np.array([arr_evolution[idx_time[j], j] for j in range(nb_crypto)])                
            current_trade = None # Useless for unsynchrnous experiences
            experiences.append(Experience_Trade(state, evolution, current_trade))

        return experiences

    def generate_train_test_database(self, historic_trades: pd.DataFrame,
                                        evolution_method: Evolution_Trade,
                                        ratio_unsynchrnous_time: float=0.66, 
                                        ratio_train_test: float=0.8
                                        ): 
        '''Function generating train/test database depending of the received trades
        ratio_unsynchrnous_time=0.66 ==> 2/3 of of training is unsychronous 
        to augment database'''
        self.cryptos_name = list(historic_trades.columns)  
        self.nb_cryptos = len(self.cryptos_name)

        # CUT TRAIN/TEST DTB
        size_dtb = len(historic_trades.index)        
        idx_cut_train_test = int(ratio_train_test*size_dtb)
        train_arr = historic_trades[:idx_cut_train_test]
        test_arr = historic_trades[idx_cut_train_test:]      
            
        # Generate Train Database
        ## Synchronous
        exp_sync_train = self._get_synchronous_experiences(train_arr, 
                                    evolution_meth=evolution_method
                                    )
        ## Unsynchronous
        len_synchronized_train = len(exp_sync_train)
        nb_train_asynchronous = int(len_synchronized_train/ratio_unsynchrnous_time)
        exp_unsync_train = self._get_unsynchronous_experiences(train_arr, nb_train_asynchronous,
                                    evolution_meth=evolution_method
                                    )
        experiences_train = exp_sync_train + exp_unsync_train
        random.shuffle(experiences_train)

        ### Generate Test Database
        experiences_test = self._get_synchronous_experiences(test_arr, 
                                    evolution_meth=None # No need to get evolution for test
                                    )        
        self.experiences_train = experiences_train # No shuffle in order to do a simulation of reality
        self.experiences_test = experiences_test