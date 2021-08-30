from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import random
from typing import Callable, List
from dataclasses import dataclass

class Scaler_Trade:

    default_scaler: StandardScaler
    nb_min_historic: List[int]          # Number of minutes between 2 values
    nb_iteration_historic: List[int]    # Number of iteration of min diff

    def __init__(self, nb_min_historic: List[int], nb_iteration_historic: List[int]):
        '''Initialization of Scaler specific to trades based on historical value'''

        self.default_scaler = StandardScaler(with_mean=False) # Supposed to be centered at 0
        self.nb_min_historic = nb_min_historic
        self.nb_iteration_historic = nb_iteration_historic

    def _get_diff_pct_trade(self, raw_historic: np.ndarray) -> np.ndarray:
        '''From historic get percentage difference of a list'''
        pct = np.diff(raw_historic, axis=0) / raw_historic[:-1, :]
        pct = np.concatenate((np.zeros((1, np.shape(pct)[1])), pct), axis=0)
        return pct

    def _set_std_2_scaler(self, list_std: List):
        self.default_scaler.fit([
            [0]*len(list_std), 
            2*np.array(list_std)]
        )

    def _get_std_list_normalize(self, list_std: List)-> List: 

        list_normalizer_std = []
        for std, nb_iter in zip(list_std, self.nb_iteration_historic):
            list_normalizer_std += [std]*nb_iter
        list_normalizer_std.reverse()
        return list_normalizer_std

    def fit(self, historic_trades: pd.DataFrame):
        '''Fit normalization based on historic data'''

        list_std = []
        arr_trades = historic_trades.to_numpy()

        # Get pct change for all trades depending on number of minutes
        for nb_min in self.nb_min_historic:
            buff_trades = arr_trades[::nb_min,:]
            pct_change = self._get_diff_pct_trade(buff_trades)
            list_std.append(np.std(pct_change))

        # Get all std of pct change (depending on minute) and send it to scaler
        list_normalizer_std = self._get_std_list_normalize(list_std)
        self._set_std_2_scaler(list_normalizer_std)

    def transform(self, state: np.ndarray):
        '''Transform state with raw values to normalized pct_change values'''
        pct_change = self._get_diff_pct_trade(state)
        return self.default_scaler.transform(pct_change)

        

@dataclass
class Experience_Trade:
    '''Correspond to one timing containing all informations based on trades to use RL
    historic        -> normalized historic of trades
    next_historic   -> normalized next historic of trades
    evolution       -> future evolution of trades (used for reward)
    current_trade   -> Current value of trades
    '''
    state: np.ndarray
    next_state: np.ndarray
    evolution: np.ndarray
    current_trade: pd.DataFrame

@dataclass
class Generator_Trade:
    '''Class that enables to generate Train/Test database especially for trades'''

    nb_min_historic: List[int]
    nb_iteration_historic: List[int]
    duration_future: int
    verbose: bool=False

    def _get_evolution(self, historic_trades: np.ndarray, evolution_method: Callable) -> np.ndarray:
        '''Call of the method to extract evolution of trades.
        Used for reward calculation during RL training'''
        return evolution_method(historic_trades)

    def _get_idx_window_historic(self, nb_iteration_historic: List[int], nb_min_historic: List[int]) -> List[int]:
        '''Function generating the window of indexes to obtain an historic state of trades'''
        if not nb_iteration_historic and not nb_min_historic:
            return None
        if len(nb_iteration_historic) != len(nb_min_historic):
            raise ValueError('nb_iteration_historic and nb_min_historic parameters must have equivalent length')
        
        idx_step = []
        for nb_iter, nb_min in zip(nb_iteration_historic, nb_min_historic):
            idx_step = idx_step + [-nb_min for _ in range(nb_iter)]
        idx_window = list(np.cumsum(idx_step) - idx_step[0])
        idx_window.reverse()
        return idx_window

    def _get_synchronous_experiences(self, historic_trades: pd.DataFrame,
                                scaler: Scaler_Trade,
                                evolution: np.ndarray=None) -> List[Experience_Trade]:
        '''Generate synchronous experiences based on duration of past and future used for input/evolution'''

        # Initialization
        experiences = []
        idx_window = self._get_idx_window_historic(self.nb_iteration_historic, self.nb_min_historic)
        min_index_research = -idx_window[0]+1
        arr_trades = historic_trades.to_numpy()

        # Loop over present
        for idx_present in range(min_index_research, np.shape(arr_trades)[0]-self.duration_future):
            
            # Extract states based on present
            idx_current_window = np.array(idx_window) + idx_present
            state = arr_trades[idx_current_window, :]          
            next_state = arr_trades[idx_current_window+1, :]   

            # Normalization of states
            state = scaler.transform(state)
            next_state = scaler.transform(state)
            
            # Evolution based on present (related to reward)
            evolution_predict = None
            if evolution: # In train mode
                evolution_predict = evolution[idx_present,:]

            # Share real trade value over present
            current_trade = arr_trades.iloc[[idx_present]]
            experiences.append(Experience_Trade(state, next_state, evolution_predict, current_trade))

        return experiences

    def _get_unsynchronous_experiences(self, historic_trades: np.ndarray, nb_experience: int,
                                        scaler: Scaler_Trade,
                                        evolution: np.ndarray=None) -> List[Experience_Trade]:
        '''Generate unsynchronous experiences (different trades on different timings to augment database. Only for train)'''
        
        # Initialization
        if nb_experience == 0:
            return None
        if evolution is None:
            raise ValueError('evolution shall not be null for unsynchronous experiences')

        arr_trades = historic_trades.to_numpy()

        experiences = []
        idx_window = self._get_idx_window_historic(self.nb_iteration_historic, self.nb_min_historic)
        min_index_research = -idx_window[0]+1
        nb_crypto = np.shape(arr_trades)[1]
        
        # Loop
        Range_t = list(range(min_index_research, np.shape(arr_trades)[0]-self.duration_future))
        for _ in range(nb_experience):

            ### Choose random present for each crypto
            idx_time = random.sample(Range_t, nb_crypto)
            state = np.zeros((len(idx_window), nb_crypto))
            next_state = np.zeros((len(idx_window), nb_crypto))
            evolution_predict = np.zeros((nb_crypto, ))

            for j in range(nb_crypto): # Over each crypto
                # Extract states based on timing
                idx_current_window = np.array(idx_window) + idx_time[j]
                state[:,j] = arr_trades[idx_current_window, j]         
                next_state[:,j] = arr_trades[idx_current_window+1, j]  

                # Evolution based on present (related to reward)
                evolution_predict[j] = evolution[idx_time[j], j]                    
            
            # Normalization of states
            state = scaler.transform(state)
            next_state = scaler.transform(state)

            current_trade = None # Useless for training
            
            experiences.append(Experience_Trade(state, next_state, evolution_predict, current_trade))

        return experiences

    def generate_train_test_database(self, historic_trades: pd.DataFrame,
                                        scaler: Scaler_Trade,
                                        evolution_method: Callable,
                                        ratio_unsynchrnous_time: float=0.66, 
                                        ratio_train_test: float=0.8
                                        ): 
        '''Function generating train/test database depending of the received trades
        ratio_unsynchrnous_time=0.66 ==> 2/3 of of training is unsychronous 
        to augment database'''
        self.cryptos_name = list(historic_trades.columns)  
        self.nb_cryptos = len(self.cryptos_name)

        # CUT TRAIN/TEST DTB
        if self.verbose:
            print('Generation of train/test database...')

        size_dtb = len(historic_trades.index)        
        idx_cut_train_test = int(ratio_train_test*size_dtb)
        train_arr = historic_trades[:idx_cut_train_test]
        test_arr = historic_trades[idx_cut_train_test:]

        # Get evolution of trades. Used for reward during training
        evolution_train = self._get_evolution(train_arr, evolution_method)             
            
        # Generate Train Database
        ## Synchronous
        exp_sync_train = self._get_synchronous_experiences(train_arr, 
                                    scaler,
                                    evolution=evolution_train
                                    )
        ## Unsynchronous
        len_synchronized_train = len(exp_sync_train)
        nb_train_asynchronous = int(len_synchronized_train/ratio_unsynchrnous_time)
        exp_unsync_train = self._get_unsynchronous_experiences(train_arr, nb_train_asynchronous,
                                    scaler,
                                    evolution=evolution_train
                                    )
        experiences_train = exp_sync_train + exp_unsync_train
        random.shuffle(experiences_train)

        ### Generate Test Database
        experiences_test = self._get_synchronous_experiences(test_arr, 
                                    scaler,
                                    )
        if self.verbose:
            print('Train/test database generated')
        return experiences_train, experiences_test