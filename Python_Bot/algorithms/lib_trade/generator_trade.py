from pandas.core.frame import DataFrame
from sklearn.preprocessing import Normalizer
import numpy as np
import pandas as pd
import random
from typing import Callable, List
from dataclasses import dataclass


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

class Generator_Trade():
    '''Class that enables to generate Train/Test database especially for trades'''
    
    verbose: bool
    normalizer: Normalizer

    def __init__(self, historic_trades: pd.DataFrame, verbose: bool=True):
        '''Initialization of Generator -> Used for preprocessing'''
        self.verbose = verbose
        if self.verbose:
            print('Fit Normalizer from database...')
        self.normalizer = Normalizer()
        self._fit_normalizer(historic_trades)

    #TODO: May be improved
    def _fit_normalizer(self, historic_trades: pd.DataFrame):
        '''Fit Normalizer to inputs
        Preprocessing: Normalize inputs in order to train efficiently Agent'''
        # Need to normalize data in order to effectively train.
        # But this transform has to be done before running in real time

        # OR maybe use diff_prc, already normalized
        self.normalizer.fit(historic_trades)
        
    def _normalize(self, historic_trades: DataFrame):
        '''Normalization of inputs
        Preprocessing: Normalize inputs in order to train efficiently Agent'''
        return self.normalizer.transform(historic_trades.to_numpy())

    def _get_evolution(self, historic_trades: np.ndarray, evolution_method: Callable):
        '''Call of the method to extract evolution of trades.
        Used for reward calculation during RL training'''
        return evolution_method(historic_trades)

    def _get_idx_window_historic(self, nb_iteration_historic: List[int], nb_min_historic: List[int]):
        '''Function generating the window of indexes to obtain an historic state of trades'''
        if len(nb_iteration_historic) != len(nb_min_historic):
            raise ValueError('nb_iteration_historic and nb_min_historic parameters must have equivalent length')
        idx_window = []
        last = 0
        for nb_iter, nb_min in zip(nb_iteration_historic, nb_min_historic):
            idx_tmp = [-nb_min*i for i in range(nb_iter)]
            idx_tmp = idx_tmp.reverse()
            idx_tmp = idx_tmp + last
            last = idx_tmp[0]
            idx_window = idx_tmp + idx_window
        return idx_window

    def _get_synchronous_experiences(self, historic_trades: pd.Dataframe,
                                nb_iteration_historic: List[int], nb_min_historic: List[int],
                                duration_future: int,
                                evolution: np.ndarray=None) -> List[Experience_Trade]:
        '''Generate synchronous experiences based on duration of past and future used for input/evolution'''

        # Initialization
        experiences = []
        idx_window = self._get_idx_window_historic(nb_iteration_historic, nb_min_historic)
        min_index_research = -idx_window[0]+1

        # Normalize
        historic_normalized = self._normalize(historic_trades)

        # Loop
        for idx_present in range(min_index_research, np.shape(historic_normalized)[0]-duration_future):

            historic_part = historic_normalized[idx_present+idx_window, :]          # State
            historic_part_next = historic_normalized[idx_present+idx_window+1, :]   # Next state

            evolution_predict = None
            if evolution: # In train mode
                evolution_predict = evolution[idx_present,:]
            current_trade = historic_trades.iloc[[idx_present]]
            experiences.append(Experience_Trade(historic_part, historic_part_next, evolution_predict, current_trade))

        return experiences

    def _get_unsynchronous_experiences(self, historic_trades: np.ndarray, nb_experience: int,
                                nb_iteration_historic: List[int], nb_min_historic: List[int],
                                duration_future: int,
                                evolution: np.ndarray=None) -> List[Experience_Trade]:
        '''Generate unsynchronous experiences (different trades on different timings to augment database. Only for train)'''
        
        # Initialization
        if nb_experience == 0:
            return None
        if evolution is None:
            raise ValueError('evolution shall not be null for unsynchronous experiences')

        experiences = []
        idx_window = self._get_idx_window_historic(nb_iteration_historic, nb_min_historic)
        min_index_research = -idx_window[0]+1
        nb_crypto = np.shape(historic_trades)[1]

        # Normalize
        historic_normalized = self._normalize(historic_trades)

        # Loop
        Range_t = list(range(min_index_research, np.shape(historic_normalized)[0]-duration_future))
        for _ in range(nb_experience):
            ### Choose random timing for each crypto
            idx_time = random.sample(Range_t, nb_crypto)
            historic_part = np.zeros((len(idx_window), nb_crypto))
            historic_part_next = np.zeros((len(idx_window), nb_crypto))
            evolution_predict = np.zeros((nb_crypto, ))

            for j in range(nb_crypto):
                historic_part[:,j] = historic_trades[idx_time[j]+idx_window, j]     # State
                historic_part_next[j] = historic_trades[idx_time[j]+idx_window, j]  # Next state
                evolution_predict[j] = evolution[idx_time[j], j]                    # Evolution related to reward
            current_trade = None # Useless for training
            
            experiences.append(Experience_Trade(historic_part, historic_part_next, evolution_predict, current_trade))

        return experiences

    def generate_train_test_database(self, historic_trades: pd.DataFrame,
                                        nb_iteration_historic: List[int], nb_min_historic: List[int],
                                        duration_future: int,
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
                                    nb_iteration_historic, nb_min_historic, 
                                    duration_future,
                                    evolution=evolution_train
                                    )
        ## Unsynchronous
        len_synchronized_train = len(exp_sync_train)
        nb_train_asynchronous = int(len_synchronized_train/ratio_unsynchrnous_time)
        exp_unsync_train = self._get_unsynchronous_experiences(train_arr, nb_train_asynchronous,
                                    nb_iteration_historic, nb_min_historic,
                                    duration_future,
                                    evolution=evolution_train
                                    )
        experiences_train = exp_sync_train + exp_unsync_train
        random.shuffle(experiences_train)

        ### Generate Test Database
        experiences_test = self._get_synchronous_experiences(test_arr, 
                                    nb_iteration_historic, nb_min_historic,
                                    duration_future,
                                    )
        if self.verbose:
            print('Train/test database generated')
        return experiences_train, experiences_test