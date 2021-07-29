from pandas.core.frame import DataFrame
from sklearn.preprocessing import Normalizer
import numpy as np
import pandas as pd
import random
from typing import Callable, List

import config
from algo_one_trade import Experience_Trade

class Train_Test_Generator_Trade():
    '''Class that enables to generate Train/Test database especially for trades'''
    
    verbose: bool
    normalizer: Normalizer

    def __init__(self, historic_trades: pd.DataFrame, verbose: bool=True):
        '''Initialization of Generator -> Used for preprocessing'''
        self.verbose = verbose
        if self.verbose:
            print('Fit Normalizer from database...')
        self.normalizer = Normalizer()
        self._normalize(historic_trades, self.normalizer)

    #TODO: May be improved
    def _fit_normalizer(self, historic_trades: pd.DataFrame, normalizer: Normalizer):
        '''Fit Normalizer to inputs
        Preprocessing: Normalize inputs in order to train efficiently Agent'''
        # Need to normalize data in order to effectively train.
        # But this transform has to be done before running in real time

        # OR maybe use diff_prc, already normalized
        normalizer.fit(historic_trades)
        

    def _normalize(self, historic_trades: DataFrame, normalizer: Normalizer):
        '''Normalization of inputs
        Preprocessing: Normalize inputs in order to train efficiently Agent'''
        return normalizer.transform(historic_trades.to_numpy())

    def _get_evolution(self, historic_trades: np.ndarray, evolution_method: Callable):
        '''Call of the method to extract evolution of trades.
        Used for reward calculation during RL training'''
        return evolution_method(historic_trades)

    def _get_synchronous_experiences(self, historic_trades: np.ndarray,
                                duration_past: int, duration_future: int,
                                evolution: np.ndarray=None) -> List[Experience_Trade]:
        '''Generate synchronous experiences based on duration of past and future used for input/evolution'''
        experiences = []
        evolution_predict = None
        for idx_start_time in range(np.shape(historic_trades)[0]-duration_past-duration_future):
            
            idx_present = idx_start_time+duration_past
            historic_part = historic_trades[idx_start_time:idx_present, :]    # State
            historic_part_next = historic_trades[idx_present, :]              # Next iteration of state

            if evolution is not None: # In train mode
                evolution_predict = evolution[idx_present-1,:]
            experiences.append(Experience_Trade(historic_part, historic_part_next, evolution_predict))

        return experiences

    def _get_unsynchronous_experiences(self, historic_trades: np.ndarray, nb_experience: int,
                                duration_past: int, duration_future: int,
                                evolution: np.ndarray=None) -> List[Experience_Trade]:
        '''Generate unsynchronous experiences (different trades on different timings to augment database. Only for train)'''
        if nb_experience == 0:
            return None
        if evolution is None:
            raise ValueError('evolution shall not be null for unsynchronous experiences')

        experiences = []
        evolution_predict = None
        Range_t = list(range(0, np.shape(historic_trades)[0]-duration_past-duration_future))
        nb_crypto = np.shape(historic_trades)[1]

        for _ in range(nb_experience):
            ### Choose random timing for each crypto
            idx_time = random.sample(Range_t, nb_crypto)
            historic_part = np.zeros((duration_past, nb_crypto))
            historic_part_next = np.zeros((nb_crypto, ))
            evolution_predict = np.zeros((nb_crypto, ))

            for j in range(nb_crypto):
                idx_present_crypto = idx_time[j] + duration_past
                historic_part[:,j] = historic_trades[idx_time[j]:idx_present_crypto, j]                               # State
                historic_part_next[j] = historic_trades[idx_time[j] + duration_past, j]                      # Next iteration of state
                evolution_predict[j] = evolution[idx_time[j] + duration_past-1,j]   # Evolution related to reward

            experiences.append(Experience_Trade(historic_part, historic_part_next, evolution_predict))

        return experiences

    def generate_train_test_database(self, historic_trades: pd.DataFrame,
                                        duration_past: int, duration_future: int,
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
        historic_trades_normalized = self._normalize(historic_trades)
        
        idx_cut_train_test = int(ratio_train_test*size_dtb)
        train_arr = historic_trades_normalized[:idx_cut_train_test]
        test_arr = historic_trades_normalized[idx_cut_train_test:]

        # Get evolution of trades. Used for reward during training
        evolution_train = self._get_evolution(train_arr, evolution_method)             
            
        # Generate Train Database
        ## Synchronous
        exp_sync_train = self._get_synchronous_experiences(train_arr, 
                                    duration_past, duration_future,
                                    evolution=evolution_train
                                    )
        ## Unsynchronous
        len_synchronized_train = len(exp_sync_train)
        nb_train_asynchronous = int(len_synchronized_train/ratio_unsynchrnous_time)
        exp_unsync_train = self._get_unsynchronous_experiences(train_arr, nb_train_asynchronous,
                                    duration_past, duration_future,
                                    evolution=evolution_train
                                    )
        experiences_train =  random.shuffle(exp_sync_train + exp_unsync_train)

        ### Generate Test Database
        experiences_test = self._get_synchronous_experiences(test_arr, 
                                    duration_past, duration_future,
                                    )
        if self.verbose:
            print('Train/test database generated')
        return experiences_train, experiences_test