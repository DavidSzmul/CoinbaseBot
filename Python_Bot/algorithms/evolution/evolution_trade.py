from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

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
        if start_check_future is None or end_check_future is None:
            raise ValueError('Parameters shall not be None')
        if start_check_future>end_check_future:
            raise ValueError('Start has to be inferior compared to end')
        # if start_check_future<0:
        #     raise ValueError('Start has to be strictly superior to zero')
        
        self._start_check_future = start_check_future
        self._end_check_future = end_check_future

    def get_evolution(self, historic: np.ndarray) -> np.ndarray:
        '''Evolution based on median'''
        evolution = np.zeros(historic.shape)

        for i in range(historic.shape[0] - self._end_check_future +1):
            array_window = historic[np.max(i + np.arange(self._start_check_future, self._end_check_future), 0), :]
            evolution[i,:] = np.median(array_window, axis=0)/historic[i,:] - 1
        return evolution
    
    def get_time_anticipation(self) ->int:
        '''Max timing to get anticipation'''
        return self._end_check_future
