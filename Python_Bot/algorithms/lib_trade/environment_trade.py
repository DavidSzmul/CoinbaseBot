### Contains Environment, Data
from dataclasses import dataclass
import numpy as np
import pandas as pd
import random
from typing import List, Tuple
from rl_lib.environment.environment import Environment
from algorithms.lib_trade.processing_trade import Experience_Trade

STD_TAXE: np.float64=1e-2 # Standard deviation of taxes (constant)

@dataclass
class Info_Trading:
    flag_change_trade: bool
    from_: str
    to_: str
    last_values: pd.DataFrame

class Environment_Compare_Trading(Environment):
    '''Environment used for Reinforcement Learning to determine the best suited trade between 2
    This Environment is currently used specifically for crypto but can easily be extended'''

    # Memorized input
    current_exp: Experience_Trade

    prc_taxes: float
    is_order_random: bool

    # Variables
    idx_current_trade: int=0
    current_trade: str=''
    previous_trade: str=''
    
    state_shape: np.ndarray
    action_shape: np.ndarray
    order_comparison: List[int]
    ctr_comparison: int=0
    nb_trade: int
    has_taxes: bool

    def __init__(self, size_historic: int, prc_taxes: float, is_order_random: bool=True):
        '''Environment dependant of the size of historic for prediction'''
        super().__init__()
        self.state_shape = np.array([2*size_historic+1])  # Historic of 2 trades + Taxes
        self.action_shape = np.array((2,))              # Choose 1rst or 2nd trade
        self.prc_taxes=prc_taxes
        self.is_order_random = is_order_random

    def _reset_order_comparison(self):
        '''Create the order to check all trades'''
        if self.idx_current_trade is None or self.idx_current_trade>=self.nb_trade or self.idx_current_trade<0:
            raise ValueError('current_trade parameter shall be contained in [0, nb_trade-1]')
        
        self.order_comparison = [c for c in range(self.nb_trade) if c != self.idx_current_trade]
        self.ctr_comparison = 0
        if self.is_order_random:
            random.shuffle(self.order_comparison)
        
    def _get_trade_to_compare(self):
        return self.order_comparison[self.ctr_comparison]

    def _get_trades_compared(self) -> Tuple:
        return (self.idx_current_trade, self._get_trade_to_compare())

    def _generate_state(self):
        '''Generation of state depending on environment
        State ->
        [   Historic of crypto currently chosen, 
            Historic of crypto possibly better,
            Percentage of taxes to be included
            ]'''
        trades_compared = self._get_trades_compared()
        state_trades = self.current_exp.state[:,trades_compared]
        return np.hstack((state_trades.T.flatten(), [self.has_taxes*self.prc_taxes/STD_TAXE]))

    def _get_reward(self, evolution: np.ndarray, trades_2_compare: List[int], flag_change_trade: bool):
        '''Generate reward depending on evolution and action'''
        if evolution is None:
                return None
        # The action to switch is relevant only if evolution of change is better than the taxes caused by switching
        return pow(-1,flag_change_trade) * (evolution[trades_2_compare[0]] - evolution[trades_2_compare[1]]) - self.has_taxes*self.prc_taxes*flag_change_trade

    def get_current_trade(self):
        return self.idx_current_trade

    def set_new_data(self, exp: Experience_Trade, current_trade: str=''):

        if current_trade != '':
            self.current_trade = current_trade
            self.previous_trade = current_trade
        
        if exp.current_trades is not None: # Trades are not defined so only the index is important
            self.idx_current_trade = exp.current_trades.columns.get_loc(self.current_trade) # Otherwise, keep last kept trade
        
        self.current_exp = exp
        self.nb_trade = exp.state.shape[1]

    def reset(self):
        '''Reset environment -> Corresponds to a new timing to compare all trades
        If idx_current_trade is defined, reset '''      
        self.has_taxes = True  # While no trade has been exchanged, taxes are to be included
        self._reset_order_comparison()
        return self._generate_state()

    def step(self, action: np.ndarray):
        '''Update Status of Env depending on action'''
        # Initialization
        self.verify_action_shape(action)
        trades_compared = self._get_trades_compared()
        flag_change_trade = (action[1]==True)
        reward = self._get_reward(self.current_exp.evolution, trades_compared, flag_change_trade)
        
        # Update internal variables depending on action
        if flag_change_trade:
            self.has_taxes=False
            self.idx_current_trade = self._get_trade_to_compare()
        
        # Verify if environment has checked all trades
        self.ctr_comparison+=1
        done = (self.ctr_comparison>=self.nb_trade-1)
        
        # Generate next state 
        if done:
            next_state = np.zeros(self.state_shape)
        else:
            next_state = self._generate_state()
            

        # Other informations used by the manager (or parent classes)
        # Important informations after each choice of trades are:
        # - The current value of trades
        info = None
        if done:

            if self.current_exp.current_trades is not None:
                self.current_trade = self.current_exp.current_trades.columns[self.idx_current_trade]
                
                flag_change_trade = (self.current_trade != self.previous_trade)
                from_ = self.previous_trade if flag_change_trade else None
                to_ = self.current_trade if flag_change_trade else None

                info = Info_Trading(flag_change_trade, from_, to_, self.current_exp.current_trades)
                self.previous_trade = self.current_trade
            
        return next_state, reward, done, info