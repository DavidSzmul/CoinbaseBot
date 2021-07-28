### Contains Environment, Data to exchange and Displayer of test for the algorithm
from dataclasses import dataclass
import numpy as np
import random
from typing import List
from rl_lib.environment.environment import Environment

@dataclass
class Experience_Trade:
    '''Correspond to one timing containing all informations based on trades to use RL
    historic    -> normalized historics of trades
    next_value  -> normalized next value of trades
    evolution   -> future evolution of trades (used for reward)'''
    historic: np.ndarray
    next_value: np.ndarray
    evolution: np.ndarray

@dataclass
class Exchanged_Var_Environment_Trading:
    '''Internal variable of environment communicated from Executor'''
    historic: np.ndarray
    next_value: np.ndarray
    evolution: np.ndarray
    current_trade: int

class Environment_Compare_Trading(Environment):
    '''Environment used for Reinforcement Learning to determine the best suited trade between 2
    This Environment is currently used specifically for crypto but can easily be extended'''

    exchanged_var: Exchanged_Var_Environment_Trading
    nb_trade: int
    order_comparison: List[int]
    ctr_comparison: int
    has_taxes: bool

    def __init__(self, state_shape: np.ndarray, prc_taxes: float):
        '''Environment dependant of the size of historic for prediction'''
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = np.array((2,)) # Choose 1rst or 2nd trade
        self.prc_taxes=prc_taxes

    def set_newTime_on_Env(self, exchanged_var: Exchanged_Var_Environment_Trading):
        '''New timing where to compare all available trades to choose the best one'''
        self.exchanged_var = exchanged_var
        
        self.nb_trade = np.shape(self.exchanged_var.historic)[1]
        self.order_comparison = random.shuffle([c for c in range(self.nb_trade) if c != self.current_trade])
        self.has_taxes = True # While no trade has been exchanged, taxes are to be included


    def get_current_trade(self):
        return self.exchanged_var.current_trade
    
    def get_trade_to_compare(self):
        return self.order_comparison[self.ctr_comparison]
    
    def update_trade_to_compare(self):
        idx_to_trade = self.get_trade_to_compare()
        self.ctr_comparison+=1
        return idx_to_trade

    def reset(self):
        '''Reset environment -> Corresponds to the new comparison between 2 trades'''
        if self.exchanged_var is None:
            raise ValueError('exchanged_var item sall not be empty')

        # Generate State-> [Historic of crypto currently chosen, 
        #                   Historic of crypto possibly better,
        #                   Percentage of taxes to be included]
        trades_2_compare =(self.exchanged_var.current_trade, self.update_trade_to_compare())
        state_trades = self.exchanged_var.historic[:,trades_2_compare]
        self.state = np.array(np.reshape(state_trades), self.has_taxes*self.prc_taxes)

        return self.state

    def step(self, action: np.ndarray):
        '''Shows what shall be the next state based on action'''

        # Initialization
        self.verify_action_shape(action)
        trades_2_compare =(self.exchanged_var.current_trade, self.get_trade_to_compare())
        flag_change_trade = (action[0]>0)

        # Get reward depending on action + evolution
        def get_reward(evolution, _trades_2_compare):
            # evolution[0]: trade currently chosen
            # evolution[1]: trade possibly better
            if evolution is None:
                return None
            # The action to switch is relevant only if evolution of change is better than the taxes caused by switching
            return pow(-1,flag_change_trade) * (evolution[_trades_2_compare[0]] - evolution[_trades_2_compare[1]]) - self.has_taxes*self.prc_taxes
        
        reward = get_reward(action, self.exchanged_var.evolution, trades_2_compare)
        
        # Update internal variables depending on action
        if flag_change_trade:
            self.has_taxes=False
            self.exchanged_var.current_trade = self.get_trade_to_compare()

        # Generate New State-> [Historic of crypto currently chosen, 
        #                   Historic of crypto possibly better,
        #                   Percentage of taxes to be included]
        state_prev_trades = self.exchanged_var.historic[1:,trades_2_compare]
        state_new_trades = self.exchanged_var.next_value[trades_2_compare]
        state_trades = np.concatenate(state_prev_trades, state_new_trades, axis=0)
        self.state = np.array(state_trades.flatten(), [self.has_taxes*self.prc_taxes])
        
        # Other infos
        done = False
        info = None
        return self.state, reward, done, info