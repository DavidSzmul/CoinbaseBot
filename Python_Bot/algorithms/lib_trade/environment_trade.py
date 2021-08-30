### Contains Environment, Data
from dataclasses import dataclass
import numpy as np
import random
from typing import List
from rl_lib.environment.environment import Environment
from algorithms.lib_trade.generator_trade import Experience_Trade, Generator_Trade


class Environment_Compare_Trading(Environment):
    '''Environment used for Reinforcement Learning to determine the best suited trade between 2
    This Environment is currently used specifically for crypto but can easily be extended'''

    experience_trade: Experience_Trade
    order_comparison: List[int]
    ctr_comparison: int=0
    nb_trade: int
    has_taxes: bool

    def __init__(self, state_shape: np.ndarray, prc_taxes: float):
        '''Environment dependant of the size of historic for prediction'''
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = np.array((2,)) # Choose 1rst or 2nd trade
        self.prc_taxes=prc_taxes

    def set_new_experience(self, experience_trade: Experience_Trade):
        '''New timing where to compare all available trades to choose the best one'''
        self.experience_trade = experience_trade
        
        self.nb_trade = np.shape(self.experience_trade.historic)[1]
        self.order_comparison = [c for c in range(self.nb_trade) if c != self.experience_trade.current_trade]
        random.shuffle(self.order_comparison)
        self.has_taxes = True # While no trade has been exchanged, taxes are to be included

    def get_state_shape(self):
        return self.state_shape
    
    def get_action_shape(self):
        return self.action_shape

    def get_current_trade(self):
        return self.experience_trade.current_trade
    
    def get_trade_to_compare(self):
        return self.order_comparison[self.ctr_comparison]
    
    def update_trade_to_compare(self):
        idx_to_trade = self.get_trade_to_compare()
        self.ctr_comparison+=1
        return idx_to_trade

    def reset(self):
        '''Reset environment -> Corresponds to the new comparison between 2 trades'''
        if self.experience_trade is None:
            raise ValueError('exchanged_var item sall not be empty')

        # Generate State-> [Historic of crypto currently chosen, 
        #                   Historic of crypto possibly better,
        #                   Percentage of taxes to be included]
        trades_2_compare =(self.experience_trade.current_trade, self.update_trade_to_compare())
        state_trades = self.experience_trade.state[:,trades_2_compare]
        self.state = np.array(np.reshape(state_trades), self.has_taxes*self.prc_taxes)

        return self.state

    def step(self, action: np.ndarray):
        '''Shows what shall be the next state based on action'''

        # Initialization
        self.verify_action_shape(action)
        trades_2_compare =(self.experience_trade.current_trade, self.get_trade_to_compare())
        flag_change_trade = (action[0]>0)

        # Get reward depending on action + evolution
        def get_reward(evolution, _trades_2_compare):
            # evolution[0]: trade currently chosen
            # evolution[1]: trade possibly better
            if evolution is None:
                return None
            # The action to switch is relevant only if evolution of change is better than the taxes caused by switching
            return pow(-1,flag_change_trade) * (evolution[_trades_2_compare[0]] - evolution[_trades_2_compare[1]]) - self.has_taxes*self.prc_taxes
        reward = get_reward(self.experience_trade.evolution, trades_2_compare)
        
        # Update internal variables depending on action
        if flag_change_trade:
            self.has_taxes=False
            self.experience_trade.current_trade = self.get_trade_to_compare()

        # Generate New State-> [Historic of crypto currently chosen, 
        #                   Historic of crypto possibly better,
        #                   Percentage of taxes to be included]
        state_prev_trades = self.experience_trade.state[1:,trades_2_compare]
        state_new_trades = self.experience_trade.next_state[trades_2_compare]
        state_trades = np.concatenate(state_prev_trades, state_new_trades, axis=0)
        self.state = np.array(state_trades.flatten(), [self.has_taxes*self.prc_taxes])
        
        # Other infos
        done = False
        info = None
        return self.state, reward, done, info