from dataclasses import dataclass
from RL_lib.Environment.Environment import Environment
import numpy as np
import random

@dataclass
class Exchanged_Var_Environment_Trading:
    '''Internal variable of environment communicated from Executor'''
    historic_2_trades: np.ndarray
    next_values_2_trades: np.ndarray
    evolution_2_trades: np.ndarray
    current_trade: int

class Environment_Compare_Trading(Environment):
    '''Environment used for Reinforcement Learning to determine the best suited trade between 2
    This Environment is currently used specifically for crypto but can easily be extended'''

    exchanged_var: Exchanged_Var_Environment_Trading

    def __init__(self, state_shape: np.ndarray, prc_taxes: float=0.01):
        '''Environment dependant of the size of historic for prediction'''
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = np.array((2,)) # Choose 1rst or 2nd trade
        self.prc_taxes=prc_taxes

    def set_exchanged_var(self, exchanged_var: Exchanged_Var_Environment_Trading):
        '''Exchanged variable in order to correctly define states depending on the mode'''
        self.exchanged_var = exchanged_var

    def get_current_trade(self):
        return self.exchanged_var.current_trade

    def step(self, action: np.ndarray):
        '''Basic Environment'''
        if np.shape(action) != self.action_shape:
            raise ValueError("Size of action is not corrsponding with environment")

        self.next_state = np.ones(self.state_shape)
        reward = max(np.sum(action), 10)
        done = False
        info = None
        # Depending on action of Agent, return new state + reward
        info = None
        reward = None
        self.next_state = None

        # Get reward depending on action + evolution
        def get_reward(action, evolution, index_current, index_compare):
            # The action to switch (action=1) is relevant only if evolution of change is better than the taxes caused by switching
            return pow(-1,(action>0)) * (evolution[index_current] - evolution[index_compare]) - self.has_taxes*self.prc_taxes
        
        if self._mode == 'train':
            reward = get_reward(action, self.last_experience['evolution'], self.current_crypto, self.order_comparison[self.ctr_studied_crypto])
        
        # Get next state (next sutdied crypto)
        flag_change_crypto = (action>0)
        if flag_change_crypto:
            self.current_crypto = self.order_comparison[self.ctr_studied_crypto]

        done = (self.ctr_studied_crypto >= len(self.order_comparison))
        if not done:
            self.has_taxes = (self.has_taxes and not flag_change_crypto)
            self.ctr_studied_crypto+=1
            self.next_state = self.last_experience['state'][:,[self.current_crypto, self.order_comparison[self.ctr_studied_crypto]]] + [self.has_taxes]

        return self.next_state, reward, done, info

    def reset(self):
        '''Reset environment'''
        #TODO
        order_comparison = random.shuffle([c for c in range(self.nb_trade) if c != self.current_trade])

        self.state = np.zeros(self.state_shape)
        return self.state