### Algorithm of choice of crypto using DQN Reinforcement Learning Algo
### Model can be continuously improved by generating augmented database
from enum import Enum
from typing import Callable
import numpy as np
import random
from typing import List

# from Algorithms.portfolio import Portfolio
from rl_lib.agent.agent import Agent
from rl_lib.agent.executor import Executor
from displayer.displayer import Displayer
from algorithms.lib_trade.portfolio import Portfolio, Portfolio_Coinbase

from algorithms.lib_trade.algo_one_trade import Experience_Trade, Exchanged_Var_Environment_Trading, Environment_Compare_Trading

class Mode_Algo(Enum):
    train = 1
    test = 2
    real_time = 3

class Main_RL_Trade_Algo:
    '''Class containing and interacting with all objects to create the algorithm'''
    def __init__(self):
        pass

class Historic_Executor(Executor):
    '''Class communicating with Environment to automatically give the right input to environment
    according to the mode'''

    historic: np.ndarray
    trades_names: List[str]
    mode: Mode_Algo
    nb_time: int
    nb_trade: int
    ctr_trade: int
    loop: Callable

    def __init__(self):
        super().__init__(len_loss=1)

    def _reset(self, current_trade: int, experiences: List[Experience_Trade], trades_name: List[str]):
        self.current_trade = current_trade
        self.trades_names = trades_name

        self.nb_time = len(experiences)
        self.nb_trade = 0
        if experiences is not None:
            self.nb_trade = np.shape(experiences[0].historic)[1]
        self.ctr_trade = 0

    def start(self, agent: Agent, env: Environment_Compare_Trading,
            mode: Mode_Algo, current_trade: int, trades_name: List[str],
            experiences: List[Experience_Trade]=None,
            portfolio: Portfolio=None,
            displayer: Displayer=None):
        '''Start Loop of Train/Test/RT'''

        self._reset(current_trade, experiences, trades_name)
        switcher = {
            Mode_Algo.train: self.start_train,
            Mode_Algo.test: self.start_test,
            Mode_Algo.real_time: self.start_real_time,
        }
        self.loop = switcher[mode]
        self.loop(agent, env, experiences, portfolio, displayer) # Run appropriate mode

    def get_exchanged_var_env_TrainTest(self, experiences: List[Experience_Trade],
            idx_time: int, current_trade: int):
        '''Generate Exchanged variable for environment for Train/Test'''

        excanged_var = Exchanged_Var_Environment_Trading(
            historic = experiences[idx_time].historic,
            next_value = experiences[idx_time].next_value,
            evolution = experiences[idx_time].evolution,
            current_trade = current_trade,
            )
        return excanged_var

    def start_train(self, agent: Agent, env: Environment_Compare_Trading,
                        experiences: List[Experience_Trade], portfolio: Portfolio,
                        displayer: Displayer):

        '''Loop for training Agent'''
        for i_time in range(self.nb_time): # For each Time
            # Select random current_trade
            self.current_trade = random.choice(list(range(self.nb_trade)))
            # Update Exchanged Variables for environment
            exchanged_var = self.get_exchanged_var_env_TrainTest(experiences, self.current_trade, i_time)
            # Update Environment to set new states
            env.set_newTime_on_Env(exchanged_var)

            for _ in range(self.nb_trade-1): # For each crypto
                # Reset Environment (compare with another trade)
                env.reset()
                # Train Agent
                self.train_cycle(agent, env)
            
            # Add displayer to check performances
            if displayer is not None:
                displayer.update(exchanged_var)

    def start_test(self, agent: Agent, env: Environment_Compare_Trading, 
                    experiences: List[Experience_Trade], portfolio: Portfolio,
                    displayer: Displayer):
        '''Loop for testing Agent'''

        if self.current_trade is None:
            raise ValueError('"current_trade" can''t be None during test phase')

        for i_time in range(self.nb_time): # For each Time
            # Update Exchanged Variables for environment
            exchanged_var = self.get_exchanged_var_env_TrainTest(experiences, self.current_trade, i_time)
            # Update Environment to set new states
            env.set_newTime_on_Env(exchanged_var)
            
            for _ in range(self.nb_trade-1): # For each crypto
                # Reset Environment (compare with another trade)
                env.reset()
                # Execute Agent
                self.execute_cycle(agent, env)
            # Determine new current trade
            previous_trade = self.current_trade
            self.current_trade = env.get_current_trade()
            
            # Update portfolio values
            last_prices = {}
            for i,name in enumerate(self.trades_names):
                last_prices[name] = exchanged_var.historic[-1, i]
            portfolio.update_last_prices(last_prices)

            # Update portfolio if current_trade has changed
            if self.current_trade is not previous_trade:
                trade_from = self.trades_names[previous_trade]
                trade_to = self.trades_names[self.current_trade]
                value_convert = portfolio.get_value(trade_from)
                portfolio.convert_money(trade_from, trade_to, value_convert)

            # Add displayer to check performances
            if displayer is not None:
                displayer.update(exchanged_var, portfolio)



    def start_real_time(self, agent: Agent, env: Environment_Compare_Trading,
                            experiences: List[Experience_Trade], portfolio: Portfolio,
                            displayer: Displayer):
        '''Loop for executing in real-Time the association Agent+Env'''
        
        if self.current_trade is None:
            raise ValueError('"current_trade" can''t be None during real-time phase')

        ### Can only be finished by user
        finished = False
        while (not finished): # For each Time
            
            ### Get real-time values of trades (wait include)
            ### Update Environment to set new states
            self.update_environment_realtime(env, self.current_trade)

            for j in range(self.nb_trade-1):
                # Reset Environment (compare with another trade)
                env.reset()
                ### Test Agent
                self.execute_cycle(agent, env)
            ### Determine new current trade
            self.current_trade = env.get_current_trade()

            # Add displayer to check performances
            if displayer is not None:
                exchanged_var=None
                finished = displayer.update(exchanged_var)


# TODO
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
        self.cryptos_name = []

        ## Experiences
        self.train_experience = []
        self.test_experience = []
        self.curent_experiences = None
        self.normalizer = None
        self._mode = None

        ## Reinitialization Mode
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

if __name__ == '__main__':
    pass
    # env = Environment_Crypto()
    # env.generate_train_test_environment()

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
    
