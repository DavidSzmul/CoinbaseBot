### Algorithm of choice of crypto using DQN Reinforcement Learning Algo
### Model can be continuously improved by generating augmented database
from typing import Callable
import numpy as np
import random

from rl_lib.agent.dqn import DQN_Agent, DQN_parameters
from rl_lib.manager.manager import Agent_Environment_Manager, RL_Train_Perfs_Historic

from algorithms.lib_trade.processing_trade import Mode_Algo
from algorithms.lib_trade.environment_trade import Environment_Compare_Trading

from displayer.displayer import Displayer



class AlgoOneTrade_Manager_Factory(Agent_Environment_Manager):
    '''Class communicating with Environment to automatically give the right input to environment
    according to the mode
    TODO: Class needs to manage the save load of model Agent easily'''

    PRC_TAXES: float=0.02
    
    def __init__(self, size_historic: int):
        # Definition of environment
        env = Environment_Compare_Trading(size_historic, self.PRC_TAXES, is_order_random=True)
        # Definition of agent
        agent = DQN_Agent(env.get_state_shape(), env.get_action_shape(), 
                            params=DQN_parameters(gamma=0.99,)
                        )
        Agent_Environment_Manager.__init__(self, agent, env, flag_return_train_perfs=True)

class AlgoOneTrade_Main:
    def __init__(self):
        super().__init__()

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
        self.__loop = switcher[mode]
        self.__loop(agent, env, experiences, portfolio, displayer) # Run appropriate mode

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
                displayer.update(exchanged_var, self.train_perfs)

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


class Main_RL_Trade_Algo:
    '''Class containing and interacting with all objects to create the algorithm'''
    mode: Mode_Algo=None
    env: Environment
    agent: Agent
    generator: Train_Test_Generator_Trade
    executor: Historic_Executor
    portfolio: Portfolio
    displayer: Displayer

    train_experience: List[Experience_Trade]
    test_experience: List[Experience_Trade]
    current_experience: List[Experience_Trade]

    crypto_names: List[str]
    crypto_historic: DataFrame

    # Properites of environment
    duration_past: int
    duration_future: int
    prc_taxes: float

    def __init__(self, model_path=None):
        
        # Determine Cryptos studied
        self.crypto_historic = Historic_coinbase_dtb.load(Historic_coinbase_dtb.Resolution_Historic.min)
        self.crypto_names = [str(c) for c in self.crypto_historic.columns]

        # Create Generator of database/preprocessing
        self.generator = Train_Test_Generator_Trade(self.crypto_historic, verbose=True)

        # Define Environment
        print('Creation of Environment...')
        self.duration_past = 180
        self.duration_future = 60
        self.prc_taxes = 0.01
        state_shape = np.array([self.duration_past,len(self.crypto_names)])
        self.env = Environment_Compare_Trading(state_shape,self.prc_taxes)
        action_shape = self.env.get_action_shape()

        # Define Agent
        print('Creation of Agent...')
        params = DQN_parameters()
        layers_NN = [64, 32]
        self.agent = DQN_Agent(state_shape, action_shape, layers_model=layers_NN,
                                name_model=model_path)
        #Create Executor for Train/Test/RT
        self.executor = Historic_Executor()

    def _evolution_method(self, future_values: np.ndarray):
        '''Definition of evolution of cryptocurrencies in order to determine reward:
            Because Reinforcement Learning enables to the Bot to determine rewards based on future rewards (Q-value with DQN),
            it is possible to estimate a relatively simple evolution used for reward'''
        # First idea consists in giving a median of future increases
        return np.median(future_values - future_values[:,0], axis=1)

    def reset_mode(self, mode: Mode_Algo):
        # Check if a new train/test database needs to be regenerated
        previous_mode = self.mode
        mode_needing_dtb = [Mode_Algo.train, Mode_Algo.test]    
        flag_regenerate = (mode in mode_needing_dtb) and (previous_mode not in mode_needing_dtb)

        if flag_regenerate:
            self.train_experience, self.test_experience = self.generator.generate_train_test_database(
                                            self, self.crypto_historic,
                                            self.duration_past, self.duration_future,
                                            evolution_method=self._evolution_method,
                                            verbose=True)
        self.mode = mode
        
        # Set current experiences to appropriate mode (no memory added because it acts as a pointer)
        if self.mode == Mode_Algo.train:
            self.current_experience = self.train_experience
        elif mode == Mode_Algo.test:
            self.current_experience = self.test_experience
        else:
            self.current_experience = None

    def execute(self, mode, initial_amount=0):

        # Reset mode if necessary
        self.reset_mode(mode)

        STABLECOIN_NAME = 'USDC-USD'
        self.portfolio = Portfolio(self.crypto_names)
        self.portfolio.add_money(initial_amount, STABLECOIN_NAME, need_confirmation=False)
        current_crypto = self.crypto_names.index(STABLECOIN_NAME)
        displayer = None

        self.executor.start(self.agent, self.env,
                            self.mode, current_crypto, self.crypto_names,
                            self.current_experience, self.portfolio,
                            displayer)

    def save_model(self,path: str):
        self.agent.save_weights(path,overwrite=True)

        
        

if __name__ == '__main__':
    main = Main_RL_Trade_Algo(model_path=None)

    # Start 1rst train
    main.execute(Mode_Algo.train)
    path = 'data/model/best'
    main.save_model(path)

    
    
