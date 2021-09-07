### Algorithm of choice of crypto using DQN Reinforcement Learning Algo
### Model can be continuously improved by generating augmented database
from typing import List

from rl_lib.agent.dqn import DQN_Agent, DQN_parameters
from rl_lib.manager.manager import Agent_Environment_Manager

from algorithms.lib_trade.processing_trade import Generator_Trade, Mode_Algo, Scaler_Trade
from algorithms.lib_trade.environment_trade import Environment_Compare_Trading
from algorithms.lib_trade.portfolio import Portfolio
from algorithms.ihm_trade.ihm_trade import StatusLoop

from displayer.displayer_rl_train import Displayer_RL_Train


class AlgoOneTrade_Manager_Factory(Agent_Environment_Manager):
    '''Class communicating with Environment to automatically give the right input to environment
    according to the mode
    TODO: Class needs to manage the save load of model Agent easily'''

    PRC_TAXES: float=0.02
    path_save_model: str
    scaler: Scaler_Trade
    generator: Generator_Trade
    env: Environment_Compare_Trading
    
    def __init__(self, nb_min_historic: List[int], nb_iteration_historic: List[int], 
                    path_load_model: str=None, path_save_model: str=None,
                    displayer_train: Displayer_RL_Train=None, displayer_test=None, displayer_rt=None
                    ):
        # Initialization
        self.path_save_model = path_save_model
        self.displayer_train = displayer_train

        ### Params changable by the user (to create new model)
        LAYERS_MODEL = [128,64]

        # Definition of scaler for preprocessing and generating historic of trades
        self.scaler = Scaler_Trade(nb_min_historic, nb_iteration_historic)
        # Need to fit scaler based on previously saved trade historic


        # Definition of generator of experiences
        self.generator = Generator_Trade(Mode_Algo.train, self.scaler)

        # Definition of environment
        env = Environment_Compare_Trading(self.generator.get_size_historic(), self.PRC_TAXES, is_order_random=True)
        # Definition of agent
        agent = DQN_Agent(env.get_state_shape(), env.get_action_shape(), 
                            params=DQN_parameters(gamma=0.99,), #Definition of all parameters intresect to DQN (independant to model NN)
                        name_model=path_load_model, # Load existing model if not None
                        layers_model=LAYERS_MODEL
                        )
        Agent_Environment_Manager.__init__(self, agent, env, flag_return_train_perfs=True)

    def set_mode(self, mode: Mode_Algo):
        self.generator.set_mode(mode)

    def save_model(self, filepath, overwrite=False):
        self.agent.save_model(self.path_save_model, overwrite=True)

    def run(self, status_loop: StatusLoop):
        if self.generator.get_mode == Mode_Algo.train:
            self.run_training(status_loop)
        elif self.generator.get_mode == Mode_Algo.test:
            self.run_testing(status_loop)
        elif self.generator.get_mode == Mode_Algo.real_time:
            self.run_real_time(status_loop)

    def run_training(self, status_loop: StatusLoop):
        self.env.idx_current_trade = 0
        while (status_loop.is_running and not self.generator.is_generator_finished()):
            #Generate new experience on environment
            self.env.set_new_episode(self.generator.get_new_experience()) 
            # Start new experience/episode
            _, perfs = self.loop_episode_train()

            # Display performances if enabled
            if self.displayer_train:
                self.displayer_train.update(perfs)

    def run_testing(self, status_loop: StatusLoop):
        # Create a new portfolio based on generator
        portfolio = Portfolio()
        previous_idx_current_trade = 0
        self.env.idx_current_trade = previous_idx_current_trade
        init_money = True
        while (status_loop.is_running and not self.generator.is_generator_finished()):

            #Update environment and porfolio
            self.env.set_new_episode(self.generator.get_new_experience()) 
            last_prices = self.env.current_exp.current_trades
            trades_names = last_prices.columns
            idx_current_trade = self.env.get_current_trade()
            portfolio.update_last_prices(last_prices)

            # Add artificially money for the initialization
            if init_money:
                portfolio.add_money(to_=trades_names[idx_current_trade], value=1000)
                init_money=False

            # Start new experience/episode
            info = self.loop_episode()
            
            # Realize conversion depending on agent decision
            if idx_current_trade is not previous_idx_current_trade:
                
                trade_from = trades_names[previous_idx_current_trade] 
                trade_to = trades_names[idx_current_trade]
                value_convert = portfolio.get_value(trade_from)
                portfolio.convert_money(trade_from, trade_to, value_convert, prc_taxes=self.PRC_TAXES)

            # Display performances if enabled
            if self.displayer_test:
                pass
                # self.displayer_train.update(perfs)


if __name__ == '__main__':
    main = Main_RL_Trade_Algo(model_path=None)

    # Start 1rst train
    main.execute(Mode_Algo.train)
    path = 'data/model/best'
    main.save_model(path)

    
    
