### Algorithm of choice of crypto using DQN Reinforcement Learning Algo
### Model can be continuously improved by generating augmented database
import time
from typing import List

from rl_lib.agent.dqn import DQN_Agent, DQN_parameters
from rl_lib.manager.manager import Agent_Environment_Manager

from algorithms.lib_trade.processing_trade import Scaler_Trade, Generator_Trade, Mode_Algo
from algorithms.lib_trade.environment_trade import Environment_Compare_Trading
from algorithms.lib_trade.portfolio import Portfolio
from algorithms.ihm_trade.ihm_trade import Abstract_RL_App
from algorithms.evolution.evolution_trade import Evolution_Trade_Median



from database import Historic_coinbase_dtb
from database.historic_generator import Historic_Coinbase_Generator

from displayer.displayer_rl_train import Displayer_RL_Train


class RL_Bot_App(Abstract_RL_App):
    '''Class communicating with IHM to train/test/rt bot'''

    PRC_TAXES: float=0.02
    path_save_model: str
    scaler: Scaler_Trade
    generator: Generator_Trade
    displayer_train: Displayer_RL_Train = None
    def __init__(self):
        ### Params changable by the user (to create new model)
        self.LAYERS_MODEL = [128,64]

    def set_displayer_train(self, displayer_train: Displayer_RL_Train):
        self.displayer_train = displayer_train
        
    def define_params(self, min_historic: List[int],nb_cycle_historic: List[int], path_agent:str):
        '''Definition of all objects required for bot'''

        # Scaler for preprocessing
        self.scaler = Scaler_Trade(min_historic, nb_cycle_historic)
        # Generator for generating experiences for environment
        self.generator = Generator_Trade(self.scaler)
        print('Preprocessing Objects Built')

        # Definition of Environment
        env = Environment_Compare_Trading(self.generator.get_size_historic(), self.PRC_TAXES, is_order_random=True)
        # Definition of Agent
        agent = DQN_Agent(env.get_state_shape(), env.get_action_shape(), 
                        params=DQN_parameters(gamma=0.99,
                                    # TO continue if necesary
                                ), #Definition of all parameters intresect to DQN (independant to model NN)
                        name_model=path_agent, # Load existing model if not None
                        layers_model=self.LAYERS_MODEL
                        )
        # Definition of Manager for RL
        self.manager_RL = Agent_Environment_Manager(agent, env, flag_return_train_perfs=True)
        print('Agent+Environment Built')

        # Setup Generator
        self.reset_generator()
        # self.update_train_test_dtb()

    def reset_generator(self):
        # Evolution method 
        EVOLUTION_METHOD = Evolution_Trade_Median(start_check_future=-15, end_check_future=1040)
        RATIO_UNSYNCHRONOUS = 0.66
        RATIO_TRAIN_TEST = 0.8

        # Get data
        fresh_data = Historic_coinbase_dtb.load()
        self.scaler.fit(fresh_data)
        # Preprocess
        self.generator.generate_train_test_database(fresh_data,EVOLUTION_METHOD,
                    ratio_unsynchrnous_time=RATIO_UNSYNCHRONOUS,
                    ratio_train_test=RATIO_TRAIN_TEST)
    

    def update_train_test_dtb(self):
        # PARAMETERS
        MAX_SIZE_DTB = 1e5  

        # Generate more fresh values
        hist_gen = Historic_Coinbase_Generator()
        hist_gen.update_dtb(maxSizeDtb=MAX_SIZE_DTB, verbose=True)

        # Setup Generator
        self.reset_generator()
        print('Ready to use BOT')

    
    @Abstract_RL_App._thread_run
    def train(self):
        # Set generator to correct mode
        self.generator.set_mode(Mode_Algo.train)

        # Initialize displayer
        if self.displayer_train:
            self.displayer_train.setup_new_window('Train', 'Training in execution')

        # Loop
        self.manager_RL.env.idx_current_trade = 0
        while (self.is_running):
            if self.generator.is_generator_finished():
                self.is_running=False
                return

            #Generate new experience on environment
            self.manager_RL.env.set_new_data(self.generator.get_new_experience()) 
            # Start new experience/episode
            _, perfs = self.manager_RL.loop_episode_train()

            # Display performances if enabled
            if self.displayer_train:
                self.displayer_train.update(perfs)
            # Print advancement
            self.generator.print_advancement()

    @Abstract_RL_App._thread_run
    def test(self):
        ctr=0
        while self.is_running:
            ctr+=1
            print('Test:', ctr)
            time.sleep(0.5)

    @Abstract_RL_App._thread_run
    def real_time(self):
        ctr=0
        while self.is_running:
            ctr+=1
            print('Real-Time:', ctr)
            time.sleep(0.1)

    def save_model(self, path: str):
        self.manager_RL.agent.save_model(path, overwrite=True)
        print('Agent Model Saved\n', path)

    # def run_testing(self, status_loop: StatusLoop):
    #     # Create a new portfolio based on generator
    #     portfolio = Portfolio()
    #     previous_idx_current_trade = 0
    #     self.env.idx_current_trade = previous_idx_current_trade
    #     init_money = True
    #     while (status_loop.is_running and not self.generator.is_generator_finished()):

    #         #Update environment and porfolio
    #         self.env.set_new_episode(self.generator.get_new_experience()) 
    #         last_prices = self.env.current_exp.current_trades
    #         trades_names = last_prices.columns
    #         idx_current_trade = self.env.get_current_trade()
    #         portfolio.update_last_prices(last_prices)

    #         # Add artificially money for the initialization
    #         if init_money:
    #             portfolio.add_money(to_=trades_names[idx_current_trade], value=1000)
    #             init_money=False

    #         # Start new experience/episode
    #         info = self.loop_episode()
            
    #         # Realize conversion depending on agent decision
    #         if idx_current_trade is not previous_idx_current_trade:
                
    #             trade_from = trades_names[previous_idx_current_trade] 
    #             trade_to = trades_names[idx_current_trade]
    #             value_convert = portfolio.get_value(trade_from)
    #             portfolio.convert_money(trade_from, trade_to, value_convert, prc_taxes=self.PRC_TAXES)

    #         # Display performances if enabled
    #         if self.displayer_test:
    #             pass
    #             # self.displayer_train.update(perfs)
    


if __name__ == '__main__':
    from algorithms.ihm_trade.ihm_trade import TkView_MVC_Trade, Controller_MVC_Trade

    # create the MVC & start the application
    list_Historic_Environement = [
        {'name': 'step x5 every 50 cycles on 5 days',
        'subfolder': 'x5_50cycles_5d',
        'min_historic': [1, 5, 25, 125],
        'nb_cycle_historic': [50, 50, 50, 50],
        },
        {'name': 'step x2 every 15 cycles on 1 days',
        'subfolder': 'x2_15cycles_5d',
        'min_historic': [1, 2, 4, 8],
        'nb_cycle_historic': [15, 15, 15, 15],
        }
    ]

    # Setup Main IHM
    view = TkView_MVC_Trade() # WIDGETS
    model = RL_Bot_App() # Corresponds to the application to execute
    c = Controller_MVC_Trade(model, view, list_Historic_Environement)
    c.setup()

    # Setup Displayer Train
    disp = Displayer_RL_Train(view.get_root(), nb_cycle_update=1)
    model.set_displayer_train(disp)

    # Start App
    c.start()
