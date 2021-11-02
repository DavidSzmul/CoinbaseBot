### Algorithm of choice of crypto using DQN Reinforcement Learning Algo
### Model can be continuously improved by generating augmented database
from algorithms.ihm_trade.displayer_perfs_trade import Displayer_Perfs_Trade
from threading import current_thread
import time
from typing import Callable, List

from rl_lib.agent.dqn import DQN_Agent, DQN_parameters
from rl_lib.manager.manager import Agent_Environment_Manager, RL_Train_Perfs

from algorithms.lib_trade.processing_trade import Scaler_Trade, Generator_Trade, Mode_Algo
from algorithms.lib_trade.environment_trade import Environment_Compare_Trading, Info_Trading
from algorithms.lib_trade.portfolio import Account, Portfolio
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

    ### CALLBACKS
    callback_train: Callable[[RL_Train_Perfs], None]=None
    callback_test: Callable[[Portfolio], None]=None
    callback_test_end: Callable[[], None]=None

    def __init__(self):
        ### Params changable by the user (to create new model)
        self.LAYERS_MODEL = [128,64]

    def set_train_callback(self, callback_loop: Callable[[RL_Train_Perfs], None]):
        self.callback_train = callback_loop

    def set_test_callback(self, callback_loop: Callable[[Portfolio], None], callback_end: Callable[[], None]):
        self.callback_test = callback_loop
        self.callback_test_end = callback_end
        
        
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
        # Evolution method: Used to define the reward of the bot
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
        trade_name_INIT = 'USDC-USD'
        self.generator.set_mode(Mode_Algo.train)
        self.manager_RL.env.set_new_data(current_trade=trade_name_INIT) 
        self.manager_RL.env.idx_current_trade = 0 # TODO: DEBUG

        # Loop
        while (self.is_running):
            
            #Generate new experience on environment
            self.manager_RL.env.set_new_data(self.generator.get_new_experience()) 
            # Start new experience/episode
            _, perfs = self.manager_RL.loop_episode_train()

            # Callback if enabled
            if self.callback_train:
                self.callback_train(perfs)
            # Print advancement
            self.generator.print_advancement()

            # End Loop
            if self.generator.is_generator_finished():
                self.is_running=False
                break

    @Abstract_RL_App._thread_run
    def test(self):
        # Set generator to correct mode
        trade_name_INIT = 'USDC-USD'
        self.generator.set_mode(Mode_Algo.test)
        portfolio = Portfolio()
        portfolio.set_account(Account(trade_name_INIT, 1))
        portfolio.add_money(trade_name_INIT, 1000) # Add 1000$ to account
        self.manager_RL.env.set_new_data(current_trade=trade_name_INIT) 
        
        # Loop
        while (self.is_running):
            
            #Generate new experience on environment
            self.manager_RL.env.set_new_data(exp=self.generator.get_new_experience()) 
            # Start new experience/episode
            info: Info_Trading = self.manager_RL.loop_episode()

            # Update portfolio based on Bot decision
            last_prices = info.last_values.to_dict('records')[0]
            portfolio.update(last_prices)
            if info.flag_change_trade:
                portfolio.convert_money(info.from_, info.to_, -1, prc_taxes=self.PRC_TAXES)

            # Callback if enabled
            if self.callback_test:
                self.callback_test(portfolio)

            # Print advancement
            # self.generator.print_advancement()

            # End Loop
            if self.generator.is_generator_finished():
                self.is_running=False
                break
        # End while
        if self.callback_test_end: # Usually used to display results
            self.callback_test_end()

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
         {'name': 'TMP: step x5 every 50 cycles on 1 days', # To enable quick tests
        'subfolder': 'x5_50cycles_1d',
        'min_historic': [1, 5, 25],
        'nb_cycle_historic': [50, 50, 50],
        },
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
    bot = RL_Bot_App() # Corresponds to the model to execute
    c = Controller_MVC_Trade(bot, view, list_Historic_Environement)
    c.setup()

    # Setup Displayer Train 
    disp_train = Displayer_RL_Train(view.get_root(), nb_cycle_update=1)
    bot.set_train_callback(disp_train.update)

    # Setup Displayer Test
    disp_test = Displayer_Perfs_Trade(view.get_root())
    bot.set_test_callback(disp_test.update, disp_test.display)

    # Start App
    c.start()
