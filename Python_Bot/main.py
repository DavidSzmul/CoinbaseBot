from algorithms.rl_algo_one_trade import AlgoOneTrade_Manager_Factory, Displayer_RL_Train
from algorithms.ihm_trade.ihm_trade import Model_MVC_Trade, TkView_MVC_Trade, Controller_MVC_Trade

if __name__=="__main__":
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

    app = AlgoOneTrade_Manager_Factory() # Corresponds to the application to execute
    model = Model_MVC_Trade(runner) # Corresponds to the model associated to the IHM
    view = TkView_MVC_Trade() # WIDGETS
    c = Controller_MVC_Trade(model, view, list_Historic_Environement)
    c.start()