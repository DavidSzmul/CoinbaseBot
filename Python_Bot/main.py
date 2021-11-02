from algorithms.rl_algo_one_trade import RL_Bot_App
from algorithms.ihm_trade.ihm_trade import TkView_MVC_Trade, Controller_MVC_Trade

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

    app = RL_Bot_App() # Corresponds to the application to execute
    # model = Easy_RL_App() # Corresponds to the model associated to the IHM
    view = TkView_MVC_Trade() # WIDGETS
    c = Controller_MVC_Trade(app, view, list_Historic_Environement)
    c.setup()
    c.start()