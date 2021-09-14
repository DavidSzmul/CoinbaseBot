import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md
from datetime import datetime
from algorithms.evolution.evolution_trade import Evolution_Trade
from database import Historic_coinbase_dtb

from dataclasses import dataclass

@dataclass
class Displayer_Evolution:
    '''Class enabling to challenge an evolution method by displaying the results'''

    evolution_method: Evolution_Trade
    trade_display: str

    def load_data(self) -> pd.DataFrame:
        df = Historic_coinbase_dtb.load()
        return df[self.trade_display] 

    def plot(self):

        input_trade = self.load_data()
        trade_adapt = input_trade.to_numpy()
        trade_adapt = trade_adapt.reshape((trade_adapt.shape[0], -1))
        evolution = self.evolution_method.get_evolution(trade_adapt)

        plt.ioff()
        fig, ax = plt.subplots(2,1,sharex=True)
        dates=[datetime.fromtimestamp(ts) for ts in input_trade.index]
        ax[0].plot(dates, input_trade.to_numpy())
        ax[1].plot(dates, evolution)
        
        plt.xticks(rotation=25)
        xfmt = md.DateFormatter('%Y-%m-%d %Hh')
        ax[1].xaxis.set_major_formatter(xfmt)
        plt.show()

if __name__ =="__main__":
    from algorithms.evolution.evolution_trade import Evolution_Trade_Median

    evolution_method = Evolution_Trade_Median(start_check_future=-15, end_check_future=1040)
    trade_display = 'BTC-USD'

    disp = Displayer_Evolution(evolution_method, trade_display)
    disp.plot()