
from collections import deque
from algorithms.lib_trade.portfolio import Portfolio
import matplotlib.pyplot as plt
import tkinter as tk

from displayer.displayer import Matplotlib_Displayer, MatplotlibPlot

class Portfolio_Perfs_Historic:
    '''Dataclass Constaining all info about Reinforcement Learning training'''
    max_deque: int          # Max size of dequeue
    nb_window: int          # Window size for min/max/avg criteria

    t: deque
    total_values: deque         # Total value of portfolio
    trades_prc: deque           # Percentages of current time
    anotations_trades: deque    # Name of changed trades


    def __init__(self, max_deque: int=1e6):
        self.max_deque = int(max_deque)
        self.reset()

    def reset(self):
        self.t = deque(maxlen=self.max_deque)
        self.total_pfl = deque(maxlen=self.max_deque)
        self.trades_prc = deque(maxlen=self.max_deque)
        self.anotations_trades = deque(maxlen=self.max_deque)
        self.chosen_trade = None


    def add(self, portfolio: Portfolio):
        if self.t:# If not empty
            self.t.append(self.t[-1]+1)
        else: # If empty
            self.t.append(0)
            
        previous_trade = self.chosen_trade
        self.chosen_trade = portfolio.get_highest_account()
        self.total_pfl.append(portfolio.get_total_value())

        if (previous_trade != self.chosen_trade):
            self.anotations_trades.append(self.chosen_trade)
            self.last_value = portfolio[self.chosen_trade].value
        
        prc = (portfolio[self.chosen_trade].value / self.last_value) - 1
        self.trades_prc.append(prc)


class Displayer_Perfs_Trade(Matplotlib_Displayer):
    '''Class Displaying performances of bot (test or real time) based on Portfolio accounts'''
    
    def __init__(self, master: tk.Tk, nb_cycle_update: int=1, title: str=None):
        '''Initialization'''
        fig, ax = plt.subplots(3, 1, sharex=True) 
        plt.close() # This enables to close correctly mainloop when app is destroyed

        Matplotlib_Displayer.__init__(self, fig, ax, master, nb_cycle_update=nb_cycle_update, title=title)
        self.perfs = Portfolio_Perfs_Historic()

    def update(self, portfolio: Portfolio):
        self.perfs.add(portfolio)
        print(f'Current Total: {self.perfs.total_pfl[-1]}$')
        # self.update_display(self.perfs)
    
    def update_display(self, perfs: Portfolio_Perfs_Historic):
        '''Update displayer based on RL perfs'''
        
        data_displayer = [
            # 1rst Subplot
            [
                MatplotlibPlot(perfs.t, perfs.trades_prc, label='Percentage Benef'),
            ],
            # 2nd Subplot
            [
                MatplotlibPlot(perfs.t, perfs.total_pfl, label='Total Portfolio'),
            ],
        ]
        Matplotlib_Displayer.update(self, data_displayer)
        for idx, annot in enumerate(perfs.anotations_trades):
            if annot is not None:
                self.ax[1].annotate(annot, (perfs.t[idx], 0))
        
    def display(self):
        data_displayer = [
            # 1rst Subplot
            [
                MatplotlibPlot(self.perfs.t, self.perfs.trades_prc, label='Percentage Benef'),
            ],
            # 2nd Subplot
            [
                MatplotlibPlot(self.perfs.t, self.perfs.total_pfl, label='Total Portfolio'),
            ],
        ]
        Matplotlib_Displayer.update(self, data_displayer)
        for idx, annot in enumerate(self.perfs.anotations_trades):
            if annot is not None:
                self.ax[1].annotate(annot, (self.perfs.t[idx], 0))
        self.fig.canvas.draw()

        
if __name__=='__main__':
    pass



