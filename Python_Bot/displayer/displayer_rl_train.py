import matplotlib.pyplot as plt
import tkinter as tk

from rl_lib.manager.manager import RL_Train_Perfs, RL_Train_Perfs_Historic
from displayer.displayer import Matplotlib_Displayer, MatplotlibPlot

class Displayer_RL_Train(Matplotlib_Displayer):
    '''Dipslayer specific to RL training'''
    
    def __init__(self, master: tk.Tk, nb_cycle_update: int=1, nb_window: int=10):
        '''Initialization'''
        fig, ax = plt.subplots(3, 1, sharex=True) 
        plt.close() # This enables to close correctly mainloop when app is destroyed

        Matplotlib_Displayer.__init__(self, fig, ax, master, nb_cycle_update=nb_cycle_update)
        self.train_perfs = RL_Train_Perfs_Historic(nb_window=10)

    def update(self, train_perf: RL_Train_Perfs):
        self.train_perfs.add(train_perf)
        self.update_display(self.train_perfs)

    def update_display(self, train_perfs: RL_Train_Perfs_Historic):
        '''Update displayer based on RL perfs'''
        
        data_displayer = [
            # 1rst Subplot
            [
                MatplotlibPlot(train_perfs.t, train_perfs.total_rewards, label='Reward'),
                MatplotlibPlot(train_perfs.t, train_perfs.total_rewards_avg, label='Average'),
                MatplotlibPlot(train_perfs.t, train_perfs.envelopes_plus, 'g--'),
                MatplotlibPlot(train_perfs.t, train_perfs.envelopes_minus, 'g--')
            ],
            # 2nd Subplot
            [
                MatplotlibPlot(train_perfs.t, train_perfs.total_losses, label='Loss Agent'),
            ],
            # 3rd Subplot
            [
                MatplotlibPlot(train_perfs.t, train_perfs.epsilons, label='Exploration Agent'),
            ],
        ]
        Matplotlib_Displayer.update(self, data_displayer)
        
if __name__=='__main__':
    pass