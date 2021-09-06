import matplotlib.pyplot as plt

from rl_lib.manager.manager import RL_Train_Perfs, RL_Train_Perfs_Historic
from displayer.displayer import Matplotlib_Displayer, MatplotlibPlot

class Displayer_RL_Train(Matplotlib_Displayer):
    '''Dipslayer specific to RL training'''
    
    def __init__(self, nb_cycle_update: int=1, init_show: bool=True):
        '''Initialization'''
        fig, ax = plt.subplots(3, 1, sharex=True)
        Matplotlib_Displayer.__init__(self, fig, ax, nb_cycle_update=nb_cycle_update, init_show=init_show)
        self.train_perfs = RL_Train_Perfs_Historic(nb_window=10)

    def update(self, train_perf: RL_Train_Perfs):
        self.train_perfs.add(train_perf)
        self.update_display(self.train_perfs)

    def update_display(self, train_perfs: RL_Train_Perfs_Historic):
        '''Update displayer based on RL perfs'''
        data_displayer = [
            # 1rst Subplot
            [
                MatplotlibPlot(train_perfs.t, train_perfs.total_rewards, 'Reward'),
                MatplotlibPlot(train_perfs.t, train_perfs.total_rewards_avg, 'Avg Reward'),
                MatplotlibPlot(train_perfs.t, train_perfs.total_rewards_std, 'Std Reward')
            ],
            # 2nd Subplot
            [
                MatplotlibPlot(train_perfs.t, train_perfs.total_losses, 'Loss Agent'),
            ],
            # 3rd Subplot
            [
                MatplotlibPlot(train_perfs.t, train_perfs.epsilons, 'Exploration Agent'),
            ],
        ]
        Matplotlib_Displayer.update(self, data_displayer)
        
if __name__=='__main__':
    pass