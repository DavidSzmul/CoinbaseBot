import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from displayer.displayer import Displayer
from rl_lib.agent.manager import Train_perfs

class Displayer_Train_MATPLOTLIB(Displayer):

    ctr_t: int
    t: deque
    epsilons: deque
    rewards: deque
    losses: deque
    min_rew: deque
    avg_rew: deque
    max_rew: deque

    def __init__(self, max_deque = 10000, delta_display=1, nb_mean=10):
        self.nb_sp=3
        self.nb_mean=nb_mean
        self.delta_display = delta_display
        self.max_deque = max_deque

        self.fig, self.ax = plt.subplots(self.nb_sp, 1, sharex=True)
        plt.close(self.fig.number)
        plt.ion()
        self.reset() 

    def reset(self):
        '''Reinitialization of internal variables'''
        self.ctr_t = 0
        self.t  = deque(maxlen=self.max_deque)
        self.epsilons = deque(maxlen=self.max_deque)
        self.rewards = deque(maxlen=self.max_deque)
        self.losses = deque(maxlen=self.max_deque)
        self.min_rew = deque(maxlen=self.max_deque)
        self.avg_rew = deque(maxlen=self.max_deque)
        self.max_rew = deque(maxlen=self.max_deque)
        self.reinit_fig()

    def reinit_fig(self):
        ### Check if plot still exists
        if not plt.fignum_exists(self.fig.number):
            self.fig, self.ax = plt.subplots(self.nb_sp, 1, sharex=True)
        else:
            for a in self.ax:
                a.clear()

    def update_stats(self, train_perfs: Train_perfs):
        self.epsilons.append(train_perfs.epsilon)
        self.rewards.append(train_perfs.total_reward)
        self.losses.append(train_perfs.total_loss)
        self.ctr_t+=1
        self.t.append(self.ctr_t)

        reward_array = np.array(self.rewards)
        self.min_rew.append(min(reward_array[-self.nb_mean:]))
        self.avg_rew.append(sum(reward_array[-self.nb_mean:])/len(reward_array[-self.nb_mean:]))
        self.max_rew.append(max(reward_array[-self.nb_mean:]))


    def update(self,train_perfs: Train_perfs):
        
        self.reinit_fig()
        self.update_stats(train_perfs)
        if self.ctr_t % self.delta_display:
            return

        ### Signals Displayed
        Signals_p0 = [
            {'name': 'Max Reward',
            'x': self.t, 'y': self.max_rew,
            'linewidth':1},
            {'name': 'Mean Reward',
            'x': self.t, 'y': self.avg_rew,
            'linewidth':1},
            {'name': 'Min Reward', 
            'x': self.t, 'y': self.min_rew,
            'linewidth':1},
            ]
        Signals_p1 ={
            'name': 'Exploration', 
            'x': self.t, 'y': self.epsilons,
            'linewidth':1}

        Signals_p2 ={
            'name': 'Loss', 
            'x': self.t, 'y': self.losses,
            'linewidth':1}
            
        ### Display
        for signal in Signals_p0:
            self.ax[0].plot(signal['x'], signal['y'], 
                    linewidth=signal['linewidth'],
                    label=signal['name'])

        self.ax[1].plot(Signals_p1['x'], Signals_p1['y'], 
                    linewidth=Signals_p1['linewidth'],
                    label=Signals_p1['name'])

        self.ax[2].plot(Signals_p2['x'], Signals_p2['y'], 
                    linewidth=Signals_p2['linewidth'],
                    label=Signals_p2['name'])

        # Enable legend
        for a in self.ax:
            a.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

if __name__=='__main__':
    d = Displayer_Train_MATPLOTLIB(max_deque=20)
    for i in range(0,100):
        if i==20:
            d.reset()
        d.update(Train_perfs(i,i,i))
    plt.ioff(), plt.show()