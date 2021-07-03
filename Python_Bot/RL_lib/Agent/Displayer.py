import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class Displayer(object):

    def __init__(self, max_deque = 10000, delta_display=1, nb_mean=10):
        self.nb_sp=3
        self.nb_mean=nb_mean
        self.delta_display = delta_display

        self.fig, self.ax = plt.subplots(self.nb_sp, 1, sharex=True)
        plt.close(self.fig.number)
        plt.ion()
        self.reinit_fig()

        self.ctr_t = 0
        self.t  = deque(maxlen=max_deque)
        self.epsilons = deque(maxlen=max_deque)
        self.rewards = deque(maxlen=max_deque)
        self.losses = deque(maxlen=max_deque)
        self.min_rew = deque(maxlen=max_deque)
        self.avg_rew = deque(maxlen=max_deque)
        self.max_rew = deque(maxlen=max_deque)

    def reinit_fig(self):
        ### Check if plot still exists
        if not plt.fignum_exists(self.fig.number):
            self.fig, self.ax = plt.subplots(self.nb_sp, 1, sharex=True)
        else:
            for a in self.ax:
                a.clear()

    def update_stats(self,epsilon, reward, loss):
        self.epsilons.append(epsilon)
        self.rewards.append(reward)
        self.ctr_t+=1
        self.t.append(self.ctr_t)
        
        self.rewards.append(reward)
        self.losses.append(loss)

        reward_array = np.array(self.rewards)
        self.min_rew.append(min(reward_array[-self.nb_mean:]))
        self.avg_rew.append(sum(reward_array[-self.nb_mean:])/len(reward_array[-self.nb_mean:]))
        self.max_rew.append(max(reward_array[-self.nb_mean:]))


    def display_historic(self,epsilon, reward, loss):
        
        self.reinit_fig()
        self.update_stats(epsilon, reward, loss)
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
    d = Displayer(max_deque=20)
    for i in range(0,100):
        d.display_historic(i, i, i)
    plt.ioff(), plt.show()