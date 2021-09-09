from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List
import matplotlib.pyplot as plt

class Displayer(ABC):
    '''Abstract Class used to display informations from another class.
    This display can be from different forms:
        -   Using Matlplotlib figures
        -   Using an associated server communicating to an associated Front-End'''

    nb_cycle_update: int
    ctr_cycle_update: int

    def __init__(self, nb_cycle_update: int=1):
        self.nb_cycle_update = nb_cycle_update
        self.ctr_cycle_update = 0
    
    @abstractmethod
    def update(self):
        '''Update of displayed informations'''

@dataclass
class MatplotlibPlot:
    x: Any
    y: Any
    label: str=None
    linewidth: int=1

class Matplotlib_Displayer(Displayer):

        def __init__(self, fig, ax, nb_cycle_update: int=1, init_show: bool=True):
            Displayer.__init__(self, nb_cycle_update=nb_cycle_update)
            self.fig = fig
            self.ax = ax
            plt.ion()
            if init_show:
                plt.show()
            self.reinit_fig() 

        def reinit_fig(self):
            '''Reinitialization of display'''
            for a in self.ax:
                a.clear()

        def update(self, data: List[List[MatplotlibPlot]]):
            
            # Verification that data is coherent
            if len(data) != len(self.ax):
                raise ValueError('data should has the same size as the number of axes')
            # Check confirmation of update
            self.ctr_cycle_update=(self.ctr_cycle_update+1)%self.nb_cycle_update
            if self.ctr_cycle_update!=0:
                return     

            # Update display
            self.reinit_fig()

            #Display only plot for the moment
            for i,d in enumerate(data):
                for signal in d:
                    self.ax[i].plot(signal.x, signal.y, 
                        linewidth=signal.linewidth,
                        label=signal.label)

            # Enable legend
            for a in self.ax:
                a.legend()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

if __name__=='__main__':
    fig, ax = plt.subplots(2, 1, sharex=True)
    disp = Matplotlib_Displayer(fig, ax, nb_cycle_update=1)

    data_displayer_1 = [
        # 1rst Subplot
        [
            MatplotlibPlot([1,2,3,4], [0,1,2,3], 'test 1.1'),
            MatplotlibPlot([2,3,4,5], [0,1,2,3], 'test 1.2'),
            MatplotlibPlot([1,2,3,4], [1,2,3,4], 'test 1.3')
        ],
        # 2nd Subplot
        [
            MatplotlibPlot([1,10], [-1,1], 'test 2'),
        ],
    ]
    data_displayer_2 = [
        # 1rst Subplot
        [
            MatplotlibPlot([1,2,3,4], [0,-1,-2,-3], 'test 1.1'),
            MatplotlibPlot([2,3,4,5], [0,-1,-2,-3], 'test 1.2'),
            MatplotlibPlot([1,2,3,4], [-1,-2,-3,-4], 'test 1.3')
        ],
        # 2nd Subplot
        [
            MatplotlibPlot([1,10], [-1,1], 'test 2'),
        ],
    ]
    disp.update(data_displayer_1)
    disp.update(data_displayer_2)

    print('done')
