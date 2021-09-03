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

class Matplotlib_Displayer(Displayer):

        def __init__(self, fig, ax, nb_cycle_update: int=1):
            super.__init__(nb_cycle_update=nb_cycle_update)
            self.fig = fig
            self.ax = ax
            plt.close(self.fig.number)
            plt.ion()
            self.reinit_fig() 

        def reinit_fig(self):
            '''Reinitialization of display'''
            for a in self.ax:
                a.clear()

        def update(self, data: List[Any]):
            
            # Check confirmation of update
            self.ctr_cycle_update=(self.ctr_cycle_update+1)%self.nb_cycle_update
            if self.ctr_cycle_update==0:
                return            
            # Update display
            self.reinit_fig()
            #Display...
