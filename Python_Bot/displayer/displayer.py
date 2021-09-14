from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List

import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.backends.backend_tkagg as  backend_tkagg
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
    fmt: Any=None
    label: str=None
    linewidth: int=1

class Matplotlib_Displayer(Displayer):

        new_window: Any=None

        def __init__(self, fig, ax, master: tk.Tk, use_toolbar: bool=True, nb_cycle_update: int=1, title: str=None):
            Displayer.__init__(self, nb_cycle_update=nb_cycle_update)
            self.fig = fig
            self.ax = ax
            self.master = master
            self.title = title
            self.canvas = None
            self.use_toolbar = use_toolbar
            self.reinit_fig() 

        def reinit_fig(self):
            '''Reinitialization of display'''
            for a in self.ax:
                a.clear()

        def exist_window(self):
            if self.new_window is None:
                return False
            return (self.new_window.winfo_exists() == 1)

        def setup_new_window(self):
            self.new_window = tk.Toplevel(self.master)
            if self.title:
                self.new_window.title(self.title)
            self.new_window.geometry("400x400")
        
            # # A Label widget to show in toplevel
            # tk.Label(self.new_window,
            #     text=text).pack()
            self.canvas = backend_tkagg.FigureCanvasTkAgg(self.fig, self.new_window)
            self.canvas.get_tk_widget().pack(side=tk.BOTTOM, expand=True) #, fill=tk.BOTH

            if self.use_toolbar:
                toolbar = backend_tkagg.NavigationToolbar2Tk(self.canvas, self.new_window)
                toolbar.update()
            self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            

        def update(self, data: List[List[MatplotlibPlot]]):
            
            # Verification that data is coherent
            if len(data) != len(self.ax):
                raise ValueError('data should has the same size as the number of axes')

            # Verify if window exists
            if not self.exist_window():
                self.setup_new_window()
            # Check confirmation of update
            self.ctr_cycle_update=(self.ctr_cycle_update+1)%self.nb_cycle_update
            if self.ctr_cycle_update!=0:
                return     

            # Update display
            self.reinit_fig()

            #Display only plot for the moment
            for i,d in enumerate(data):
                for signal in d:
                    if signal.fmt:
                        self.ax[i].plot(signal.x, signal.y, signal.fmt,
                            linewidth=signal.linewidth,
                            label=signal.label)

                    else:
                        self.ax[i].plot(signal.x, signal.y, 
                            linewidth=signal.linewidth,
                            label=signal.label)

            # Enable legend
            for a in self.ax:
                a.legend()
            self.fig.canvas.draw()


if __name__=='__main__':

    data_displayer= [
        [
            # 1rst Subplot
            [
                MatplotlibPlot([1,2,3,4], [0,1,2,3], label='test 1.1'),
                MatplotlibPlot([2,3,4,5], [0,1,2,3], label='test 1.2'),
                MatplotlibPlot([1,2,3,4], [1,2,3,4], label='test 1.3')
            ],
            # 2nd Subplot
            [
                MatplotlibPlot([1,10], [-1,1], label='test 2'),
            ],
        ],
        [
            # 1rst Subplot
            [
                MatplotlibPlot([1,2,3,4], [0,-1,-2,-3], label='test 1.1'),
                MatplotlibPlot([2,3,4,5], [0,-1,-2,-3], label='test 1.2'),
                MatplotlibPlot([1,2,3,4], [-1,-2,-3,-4], label='test 1.3')
            ],
            # 2nd Subplot
            [
                MatplotlibPlot([1,10], [-1,1], label='test 2'),
            ],
        ],
        [
            # 1rst Subplot
            [
                MatplotlibPlot([1,2,3,4], [0,1,2,3], label='test 1.1'),
                MatplotlibPlot([2,3,4,5], [0,1,2,3], label='test 1.2'),
                MatplotlibPlot([1,2,3,4], [1,2,3,4], label='test 1.3')
            ],
            # 2nd Subplot
            [
                MatplotlibPlot([1,10], [-1,1], label='test 2'),
            ],
        ],
        [
            # 1rst Subplot
            [
                MatplotlibPlot([1,2,3,4], [0,-1,-2,-3], label='test 1.1'),
                MatplotlibPlot([2,3,4,5], [0,-1,-2,-3], label='test 1.2'),
                MatplotlibPlot([1,2,3,4], [-1,-2,-3,-4], label='test 1.3')
            ],
            # 2nd Subplot
            [
                MatplotlibPlot([1,10], [-1,1], 'r', label='test 2'),
            ],
        ],
    ]
    
    # Initialisation
    fig, ax = plt.subplots(2, 1, sharex=True)
    plt.close() # This enables to close correctly mainloop when app is destroyed
    root = tk.Tk()
    disp = Matplotlib_Displayer(fig, ax, root, nb_cycle_update=1)
    disp.setup_new_window()

    DELTA_TIME=1000
    for i, d in enumerate(data_displayer):
        root.after((i+1)*DELTA_TIME, disp.update, d)
    root.mainloop()
    print('done')
