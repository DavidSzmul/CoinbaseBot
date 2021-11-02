import unittest
import numpy as np
import matplotlib.pyplot as plt
from displayer.displayer import Matplotlib_Displayer, MatplotlibPlot
import tkinter as tk
class TestMatplotlib_Displayer(unittest.TestCase):
    
    # @classmethod
    # def setUpClass(self):
    #     '''Init Class'''

    def test_safety_length(self):
        fig, ax = plt.subplots(3, 1, sharex=True)
        master = tk.Tk()
        disp = Matplotlib_Displayer(fig, ax, master, nb_cycle_update=1)
        data_displayer_BAD = [
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
        self.assertRaises(ValueError, disp.update, data_displayer_BAD)

    def test_update(self):
        '''Test MatplotlibDisplayer Class'''

        fig, ax = plt.subplots(2, 1, sharex=True)
        master = tk.Tk()
        disp = Matplotlib_Displayer(fig, ax, master, nb_cycle_update=1)

        data_displayer_1 = [
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
        ]
        data_displayer_2 = [
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
        ]
        disp.update(data_displayer_1)
        disp.update(data_displayer_2)
        print('done')

# run the actual unittests
if __name__ =="__main__":
    unittest.main()
