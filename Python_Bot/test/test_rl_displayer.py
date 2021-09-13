import unittest
import tkinter as tk
from displayer.displayer_rl_train import Displayer_RL_Train
from rl_lib.manager.manager import RL_Train_Perfs_Historic, RL_Train_Perfs

class TestMatplotlib_Displayer(unittest.TestCase):
    
    # @classmethod
    # def setUpClass(self):
    #     '''Init Class'''

    def test_update(self):
        '''Test MatplotlibDisplayer Class'''
        root = tk.Tk()
        disp = Displayer_RL_Train(root, nb_cycle_update=1, title='Test')
        # historic = RL_Train_Perfs_Historic(max_deque=100, nb_window=2)

        data = [
            [1.1,1,1],
            [2.1,4,1],
            [3.1,0,1],
            [4.1,2,1],
            [4.1,2,0.9]
        ]
        for d in data:
            disp.update(RL_Train_Perfs(*d))

# run the actual unittests
if __name__ =="__main__":
    unittest.main()
