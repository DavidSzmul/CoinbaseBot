import unittest
from displayer.displayer_rl_train import Displayer_RL_Train
from rl_lib.manager.manager import RL_Train_Perfs_Historic, RL_Train_Perfs

class TestMatplotlib_Displayer(unittest.TestCase):
    
    # @classmethod
    # def setUpClass(self):
    #     '''Init Class'''

    def test_update(self):
        '''Test MatplotlibDisplayer Class'''

        disp = Displayer_RL_Train(nb_cycle_update=1, init_show=False)
        historic = RL_Train_Perfs_Historic(max_deque=100, nb_window=2)

        data = [
            [1.1,1,1],
            [2.1,4,1],
            [3.1,0,1],
            [4.1,2,1],
            [4.1,2,0.9]
        ]
        for d in data:
            historic.add(RL_Train_Perfs(*d))
            disp.update(historic)

# run the actual unittests
if __name__ =="__main__":
    unittest.main()
