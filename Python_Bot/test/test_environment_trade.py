import unittest
import pandas as pd
import numpy as np

class TestNormalizerTrade(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        '''Init Class'''
        # Initialize a fixed Experience to use in parameter for environment
        trades = ['BTC', 'ETH', 'BTH', 'ETC']

        SIZE = 1000
        OFFSET = 10000
        indexes = np.arange(OFFSET,OFFSET+60*SIZE,60)

        arr = np.zeros(shape=(SIZE, len(trades)))
        for clm in range(np.shape(arr)[1]):
            arr[:,clm] = np.arange(SIZE*clm, SIZE*(clm+1)) + 1 
        self.data = pd.DataFrame(data=arr, index=indexes, columns=trades)

    def test_get_pct_change(self):
        '''Test Pct Change'''
        raw_list = np.array([
            [100, 1000, 200],
            [100, 2000, 1000],
            [200, 1000, 500],
        ])
        expected_result = np.array([


# run the actual unittests
if __name__ =="__main__":
    unittest.main()
