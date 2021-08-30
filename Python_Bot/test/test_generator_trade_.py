import unittest
import pandas as pd
import numpy as np
from algorithms.lib_trade.generator_trade import Experience_Trade, Generator_Trade

class TestGeneratorTrade(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        '''Init Class'''
        # Initialize a fixed dataframe
        # Generate a fixed linear dataframe for many trades
        trades = ['BTC', 'ETH', 'BTC', 'ETC']
        SIZE = 1000
        OFFSET = 10000
        indexes = np.arange(OFFSET,OFFSET+60*SIZE,60)
        
        arr = np.zeros(shape=(SIZE, len(trades)))
        for clm in range(np.shape(arr)[1]):
            arr[:,clm] = np.arange(SIZE*clm, SIZE*(clm+1)) + 1 

        self.data = pd.DataFrame(data=arr, index=indexes, columns=trades)

    @classmethod
    def tearDownClass(self):
        '''End Class'''
        pass
    
    def test_get_diff_pct_trade(self):
        '''Test Diff Pct'''
        raw_list = np.array([
            [100, 1000, 200],
            [100, 2000, 1000],
            [200, 1000, 500],
        ])
        expected_result = np.array([
            [0, 0, 0],
            [0, 1, 4],
            [1, -0.5, -0.5],
        ])
        gen = Generator_Trade(self.data, verbose=False)
        
        output_list = gen._get_diff_pct_trade(raw_list)
        test_valid = np.all(output_list==expected_result)
        self.assertTrue(test_valid)


    def test_normalization(self):
        '''Test Normalization'''
        # raise ValueError('tmp')
        gen = Generator_Trade(self.data, verbose=False)
        data_raw = self.data.to_numpy()
        data_norm = gen._normalize(self.data)

        test_valid = np.all(data_raw.flatten() > data_norm.flatten())
        self.assertTrue(test_valid)

    

    def test_get_index_window_safety(self):
        '''Test Index Window Safety'''
        nb_iteration = [4, 4, 4]
        nb_min = [1, 10, 100, 1000]
        gen = Generator_Trade(self.data, verbose=False)

        ### For different size
        self.assertRaises(ValueError, gen._get_idx_window_historic, nb_iteration, nb_min)
        ### For empty size
        self.assertIsNone(gen._get_idx_window_historic(None, None))

    def test_index_window_coherence(self):
        '''Test Index Window Coherence'''
        nb_iteration = [4, 4, 4]
        nb_min = [1, 10, 100]
        valid_answer = [-443, -343, -243, -143, 
                        -43, -33, -23, -13, 
                        -3, -2 ,-1, 0]

        gen = Generator_Trade(self.data, verbose=False)
        idx_window = gen._get_idx_window_historic(nb_iteration, nb_min)

        test_valid = np.all(idx_window==valid_answer)
        self.assertTrue(test_valid)

    def test_synchronous_generator(self):
        
        nb_iteration = [4, 4, 4]
        nb_min = [1, 10, 100]
        duration_future = 20
        gen = Generator_Trade(self.data, verbose=False)
        sync_exp = gen._get_synchronous_experiences(self.data, nb_iteration, nb_min, duration_future,
                                                    )

        is_evolution_none_exp = [s.evolution is None for s in sync_exp]
        test_valid = np.all(is_evolution_none_exp)
        self.assertTrue(test_valid)

# run the actual unittests
if __name__ =="__main__":
    unittest.main()
