import unittest
import pandas as pd
import numpy as np
from algorithms.lib_trade.processing_trade import *

class TestScalerTrade(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        '''Init Class'''
        # Initialize a fixed dataframe
        # Generate a fixed linear dataframe for many trades
        trades = ['BTC', 'ETH', 'BTH', 'ETC']
        SIZE = 1000
        OFFSET = 10000
        indexes = np.arange(OFFSET,OFFSET+60*SIZE,60)

        arr = np.zeros(shape=(SIZE, len(trades)))
        for clm in range(np.shape(arr)[1]):
            arr[:,clm] = np.arange(SIZE*clm, SIZE*(clm+1)) + 1 
        self.data = pd.DataFrame(data=arr, index=indexes, columns=trades)

    def test_get_index_window_safety(self):
        '''Test Index Window Safety'''

        ### For different size
        nb_min = [1, 10, 1000]
        nb_iteration = [4, 4]
        scaler = Scaler_Trade(nb_min, nb_iteration)
        self.assertRaises(ValueError, scaler.get_idx_window_historic)

        ### For empty size
        scaler = Scaler_Trade(None, None)
        self.assertIsNone(scaler.get_idx_window_historic())

    def test_get_index_window_coherence(self):
        '''Test Index Window Coherence'''
        nb_min = [1, 10, 100]
        nb_iteration = [4, 4, 4]
        valid_answer = [-443, -343, -243, -143, 
                        -43, -33, -23, -13, 
                        -3, -2 ,-1, 0]

        # Scaler definition
        scaler = Scaler_Trade(nb_min, nb_iteration)
        idx_window = scaler.get_idx_window_historic()

        test_valid = np.all(idx_window==valid_answer)
        self.assertTrue(test_valid)

    def test_get_pct_change(self):
        '''Test Pct Change'''
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
        scaler = Scaler_Trade(None, None)
        output_list = scaler.get_pct_change(raw_list)

        test_valid = np.all(output_list==expected_result)
        self.assertTrue(test_valid)

    def test_reset_scaler(self):
        '''Test manual scaler'''
        std_list = [1, 1, 1e1, 1e2, 1e3]

        prc_2_normalize = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7e1, 8e1, 9e1],
            [-1e2, -2e2, -3e2],
            [0, 0, 1e2],
        ])
        expected_output = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [-1, -2, -3],
            [0, 0, 0.1],
        ])
        
        scaler = Scaler_Trade(None, None)
        scaler.reset_scaler(std_list)
        prc_normalized = scaler.transform_prc_change(prc_2_normalize)
        
        test_valid = np.all(prc_normalized==expected_output)
        self.assertTrue(test_valid)


    def test_get_std_list_normalize(self):
        '''Test that list of std based on parameters is correct'''
        nb_min_historic = [1, 2, 3]
        nb_iteration_historic = [2, 4, 2]
        scaler = Scaler_Trade(nb_min_historic, nb_iteration_historic)

        std_input = [1e3, 1, 40]
        expected_output = 2*[40] + 4*[1] + 2*[1e3]
        output = scaler._get_std_list_normalize(std_input)

        test_valid = np.all(output==expected_output)
        self.assertTrue(test_valid)

    def test_fit(self):
        '''Test normalization fitting for state'''
        nb_min_historic = [1, 10, 100]
        nb_iteration_historic = [2, 2, 2]
        scaler = Scaler_Trade(nb_min_historic, nb_iteration_historic)
        scaler.fit(self.data)

        std = scaler.default_scaler.scale_
        cdt_valid = (
            len(std)==6 and
            std[0]==std[1] and std[2]==std[3] and std[4]==std[5] and
            std[0]>std[2] and std[2]>std[4] # Reverse because long term pct change are on start
        )
        # Verification of length of std
        self.assertTrue(cdt_valid)

    def test_transform(self):
        '''Test transform is working correclty'''
        nb_min_historic = [1, 10, 100]
        nb_iteration_historic = [2, 2, 2]
        scaler = Scaler_Trade(nb_min_historic, nb_iteration_historic)
        scaler.fit(self.data)

class TestEvolutionTradeMedian(unittest.TestCase):

    def test_init(self):
        self.assertRaises(ValueError, Evolution_Trade_Median, None, None)
        self.assertRaises(ValueError, Evolution_Trade_Median, 1, None)
        self.assertRaises(ValueError, Evolution_Trade_Median, None, 2)
        self.assertRaises(ValueError, Evolution_Trade_Median, 0, 2)
        self.assertRaises(ValueError, Evolution_Trade_Median, 3, 2)

        obj = Evolution_Trade_Median(2,10)

    def test_get(self):
        start = 2
        end = 5
        obj = Evolution_Trade_Median(start, end)

        trades = np.array([
            [10, 10],
            [10, 10],
            [27, 1],
            [30, 2],
            [25, 1],
            [25, 5],
            [15, 6],
            [17, 10],
        ])
        expected_output = np.array([
            [27, 1],
            [25, 2],
            [25, 5],
            [17, 6],
            [np.NaN, np.NaN],
            [np.NaN, np.NaN],
            [np.NaN, np.NaN],
            [np.NaN, np.NaN],
        ])/trades - 1
        expected_output[4:,:] = 0

        output = obj.get_evolution(trades)

        test_valid = np.all(output==expected_output)
        self.assertTrue(test_valid)


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

    def test_synchronous_generator(self):

        # Create and fit scaler
        nb_min = [1, 10, 100]
        nb_iteration = [4, 4, 4]
        scaler = Scaler_Trade(nb_min, nb_iteration)
        scaler.fit(self.data)

        # Generate Evolution Method
        start_future = 60
        end_future = 120
        evolution_meth = Evolution_Trade_Median(start_future, end_future)

        # Generate synchronous experiences
        gen = Generator_Trade(mode=None, scaler=scaler) 
        sync_exp = gen._get_synchronous_experiences(self.data, evolution_meth)
        print('Syncrhonous Generated')

    def test_unsynchronous_generator(self):

        # Create and fit scaler
        nb_min = [1, 10, 100]
        nb_iteration = [4, 4, 4]
        scaler = Scaler_Trade(nb_min, nb_iteration)
        scaler.fit(self.data)

        # Generate Evolution Method
        start_future = 60
        end_future = 120
        evolution_meth = Evolution_Trade_Median(start_future, end_future)

        # Generate synchronous experiences
        nb_experiences = 10
        gen = Generator_Trade(mode=None, scaler=scaler) 
        unsync_exp = gen._get_unsynchronous_experiences(self.data, nb_experiences, evolution_meth)
        self.assertTrue(len(unsync_exp)==nb_experiences)
        print('Unsyncrhonous Generated')

    def test_generate_database(self):
        # Create and fit scaler
        nb_min = [1, 10, 100]
        nb_iteration = [4, 4, 4]
        scaler = Scaler_Trade(nb_min, nb_iteration)
        scaler.fit(self.data)

        # Generate Evolution Method
        start_future = 60
        end_future = 120
        evolution_meth = Evolution_Trade_Median(start_future, end_future)

        # Generate synchronous experiences
        gen = Generator_Trade(mode=Mode_Algo.train, scaler=scaler) 
        gen.generate_train_test_database(self.data, evolution_meth)
        
        exp = gen.get_new_experience()
        self.assertIsNotNone(exp)


# run the actual unittests
if __name__ =="__main__":
    unittest.main()
