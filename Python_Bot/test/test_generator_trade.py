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

    def test_normalization(self):
        gen = Generator_Trade(self.data, verbose=False)
        data_raw = self.data.to_numpy()
        data_norm = gen._normalize(self.data)

        test_valid = np.all(data_raw.flatten() > data_norm.flatten())
        self.assertTrue(test_valid)

    def test_index_window_safety(self):
        nb_iteration = [4, 4, 4]
        nb_min = [1, 10, 100, 1000]
        gen = Generator_Trade(self.data, verbose=False)
        self.assertRaises(ValueError, gen._get_idx_window_historic, nb_iteration, nb_min)

    def test_index_window_empty(self):
        gen = Generator_Trade(self.data, verbose=False)
        self.assertIsNone(gen._get_idx_window_historic(None, None))

    def test_index_window_coherence(self):
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
        gen = Generator_Trade(self.data.iloc, verbose=False)
        sync_exp = gen._get_synchronous_experiences(self.data, nb_iteration, nb_min, duration_future,
                                                    )

        is_evolution_none_exp = [s.evolution is None for s in sync_exp]
        test_valid = np.all(is_evolution_none_exp)
        self.assertTrue(test_valid)


    # def test_compute_tax_electric(self):
    #     v = VehicleInfo("BMW", True, 10000)
    #     self.assertEqual(v.compute_tax(), 200)

    # def test_compute_tax_exemption(self):
    #     v = VehicleInfo("BMW", False, 10000)
    #     self.assertEqual(v.compute_tax(5000), 250)
    
    # def test_compute_tax_exemption_negative(self):
    #     v = VehicleInfo("BMW", False, 10000)
    #     self.assertRaises(ValueError, v.compute_tax, -5000)

    # def test_compute_tax_exemption_high(self):
    #     v = VehicleInfo("BMW", False, 10000)
    #     self.assertEqual(v.compute_tax(20000), 0)

    # def test_can_lease_false(self):
    #     v = VehicleInfo("BMW", False, 10000)
    #     self.assertEqual(v.can_lease(5000), False)

    # def test_can_lease_true(self):
    #     v = VehicleInfo("BMW", False, 10000)
    #     self.assertEqual(v.can_lease(15000), True)

    # def test_can_lease_negative_income(self):
    #     v = VehicleInfo("BMW", False, 10000)
    #     self.assertRaises(ValueError, v.can_lease, -5000)

# run the actual unittests
if __name__ =="__main__":
    unittest.main()
