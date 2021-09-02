import unittest
import numpy as np
import numpy.matlib as mb
from algorithms.lib_trade.environment_trade import Experience_Trade, Environment_Compare_Trading

class TestEnvironmentTrade(unittest.TestCase):
    
    # @classmethod
    # def setUpClass(self):
    #     '''Init Class'''

    def test_get_reward(self):
        '''Test Reward Function based on evolution'''

        # Initialization
        size_historic = 10
        prc_tax = 1e-1
        env = Environment_Compare_Trading(size_historic, prc_tax)
        evolution = np.array([1,-1,0])

        # Verification depending on taxes
        env.has_taxes = True
        self.assertEqual(env._get_reward(evolution, [0, 1], True), -2.1)
        self.assertEqual(env._get_reward(evolution, [0, 1], False), 2)

        env.has_taxes = False
        self.assertEqual(env._get_reward(evolution, [1, 0], True), 2)
        self.assertEqual(env._get_reward(evolution, [1, 0], False), -2)

        env.has_taxes = True    
        self.assertEqual(env._get_reward(evolution, [0, 2], True), -1.1)

    def test_reset_order_comparison(self):
        # Initialization
        size_historic = 10
        prc_tax = 1e-1
        env = Environment_Compare_Trading(size_historic, prc_tax, is_order_random=False)

        # Verification
        env.current_trade = 1
        env.nb_trade = 4
        env._reset_order_comparison()
        self.assertTrue(np.all(env.order_comparison == [0,2,3]))

    def test_reset(self):
        # Initialization
        size_historic = 3
        prc_tax = 1e-2
        env = Environment_Compare_Trading(size_historic, prc_tax, is_order_random=False)

        # Define simple state for experiences
        state_exp = np.array([
            [0,1,2,3,4],
            [0,1,2,3,4],
            [0,1,2,3,4],
        ])
        # Verification
        self.assertTrue(np.all(env.reset(Experience_Trade(state_exp, None, 1)
                                        )==np.array([1,1,1,0,0,0,1])))
        self.assertTrue(np.all(env.reset(Experience_Trade(state_exp+1, None, 1)
                                        )==np.array([2,2,2,1,1,1,1])))
        self.assertTrue(np.all(env.reset(Experience_Trade(state_exp, None, 0)
                                        )==np.array([0,0,0,1,1,1,1])))

    def test_step_doneOk(self):
        # Initialization
        size_historic = 3
        prc_tax = 1e-2
        env = Environment_Compare_Trading(size_historic, prc_tax, is_order_random=False)

        # Define simple experience
        current_trade = 2
        state_exp = np.array([
            [0,1,2,3],
            [0,1,2,3],
            [0,1,2,3],
        ])
        exp = Experience_Trade(state_exp, None, current_trade)
        
        # Verification
        _ = env.reset(exp)
        _, _, done, _ = env.step(np.array([1,0]))
        self.assertFalse(done)
        _, _, done, _ = env.step(np.array([0, 1]))
        self.assertFalse(done)
        ## End of env
        _, _, done, info = env.step(np.array([1,0]))
        self.assertTrue(done)
        self.assertEqual(info['current_trade'], 1)

    def test_step_actionEffectOnTax(self):
        # Initialization
        size_historic = 3
        prc_tax = 1e-2
        env = Environment_Compare_Trading(size_historic, prc_tax, is_order_random=False)

        # Define simple experience
        current_trade = 1
        state_exp = np.array([
            [0,1,2,3,4],
            [0,1,2,3,4],
            [0,1,2,3,4],
        ])
        exp = Experience_Trade(state_exp, None, current_trade)
        
        # Verification
        _ = env.reset(exp)
        state, _, _, _ = env.step(np.array([1,0]))
        self.assertTrue(env.has_taxes)
        _, _, _, _ = env.step(np.array([1,0]))
        self.assertTrue(env.has_taxes)
        _, _, _, _ = env.step(np.array([0,1]))
        self.assertFalse(env.has_taxes)
        _, _, _, _ = env.step(np.array([1,0]))
        self.assertFalse(env.has_taxes)

    def test_step(self):
        # Initialization
        size_historic = 3
        prc_tax = 1e-2
        env = Environment_Compare_Trading(size_historic, prc_tax, is_order_random=False)

        # Define simple experience
        current_trade = 1
        state_exp = np.array([
            [0,1,2,3,4],
            [0,1,2,3,4],
        ])
        exp = Experience_Trade(state_exp, None, current_trade)
        
        # Verification
        state = env.reset(exp)
        self.assertTrue(np.all(state == np.array([1,1,0,0,1])))
        state, _, _, _ = env.step(np.array([1,0]))
        self.assertTrue(np.all(state == np.array([1,1,2,2,1])))
        state, _, _, _ = env.step(np.array([0,1]))
        self.assertTrue(np.all(state == np.array([2,2,3,3,0])))
        state, _, _, _ = env.step(np.array([1,0]))
        self.assertTrue(np.all(state == np.array([2,2,4,4,0])))
        state, _, _, _ = env.step(np.array([1,0]))
        self.assertIsNone(state)

# run the actual unittests
if __name__ =="__main__":
    unittest.main()
