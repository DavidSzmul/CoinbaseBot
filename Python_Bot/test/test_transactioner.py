import unittest
from algorithms.lib_trade.transactioner import DefaultTransactioner

class TestDefaultTransactioner(unittest.TestCase):

    def test_convert(self):
        
        transactioner = DefaultTransactioner(flag_success=True, prc_tax=0)
        transaction = transactioner.convert('aa','bb',10)
        self.assertEqual(transaction.success, True)
        self.assertEqual(transaction.amount_from, 10)
        self.assertEqual(transaction.amount_to, 10)

        transactioner.set_flag_success(False)
        transactioner.set_prc_tax(0.1)
        transaction = transactioner.convert('aa','bbc',10)
        self.assertEqual(transaction.success, False)
        self.assertEqual(transaction.amount_from, 10)
        self.assertEqual(transaction.amount_to, 9)

# run the actual unittests
if __name__ =="__main__":
    unittest.main()