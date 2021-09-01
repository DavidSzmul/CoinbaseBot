import unittest
import numpy as np
from algorithms.lib_trade.portfolio import Portfolio, Portfolio_with_Transactioner

class TestPortfolio(unittest.TestCase):
    
    # @classmethod
    # def setUpClass(self):
    #     '''Init Class'''
        
    def test_convert(self):
        '''Test Conversion of 2 cryptos'''

        crypto_names = ['USDC-USD', 'BTC-USD', 'Toto']
        last_prices = {'USDC-USD':1, 'BTC-USD': 30000}
        portfolio = Portfolio(crypto_names)
        portfolio.update_last_prices(last_prices)
        portfolio.add_money(50, to_= 'USDC-USD')
        portfolio.convert_money('USDC-USD','BTC-USD', 40, prc_taxes=0.01)
        portfolio.convert_money('USDC-USD','BTC-USD', 40, prc_taxes=0.01)
        portfolio.convert_money('BTC-USD','USDC-USD', 50, prc_taxes=0.01) 


# run the actual unittests
if __name__ =="__main__":
    unittest.main()
