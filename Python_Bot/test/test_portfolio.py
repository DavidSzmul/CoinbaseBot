import unittest
import numpy as np
from algorithms.lib_trade.portfolio import Account, Portfolio, Portfolio_with_Transactioner

class TestAccount(unittest.TestCase):

    def test_safety(self):
        
        self.assertRaises(ValueError, Account, 'test', -1)
        account = Account('test', last_price=1)
        self.assertRaises(ValueError, account.set_value, -1)
        self.assertRaises(ValueError, account.set_amount, -1)
        self.assertRaises(ValueError, account.update, 0)

    def test_update(self):
        account = Account('test', last_price=2)
        account.set_amount(30)
        self.assertEqual(account.amount, 30)
        self.assertEqual(account.value, 60)
        account.set_value(30)
        self.assertEqual(account.value, 30)
        self.assertEqual(account.amount, 15)
        account.update(20)
        self.assertEqual(account.amount, 15)
        self.assertEqual(account.value, 300)

class TestPortfolio(unittest.TestCase):
    
    # @classmethod
    # def setUpClass(self):
    #     '''Init Class'''

    def test_portfolio_init_dict(self):
        account_last_prices = {'trade_1': 1, 'trade_2': 2, 'trade_unknown': 12}
        portfolio = Portfolio(last_prices=account_last_prices)

        # Verification all keys are equivalent
        for k in portfolio.keys():
            self.assertTrue(k in account_last_prices.keys())
        for a in account_last_prices.keys():
            self.assertTrue(a in portfolio.keys())

    def test_portfolio_init_accounts(self):
        accounts = [Account('trade_1', 1), Account('trade_2', 1), Account('trade_3', 1, amount=12)]
        portfolio = Portfolio(accounts=accounts)

        for a in accounts:
            self.assertEqual(a, portfolio[a.name])
        

    def test_update_account(self):

        account_last_prices = {'trade_1': 1, 'trade_2': 1}
        portfolio = Portfolio(account_last_prices)

        # Set known account
        account_known = Account('trade_2', last_price=2, amount=5)
        portfolio.set_account(account_known)

        self.assertEqual(portfolio['trade_2'].amount, 5)
        self.assertEqual(portfolio['trade_2'].last_price, 2)
        self.assertEqual(portfolio['trade_2'].value, 10)

        # Set unknown account
        account_unknown = Account('unknown', last_price=2, amount=10)
        portfolio.set_account(account_unknown)

    def test_add_money(self):

        account_last_prices = {'trade_1': 1, 'trade_2': 1}
        portfolio = Portfolio(account_last_prices)
        portfolio.add_money('trade_1', 20)

        self.assertTrue(False)

        
    # def test_convert(self):
    #     '''Test Conversion of 2 cryptos'''

    #     crypto_names = ['USDC-USD', 'BTC-USD']
    #     last_prices = {'USDC-USD':1, 'BTC-USD': 30000}
    #     portfolio = Portfolio(crypto_names)
    #     portfolio.update(last_prices)
    #     portfolio.add_money(50, to_= 'USDC-USD')
    #     portfolio.convert_money('USDC-USD','BTC-USD', 40, prc_taxes=0.01)
    #     portfolio.convert_money('USDC-USD','BTC-USD', 25, prc_taxes=0)
    #     portfolio.convert_money('USDC-USD','BTC-USD', 9999, prc_taxes=0.01) 


# run the actual unittests
if __name__ =="__main__":
    unittest.main()
