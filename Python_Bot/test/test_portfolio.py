from algorithms.lib_trade.transactioner import DefaultTransactioner, Transactioner
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

    def test_set_account(self):

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

    def test_portfolio_set_accounts(self):
        portfolio = Portfolio()
        accounts = [Account('trade_1', 1), Account('trade_2', 1), Account('trade_3', 1, amount=12)]
        portfolio.set_accounts(accounts)


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

    def test_add_money(self):

        account_last_prices = {'trade_1': 0.1, 'trade_2': 1}
        portfolio = Portfolio(account_last_prices)
        portfolio.add_money('trade_1', 20)
        self.assertEqual(portfolio['trade_1'].value, 20)
        self.assertEqual(portfolio['trade_2'].value, 0)
        portfolio.add_money('trade_1', 20)
        self.assertEqual(portfolio['trade_1'].value, 40)
        self.assertRaises(ValueError, portfolio.add_money,'unknown', 20)

        
    def test_convert(self):
        '''Test Conversion of 2 cryptos'''
        last_prices = {'USDC-USD':1, 'BTC-USD': 30000}
        portfolio = Portfolio()
        portfolio.update(last_prices)
        portfolio.add_money('USDC-USD', 200)
        portfolio.convert_money('USDC-USD','BTC-USD', 100, prc_taxes=0.1)
        self.assertEqual(portfolio['USDC-USD'].value, 100)
        self.assertEqual(portfolio['BTC-USD'].value, 90)

        portfolio.convert_money('BTC-USD','USDC-USD', 50, prc_taxes=0)
        self.assertEqual(portfolio['USDC-USD'].value, 150)
        self.assertEqual(portfolio['BTC-USD'].value, 40)
        portfolio.convert_money('USDC-USD','BTC-USD', 9999, prc_taxes=0) 
        self.assertEqual(portfolio['USDC-USD'].value, 0)
        self.assertEqual(portfolio['BTC-USD'].value, 190)

class TestPortfolioWithTransactioner(unittest.TestCase):
    
    # @classmethod
    # def setUpClass(self):
    #     '''Init Class'''

    def test_convert(self):
        '''Test Conversion of 2 cryptos'''

        # Initialization
        transactioner = DefaultTransactioner(flag_success=True, prc_tax=0.1)
        last_prices = {'USDC-USD':1, 'BTC-USD': 1, 'Unknown':1}
        portfolio = Portfolio_with_Transactioner(transactioner, last_prices=last_prices)
        portfolio['USDC-USD'].add_amount(200)

        # Success with tax
        portfolio.convert_money('USDC-USD','BTC-USD', 100)
        self.assertEqual(portfolio['USDC-USD'].amount, 100)
        self.assertEqual(portfolio['BTC-USD'].amount, 90)

        # Success without tax
        transactioner.prc_tax=0
        portfolio.convert_money('BTC-USD','USDC-USD', 90)
        self.assertEqual(portfolio['USDC-USD'].value, 190)
        self.assertEqual(portfolio['BTC-USD'].value, 0)
        
        # Fail without change
        transactioner.flag_success=False
        portfolio.convert_money('BTC-USD','USDC-USD', 100000)
        self.assertEqual(portfolio['USDC-USD'].value, 190)
        self.assertEqual(portfolio['BTC-USD'].value, 0)


# run the actual unittests
if __name__ =="__main__":
    unittest.main()
