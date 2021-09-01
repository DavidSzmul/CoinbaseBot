from typing import Dict, List
from dataclasses import dataclass

from algorithms.lib_trade.transactioner import Transactioner

class Account:
    name: str
    last_price: float
    amount: float=0
    value: float=0

    def __init__(self, name: str, last_price: float, amount: float=0):
        self.name = name
        self.amount = amount 
        self.update(last_price)
    
    def update(self, last_price: float):
        '''Update real value of trade'''
        if last_price<=0:
            raise ValueError('last_price shall be striclty superior to zero')
        self.last_price = last_price
        self.value = self.amount*self.last_price

    def set_amount(self, amount: float):
        '''Calculate real value of trade'''
        if amount<0:
            raise ValueError('amount shall be superior to zero')
        self.amount = amount
        self.value = amount*self.last_price

    def set_value(self, value: float):
        '''Set value on account'''
        if value<0:
            raise ValueError('value shall be superior to zero')
        self.value = value
        self.amount = value/self.last_price

    def add_value(self, value: float):
        self.set_value(self.value+value)

    def add_amount(self, amount: float):
        self.set_amount(self.amount+amount)


class Portfolio(Dict[str, Account]):

    __total_value: int=0

    def __init__(self, last_prices: Dict[str, float]=None, accounts: List[Account]=None):
        '''Dictionary containing all accounts of cryptocurrency'''
        super().__init__()
        if accounts:
            for a in accounts:
                self[a.name] = a
            return
        elif last_prices:
            for k,price in last_prices.items():
                self[k] = Account(k, price)
            return
        raise ValueError('At least one parameter has to be used')

    def set_account(self, account: Account):
        self[account.name] = account

    def _update_total(self):
        '''Update total value of portfolio'''
        self.__total_value=0
        for k in self.keys():
            self.__total_value += self[k].value

    def update(self, last_prices: Dict):
        '''Update last price of each cryptocurrency account'''
        for k in last_prices.keys():
            if k in self:
                self[k].update(last_prices[k])
        self._update_total()

    def add_money(self, to_: str, value: float):
        '''Add virtualy money into a specific account'''

        if to_ not in self.keys():
            raise ValueError(f'{to_} not included in portfolio')
        self[to_].add_value(value)
        self._update_total()

    def convert_money(self, from_: str, to_: str, value: float, prc_taxes: float=0):
        '''Conversion from one account to another'''
        if from_ not in self:
            raise ValueError(from_+' is not included in portfolio')
        if to_ not in self:
            raise ValueError(to_+' is not included in portfolio')

        # SELL/BUY Ammounts
        value_sell = min(value, self[from_].value)
        amount_sell = value_sell/self[from_].last_price
        amount_buy = value_sell*(1-prc_taxes)/self[to_].last_price

        # Modify into accounts
        self[from_].add_amount(-amount_sell)
        self[to_].add_amount(amount_buy)
        self._update_total()
    
    def display_value(self):
        print('Last update of portfolio')
        for k in self.keys():
            print(k, ': ',self[k].value)
        print(f'TOTAL: {self.__total_value}')
        print('')

    def get_total_value(self):
        return self.__total_value

    
class Portfolio_with_Transactioner(Portfolio):

    transactioner: Transactioner
    
    def __init__(self, account_names: List[str], transactioner: Transactioner):
        '''Create Coinbase Portfolio with associated cryptos'''

        super().__init__(account_names)
        # Define StableCoin where real money comes in
        self.stableCoin = 'USDC-USD'
        self[self.stableCoin].last_price = 1
        # Add a transactioner to do all required transactions
        self.transactioner = transactioner

    def add_money(self, value: float, to_: str=None, need_confirmation : bool=False):
        '''Add money (inherited method, with stableCoin put at default'''
        if to_ is None:
            to_ = self.stableCoin
        super().add_money(value, to_, need_confirmation)

    def convert_money(self, from_: str, to_: str, value: float, prc_taxes: float=0):
        '''Realise real conversion before adding it to the portfolio'''
        # Do transaction using Scrapping
        self.transactioner.convert(from_, to_, value)
        # Inherite
        super().convert_money(from_, to_, value, prc_taxes)

if __name__ == '__main__':
    pass
    # from database import Historic_coinbase_dtb
    # crypto_study = [d['coinbase_name'] for d in Historic_coinbase_dtb.load_studied_crypto()]
    # last_prices = {'USDC-USD':1, 'BTC-USD': 30000}

    # Ptf_test = Portfolio(crypto_study)
    # Ptf_test.update_last_prices(last_prices)
    # Ptf_test.add_money(50, to_= 'USDC-USD', need_confirmation=False)

    # Ptf_test.convert_money('USDC-USD','BTC-USD', 40, prc_taxes=0.01)
    # Ptf_test.display()

    # Ptf_test.convert_money('USDC-USD','BTC-USD', 40, prc_taxes=0.01)
    # Ptf_test.display()

    # Ptf_test.convert_money('BTC-USD','USDC-USD', 50, prc_taxes=0.01)
    # Ptf_test.display()
