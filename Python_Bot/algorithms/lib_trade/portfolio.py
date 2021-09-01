from typing import Dict, List
from dataclasses import dataclass

from algorithms.lib_trade.transactioner import Transactioner

@dataclass
class Account:
    amount: float=0
    value: float=None
    last_price: float=None

class Portfolio(Dict):

    total_value: int=0

    def __init__(self, account_names: List[str]):
        '''Dictionary containing all accounts of cryptocurrency'''
        super().__init__()
        for a in account_names:
            self[a] = Account()

    def update_last_prices(self, last_prices: Dict):
        '''Update last price of each cryptocurrency account'''
        for k in last_prices.keys():
            if k in self:
                self[k].last_price = last_prices[k]
        self.update_values()

    def update_total(self):
        '''Update total value of portfolio'''
        self.total_value=0
        for k in self.keys():
            self.total_value += self[k].value

    def update_values(self, keys: List[str]=None):
        '''Update values of each account'''
        if keys is None:
            keys=self.keys()
        self.total_value=0
        for k in self.keys():
            self[k].value = self[k].amount*self[k].last_price
        self.update_total()

    def add_money(self, value: float, to_: str=None, need_confirmation :bool=False):
        '''Add virtualy money into a specific account'''

        if to_ is None:
            print('No money added')
            return
        if need_confirmation:
            confirmation = input("Please confirm that you are human. Write 'YES'\n")=='YES'
            if not confirmation:
                print('Adding Money to portfolio canceled.')
                return
        self[to_].value += value
        self[to_].amount = value/self[to_].last_price
        self.update_values()

    def convert_money(self, from_: str, to_: str, value: float, prc_taxes: float=0):
        '''Conversion from one account to another'''

        if from_ not in self:
            raise ValueError(from_+' is not included in portfolio')
        if to_ not in self:
            raise ValueError(to_+' is not included in portfolio')

        # SELL
        value_sell = min(value,self[from_].value)
        amount_sell = value_sell/self[from_].last_price
        self[from_].amount -= amount_sell

        # BUY with taxes
        amount_buy = value_sell*(1-prc_taxes)/self[to_].last_price
        self[to_].amount += amount_buy

        # Update Portfolio
        self.update_values(keys=[from_,to_])
    
    def display(self):
        print('Last update of portfolio')
        for k in self.keys():
            print(k, ': ',self[k].amount)
        print(f'TOTAL: {self.total_value}')
        print('')
    
    def get_value(self,from_):
        return self[from_].value

    def get_total_value(self):
        return self.total_value

    
class Portfolio_with_Transactioner(Portfolio):

    transactioner: Transactioner
    
    def __init__(self, account_names: List[str], transactioner: transactioner):
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
