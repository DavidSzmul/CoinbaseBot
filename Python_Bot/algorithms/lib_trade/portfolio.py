from typing import Dict, List
import numpy as np

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
            self.set_accounts(accounts)

        elif last_prices:
            for k,price in last_prices.items():
                self[k] = Account(k, price)
        

    def set_account(self, account: Account):
        self[account.name] = account

    def set_accounts(self, accounts: List[Account]):
        for a in accounts:
            self.set_account(a)

    def _update_total(self):
        '''Update total value of portfolio'''
        self.__total_value=0
        for k in self.keys():
            self.__total_value += self[k].value

    def update(self, last_prices: Dict[str, float]):
        '''Update last price of each cryptocurrency account'''
        for k, last_price in last_prices.items():
            # Add account if new last price not included into portfolio
            if k not in self:
                # Expect that no amount is included
                self.set_account(Account(k, last_price, amount=0)) 
            else:
                self[k].update(last_price)
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

        # CLAMP Value
        if value<0: # Need to convert all money on account
            value = self[from_].value
        value = min(value, self[from_].value)

        # SELL/BUY Ammounts
        value_sell = value
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

    def get_highest_account(self):
        values = [v.value for v in self.values()]
        idx = np.argmax(values)[0]
        return self.keys()[idx]

    
class Portfolio_with_Transactioner(Portfolio):

    transactioner: Transactioner
    
    def __init__(self, transactioner: Transactioner, last_prices: Dict[str, float]=None, accounts: List[Account]=None):
        '''Create Portfolio with associated Transactioner'''
        super().__init__(last_prices=last_prices, accounts=accounts)
        self.transactioner = transactioner

    def convert_money(self, from_: str, to_: str, value: float) -> bool:
        '''Realise real conversion before adding it to the portfolio'''
        # Hidden taxes that are contained inside transaction
        transaction=self.transactioner.convert(from_, to_, value)
        if transaction.success:
            # Confirm transaction on Accounts
            self[from_].add_amount(-transaction.amount_from)
            self[to_].add_amount(transaction.amount_to)   
        return transaction.success
         

if __name__ == '__main__':
    pass