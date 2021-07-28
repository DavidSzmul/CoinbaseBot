from typing import Dict, List
from dataclasses import dataclass
from Database import Historic_dtb

@dataclass
class Account:
    ammount: float=0
    value: float=0
    last_price: float=0

class Portfolio(Dict):

    total_value: int=0

    def __init__(self,*arg,**kw):
        '''Dictionary containing all accounts of cryptocurrency'''
        super().__init__(*arg,**kw)

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
            self[k].value = self[k].ammount*self[k].last_price
        self.update_total()

    def add_money(self, value: float, to_: str=None, need_confirmation :bool=True):
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
        self[to_].ammount = value/self[to_].last_price
        self.update_values()

    def convert_money(self, from_, to_, value, prc_taxes=0):
        '''Conversion from one account to another'''

        if from_ not in self:
            raise ValueError(from_+' is not included in portfolio')
        if to_ not in self:
            raise ValueError(to_+' is not included in portfolio')

        # SELL
        value_sell = min(value,self[from_].value)
        ammount_sell = value_sell/self[from_].last_price
        self[from_].ammount -= ammount_sell

        # BUY with taxes
        ammount_buy = value_sell*(1-prc_taxes)/self[to_].last_price
        self[to_].ammount += ammount_buy

        # Update Portfolio
        self.update_values(keys=[from_,to_])
    
    def display(self):
        print('Last update of portfolio')
        for k in self.keys():
            print(k, ': ',self[k].ammount)
        print(f'TOTAL: {self.total_value}')
        print('')
    
    def get_total_value(self):
        return self.total_value

    

class Portfolio_Coinbase(Portfolio):

    def __init__(self,*arg,**kw):
        super().__init__(*arg,**kw)
        '''Create Coinbase Portfolio with associated cryptos'''

        crypto_study = [d['coinbase_name'] for d in Historic_dtb.load_studied_crypto()]
        for c in crypto_study:
            self[c] = Account()

        # Define StableCoin where real money comes in
        self.stableCoin = 'USDC-USD'
        self[self.stableCoin].last_price = 1

    def add_money(self, value: float, to_: str=None, need_confirmation :bool=True):
        '''Add money (inherited method, with stableCoin put at default'''
        if to_ is None:
            to_ = self.stableCoin
        super().add_money(value, to_, need_confirmation)

if __name__ == '__main__':
    Ptf_test = Portfolio_Coinbase()
    Ptf_test.add_money(50, need_confirmation=False)
    last_prices = {'USDC-USD':1, 'BTC-USD': 30000}
    Ptf_test.update_last_prices(last_prices)

    Ptf_test.convert_money('USDC-USD','BTC-USD', 40, prc_taxes=0.01)
    Ptf_test.display()

    Ptf_test.convert_money('USDC-USD','BTC-USD', 40, prc_taxes=0.01)
    Ptf_test.display()

    Ptf_test.convert_money('BTC-USD','USDC-USD', 50, prc_taxes=0.01)
    Ptf_test.display()
