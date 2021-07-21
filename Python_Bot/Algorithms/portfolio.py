from typing import Dict, List
from dataclasses import dataclass
from Database import Historic_dtb

@dataclass
class Crypto_Account:
    ammount: float=0
    value: float=0
    last_price: float=0

class Portfolio_Coinbase(dict):

    def __init__(self,*arg,**kw):
        super().__init__(*arg,**kw)

        # Create Account for each cryptocurrency
        crypto_study = [d['coinbase_name'] for d in Historic_dtb.load_studied_crypto()]
        for c in crypto_study:
            self[c] = Crypto_Account()

        # Define StableCoin where real money comes in
        self.stableCoin = 'USDC-USD'
        self[self.stableCoin].last_price = 1
        self.total_value = 0
    
    def update_last_prices(self, last_historic: Dict):
        #### TODO
        for k in last_historic.keys():
            if k in self:
                self[k].last_price = last_historic[k]
        self.update_values()
            
    def update_total(self):
        self.total_value=0
        for k in self.keys():
            self.total_value += self[k].value

    def update_values(self, keys: List[str]=None):
        if keys is None:
            keys=self.keys()
        self.total_value=0
        for k in self.keys():
            self[k].value = self[k].ammount*self[k].last_price
        self.update_total()

    def add_money(self, value, need_confirmation = True):
        # Add USDT in portfolio in order for algorithms to be able to do transactions
        # CAUTION: This action has to be made only by a human, need confirmation
        if need_confirmation:
            confirmation = input("Please confirm that you are human. Write 'YES'\n")=='YES'
            if not confirmation:
                print('Adding Money to portfolio canceled.')
                return
        self[self.stableCoin].value += value
        self[self.stableCoin].ammount = value/self[self.stableCoin].last_price
        self.update_values()

    def convert_money(self, from_, to_, value, prc_taxes=0):
        # TODO Add update of ammount

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
