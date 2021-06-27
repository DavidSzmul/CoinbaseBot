import os 
import config
import json

CRYPTO_STUDY_FILE = os.path.join(config.DATA_DIR, 'dtb/CRYPTO_STUDIED.json')

class Portfolio(dict):

    def __init__(self,*arg,**kw):
        super().__init__(*arg,**kw)
        # Read Crypto
        with open(CRYPTO_STUDY_FILE) as f:
            data = json.load(f)
        crypto_study = [d['coinbase_name'] for d in data]
        for c in crypto_study:
            self[c] = {'ammount': 0, 'value': 0, 'last-price': 0}
        self.stableCoin = 'USDC-USD'
    
    def update_last_prices(self, last_historic):
        #### TODO
        for k in last_historic.keys():
            if k in self:
                self[k]['last-price'] = last_historic[k]
        self.update_values()
            
    def update_values(self, keys=[]):
        if len(keys)==0:
            keys=self.keys()
        for k in self.keys():
            self[k]['value'] = self[k]['ammount']*self[k]['last-price']

    def add_money(self, value, need_confirmation = True):
        # Add USDT in portfolio in order for algorithms to be able to do transactions
        # CAUTION: This action has to be made only by a human, need confirmation
        if need_confirmation:
            confirmation = input("Please confirm that you are human. Write 'YES'\n")=='YES'
            if not confirmation:
                print('Adding Money to portfolio canceled.')
                return
        self[self.stableCoin]['value'] = value
        self[self.stableCoin]['ammount'] = value/self[self.stableCoin]['last-price']

    def convert_money(self, from_, to_, value, prc_taxes=0):
        # TODO Add update of ammount

        if from_ not in self:
            raise ValueError(from_+' is not included in portfolio')
        if to_ not in self:
            raise ValueError(to_+' is not included in portfolio')

        # SELL
        value_sell = min(value,self[from_]['value'])
        ammount_sell = value_sell/self[from_]['last-price']
        self[from_]['ammount'] -= ammount_sell

        # BUY with taxes
        ammount_buy = value_sell*(1-prc_taxes)/self[to_]['last-price']
        self[to_]['ammount'] += ammount_buy

        self.update_values(keys=[from_,to_])


    def display(self):
        print('Last update of portfolio')
        for k in self.keys():
            print(k, ': ',self[k]['ammount'])
        print('')

if __name__ == '__main__':
    Ptf_test = Portfolio()
    last_prices = {'USDC-EUR':1, 'BTC-EUR': 30000}
    Ptf_test.update_last_prices(last_prices)
    Ptf_test.add_money(50, need_confirmation=False)
    Ptf_test.convert_money('USDC-EUR','BTC-EUR', 40, prc_taxes=0.01)
    Ptf_test.display()

    Ptf_test.convert_money('USDC-EUR','BTC-EUR', 40, prc_taxes=0.01)
    Ptf_test.display()

    Ptf_test.convert_money('BTC-EUR','USDC-EUR', 50, prc_taxes=0.01)
    Ptf_test.display()
