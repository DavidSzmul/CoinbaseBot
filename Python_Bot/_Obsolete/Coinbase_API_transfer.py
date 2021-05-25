# Not working because transfer_money is not working in reality. And no one found a suitable solution for this, problem is server-side.
# The only solution using API is to buy/sell that adds +10% fee on transactions -> Not gonna happen
from Coinbase_API.encryption import Coinbase_cryption
from coinbase.wallet.client import Client
from coinbase.wallet.model import APIObject
from coinbase.wallet.error import CoinbaseError

import json
import pandas as pd

CoinBase_pswd = Coinbase_cryption()
api_cood = CoinBase_pswd.decrypt_api()

client = Client(api_cood[0], api_cood[1])
#Create dict id
accounts = client.get_accounts()['data']
selected_crypto = ['BTC', 'ETH']
dict_selected_crypto = {}
for s in selected_crypto:
    for a in accounts:
        if a["currency"] == s:
            dict_selected_crypto[s]=a
# print(dict_selected_crypto)

##### Verify that account is correct
account_BTC = client.get_account(dict_selected_crypto['BTC']['id'])

# tx = client.transfer_money(dict_selected_crypto['ETH']['id'],
#                            to=dict_selected_crypto['BTC']['id'],
#                            amount='10', currency="EUR",
#                            description="First Transfer using API") # Transfer not working

# try:
#     tx = account_BTC.transfer_money(to=dict_selected_crypto['ETH']['id'],
#                                     amount='10', currency="EUR",
#                                     description="First Transfer using API") # Transfer not working
#     print('Sucess')
# except CoinbaseError as e:
#     print(e.response.text)

r = client._post('v2', "trades", data={
    "amount":"100",
    "amount_asset":"EUR",
    "amount_from":"input",
    "source_asset":dict_selected_crypto['BTC']['id'],
    "target_asset":dict_selected_crypto['ETH']['id']
    }
)
result = r.json()
trade_id = result['data']['id']
client._post("v2", "trades", trade_id, "commit")


