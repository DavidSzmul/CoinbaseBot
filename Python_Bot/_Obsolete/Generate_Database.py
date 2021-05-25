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

result = r.json()
trade_id = result['data']['id']
client._post("v2", "trades", trade_id, "commit")


