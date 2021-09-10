import os
import json
import pandas as pd
from enum import Enum

import config

CRYPTO_STUDY_FILE = os.path.join(config.DATA_DIR, 'dtb/CRYPTO_STUDIED.json')
STORE = os.path.join(config.DATA_DIR, 'dtb/store.h5')

class Resolution_Historic(Enum):
    ''' Structure related to the load of Historic'''
    min = 'min'
    hour = 'hour'
    day = 'day'

def load_studied_crypto(path: str=None):
    ''' Load of studied cryptos'''
    if path is None:
        path = CRYPTO_STUDY_FILE
    with open(path) as f:
        studied_crypto = json.load(f)
    return studied_crypto

def load(resolution: Resolution_Historic=Resolution_Historic.min, path: str=None) -> pd.DataFrame:
    '''Load Historic of crypto values'''
    # Path of storage (create new file if empty)
    if path is None:
        path = STORE
    store = pd.HDFStore(path)
    # Check if store is empty
    if resolution.value not in store:
        return pd.DataFrame(index=[0])
    df = store[resolution.value]
    store.close()
    
    # Remove cryptos currently not studied
    cryptos_name = [d['coinbase_name'] for d in load_studied_crypto()]
    crypto_to_remove = [c for c in df.columns if c not in cryptos_name]
    df = df.drop(columns=crypto_to_remove)
    return df

def save(resolution: Resolution_Historic, df: pd.DataFrame, path: str=None):
    # Path of storage
    if path is None:
        path = STORE
    store = pd.HDFStore(path)
    store[resolution.value] = df
    store.close()