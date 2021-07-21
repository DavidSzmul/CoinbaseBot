import config
import os
import json
import pandas as pd
from enum import Enum

CRYPTO_STUDY_FILE = os.path.join(config.DATA_DIR, 'dtb/CRYPTO_STUDIED.json')
STORE = os.path.join(config.DATA_DIR, 'dtb/store.h5')

class Resolution_Historic(Enum):
    ''' Structure related to the load of Historic'''
    min = 'min'
    hour = 'hour'
    day = 'day'

def load_studied_crypto():
    ''' Load of studied cryptos'''
    with open(CRYPTO_STUDY_FILE) as f:
        studied_crypto = json.load(f)
    return studied_crypto

def load(resolution: Resolution_Historic) -> pd.DataFrame:
    '''Load Historic of crypto values'''
    
    # Path of storage
    store = pd.HDFStore(STORE)
    df = store[resolution.value]

    # Remove cryptos currently not studied
    cryptos_name = [d['coinbase_name'] for d in load_studied_crypto()]
    crypto_to_remove = [c for c in df.columns if c not in cryptos_name]
    df = df.drop(columns=crypto_to_remove)
    return df

def save(resolution: Resolution_Historic, df: pd.DataFrame):
    # Path of storage
    store = pd.HDFStore(STORE)
    store[resolution.value] = df