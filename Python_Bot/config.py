import os
import pandas as pd
import json

ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
DRIVER_PATH = os.path.join(ROOT_DIR,"chromedriver.exe")

def load_studied_crypto():

    CRYPTO_STUDY_FILE = os.path.join(DATA_DIR, 'dtb/CRYPTO_STUDIED.json')
    with open(CRYPTO_STUDY_FILE) as f:
        studied_crypto = json.load(f)
    return studied_crypto

def load_df_historic(resolution):
    # resolution 'min', 'hour' or 'day'
    STORE = os.path.join(DATA_DIR, 'dtb/store.h5')
    store = pd.HDFStore(STORE)
    df = store[resolution]

    studied_crypto = load_studied_crypto()
    cryptos_name = [d['coinbase_name'] for d in studied_crypto]
    crypto_to_remove = [c for c in df.columns if c not in cryptos_name]
    df = df.drop(columns=crypto_to_remove)

    return df
