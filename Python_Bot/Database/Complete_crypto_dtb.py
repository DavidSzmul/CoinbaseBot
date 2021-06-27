import pandas as pd
import numpy as np
from numpy import delete, random
import datetime
import time
import os
import json
# import coinbase
from datetime import date, datetime, timedelta
import urllib
import requests

import config

CRYPTO_STUDY_FILE = os.path.join(config.DATA_DIR, 'dtb/CRYPTO_STUDIED.json')
STORE = os.path.join(config.DATA_DIR, 'dtb/store.h5')

class Dtb_crypto_historic():
    # Database that autocomplete containing all historic of requested crypto with 1min resolution
    def __init__(self, maxSizeDtb=1e3, 
                 resolution = 'min'):
                 
        self.possible_resolutions = {'min': 60, 'hour': 3600, 'day': 86400}
        assert resolution in self.possible_resolutions, "Resolution must be: 'min', 'hour' or 'day'"

        self.maxSizeDtb = maxSizeDtb
        self.resolution = resolution
        self.time_resolution = self.possible_resolutions[self.resolution]
        self.store = pd.HDFStore(STORE)
        self.df = self.load()
        with open(CRYPTO_STUDY_FILE) as f:
            data = json.load(f)
        crypto_study = [d['coinbase_name'] for d in data]

    def update_dtb(self, crypto_names, verbose = False):
        DELAY_HOUR_REAL_TIME = 4 # Delay to take in order that values are in historic
        dt = timedelta(hours=DELAY_HOUR_REAL_TIME)
        now = datetime.now()
        ts = int(((now-dt).timestamp()//60)*60)

        # 1) Complete dataframe with new timestamps + studied crytpo
        if verbose: print('Adding missing values in historic...')

        def full_df():
            # Complete columns
            missing_crypto = [crypto for crypto in crypto_names if crypto not in self.df.columns]
        
            if len(missing_crypto)>0:
                new_clmn = pd.DataFrame(np.NaN, columns=missing_crypto, index=self.df.index)
                self.df = pd.concat([self.df, new_clmn], axis=1)

            # Complete row
            tmax = np.floor(ts/self.time_resolution)*self.time_resolution
            tmin = tmax-(self.maxSizeDtb-1)*self.time_resolution
            trange = np.arange(tmin, tmax+self.time_resolution, self.time_resolution)

            current_max_t = self.df.index.max() # Time already in dtb
            missing_t = [int(t) for t in trange if t > current_max_t]
            new_row = pd.DataFrame(np.NaN, columns=self.df.columns, index=missing_t)
            self.df = pd.concat([self.df, new_row], axis=0, verify_integrity=True)

            # Tronquate with too old time
            self.df = self.df.truncate(before=tmin, axis=0)
            if verbose:
                print(f'{len(missing_t)} new timing to add')
                print(f'{len(missing_crypto)} new crypto add')

            # Call API 
            for crypto in self.df.columns:
                api_crypto_dict = call_api_historic(start=max(tmin, current_max_t), end=tmax, step=self.time_resolution, crypto = crypto)
                for k,v in api_crypto_dict.items():
                    self.df.loc[k][crypto] = v['open']
        full_df()

        # 2) Complete Delta Time missing for each crypto-currency
        if verbose: print('Removing NaN values in historic...')

        def fill_df():
            # Determine NaN values and correct them
            flag_array_is_nan = self.df.isnull().values
            index_array_is_nan =  [(i, j) for i in range(len(flag_array_is_nan)) 
                                            for j in range(len(flag_array_is_nan[0])) if flag_array_is_nan[i,j]]
            if verbose: print(f'{len(index_array_is_nan)} NaN values to remove')
            # for x,y in index_array_is_nan:
            #     t_event = self.df.iloc[[x]].index[0]
            #     crypto = self.df.iloc[[x]].columns[y]

            #     # Coinbase API call, but it happens that some values are missing
            #     buffer = call_api_historic(start=t_event, end=t_event, step=self.time_resolution, crypto = crypto)
            #     if len(buffer)>0:
            #         self.df.iloc[x,y]=list(buffer.values())[0]['open']

            # If Nan values remains fo interpolation
            self.df=(self.df.fillna(method='ffill') + self.df.fillna(method='bfill'))/2
            self.df= self.df.fillna(method='ffill')
            self.df= self.df.fillna(method='bfill')
        fill_df()

    def save(self):
        self.store[self.resolution] = self.df

    def load(self):
        if self.resolution in self.store:
            df = self.store[self.resolution]
        else:
            df = pd.DataFrame(index=[0])
        return df


# Create a function to fetch the data
def call_api_historic(start=0, end=0, step=60, crypto = 'BTC-EUR'):

    MAX_REQUEST = 300
    dt = timedelta(seconds=step*(MAX_REQUEST-1))

    def date2coinbaseDate(t):
        return t.isoformat(timespec='seconds') + '+00:00'

    def get_data(t_start, t_end):
        OFFSET_SEC_API = 7200 # For unknown reason, an offset of 7200s is added between start and stop from API
        date_start = datetime.fromtimestamp(t_start-OFFSET_SEC_API)
        date_end = datetime.fromtimestamp(t_end-OFFSET_SEC_API)

        # Generate Coinase Url
        params = {'start': date2coinbaseDate(date_start), 'end': date2coinbaseDate(date_end), 'granularity': step}
        url = f"https://api.pro.coinbase.com/products/{crypto}/candles?{urllib.parse.urlencode(params)}"
        # Request
        r = requests.get(url)
        succeed = str(r.status_code)[0]=='2'
        if not succeed:
            raise ConnectionError(r.json()['message'])
        data = r.json()

        # Verification that timestamps are coherent
    #     if len(data)>0 and (data[0][0] != t_end or data[-1][0] != t_start):
    #         raise AssertionError(f'Timestamp not synchronized with Coinbase API.\n\
    # Delta start: {data[0][0] - t_end}s; Delta end: {data[-1][0] - t_start}s')

        # Use second line that corresponds to lowest price
        data_serie = {}
        for d in data:
            data_serie[d[0]] = {'open':d[3], 'volume':d[-1]}
        return data_serie
    
    data_serie = {}
    while start <= end:
        date_start = datetime.fromtimestamp(start)
        date_max = datetime.timestamp(date_start+dt)
        end_buffer = min(date_max, end)
        data_buffer = get_data(start, end_buffer)
        data_serie = {**data_serie, **data_buffer}

        start = end_buffer+step
        start = start + step
    return data_serie

with open(CRYPTO_STUDY_FILE) as f:
        data = json.load(f)
        crypto_study = [d['coinbase_name'] for d in data]

if __name__ =="__main__":

    # if os.path.exists(STORE):
    #     os.remove(STORE)
    Dtb = Dtb_crypto_historic(resolution='min')
    Dtb.update_dtb(crypto_study, verbose=True)
    print(Dtb.df)
    Dtb.save()
    print('Done updating data')