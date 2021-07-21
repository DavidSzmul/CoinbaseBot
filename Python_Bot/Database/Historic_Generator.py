import pandas as pd
import numpy as np
import datetime
import os
from abc import ABC, abstractmethod
from typing import List
import json
from datetime import datetime, timedelta
import urllib
import requests

from Database import Historic_dtb

class Historic_Generator(ABC):
    '''Abstract Class to let the possibility to generate Historic based on multiple method'''

    @abstractmethod
    def update_dtb(self, maxSizeDtb: int,
                    resolution: Historic_dtb.Resolution_Historic, 
                    path: str,
                    verbose: bool):
        '''Update Historic based on current time'''

class Historic_Coinbase_Generator(Historic_Generator):
    '''Update Historic of crypto-values based on Coinbase API'''

    def isResolutionValid(self, resolution: Historic_dtb.Resolution_Historic):
        res_historic = Historic_dtb.Resolution_Historic 
        self.possible_resolutions = {res_historic.min: 60, res_historic.hour: 3600, res_historic.day: 86400}
        assert resolution in self.possible_resolutions, "Resolution must be contained in Resolution_Historic"
        return True

    def getTimeResolution(self, resolution: Historic_dtb.Resolution_Historic):
        self.isResolutionValid(resolution)
        return self.possible_resolutions[self.resolution]

    def update_dtb(self, maxSizeDtb: int=1e4, 
                    resolution:Historic_dtb.Resolution_Historic = Historic_dtb.Resolution_Historic.min, 
                    path: str=None,
                    verbose = False):
        '''Complete Database based on current time'''
        
        # Initialization
        self.maxSizeDtb = maxSizeDtb
        self.resolution = resolution
        self.time_resolution = self.getTimeResolution(resolution)
        
        DELAY_HOUR_REAL_TIME = 4 # Delay to take in order that values are in historic
        dt = timedelta(hours=DELAY_HOUR_REAL_TIME)
        now = datetime.now()
        ts = int(((now-dt).timestamp()//60)*60)
        
        # Load Database
        crypto_names = [d['coinbase_name'] for d in Historic_dtb.load_studied_crypto()]
        df = Historic_dtb.load(self.resolution, path)
        if verbose: print('Database loaded...')

        # 1) Complete dataframe with new timestamps + studied crytpo
        if verbose: print('Adding missing values in historic...')

        def full_df(df):
            '''Complete missing values by calling API'''
            missing_crypto = [crypto for crypto in crypto_names if crypto not in df.columns]
        
            if len(missing_crypto)>0:
                new_clmn = pd.DataFrame(np.NaN, columns=missing_crypto, index=df.index)
                df = pd.concat([df, new_clmn], axis=1)

            # Complete row
            tmax = np.floor(ts/self.time_resolution)*self.time_resolution
            tmin = tmax-(self.maxSizeDtb-1)*self.time_resolution
            trange = np.arange(tmin, tmax+self.time_resolution, self.time_resolution)

            current_max_t = df.index.max() # Time already in dtb
            missing_t = [int(t) for t in trange if t > current_max_t]
            new_row = pd.DataFrame(np.NaN, columns=df.columns, index=missing_t)
            df = pd.concat([df, new_row], axis=0, verify_integrity=True)

            # Tronquate with too old time
            df = df.truncate(before=tmin, axis=0)
            if verbose:
                print(f'{len(missing_t)} new timing to add')
                print(f'{len(missing_crypto)} new crypto add')

            # Call API 
            for crypto in df.columns:
                api_crypto_dict = call_Coinbase_api(start=max(tmin, current_max_t), end=tmax, step=self.time_resolution, crypto = crypto)
                for k,v in api_crypto_dict.items():
                    df.loc[k][crypto] = v['open']
            return df

        df=full_df(df)

        # 2) Complete Delta Time missing for each crypto-currency
        if verbose: print('Removing NaN values in historic...')

        def fill_nan_df(df):
            '''Complete missing values by removing NaN values'''

            flag_array_is_nan = df.isnull().values
            index_array_is_nan =  [(i, j) for i in range(len(flag_array_is_nan)) 
                                            for j in range(len(flag_array_is_nan[0])) if flag_array_is_nan[i,j]]
            if verbose: print(f'{len(index_array_is_nan)} NaN values to remove')

            # If Nan values remains fo interpolation
            df=(df.fillna(method='ffill') + df.fillna(method='bfill'))/2
            df= df.fillna(method='ffill')
            df= df.fillna(method='bfill')
            return df
        df=fill_nan_df(df)

        # Save Database
        Historic_dtb.save(self.resolution, df, path)
        if verbose: print('Database saved...')


def call_Coinbase_api(start: int=0, end: int=0, step: int=60, crypto: str='BTC-USD'):
    '''Function calling Coinbase API to obtain historic of cryptocurrencies'''

    MAX_REQUEST = 300
    dt = timedelta(seconds=step*(MAX_REQUEST-1))

    def date2coinbaseDate(t):
        return t.isoformat(timespec='seconds') + '+00:00'

    def get_data(t_start, t_end):
        OFFSET_SEC_API = 7200 # For unknown reason, an offset of 7200s is added between start and stop from API
        date_start = datetime.fromtimestamp(t_start-OFFSET_SEC_API)
        date_end = datetime.fromtimestamp(t_end-OFFSET_SEC_API)

        if crypto=='USDC-USD':
            time=np.arange(t_start, t_end, 60)
            data_serie={}
            for t in time:
                data_serie[t] = {'open':1, 'volume':0}
            return data_serie

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


if __name__ =="__main__":

    resolution = Historic_dtb.Resolution_Historic.min
    Dtb = Historic_Coinbase_Generator()
    Dtb.update_dtb(maxSizeDtb=1e3,
                    resolution=resolution,
                    verbose=True)
    print('Done updating data')