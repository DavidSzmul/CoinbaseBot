from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import datetime
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import urllib
import requests

from database import Historic_coinbase_dtb

def date2coinbaseDate(t):
    return t.isoformat(timespec='seconds') + '+00:00'

class Historic_Generator(ABC):
    '''Abstract Class to let the possibility to generate Historic based on multiple method'''

    @abstractmethod
    def update_dtb(self, maxSizeDtb: int,
                    resolution, 
                    path: str,
                    verbose: bool):
        '''Update Historic based on current time'''

class Historic_Coinbase_Generator(Historic_Generator):
    '''Update Historic of crypto-values based on Coinbase API'''

    MAX_REQUEST: int=300        # Maximum array on one request from Coinbase API
    DELAY_HOUR_REAL_TIME: int=4 # Delay to take in order that values are included in API (unable to get real time values from API)

    ctr_request: int
    ctr_update_display: int
    nb_update_display: int
    nb_request: int

    def reset_display_requests(self, nb_nan_to_complete: int, nb_update_display: int=10):
        self.ctr_request=0
        self.ctr_update_display=0
        self.nb_update_display = nb_update_display
        self.nb_request = int(nb_nan_to_complete/self.MAX_REQUEST)+1

    def update_display_requests(self):
        self.ctr_request+=1
        self.ctr_update_display = (self.ctr_update_display+1)%self.nb_update_display
        if self.ctr_update_display==0:
            self.nb_request = max(self.ctr_request, self.nb_request) # It is only an approximation
            prc_advance = self.ctr_request/self.nb_request*100
            prc_str = '%.2f%%' % prc_advance
            print(f' Generation Database:  {self.ctr_request}/{self.nb_request}  //  ' + prc_str)


    def getTimeResolution(self, resolution: Historic_coinbase_dtb.Resolution_Historic):
        # Is resolution valid
        res_historic = Historic_coinbase_dtb.Resolution_Historic 
        POSSIBLE_RESOLUTIONS = {res_historic.min: 60, res_historic.hour: 3600, res_historic.day: 86400}
        if resolution not in POSSIBLE_RESOLUTIONS:
            raise ValueError("Resolution must be contained in Resolution_Historic")

        return POSSIBLE_RESOLUTIONS[resolution]

    def _get_max_time_database(self):
        '''Max time possible to call API (based on current time)'''
        dt = timedelta(hours=self.DELAY_HOUR_REAL_TIME)
        now = datetime.now()
        ts = int(((now-dt).timestamp()//60)*60)
        return ts

    def _synchronize_dtb(self, df: pd.DataFrame, list_crypto: List[str], trange: np.ndarray, verbose: bool=False) -> pd.DataFrame:
        '''Synchronize Database based on current timing and demanded cryptos(only with Nan values)'''

        if verbose: print('Synchronizing Database...')
        # Tronquate with too old time
        df = df.truncate(before=trange[0], axis=0)

        # Fill Database with new added cryptos
        missing_crypto = [crypto for crypto in list_crypto if crypto not in df.columns]
        if len(missing_crypto)>0:
            new_clmn = pd.DataFrame(np.NaN, columns=missing_crypto, index=df.index)
            df = pd.concat([df, new_clmn], axis=1)      

        current_max_t = df.index.max() # Time already in dtb
        if np.isnan(current_max_t):
            current_max_t = 0
        missing_t = [int(t) for t in trange if t > current_max_t]

        if verbose:
            print(f'{len(missing_t)} new timing to add')
            print(f'{len(missing_crypto)} new crypto add')
            
        new_row = pd.DataFrame(np.NaN, columns=df.columns, index=missing_t)
        df = pd.concat([df, new_row], axis=0, verify_integrity=True)
        return df


    def _fillna_dtb(self, df: pd.DataFrame, verbose: bool=False):
        '''Complete missing values by removing NaN values'''

        # In case of error of connection with Coinbase -> Database partly complete, cut nan values at the end
        all_na_timing = df.isna().sum(axis=1).to_numpy()
        indexes_all_na_timing = np.nonzero(all_na_timing == len(df.columns))[0]
        if indexes_all_na_timing.size > 0: # End of timing contains nan values
            # Tronquate nan at end time
            t_remove_after = df.index[indexes_all_na_timing[0]]
            df = df.truncate(after=t_remove_after, axis=0)

        flag_array_is_nan = df.isnull().values
        index_array_is_nan =  [(i, j) for i in range(len(flag_array_is_nan)) 
                                        for j in range(len(flag_array_is_nan[0])) if flag_array_is_nan[i,j]]
        
        # Display
        if verbose: print(f'{len(index_array_is_nan)} NaN values to remove')

        # If Nan values remains fo interpolation
        df=(df.fillna(method='ffill') + df.fillna(method='bfill'))/2
        df= df.fillna(method='ffill')
        df= df.fillna(method='bfill')
        return df

    def _call_Coinbase_api(self, crypto: str, t_start: int, t_end: int, t_step: int) -> Tuple[Dict, bool]:
        
        # Constants
        STABLECOIN_NAME = 'USDC-USD'
        OFFSET_SEC_API = 7200 # For unknown reason, an offset of 7200s is added between start and stop from API response
        
        date_start = datetime.fromtimestamp(t_start-OFFSET_SEC_API)
        date_end = datetime.fromtimestamp(t_end-OFFSET_SEC_API)

        if crypto==STABLECOIN_NAME:
            time=np.arange(t_start, t_end, 60)
            data_serie={}
            for t in time:
                data_serie[(t, crypto)] = {'open':1, 'volume':0}
            return data_serie, True

        # Generate Coinase Url
        params = {'start': date2coinbaseDate(date_start), 'end': date2coinbaseDate(date_end), 'granularity': t_step}
        url = f"https://api.pro.coinbase.com/products/{crypto}/candles?{urllib.parse.urlencode(params)}"
        # Request
        r = requests.get(url)
        flag_success = str(r.status_code)[0]=='2'
        if not flag_success:
            return {}, flag_success
            # raise ConnectionError(r.json()['message'])
        data = r.json()

        # Use second line that corresponds to lowest price
        data_serie = {}
        for d in data:
            data_serie[(d[0], crypto)] = {'open':d[3], 'volume':d[-1]}
        
        return data_serie, flag_success


    def _complete_dtb(self, df: pd.DataFrame, step: int, verbose: bool=False):
        '''Function calling Coinbase API to obtain historic of cryptocurrencies'''

        # Initialization
        all_na_timing = df.isna().sum(axis=1).to_numpy()
        indexes_all_na_timing = np.nonzero(all_na_timing == len(df.columns))[0]
        if indexes_all_na_timing.size == 0: # No new timing. TODO: DEBUG in order to take into account if new crypto
            return df
        start_new_time = df.index[indexes_all_na_timing[0]]
        end_new_time = df.index[indexes_all_na_timing[-1]]

        dt_s = step*(self.MAX_REQUEST-1)
        # dt_timestamp = timedelta(seconds=dt_s)

        list_crypto = list(df.columns)
        if verbose:
            self.reset_display_requests(df.isna().sum().sum())


        # 1rst part: Complete old timings for new cryptos 
        missing_cryptos = df.columns[np.nonzero(df.isna().sum(axis=0).to_numpy() == len(df.index))[0]]
        if start_new_time != df.index[0] and len(missing_cryptos)>0:
            
            t_sart_old_time = df.index[0]
            t_end_old_time = start_new_time

            data_serie = {}
            for t_cycle in np.arange(t_sart_old_time, t_end_old_time+dt_s, dt_s):
                for crypto in missing_cryptos:
                    data_buffer, success = self._call_Coinbase_api(crypto, t_cycle, min(t_cycle+dt_s, t_end_old_time-step), step)
                    if not success:
                        break
                    data_serie = {**data_serie, **data_buffer}
                if not success: # Problem of connection
                    break
                # Completion
                for k,v in data_serie.items():
                    df.at[k[0], k[1]] = v['open']
                # Display
                if verbose: self.update_display_requests()


        # 2nd part: Complete new timings (all cryptos are nan)
        date_start = start_new_time
        while date_start <= end_new_time:
            # Part Time Range
            date_max = min(date_start+dt_s, end_new_time)

            # Call of API to get values
            data_serie = {}
            for crypto in list_crypto:
                data_buffer, success = self._call_Coinbase_api(crypto, date_start, date_max, step)
                if not success:
                    break
                data_serie = {**data_serie, **data_buffer}
 
            if not success: # Problem of connection
                break
            # Completion
            for k,v in data_serie.items():
                df.at[k[0], k[1]] = v['open']

            # Continue
            date_start = date_max+step
            if verbose: self.update_display_requests()

        # End of while
        return df


    def get_database(self, resolution, path: str=None):
        list_crypto = [d['coinbase_name'] for d in Historic_coinbase_dtb.load_studied_crypto()]
        df = Historic_coinbase_dtb.load(resolution, path)
        return df, list_crypto

    def update_dtb(self, maxSizeDtb: int=1e5, 
                    resolution:Historic_coinbase_dtb.Resolution_Historic = Historic_coinbase_dtb.Resolution_Historic.min, 
                    path: str=None, verbose = False):
        '''Complete Database based on current time'''
        
        # Define Timings
        time_resolution = self.getTimeResolution(resolution)
        tmax = np.floor(self._get_max_time_database()/time_resolution)*time_resolution
        tmin = tmax-(maxSizeDtb-1)*time_resolution
        trange = np.arange(tmin, tmax+time_resolution, time_resolution)
        
        # Load Database + Cryptos to study
        df, list_crypto = self.get_database(resolution, path)
        if verbose: print('Old database loaded...')

        df = self._synchronize_dtb(df, list_crypto, trange, verbose)
        df = self._complete_dtb(df, time_resolution, verbose)
        df = self._fillna_dtb(df, verbose)

        # Save Database
        Historic_coinbase_dtb.save(resolution, df, path)
        if verbose: print('Database refreshed saved...')

    def verify_all_crypto_valid(self) -> bool:
        '''Verifiy that all crypto into text list are valid inside Coinbase API'''
        resolution = Historic_coinbase_dtb.Resolution_Historic.min
        _, list_cryptos = self.get_database(resolution)
        ts = self._get_max_time_database()
        
        OFFSET = 500000
        NB_DATA = 60
        STEP = 60
        tmax = ts - OFFSET
        tmin = tmax-(NB_DATA-1)*STEP

        DELTA_MAX_MISS = 10

        result = True
        for crypto in list_cryptos:
            data_buffer, success = self._call_Coinbase_api(crypto, tmin, tmax, STEP)
            if not success:
                result = False
                print(f'{crypto} has connection issues')
            elif len(data_buffer)==0:
                result = False
                print(f'{crypto} had only non defined values')
            elif NB_DATA - len(data_buffer)>=DELTA_MAX_MISS:
                result = False
                print(f'{crypto} misses {NB_DATA-len(data_buffer)} values')
        if result:
            print('All cryptos valid')
        return result


if __name__ =="__main__":

    resolution = Historic_coinbase_dtb.Resolution_Historic.min
    Dtb = Historic_Coinbase_Generator()

    # Dtb.update_dtb(maxSizeDtb=1e3, resolution=resolution, verbose=True)
    # print('Done updating data')

    Dtb.verify_all_crypto_valid()
    