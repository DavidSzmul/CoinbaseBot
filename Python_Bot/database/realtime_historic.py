import os, sys
import asyncio
import time
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from collections import deque

import config

options = Options()
options.headless = False
options.add_argument("--window-size=1920,1200")

async def refresh_task_every(task, delta_t = 60):
    while True:
        t_sart = time.time()
        # Do task
        task()
        # Sleep current period of time
        t_sleep = delta_t-time.time()+t_sart
        if t_sleep<0:
            print(f'WARNING: Task takes more than {delta_t}s to execute')
        await asyncio.sleep(max(t_sleep,0))

class Scrapping_Coinbase(object):

    def __init__(self, max_len = 60):
        self.driver = webdriver.Chrome(config.DRIVER_PATH, options=options)
        self.crypto_historic = deque(maxlen=max_len)


    def refresh_task_every(self, delta_t = 60):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(refresh_task_every(self.refresh_crypto_value, delta_t=delta_t))


    def refresh_crypto_value(self):
        NB_PAGE = 3
        MAX_TRY = 10

        t_start = time.time()
        crypto_dict = {}
        for page in range(1,NB_PAGE+1):
            self.driver.get(f"https://www.coinbase.com/fr/price/s/listed?page={page}")
            ctr_try = 0
            while True:
                ctr_try+=1
                grid = self.driver.find_elements_by_css_selector("tbody")
                if grid[0].text!='':
                    list_crypto = np.array(grid[0].text.replace(',','.').replace('\u202f','').replace(' €','').split('\n'))
                    list_crypto = list_crypto.reshape(len(list_crypto)//8,8)
                    for l in list_crypto:
                        crypto_dict[l[1]+'-USD']=float(l[2])
                    break
                print('Retry')
                if ctr_try>=MAX_TRY:
                    raise AssertionError('Too many try to get crypto data')
        self.crypto_historic.append(crypto_dict)

if __name__=='__main__':
    scrap = Scrapping_Coinbase()
    ### Directly inside class
    # scrap.refresh_task_every()
    scrap.refresh_crypto_value()
    print('hello')