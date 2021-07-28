### Modifications due to changes in the coinbase website

import sys, os, time
import numpy as np
import pickle
import tkinter.messagebox
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

import config
from coinbase_api.encryption import Coinbase_cryption


COINBASE_SITE = "https://www.coinbase.com/dashboard"
FILE_COOKIE = config.COINBASE_SCRAPPER_COOKIES
DRIVER_PATH = config.DRIVER_PATH

dict_id = {
    'accept_cookie': '.sc-AxjAm',
    'buy_sell': '.kMuznm',
    'convert': "[data-element-handle='folder-tab-convert']",
    'amount': '.ivDhsu',
    'from': "[data-element-handle='convert-from-selector']",
    'to': "[data-element-handle='convert-to-selector']",
    'grid': '.gLLsql',
    'crypto_descritpion': ".cds-typographyResets-tjmople.cds-body-b5itfzw.cds-foreground-f1itepcl.cds-transition-ty35lnf.cds-start-s1q2d0t0.cds-truncate-t1hgsao8",
    'preview': '.isVEuC',
    'confirm': '.isVEuC',
    'consult': '.jwGeTR'
}
def try_(fcn):
    try:
        fcn
        return True
    except:
        return False

class Coinbase_Transaction_Scrapper:

    def __init__(self, first_connection=False):
        self.email = 'david.szmul@gmail.com'
        self.coinbase_enc = Coinbase_cryption()
        options = Options()
        options.headless = False
        options.add_argument("--window-size=1920,1080")

        self.driver = webdriver.Chrome(DRIVER_PATH, options=options)
        self.driver.get(COINBASE_SITE)
        time.sleep(1)
 
        if first_connection:
            self._login()
            self._save_cookies() # Save in order to avoid OAuth2 next times
        else:
            self._load_cookies()
            self._login()
        # self._wait(dict_id['accept_cookie']).click()

    def _save_cookies(self):
        tkinter.Tk().withdraw()
        tkinter.messagebox.showinfo('Authorization', 'Confirm OAuth2 and ask to wait 30 days before asking again\nDo you want to continue ?')        
        # Saving Cookie
        cookies = self.driver.get_cookies()
        with open(FILE_COOKIE, 'wb') as f:      
            pickle.dump(cookies, f)

    def _load_cookies(self):
        with open(FILE_COOKIE, 'rb') as f:      
            cookies = pickle.load(f)
        for c in cookies:
            if isinstance(c.get("expiry"), float):
                c["expiry"] = int(c["expiry"])
            self.driver.add_cookie(c)  

    def _login(self):
        self.driver.find_element_by_id("email").send_keys(self.email)
        self.driver.find_element_by_id("password").send_keys(self.coinbase_enc.decrypt_pswd())
        self.driver.find_element_by_id("stay_signed_in").click()
        self.driver.find_element_by_id("signin_button").click()

    def convert(self, from_: str, to_: str, amount):
        self._wait(dict_id['buy_sell'], idx_list=0).click()
        time.sleep(2) # High timing due to low internet')
        for c in self._wait(dict_id['convert'], unique_=False): # Multiple object with same properties, try all of them
            try:
                c.click()
                break
            except:
                continue
        time.sleep(2) # High timing due to low internet')
        self._wait(dict_id['amount'],idx_list=1).send_keys(amount)

        def click_on_crypto(crypto):
            NB_DIV_PER_CRYPTO = 3
            IDX_CRYPTO_NAME = 1
            MAX_DELAY = 5

            crypto_descritpions = self._wait(dict_id['crypto_descritpion'], unique_=False)

            # Wait while name is not empty (time for loading page)
            t_start = time.time()
            while True:
                if time.time()-t_start>=MAX_DELAY:
                    raise ConnectionError('Page is not loading to find crypto')
                if all([c.text!='' for c in crypto_descritpions]):
                    break

            
            
            # Check if name of crypto corresponds then return
            for c in crypto_descritpions:
                if c.text==crypto:
                    c.click()
                    time.sleep(0.5)
                    return 
            raise AssertionError('Crypto '+ crypto +' not found')

        # From
        self._wait(dict_id['from']).click()
        click_on_crypto(from_)

        # To
        self._wait(dict_id['to']).click()
        click_on_crypto(to_)

        # Previsualisation + Confirmation
        self._wait(dict_id['preview'], idx_list=2).click()
        time.sleep(2)
        self._wait(dict_id['confirm'], idx_list=2).click()
        self._wait(dict_id['consult'])
        
        # Display transaction
        print(f'Conversion of {amount}€ from {from_} to {to_} confirmed')
        self.driver.refresh()
        time.sleep(2) #Need to wait before doing 1rst transfer

    def _wait(self, css_selector, obj = None, idx_list=None, max_delay=10, unique_ = True):
        if obj is None:
            obj = self.driver

        t_start = time.time()
        while time.time()-t_start<max_delay:
            items = obj.find_elements_by_css_selector(css_selector)
            if len(items)==1:
                try:
                    if not unique_:
                        continue
                    return items[0]
                except:
                    continue
                return
            elif len(items)>1: # Selector must be unique
                if idx_list is None and unique_: # Unknown index in multiple list
                    raise AssertionError('Selector'+ css_selector+' is not unique: '+str(len(items))+' instances\n Need to add "idx_list" in parameter')

                elif not unique_:
                    return items
                elif len(items)>idx_list:
                    return items[idx_list]
        raise ConnectionError('Unable to find css selector: '+css_selector)

if __name__=="__main__":
    # First Connection
    autoSelect = Coinbase_Transaction_Scrapper(first_connection=False)
    time.sleep(4) #Need to wait before doing 1rst transfer
    conversion_done =  try_(autoSelect.convert('ETH', 'BTC', 5)) 
    # print(conversion_done)
    # conversion_done =  autoSelect.convert('BTC', 'ETH', 5)