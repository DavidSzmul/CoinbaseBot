import sys, os, time
import numpy as np
import pickle
import tkinter.messagebox
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from Coinbase_API.encryption import Coinbase_cryption


COINBASE_SITE = "https://www.coinbase.com/dashboard"
FILE_COOKIE = os.getenv('COINBASE_PATH')+'COOKIE.pkl'

dict_id = {
    'accept_cookie': '.sc-AxjAm',
    'buy_sell': '.kMuznm',
    'convert': "[data-element-handle='folder-tab-convert']",
    'ammount': '.ivDhsu',
    'from': "[data-element-handle='convert-from-selector']",
    'to': "[data-element-handle='convert-to-selector']",
    'grid': '.dpKzAY',
    'preview': '.isVEuC',
    'confirm': '.isVEuC',
    'consult': '.jwGeTR'
}

class AutoSelector(object):
    def __init__(self, first_connection=False):
        self.email = 'david.szmul@gmail.com'
        self.coinbase_enc = Coinbase_cryption()
        DRIVER_PATH = os.path.join(os.getcwd(),"chromedriver.exe")
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
        time.sleep(2) #Need to wait before doing 1rst transfer

    def convert(self, from_, to_, ammount):
        try:
            self._wait(dict_id['buy_sell'], idx_list=0).click()
            time.sleep(1)
            for c in self._wait(dict_id['convert'], unique_=False): # Multiple object with same properties, try all of them
                try:
                    c.click()
                    break
                except:
                    continue
            self._wait(dict_id['ammount'],idx_list=1).send_keys(ammount)

            def click_on_crypto(crypto):
                NB_DIV_PER_CRYPTO = 3
                IDX_CRYPTO_NAME = 1
                MAX_DELAY = 5
                grid = self._wait(dict_id['grid'])
                divList = self._wait('div', obj=grid, unique_=False) 
                childs_crypto = [divList[i] for i in range(len(divList)) if i%NB_DIV_PER_CRYPTO==IDX_CRYPTO_NAME] # Child containing name of crypto
                
                # Wait while name is not empty (time for loading page)
                t_start = time.time()
                while True:
                    if time.time()-t_start>=MAX_DELAY:
                        raise ConnectionError('Page is not loading to find crypto')
                    if all([c.find_element_by_tag_name('p').text!='' for c in childs_crypto]):
                        break

                # Check if name of crypto corresponds then return
                for c in childs_crypto:
                    if c.find_element_by_tag_name('p').text==crypto:
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
            print(f'Conversion of {ammount}€ from {from_} to {to_} confirmed')
            self.driver.refresh()
            time.sleep(2) #Need to wait before doing 1rst transfer
            return True
        except : # Problem during conversion
            return False

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
    autoSelect = AutoSelector(first_connection=False)
    conversion_done =  autoSelect.convert('ETH', 'BTC', 5)  # Dogecoin not implemented on coinbase
    conversion_done =  autoSelect.convert('BTC', 'ETH', 5)  # Dogecoin not implemented on coinbase
    END = input('END')
    # autoSelect.open_coinbase()
    # print(conversion_done)