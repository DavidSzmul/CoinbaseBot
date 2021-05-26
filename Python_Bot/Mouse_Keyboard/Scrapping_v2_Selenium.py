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
    'buy_sell': '.cnRbMa',
    'convert': '.gfOswQ',
    'ammount': '.ivDhsu',
    'from': '.kBxaMV',
    'to': '.bcFGxw',
    'grid': '.dpKzAY',
    'preview': '.isVEuC',
    'confirm': '.cnRbMa',
    'consult': 'dmyFvF'


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
        self._wait(dict_id['accept_cookie']).click()
        # self.driver.find_element_by_css_selector(dict_id['accept_cookie']).click()

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

    def convert(self, from_, to_, ammount):
        self._wait(dict_id['buy_sell']).click()
        self._wait(dict_id['convert']).click()
        self._wait(dict_id['ammount']).send_keys(ammount)

        def click_on_crypto(grid, crypto):
            divList = grid.find_elements_by_tag_name("div")
            print(divList)

        # From
        self._wait(dict_id['from']).click()
        grid = self._wait(dict_id['grid'])
        click_on_crypto(grid, from_)

         # To
        self._wait(dict_id['to']).click()
        grid = self._wait(dict_id['grid'])
        click_on_crypto(grid, to_)
        

    def _find_template(self, path_template, thr=0.99, verbose=False):
        screen = pg.screenshot()
        main_image = np.array(screen)[:, :, ::-1].copy()
        gray_image = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
        template = cv2.imread(path_template, 0)
        if template is None:
            raise AssertionError('Template is not found')
        width, height = template.shape[::-1] #get the width and height
        match = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
        max_match = np.max(match)
        position = np.where(match >= thr) #get the location of template in the image

        points = list(zip(*position[::-1]))
        if len(points)==0:
            return None, max_match
        if len(points)>1:
            raise AssertionError('Multiple template detected, not robust')

        point = (points[0][0] + width/2, points[0][1] + height/2)
        if not verbose:
            return point, max_match
        return point, max_match, (width, height)

    def _wait_confirmation_template(self, path_template, thr=0.99, max_delay=5):
        point = None
        start_time = time.time()
        while point is None:
            point, max_match = self._find_template(path_template, thr=thr)
            print(max_match)
            if time.time() - start_time > max_delay:
                raise AssertionError('Time waiting template elapsed')
        return point
    
    def _wait(self, css_selector, max_delay=10):
        t_start = time.time()
        while time.time()-t_start<max_delay:
            items = self.driver.find_elements_by_css_selector(css_selector)
            if len(items)==1:
                try:
                    return items[0]
                except:
                    continue
                return
            elif len(items)>1: # Selector must be unique
                raise AssertionError('Selector'+ css_selector+' is not unique: '+str(len(items))+' instances')
        raise ConnectionError('Unable to find or click css selector: '+css_selector)

        point = self._wait_confirmation_template(path_template, thr=thr, max_delay=max_delay)
        pg.click(point[0], point[1])

    def _click_crypto(self, crypto, thr=0.995, delay_scroll=1e-1, nb_try=1, nb_click_to_finish=15):

        ctr_click_to_finish = 0
        self._wait_confirmation_template(template_Selector['coinbase']['select'])
        while True:
            time.sleep(delay_scroll)
            point, _ = self._find_template(template_Selector['crypto'][crypto], thr=thr)
            if point is not None:
                break
            if int(ctr_click_to_finish/nb_click_to_finish)%2 == 0:
                pg.click(SCROLL_DOWN_POS[0], SCROLL_DOWN_POS[1])
            else:
                pg.click(SCROLL_UP_POS[0], SCROLL_UP_POS[1])
            ctr_click_to_finish+=1

            if ctr_click_to_finish > 2*nb_try*nb_click_to_finish:
                raise AssertionError('Impossible to find demanded crypto')
        pg.click(point[0], point[1])     

    def open_coinbase(self):
        # Click icon
        self._click_template(template_Selector['chrome']['icon_chrome'])
        # Wait load
        self._wait_confirmation_template(template_Selector['chrome']['icon_search'])
        # Type coinbase
        keyboard.write(coinbase_site)
        pg.typewrite(["enter"])

    def new_conversion(self, _from, _to, amount):
        """ Amount in Euro """
        try:
            # Goto conversion
            self._click_template(template_Selector['coinbase']['buy'])
            self._click_template(template_Selector['coinbase']['convert'])
            # Write Amount
            self._click_template(template_Selector['coinbase']['amount'])
            keyboard.write(str(amount))
            # Search crypto from/to
            self._click_template(template_Selector['coinbase']['from'])
            self._click_crypto(_from)
            self._click_template(template_Selector['coinbase']['to'])
            self._click_crypto(_to)
            # Confirm transaction
            self._click_template(template_Selector['coinbase']['preconfirm'], thr=0.97)
            self._click_template(template_Selector['coinbase']['confirm'], thr=0.97)
            # Check validation
            self._wait_confirmation_template(template_Selector['coinbase']['trans_confirmed'])
        except  AssertionError as error:
            print(error)
            return False
        pg.click(point[0], point[1])
        return True

if __name__=="__main__":
    # First Connection
    autoSelect = AutoSelector(first_connection=False)
    conversion_done =  autoSelect.convert('ETH', 'BTC', 5)  # Dogecoin not implemented on coinbase
    END = input('END')
    # autoSelect.open_coinbase()
    # print(conversion_done)