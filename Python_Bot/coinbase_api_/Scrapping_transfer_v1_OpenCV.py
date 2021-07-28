import sys, os, time
import numpy as np
import pyautogui as pg
import keyboard
import cv2

path_template = 'Mouse_Keyboard\\template'
path_chrome = os.path.join(path_template, 'chrome')
path_coinbase = os.path.join(path_template, 'coinbase')
path_crypto = os.path.join(path_template, 'crypto')
coinbase_site = "https://www.coinbase.com/dashboard"

SCROLL_UP_POS = (1170, 327)
SCROLL_DOWN_POS = (1170, 898)
FINAL_POS = (1500, 400)

template_Selector = {
    'chrome': {
        'icon_chrome': os.path.join(path_chrome, 'icon_chrome.png'),
        'icon_search': os.path.join(path_chrome, 'icon_search.png')
    },
    'coinbase': {
        'buy': os.path.join(path_coinbase, 'BUY.png'),
        'convert': os.path.join(path_coinbase, 'CONVERT.png'),
        'from': os.path.join(path_coinbase, 'FROM.png'),
        'to': os.path.join(path_coinbase, 'TO.png'),
        'amount': os.path.join(path_coinbase, 'AMOUNT.png'),
        'select': os.path.join(path_coinbase, 'SELECT.png'),
        'preconfirm': os.path.join(path_coinbase, 'PRECONFIRM.png'),
        'confirm': os.path.join(path_coinbase, 'CONFIRM.png'),
        'trans_confirmed': os.path.join(path_coinbase, 'TRANS_CONFIRMED.png'),
    },
    'crypto': {
        'BTC': os.path.join(path_crypto, 'BTC.png'),
        'BTH': os.path.join(path_crypto, 'BTH.png'),
        'EOS': os.path.join(path_crypto, 'EOS.png'),
        'ETC': os.path.join(path_crypto, 'ETC.png'),
        'ETH': os.path.join(path_crypto, 'ETH.png'),
        'DGC': os.path.join(path_crypto, 'DGC.png'),
    }
}

class AutoSelector(object):
    def __init__(self):
        pass

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
    
    def _click_template(self, path_template, thr=0.99, max_delay=10):
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
    autoSelect = AutoSelector()
    autoSelect.open_coinbase()
    conversion_done =  autoSelect.new_conversion('DGC', 'ETH', 5)  # Dogecoin not implemented on coinbase
    print(conversion_done)