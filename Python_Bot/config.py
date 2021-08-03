import os

ROOT_DIR = os.path.dirname(__file__)                    # Main root directory
DATA_DIR = os.path.join(ROOT_DIR, 'data')               # Folder containing all data for the application
DRIVER_PATH = os.path.join(ROOT_DIR,"chromedriver.exe") # Path to the driver for scrapping

# Paths relating to scrapping of coinbase
COINBASE_SCRAPPER_ENV = 'COINBASE_PATH'
COINBASE_SCRAPPER_PATH = os.getenv(COINBASE_SCRAPPER_ENV)

if COINBASE_SCRAPPER_PATH is not None:
    COINBASE_SCRAPPER_KEY = COINBASE_SCRAPPER_PATH + 'KEY.txt'
    COINBASE_SCRAPPER_PSWD = COINBASE_SCRAPPER_PATH + 'PSWD.txt'
    COINBASE_SCRAPPER_API = COINBASE_SCRAPPER_PATH + 'API.txt'
    COINBASE_SCRAPPER_COOKIES = COINBASE_SCRAPPER_PATH + 'COOKIE.pkl'
else:
    COINBASE_SCRAPPER_KEY = ''
    COINBASE_SCRAPPER_PSWD = ''
    COINBASE_SCRAPPER_API = ''
    COINBASE_SCRAPPER_COOKIES = ''

def use_GPU_TF(flag_use_GPU: bool=True):
    '''Enforce the use of GPU or CPU for Tensorflow'''
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=["-1", "0"][flag_use_GPU]