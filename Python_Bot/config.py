import os

ROOT_DIR = os.path.dirname(__file__)                    # Main root directory
DATA_DIR = os.path.join(ROOT_DIR, 'data')               # Folder containing all data for the application
DRIVER_PATH = os.path.join(ROOT_DIR,"chromedriver.exe") # Path to the driver for scrapping


def use_GPU_TF(flag_use_GPU: bool=True):
    '''Enforce the use of GPU or CPU for Tensorflow'''
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=["-1", "0"][flag_use_GPU]