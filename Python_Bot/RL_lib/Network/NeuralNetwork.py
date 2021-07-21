from typing import List
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def test_GPU():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

class NetworkGenerator(object):
    """Class generating neural network for specific agents"""
    # Model based on simple consecutive Dense Layer, automately adapted to the State/Action space

    def create_DQN_Model(self, state_shape: np.ndarray, action_shape: np.ndarray, 
                            layers: List[int] = [16, 16], learning_rate: float = 1e-3):

        model = Sequential()
        model.add(Dense(layers[0], activation='relu', input_shape=state_shape))
        for layer in layers[1:]:
            model.add(Dense(layer, activation='relu'))
        model.add(Dense(action_shape, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
        return model

if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    flag_use_GPU = True
    os.environ["CUDA_VISIBLE_DEVICES"]=["-1", "0"][flag_use_GPU]

    Generator = NetworkGenerator()
    test_GPU()
    model = Sequential()
    # model = Generator.create_SimpleModel()


