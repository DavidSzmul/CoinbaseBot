import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.optimizers import Adam

def test_GPU():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

class NetworkGenerator():
    # Model based on simple consecutive Dense Layer, automately adapted to the State/Action space
    def __init__(self):
        pass

    def create_SimpleModel(self, env, layers = [16, 16], learning_rate = 1e-3):

        model = Sequential()
        model.add(Dense(layers[0], activation='relu', input_shape=env.observation_space.shape))
        model.add(BatchNormalization())
        for layer in layers[1:]:
            model.add(Dense(layer, activation='relu'))
            model.add(BatchNormalization())

        ### Sigmoid between [0,1] for discrete values
        ### Tanh between [-1,1] for box values
        ### Linear for DQN
        model.add(Dense(env.action_space.n, activation='linear'))
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


