import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Normalization, BatchNormalization


def AAN_Model(neurons_inp=50, neurons_h1=40, act_h1='relu', neurons_h2=8, act_h2='linear', neuron_out=3, act_out='linear'): 

    model = Sequential()

    model.add(Input(shape=neurons_inp,))
    model.add(Dense(neurons_h1, activation=act_h1))
    model.add(BatchNormalization())
    model.add(Dense(neurons_h2, activation=act_h2))
    model.add(BatchNormalization()) 
    model.add(Dense(neuron_out, activation=act_out))

    return model 