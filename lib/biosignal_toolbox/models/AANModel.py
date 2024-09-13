import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Normalization


def AAN_Model(neurons_inp=160, act_inp='relu', neurons_h1=40, act_h1='relu', neurons_h2=8, act_h2='linear'): 

    model = Sequential()
    
    # what is shape of: self.train_inp_r[1].__len__() ? 
    model.add(Dense(neurons_inp, input_dim=neurons_inp, activation=act_inp)) # 110 inputs in first layer, 7 neurons in hidden layer
    model.add(Dense(neurons_h1, activation=act_h1)) #second hidden layer with x neurons /softplus, relu, linear
    if neurons_h2 >0:
        model.add(Dense(neurons_h2, activation=act_h2)) 
    model.add(Dense(1, activation='linear')) # 1 neuron in output layer

    return model 