import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Normalization


def AAN_Model(layer1_neurons = 200, layer2_neurons = 15, feature_dim = 200): 

    model = Sequential()
    
    # what is shape of: self.train_inp_r[1].__len__() ? 
    model.add(Dense(layer1_neurons, input_dim = feature_dim, activation= 'relu')) # 110 inputs in first layer, 7 neurons in hidden layer
    model.add(Dense(layer2_neurons, activation = 'relu')) #second hidden layer with x neurons /softplus, relu, linear
    model.add(Dense(1, activation = 'linear')) # 1 neuron in output layer

    return model 