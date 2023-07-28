import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense


def MLP_Model(x_train, leaky_alpha = 0.5, first_layer_units = 8, second_layer_units = 8, third_layer_units = 8, activation = "sigmoid", n_classes = 2, dropout_rate = 0.2): 

    # MLP setup
    model = Sequential()
    model.add(Dense(units=first_layer_units, input_shape=(x_train.shape[1],)))
    model.add(tf.keras.layers.LeakyReLU(alpha=leaky_alpha))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=second_layer_units))
    model.add(tf.keras.layers.LeakyReLU(alpha=leaky_alpha))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=third_layer_units)) 
    model.add(tf.keras.layers.LeakyReLU(alpha=leaky_alpha))
    model.add(Dense(units=n_classes-1, activation=activation))

    return model 



