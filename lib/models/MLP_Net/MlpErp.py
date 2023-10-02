import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Normalization


def MLP_Model(x_train, leaky_alpha = 0.5, first_layer_units = 64, second_layer_units = 32, third_layer_units = 10, activation = "sigmoid", n_classes = 2, dropout_rate = 0.5, use_norm_layer = False): # 8 8 8 0.2 drop

    # MLP setup
    model = Sequential()
    if(use_norm_layer): 
        norm_layer = Normalization()
        norm_layer.adapt(x_train)
        model.add(norm_layer)

    model.add(BatchNormalization())
    model.add(Dense(units=first_layer_units, input_shape=(x_train.shape[1],)))
    model.add(tf.keras.layers.LeakyReLU(alpha=leaky_alpha))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(Dense(units=second_layer_units))
    model.add(tf.keras.layers.LeakyReLU(alpha=leaky_alpha))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(Dense(units=third_layer_units)) 
    model.add(tf.keras.layers.LeakyReLU(alpha=leaky_alpha))
    model.add(Dense(units=n_classes-1, activation=activation))

    return model 



