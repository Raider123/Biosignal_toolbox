import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout
from tensorflow.keras import regularizers


def AAN_Model(neurons_inp=128, neurons_h1=128, act_h1='relu', neurons_h2=64, act_h2='relu', neurons_h3=32,  act_h3='relu', neurons_h4=32,  act_h4='relu',neuron_out=3, act_out='linear'): 

    model = Sequential()

    model.add(Input(shape=neurons_inp,))
    model.add(Dense(neurons_h1, activation=act_h1, kernel_regularizer=regularizers.l2(1e-3)))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    if neurons_h2:
        model.add(Dense(neurons_h2, activation=act_h2, kernel_regularizer=regularizers.l2(1e-4)))
        model.add(Dropout(0.1))
        model.add(BatchNormalization()) 
    if neurons_h3:
        model.add(Dense(neurons_h3, activation=act_h3, kernel_regularizer=regularizers.l2(1e-4)))
        model.add(Dropout(0.1))
        model.add(BatchNormalization())
    if neurons_h4:
        model.add(Dense(neurons_h4, activation=act_h4, kernel_regularizer=regularizers.l2(1e-5)))
        model.add(Dropout(0.1))
        model.add(BatchNormalization())
    model.add(Dense(neuron_out, activation=act_out, bias_initializer='zeros'))

    return model 


def Autoencoder():
    input_dim = 400 #3168
    bottleneck_dim = 100
    output_dim = input_dim
    #! ***** Encoder ***** !#
    input_layer = Input(shape=(input_dim,))
    encoder_layer = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(input_layer)
    encoder_layer = Dropout(0.3)(encoder_layer)
    encoder_layer = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(input_layer)
    encoder_layer = Dropout(0.3)(encoder_layer)
    # encoder_layer = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(encoder_layer)
    # encoder_layer = Dropout(0.3)(encoder_layer)
    encoded_layer = Dense(bottleneck_dim, activation='relu')(encoder_layer)

    #! ***** Decoder ***** !#
    decoder_layer = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(encoded_layer)
    decoder_layer = Dropout(0.3)(decoder_layer)
    encoder_layer = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(input_layer)
    encoder_layer = Dropout(0.3)(encoder_layer)
    # decoder_layer = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(decoder_layer)
    # decoder_layer = Dropout(0.3)(decoder_layer)
    decoded_layer = Dense(output_dim, activation='relu')(decoder_layer)

    #! ***** Model Definition ***** !#
    autoencoder_model = Model(inputs=input_layer, outputs=decoded_layer, name='autoencoder_model')
    encoder_model = Model(inputs=input_layer, outputs=encoded_layer, name='encoder_model')

    return autoencoder_model, encoder_model
