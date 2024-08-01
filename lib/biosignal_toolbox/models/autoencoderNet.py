import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, RepeatVector, SimpleRNN
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Activation, UpSampling2D, Layer, Conv1D, Conv2DTranspose, Reshape
from tensorflow.keras.layers import Lambda, LSTM, Reshape, TimeDistributed, Flatten, Dense, Input
import keras.backend as K
from tensorflow.keras import initializers
from scipy import signal
import numpy as np 
from keras import layers
import copy 
import matplotlib.pyplot as plt

def Autoencoder_Net(Chans = 34, Samples =512, ks1= 7, ks2 = 5, ks3 = 3, F = 16): 
    
    # Autoencoder setup
    #**************************
    #********Encoder***********
    #**************************
    # first block 
    model = Sequential()
    model.add(Conv2D(F, kernel_size = (1, ks1), padding = 'same', input_shape = (Chans, Samples, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(F, kernel_size = (1, ks1), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1, 2))) 
    # second block 
    model.add(Conv2D(F, kernel_size = (1, ks1), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(F, kernel_size = (1, ks1), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1, 2))) 
    # third block 
    model.add(Conv2D(F, kernel_size = (1, ks2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(F, kernel_size = (1, ks2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1, 2))) 
    # fourth block 
    model.add(Conv2D(F, kernel_size = (1, ks3), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(F, kernel_size = (1, ks3), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    #**************************
    #********Decoder***********
    #**************************
    # first block 
    model.add(UpSampling2D(size=(1, 2))) 
    model.add(Conv2D(F, kernel_size = (1, ks3), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(F, kernel_size = (1, ks3), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # second block 
    model.add(UpSampling2D(size=(1, 2))) 
    model.add(Conv2D(F, kernel_size = (1, ks3), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(F, kernel_size = (1, ks3), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # third block 
    model.add(UpSampling2D(size=(1, 2))) 
    model.add(Conv2D(F, kernel_size = (1, ks3), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(F, kernel_size = (1, ks3), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # fourth block 
    model.add(Conv2D(F, kernel_size = (1, ks3), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(1, kernel_size = (1, ks3), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    return model 


class AddArrayLayer(Layer):
    def __init__(self, array_initializer='ones', **kwargs):
        super(AddArrayLayer, self).__init__(**kwargs)
        self.array_initializer = tf.keras.initializers.get(array_initializer)

    def build(self, input_shape):
        self.my_array = self.add_weight(name='my_array',shape=(input_shape[-1],), initializer=self.array_initializer, trainable=True)
        super(AddArrayLayer, self).build(input_shape)

    def call(self, inputs):
        return inputs * self.my_array

    def compute_output_shape(self, input_shape):
        return input_shape


class CustomScalingLayer(tf.keras.layers.Layer):
    def __init__(self, Samples, **kwargs):
        super(CustomScalingLayer, self).__init__(**kwargs)
        self.Samples = Samples

    def build(self, input_shape):
        self.offset_factors = self.add_weight(name='offset_factors',shape=(self.Samples), initializer='zeros', trainable=True)
        self.scale_factors = self.add_weight(name='scale_factors',shape=(self.Samples), initializer='ones', trainable=True)
        super(CustomScalingLayer, self).build(input_shape)

    def call(self, inputs):
        #print("type of input", inputs.shape)
        #print("shape of scale factors: ", tf.reshape(self.scale_factors, [1, 1, -1]))
        return (inputs *  tf.reshape(self.scale_factors, [1, 1, -1])) + tf.reshape(self.offset_factors, [1, 1, -1]) # scale the final outputs 

def toUV(x): 
    return x*1000000

def toVolts(x): 
    return x/1000000

# Define a custom constraint class
class NormalizeWeightsConstraint(tf.keras.constraints.Constraint):
    def __call__(self, w):
        return w / tf.reduce_sum(w)


def FilterNet(Chans = 34, kernel = 200, kernel1 = 10, kernel2 = 5, kernel3 = 100, Samples =1024, F = 4, F1 = 16): 
    
    # Autoencoder setup
    #**************************
    #********Encoder***********
    #**************************
    #offset_value = tf.Variable(initial_value=0.01, trainable=True) # use offset to be trained later 
    
    # first block 
    model = Sequential() # linear best 
    model.add(Conv2D(F, kernel_size = (1, kernel1), padding = 'same', input_shape = (Chans, Samples, 1), use_bias = False))
    model.add(Activation('linear'))
    model.add(Lambda(toUV)) # ensure that the calculations are in range 
    model.add(Conv2D(F, kernel_size = (1, kernel2), padding = 'same', use_bias = False))#, kernel_constraint=NormalizeWeightsConstraint()))
    model.add(Activation('linear'))
    model.add(AveragePooling2D(pool_size=(1, 2))) 
    # model.add(Lambda(lambda x: x *100))
    model.add(Conv2D(F, kernel_size = (1, kernel3), padding = 'same', use_bias = False))#, kernel_constraint=NormalizeWeightsConstraint()))
    model.add(Activation('linear'))
    model.add(AveragePooling2D(pool_size=(1, 2))) 
    # maximal pooled here 
    model.add(Conv2D(F1, kernel_size = (1, kernel), padding = 'same', use_bias = False))#, kernel_constraint=NormalizeWeightsConstraint()))
    model.add(Activation('linear'))
    model.add(UpSampling2D(size=(1, 2))) 
    model.add(Conv2D(F, kernel_size = (1, kernel3), padding = 'same', use_bias = False))#, kernel_constraint=NormalizeWeightsConstraint()))
    model.add(Activation('linear'))
    model.add(UpSampling2D(size=(1, 2))) 
    model.add(Conv2D(1, kernel_size = (1, kernel1), padding = 'same', use_bias = False))#, kernel_constraint=NormalizeWeightsConstraint()))
    model.add(Lambda(toVolts))
    # # extra custom layer
    # model.add(Reshape((Chans, Samples)))  # reshape input 
    # model.add(CustomScalingLayer(Samples)) 
    # model.add(Reshape((Chans, Samples, 1)))  # reshape input
    # model.add(Conv2D(F, kernel_size = (1, kernel3), padding = 'same', use_bias = False))
    # model.add(Activation('linear'))
    # model.add(Conv2D(1, kernel_size = (1, 1), padding = 'same', use_bias = False)) 
    # model.add(Lambda(toVolts))
    

    #model.add(TimeDistributed(AddArrayLayer())) #layer wise offset correction at the end
    
    return model 


def FilterNetRNN(Chans=34, kernel=100, Samples=512, F=8):
    # Input layer
        # Input layer
    inputs = Input(shape=(Chans, Samples, 1))

    # Convolutional layers
    x = Conv2D(F, kernel_size=(1, kernel), padding='same', use_bias=False)(inputs)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=(1, 2))(x)

    x = Conv2D(F, kernel_size=(1, kernel), padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=(1, 2))(x)

    x = Conv2D(F, kernel_size=(1, kernel), padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = UpSampling2D(size=(1, 2))(x)

    x = Conv2D(F, kernel_size=(1, kernel), padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = UpSampling2D(size=(1, 2))(x)

    x = Conv2D(F, kernel_size=(1, kernel), padding='same', use_bias=False)(x)
    x = Activation('relu')(x)

    # Reshape for SimpleRNN
    x = Reshape((Chans, -1))(x)

    # Add SimpleRNN layer
    rnn_units = 64  # Adjust the number of units according to your preference
    x = SimpleRNN(rnn_units, activation='relu', return_sequences=True)(x)

    # Reshape back to the 4D shape expected by the subsequent convolutional layer
    x = Reshape((Chans, Samples // 4, rnn_units))(x)

    # Final Conv2D layer with 1 filter
    outputs = Conv2D(1, kernel_size=(1, kernel), padding='same', use_bias=False)(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# def LSTMFilter(Chans=34, Samples=512):
#     # Input layer
#     input_shape = (Chans, Samples, 1)
#     inputs = Input(shape=input_shape)
#     model = Sequential()
#     model.add(LSTM(100, activation='relu', input_shape=input_shape))
#     model.add(TimeDistributed(Dense(1)))
 
#     return model

def Conv2D2KernelLayer(Chans = 34, kernel =150, Samples =512, F = 4): 
    
    b, a = signal.iirfilter(2, [0.5, 4.0], btype='bandpass', ftype='butter', output='ba', fs=500)
    impulse_response = signal.impulse((b, a), N = kernel)[1]
    init = np.zeros((F, kernel))
    for i in range(0, F): 
        init[i, :]= impulse_response/np.sum(impulse_response)

    print(init.shape)
    
    custom_initializer = tf.constant_initializer(init)
    plt.figure()
    plt.plot(impulse_response/np.sum(impulse_response))
    plt.show()

    # first block 
    model = Sequential() # linear best 
    model.add(Conv2D(F, kernel_size = (1, kernel), padding = 'same', input_shape = (Chans, Samples, 1), use_bias = False, kernel_initializer=custom_initializer))#custom_initializer))
    model.add(Activation('linear'))
    model.add(Conv2D(F, kernel_size = (1, kernel), padding = 'same', use_bias = False))#custom_initializer))
    model.add(Activation('linear'))
    #model.add(Conv2DWithTwoKernels(F, kernel_size = (1, kernel), padding = 'same', input_shape = (Chans, Samples, 1), use_bias = False, kernel_initializer=custom_initializer))#custom_initializer))
    model.add(Conv2D(1, kernel_size = (1, kernel), padding = 'same', use_bias = False)) # offset correction 

    return model 

def MLPFilter(Chans = 34, Samples =512): 

    #layers.TimeDistributed(layer)

    # first block 
    model = Sequential() # linear best 
    model.add(Input(shape = (Chans, Samples, 1)))
    model.add(layers.TimeDistributed((Dense(units=32, use_bias=False))))#custom_initializer))
    model.add(Activation('linear'))
    model.add(BatchNormalization())
    model.add(layers.TimeDistributed((Dense(units=4, use_bias=False))))#custom_initializer))
    model.add(Activation('linear'))
    model.add(BatchNormalization())
    model.add(layers.TimeDistributed((Dense(units=32, use_bias=False))))#custom_initializer))
    model.add(Activation('linear'))
    model.add(BatchNormalization())
    # model.add(BatchNormalization())
    model.add(layers.TimeDistributed((Dense(units=Samples, use_bias=False))))
    return model 