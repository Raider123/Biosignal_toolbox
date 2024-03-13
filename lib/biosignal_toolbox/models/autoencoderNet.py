import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Activation, UpSampling2D, Layer, Conv1D
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



def FilterNet(Chans = 34, kernel = 100, Samples =512, F = 8): 
    
    # Autoencoder setup
    #**************************
    #********Encoder***********
    #**************************
    # first block 
    model = Sequential() # leaky_relu best 
    model.add(Conv2D(F, kernel_size = (1, kernel), padding = 'same', input_shape = (Chans, Samples, 1), use_bias = False))
    model.add(Activation('leaky_relu'))
    model.add(AveragePooling2D(pool_size=(1, 2))) 
    model.add(Conv2D(F, kernel_size = (1, kernel), padding = 'same', use_bias = False))
    model.add(Activation('leaky_relu'))
    model.add(AveragePooling2D(pool_size=(1, 2))) 
    model.add(Conv2D(F, kernel_size = (1, kernel), padding = 'same', use_bias = False))
    model.add(Activation('leaky_relu'))
    model.add(UpSampling2D(size=(1, 2))) 
    model.add(Conv2D(F, kernel_size = (1, kernel), padding = 'same', use_bias = False))
    model.add(Activation('leaky_relu'))
    model.add(UpSampling2D(size=(1, 2))) 
    model.add(Conv2D(F, kernel_size = (1, kernel), padding = 'same', use_bias = False))
    model.add(Activation('leaky_relu'))
    model.add(Conv2D(1, kernel_size = (1, kernel), padding = 'same', use_bias = False)) # offset correction 

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


def FilterNetV2(Chans = 34, kernel = 100, Samples =512, F = 8): 
    
    # Autoencoder setup
    #**************************
    #********Encoder***********
    #**************************
    #offset_value = tf.Variable(initial_value=0.01, trainable=True) # use offset to be trained later 

    # first block 
    model = Sequential() # leaky_relu best 
    model.add(Conv2D(F, kernel_size = (1, kernel), padding = 'same', input_shape = (Chans, Samples, 1), use_bias = False))
    model.add(Activation('leaky_relu'))
    model.add(AveragePooling2D(pool_size=(1, 2))) 
    model.add(Conv2D(F, kernel_size = (1, kernel), padding = 'same', use_bias = False))
    model.add(Activation('leaky_relu'))
    model.add(AveragePooling2D(pool_size=(1, 2))) 
    model.add(Conv2D(F, kernel_size = (1, kernel), padding = 'same', use_bias = False))
    model.add(Activation('leaky_relu'))
    model.add(UpSampling2D(size=(1, 2))) 
    model.add(Conv2D(F, kernel_size = (1, kernel), padding = 'same', use_bias = False))
    model.add(Activation('leaky_relu'))
    model.add(UpSampling2D(size=(1, 2))) 
    model.add(Conv2D(F, kernel_size = (1, kernel), padding = 'same', use_bias = False))
    model.add(Activation('leaky_relu'))
    model.add(Conv2D(1, kernel_size = (1, kernel), padding = 'same', use_bias = False)) 
    model.add(TimeDistributed(AddArrayLayer())) #layer wise offset correction at the end
    
    return model 

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
    model = Sequential() # leaky_relu best 
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
    model = Sequential() # leaky_relu best 
    model.add(Input(shape = (Chans, Samples, 1)))
    model.add(layers.TimeDistributed((Dense(units=32, use_bias=False))))#custom_initializer))
    model.add(Activation('leaky_relu'))
    model.add(BatchNormalization())
    model.add(layers.TimeDistributed((Dense(units=4, use_bias=False))))#custom_initializer))
    model.add(Activation('leaky_relu'))
    model.add(BatchNormalization())
    model.add(layers.TimeDistributed((Dense(units=32, use_bias=False))))#custom_initializer))
    model.add(Activation('leaky_relu'))
    model.add(BatchNormalization())
    # model.add(BatchNormalization())
    model.add(layers.TimeDistributed((Dense(units=Samples, use_bias=False))))
    return model





# # if you want to write your own one 
# class IdentityLayer(Layer):
#     def __init__(self, **kwargs):
#         super(IdentityLayer, self).__init__(**kwargs)

#     def call(self, inputs):
#         return inputs

#     def compute_output_shape(self, input_shape):
#         return input_shape

# def ExampleLayer(Chans = 34, Samples =512): 

#     model = Sequential() # leaky_relu best 
#     model.add(IdentityLayer(input_shape=(Chans, Samples, 1)))
#     return model



# class NumpyLayer(Layer):
#     def __init__(self, **kwargs):
#         super(NumpyLayer, self).__init__(**kwargs)
        

#     def call(self, inputs):
#         print("shape tensor", tf.shape(inputs))
#         print("is tensor", tf.is_tensor(inputs))
#         tf.compat.v1.disable_eager_execution()

#         with tf.compat.v1.Session() as sess:
#             tensor_value = sess.run(inputs)

#         print("numpy arr", tensor_value.shape)

#         outputs = inputs 
        
#         return outputs

#     def compute_output_shape(self, input_shape):
#         return input_shape


# def SimpleNumpyLayer(Chans = 34, Samples =512): 

#     model = Sequential() # leaky_relu best 
#     model.add(Input(shape = (Chans, Samples, 1)))
#     model.add(NumpyLayer())
#     return model


# class Conv2DWithTwoKernels(layers.Layer):
#     def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs):
#         super(Conv2DWithTwoKernels, self).__init__()
#         self.filters = filters
#         self.kernel_size = kernel_size
#         self.strides = strides
#         self.padding = padding
#         self.data_format = data_format
#         self.dilation_rate = dilation_rate
#         self.activation = activation
#         self.use_bias = use_bias
#         self.kernel_initializer = kernel_initializer
#         self.bias_initializer = bias_initializer
#         self.kernel_regularizer = kernel_regularizer
#         self.bias_regularizer = bias_regularizer
#         self.activity_regularizer = activity_regularizer
#         self.kernel_constraint = kernel_constraint
#         self.bias_constraint = bias_constraint
#         self.conv1 = layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, data_format=self.data_format, dilation_rate=self.dilation_rate, activation=self.activation, use_bias=self.use_bias, kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer, kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer, kernel_constraint=self.kernel_constraint, bias_constraint=self.bias_constraint, **kwargs)
#         self.conv2 = layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, data_format=self.data_format, dilation_rate=self.dilation_rate, activation=self.activation, use_bias=self.use_bias, kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer, kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer, kernel_constraint=self.kernel_constraint, bias_constraint=self.bias_constraint, **kwargs)

#     def call(self, inputs):

#         out = self.conv1(inputs)
#         return out #/ self.conv2(out)