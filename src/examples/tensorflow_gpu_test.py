import sys 
import tensorflow as tf 
import tensorflow.keras as ks

print("Tensorflow version: ", tf.__version__)
print("Keras version: ", ks.__version__)
print("Python version: ", sys.version) 

gpu = len(tf.config.list_physical_devices('GPU'))>0 
print("GPU is", "available" if gpu else "NOT AVAILABLE")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#is_cuda_gpu_available = tf.test.is_gpu_available(cuda_only=True)
#print("")
print(is_cuda_gpu_available)

