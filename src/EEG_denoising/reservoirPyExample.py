import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge, Input, Concat, ESN
import numpy as np
import matplotlib.pyplot as plt

# ******************************
# ********* user params ********
# ******************************


reservoir_size = 1000
leak_rate = 0.5 # 0.5 
spectral_radius = 0.4 #0.9 


# ******************************
# ********* main code **********
# ******************************
rpy.verbosity(0)  # no need to be too verbose here
#rpy.set_seed(42)  # make everything reproducible!

# create a reservoir: 
# We will first create a reservoir for our ESN, with 100 neurons. Reservoirs can be created using the Reservoir class.

# We change the value of two hyperparameters: - lr: the leaking rate, which controls the time constant of the neurons;
# - sr: the spectral radius of the recurrent connections in the reservoir. It controls the chaoticity of the reservoir dynamics. 
reservoir = Reservoir(reservoir_size, lr=leak_rate, sr=spectral_radius)
readout = Ridge(ridge=1e-12) # ridge regressor readout
data = Input()
concatenate = Concat()

reservoir <<= readout 

esn_model = data >> readout

#esn_model = [data, data >> reservoir, data >> reservoir] >> concatenate >> readout
# create an example sine signal 

# X = np.sin(np.linspace(0, 6*np.pi, 100)).reshape(-1, 1) +np.sin(np.linspace(0, 6*np.pi, 100)).reshape(-1, 1) +np.sin(np.linspace(0, 0.1*np.pi, 100)).reshape(-1, 1) +1
# Y = np.sin(np.linspace(0, 6*np.pi, 100)).reshape(-1, 1)
# print(f"shape of X: {X.shape}")

# load real EEG train data 
X_EEG = np.load("X.npy", allow_pickle=False) # trials, channels, sampels , windows 
Y_EEG = np.load("Y.npy", allow_pickle=False)

X_EEG_test = np.load("X_test.npy", allow_pickle=False)
Y_EEG_test = np.load("Y_test.npy", allow_pickle=False)

print(type(X_EEG))
print(X_EEG.shape)

example_trial = 20
example_channel = 7
# X_train = np.expand_dims(X_EEG[example_trial, example_channel, :, 0], axis = 1) 
# Y_train = np.expand_dims(Y_EEG[example_trial, example_channel, :, 0], axis= 1) 
# X_test = np.expand_dims(X_EEG[example_trial+1, example_channel, :, 0], axis= 1) 
# Y_test = np.expand_dims(Y_EEG[example_trial+1, example_channel, :, 0], axis= 1) 

X_train = np.expand_dims(X_EEG, axis = 1) 
Y_train = np.expand_dims(Y_EEG, axis= 1) 
X_test = np.expand_dims(X_EEG_test, axis= 1) 
Y_test = np.expand_dims(Y_EEG_test, axis= 1) 

print("train shape x: ", X_train.shape) 
X_train_new = X_train[0:500, :]
Y_train_new = X_train[1:501, :]

# train data
# X_train = X[0:50]
# Y_train = Y[0:50]
# X_test =  X[51:]
# Y_test =  Y[51:]

# train model 
esn_model = esn_model.fit(X_train_new, Y_train_new, warmup=10)
# print flags for checking 
print(reservoir.is_initialized, readout.is_initialized, readout.fitted)


# predict on new samples 
Y_pred = esn_model.run(X_train_new)

plt.figure(figsize=(10, 3))
plt.title("A sine wave and its future.")
plt.xlabel("$t$")
plt.plot(Y_pred, label="Y predict", color="purple")
plt.plot(X_train_new, label="X", color="blue")
plt.plot(Y_train_new, label="Y target", color="red")
plt.legend()

plt.show()


# # predict on new samples 
# Y_pred = esn_model.run(X_test)


# plt.figure(figsize=(10, 3))
# plt.title("A sine wave and its future.")
# plt.xlabel("$t$")
# plt.plot(Y_pred, label="Y predict", color="purple")
# plt.plot(Y_test, label="Y_test (target)", color="red")
# plt.plot(X_test, label="X_test", color="blue")
# plt.legend()
# plt.show()

