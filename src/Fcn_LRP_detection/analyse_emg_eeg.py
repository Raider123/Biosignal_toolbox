
# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from time import perf_counter
import sys


# own libs
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/mne_machine_learning"
sys.path.append(proj_path+"/lib") # path to lib folder
import eeg_lib
import emg_lib

data_path = proj_path+"/data/"


# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************

eeg_file = data_path+"JV43_intentional_unilateral_34ch_05_4Hz_train_2.npy"
emg_file = data_path+"JV43_intentional_unilateral_emg_train_2.npy"
time_axis_file=data_path+ "time_axis_eeg_epochs.npy"
channel_names_file_eeg = data_path+"remaining_eeg_channel_names.npy"

# *********************************************************************************
# ***************** Load train, test, val sets for every iteration ****************
# *********************************************************************************

# load each individual train, val and test sets (preprocessed)
train_eeg = np.load(eeg_file)
train_emg = np.load(emg_file)
time_axis = np.load(time_axis_file)
eeg_channels = np.load(channel_names_file_eeg)

train_eeg_average = np.mean(train_eeg, axis = 0)
train_emg_average = np.mean(train_emg, axis = 0)


train_eeg_c1 = train_eeg_average[list(eeg_channels).index("C1"), :]

print(train_eeg_average.shape)

plt.figure()
plt.plot(train_eeg_c1)
plt.plot(train_emg_average[0,:])
plt.show()