# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import sys
from tensorflow.keras.models import load_model
#tf.config.set_visible_devices([], 'GPU')


# own libs
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/mne_machine_learning"
sys.path.append(proj_path+"/lib") # path to lib folder
import eeg_lib


# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************
data_path = proj_path+"/data/"
results_path = proj_path+"/results/"
subject_names = ["JV43","RA12", "JV43", "AV82", "UP28", "XP01", "ZS27", "JD68", "QS70"] # specify which subjects data should be evaluated
scenario_name = "intentional_unilateral"
model1_name = "_EEGNet"
preprocessed_data_filename_end = "34ch_05_4Hz"
preprocessed_data_filename_end_f = "34ch_01_40Hz"
iteration = 2


# training windows and features
test_windows = ["bis-2500", "bis-2050", "bis-100", "bis0"]
window_labels_test = [0.0, 0.0, 1.0, 1.0]

# preprocessing
# window wise metric evaluation
window_size = 1000 #windowsize in ms (analog to pySPACE evaluation)
window_step = 50 # stepsize in ms (analog to pySPACE evaluation)
f_samp_eeg = 500 # in Hz


num_classes = 2

# *********************************************************************************
# ***************** Load data  ****************************************************
# *********************************************************************************

# basic window extraction 
lrp_epochs_test_scaled = np.load(data_path+"RA12"+"_"+scenario_name+preprocessed_data_filename_end+"_val_"+str(iteration)+".npy")


test_windows_EEG, num_of_windows, wind_names = eeg_lib.windowEEGEpochs(lrp_epochs_test_scaled, f_samp_eeg, window_size, window_step)
test_windows_EEG_select = eeg_lib.windowSelection(test_windows_EEG, wind_names, test_windows)
y_test = eeg_lib.setWindowLabels(test_windows_EEG_select, window_labels_test) # window labels 
y_test_EEGNet = to_categorical(y_test, num_classes)

# specific preprocessing
x_test_EEG_net = eeg_lib.reshapeWindowsForCNNnets(test_windows_EEG_select)


model_EEGNet = load_model('RA12_EEGNet_05_40Hz_base0.h5')


test_predictions = model_EEGNet.predict(x_test_EEG_net)[:, 1]
print(test_predictions)
y_test = y_test_EEGNet[:, 1]


test_pred_labels = np.array([0 if score <0.5 else 1 for score in test_predictions])
tnr_test, tpr_test, acc_test, ba_test = eeg_lib.calcTestAccAndRates(test_pred_labels.flatten(), y_test.flatten())


print("")
print("Single trial metrics val data windows:")
print("TNR: ",np.round(tnr_test, 3))
print("TPR: ",np.round(tpr_test, 3))
print("Acc: ", np.round(acc_test, 3))
print("BA: ", np.round(ba_test, 3))
print("")


