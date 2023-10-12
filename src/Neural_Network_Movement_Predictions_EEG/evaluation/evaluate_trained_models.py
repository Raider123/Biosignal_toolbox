# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from time import perf_counter
import sys
import copy 

# # own libs 
from biosignal_toolbox.eeg_lib import EEGData
from biosignal_toolbox.ML_lib import MLModel


# own libs
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/mne_machine_learning"
#sys.path.append(proj_path+"/lib/biosignal_toolbox") # path to lib folder

# models 
from biosignal_toolbox.models.CNNnets import EEGNet

# own model 
from biosignal_toolbox.models.MlpErp import MLP_Model

print(tf.config.experimental.list_physical_devices('GPU'))

# disable GPU for testing
#tf.config.set_visible_devices([], 'GPU')


# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************

data_path = proj_path+"/data/"
results_path = proj_path+"/results/"
subject_names = ["JV43","RA12", "JV43", "AV82", "UP28", "XP01", "ZS27", "JD68", "QS70"] # specify which subjects data should be evaluated
interations = [0, 1, 2] # the evaluation numbers which train test permutations are used
scenario_name = "intentional_unilateral"
preprocessed_data_filename_end = "34ch_raw_no_scale"  #"34ch_raw_no_scale" # TODO: implement online filter and normalization  
#eval_name = "fcn_network_results_34ch_MLP_scalings_test"

# model names 
MLP_eval_name = "fcn_network_results_34ch_MLP_online_no_norm"
EEGNet_eval_name = "fcn_network_results_34ch_EEGNet_online"

f_samp_eeg = 500 #sample Frequency of eeg


# training params 
loss_fcn =  "binary_crossentropy" #tf.keras.losses.Hinge()
optimizer  = "adam" # Nadam for MLP 

metrics = "accuracy"

# training windows and features
test_windows = ["bis-2500", "bis-2050", "bis-100", "bis0"]
window_labels_test = [0.0, 0.0, 1.0, 1.0] #np.zeros((81)) 


features = "fusion" # which features to be used for classification, "timepoints" or "meanfreqs" or "fusion" (combine both)
feature_indices_windows = np.arange(900, 1000, step = 1) # numpy array with time feature indices, (950, 1000) means last 100 ms of a window are used 

# window wise metric evaluation
window_size = 1000 #windowsize in ms (analog to pySPACE evaluation)
window_step = 50 # stepsize in ms (analog to pySPACE evaluation)


# *********************************************************************************
# ***************** Main processing and classification loop ***********************
# *********************************************************************************


# load the time axis of the epoched data
time_axis_eeg_batch = np.load(data_path+"time_axis_eeg_epochs.npy")

# example data 
subject = "JV43"
iteration = 0

# load each individual train, val and test sets (preprocessed)
lrp_epochs_val_scaled = np.load(data_path+subject+"_"+scenario_name+preprocessed_data_filename_end+"_test_"+str(iteration)+".npy")
channel_names = np.load(data_path+"remaining_eeg_channel_names"+".npy")

print(lrp_epochs_val_scaled.shape)
# **********************************************************************************
# ********************* Preprocessing for data of both networks ********************
# **********************************************************************************
#load
EEG_val = EEGData(format = "NumpyEpochs", epochs = lrp_epochs_val_scaled, f_samp = f_samp_eeg, channel_names = list(channel_names))

# window EEG epochs 
EEG_val.windowEEGEpochs(window_size, window_step)

# window selection 
EEG_val_MLP = EEG_val
EEG_val_MLP.windowSelection(test_windows) 
EEG_val_freq_MLP = copy.deepcopy(EEG_val_MLP) # for frequency features 
EEG_val_EEGNet = copy.deepcopy(EEG_val_MLP) # for EEGNet


# ******** MLP processing *******************

# bandpass filter data 
EEG_val_MLP.FilterWindows(f_low = 5.0, f_high = 0.3, filter_type = "scipy_butter", order=2, show_response = False) 

# # specify the window labels 
EEG_val_MLP.setWindowLabels(window_labels_test)
EEG_val_EEGNet.setWindowLabels(window_labels_test)

# time domain features (MLP)
EEG_val_MLP.featureExtractionFromWindows(feature_type = "timepoints", feature_indices_windows = feature_indices_windows)
EEG_val_freq_MLP.featureExtractionFromWindows(feature_type = "freqBandPower")

# feauture combination 
x_val_freq = EEG_val_freq_MLP.getFeatures() # get features of freq
EEG_val_MLP.addFeatures(x_val_freq) # add frequency domain features 

# input features network 
x_val_MLP = EEG_val.getFeatures()
y_val_MLP = EEG_val.getTrainLabels()


# *********** EEGNet processing *******************

EEG_val_EEGNet.FilterWindows(f_low = 40.0, f_high = 0.3, filter_type = "scipy_butter", order=2, show_response = False) 
# reshape windows for net 
EEG_val_EEGNet.reshapeWindowsForCNNnets()

EEG_val_EEGNet.labelsToCategorical(num_classes = 2)
# get train windows 
x_val_EEGNet = EEG_val_EEGNet.getWindows()
y_val_EEGNet = EEG_val_EEGNet.getTrainLabels()


# ********** Load and apply models ***********

MLP_model = MLModel(use_input_norm = True) 
MLP_model.loadModel(path =data_path, filename = subject+"_"+scenario_name+MLP_eval_name+"_model_"+str(iteration))


# predict and get results 
MLP_model.predict(data = x_val_MLP, labels = y_val_MLP, encoding = "binary", show_results = True)
perf_results = MLP_model.getPerfResults()

print("perf results MLP", perf_results)



# model setup and training 

# create EEGNet model for training 
# print("Use EEGNet")
# train_wind_shape = EEG_train.getWindows().shape # get train data shape for network 
# model_EEGNet = load #
# EEGNet_model = MLModel(model = model_EEGNet, train_epochs= n_epochs, batch_size=n_batch_size, class_weights={0: weight_no_lrp_class, 1: weight_lrp_class}, x_train=x_train, y_train= y_train, x_val = x_val, y_val = y_val, callbacks=[early_callback], loss_fcn=loss_fcn, optimizer=optimizer,metrics=metrics)


# # predict and get results 
# EEGNet_model.predict(data = x_val, labels = y_val, encoding = "onehotencoding", show_results = True)
# perf_results = EEGNet_model.getPerfResults()
    
            


