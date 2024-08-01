# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
from time import perf_counter
import copy 
import tensorflow as tf 

# # own libs 
from biosignal_toolbox.eeg_lib import EEGData
from biosignal_toolbox.ML_lib import MLModel
import biosignal_toolbox.ML_pipelines_lib as pipeline

# own model 
from biosignal_toolbox.models.MlpErp import MLP_Model

import pickle


# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************

# own libs
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox"

data_path = proj_path+"/data/"
results_path = proj_path+"/results/"

#store_file_name = "JV43_set1"

# use LSL file recorded 
train_file_list = [ "20211216_r_RA12_intentional_unilateral_set3.vhdr", "20211216_r_RA12_intentional_unilateral_set1.vhdr", "20211216_r_RA12_intentional_unilateral_set2.vhdr"]


# subject params 
subject = "RA12"  # "JV43", "AV82", "UP28", "XP01", "ZS27", "JD68", "QS70"] # specify which subjects data should be evaluated
iteration = 0 # the evaluation numbers which train test permutations are used
scenario_name = "intentional_unilateral"

# train_iter = "12"
# test_iter = "3" 

# train_iter = "23"
# test_iter = "1"

train_iter = "31"
test_iter = "2"

#machine learning params
num_classes = 2

# fcn model parameter 
n_epochs = 300 #300 training epochs (max since early stopping is used)
n_batch_size_MLP = 64 #64 for MLP

weight_no_lrp_class = 0.5 # weight for the both classes for training (loss function weighting, has to sum to 1 !)
weight_lrp_class = 0.5
early_stopping_patience = 50 # 50 


# training params 
loss_fcn =  "binary_crossentropy" #tf.keras.losses.Hinge()
optimizer  = "adam" # Nadam for MLP 
metrics = "accuracy"


# training windows and features
train_windows = ["bis-2200", "bis-2050", "bis-100", "bis0"]#, for classical ml approach 
#test_windows = ["bis-2500", "bis-2050", "bis-150", "bis-100"] # ["bis-2500", "bis-2050", "bis-100", "bis0"]

#window_labels_train = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]# 
window_labels_train = [0.0, 0.0, 1.0, 1.0]# alternative 
#window_labels_test = [0.0, 0.0, 1.0, 1.0] #  [0.0, 0.0, 1.0, 1.0]


feature_indices_windows = np.array([850, 900, 950, 999]) # indices inside the window -_> 4 samples of the last 200 ms 
use_norm_layer = False # use the input norm layer 

n_xDAWN = 4

# validation trials used for performance evaluation 
n_val_trials = 40

# window wise metric evaluation
window_size = 1000 # windowsize in ms (analog to pySPACE evaluation) + add 100 ms for cutting after filtering 
window_step = 50 # stepsize in ms (analog to pySPACE evaluation)

marker_number = 100 # onset markernumber (Qualisys)
onset_number = marker_number
error_number = 3 # number of the error marker 


# eeg channel that are kept (inverse_keep_channel = False) or dropped (inverse_keep_channel = True) for further evaluations, empty list meaning all channels are kept 
inverse_keep_channel = True # standard: True 
#channel_list = [] # do not drop channels
channel_list = ["F5", "F6", "x_dir", "y_dir", "z_dir", "FP1", "FP2", "F8", "T7", "T8", "TP9", "TP10", "P7", "P8", "PO9", "O1", "OZ", "O2", "PO10", "AF7", "AF3", "AF4", "AF8", "FT9", "FT7", "FT8", "FT10", "TP7", "TP8", "PO7", "PO3", "POZ", "PO4", "PO8", "F7"]

# just remap the parameters (need to be adapted)
t1 = -5.0
t2 = 0.0

# *********************************************************************************
# ***************** Main processing and classification loop ***********************
# *********************************************************************************


# init performance results list
perf_results_total_MLP = []
perf_results_total_EEGNet = []

# load the time axis of the epoched data
time_axis_eeg_batch = np.load(data_path+"time_axis_eeg_epochs.npy")

# measure execution time
time_start = perf_counter()

# init early stopping 
early_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=0,patience=early_stopping_patience,verbose=0,mode="auto",baseline=None,restore_best_weights=True)

# *********************************************************************************
# ***************** Load train, test, val sets for every iteration ****************
# *********************************************************************************


#  loading and epoching for training   
EEG_data = EEGData(format = "Brainvision", filenames = train_file_list, data_path = data_path)
#EEG_data = EEGData(format = "Recorded_LSL_stream", filenames = train_file_LSL, data_path = data_path, f_samp = f_samp_eeg, channel_names = channel_names)

EEG_data_train_xDAWN = copy.deepcopy(EEG_data)
EEG_data.rereferencingEpoching(marker_number, error_number, channel_list, inverse_keep_channel = inverse_keep_channel, t1 = t1, t2= t2, f_lowpass = 4.0, f_highpass = 0.5, apply_filter=True)


# only for now, make pretty later 
EEG_data_train_xDAWN.rereferencingEpoching(marker_number, error_number, channel_list, inverse_keep_channel = inverse_keep_channel, t1 = -1.0, t2= t2, f_lowpass = 4.0, f_highpass = 0.5, apply_filter=True)


EEG_train, EEG_val = EEG_data.splitTrainTestEpochs(n_test_epochs=n_val_trials) # split in train and val_test 
#EEG_test, EEG_val_1 = EEG_val.splitTrainTestEpochs(n_test_epochs=int(n_val_trials/2)) # split in train and val_test 

#EEG_val, EEG_test = EEG_val_test.splitTrainTestEpochs(n_test_epochs=5) # split into val and test 
EEG_data_train_xDAWN_train, h = EEG_data_train_xDAWN.splitTrainTestEpochs(n_test_epochs=n_val_trials) # split in train and val_test 


channel_names = EEG_data.getChannelNames()
print("channel names", channel_names)
print("channel length", len(channel_names))
print("")

# **********************************************************************************
# ********************* Preprocessing for data of both networks ********************
# **********************************************************************************

times, epoch = EEG_train.getEpochs()
print(epoch.shape)


# train xDAWN 
xd_trained = EEG_data_train_xDAWN_train.xDAWNSpatialfilter(n_components = n_xDAWN, markernumber = marker_number, processing_type="fit", return_filter = True)


# first stage processing for both methods and train validation data 
EEG_train = pipeline.classicFirstStagePreprocessing(EEG_train, window_size, window_step, train_windows, xd_trained, n_xDAWN)
EEG_val = pipeline.classicFirstStagePreprocessing(EEG_val, window_size, window_step, train_windows, xd_trained, n_xDAWN)

print("window shape", EEG_train.getWindows().shape)


# get features by running processing pipeline 
x_train, y_train = pipeline.classicLRPpreprocessing(EEG_train, window_labels_train, feature_indices_windows)
x_val, y_val = pipeline.classicLRPpreprocessing(EEG_val, window_labels_train, feature_indices_windows)

print("feature shape", x_train.shape)


# Load model with norm layer  
MLP = MLP_Model(x_train, use_norm_layer = use_norm_layer)
MLP_model = MLModel(model = MLP, type= "keras")
MLP_model.trainModel(save_trained_model = False, model_filename =data_path+subject+"_"+scenario_name+"_model_MLP_"+str(iteration), train_epochs= n_epochs, batch_size=n_batch_size_MLP, class_weights=None, x_train=x_train, y_train= y_train, x_val = x_val, y_val = y_val, callbacks=[early_callback], loss_fcn=loss_fcn, optimizer=optimizer,metrics=metrics)


# predict and get results 
print("predict MLP net")
MLP_model.predict(data = x_train, labels = y_train, encoding = "binary", show_results = True, show_pred_time = False, eval_type = "offline")
perf_results_MLP = MLP_model.getPerfResults()


# perf results 
perf_results_total_MLP.append(perf_results_MLP[0:3])


# # save res 
# np.savetxt(results_path+result_file_name, perf_results_total_MLP, delimiter=",", fmt = "%1.8f", )
# np.savetxt(results_path+result_file_name, perf_results_total_EEGNet, delimiter=",", fmt = "%1.8f", )


print("all done")
print("")
print("execution time: ")
print(perf_counter()-time_start)


# save the features in a dictionary 
print(f"saving features")

# Create the dictionary
train_dict = {"data": x_train, "target": y_train}

# Create the dictionary
val_test_dict = {"data": x_val, "target": y_val}

#print(train_dict["data"].shape)

# Specify the file path where you want to save the dictionary
train_file = data_path+subject+"_train_"+train_iter+".pickle"
test_file = data_path+subject+"_val_test_"+test_iter+".pickle"

# Open the file in binary write mode and save the dictionary using pickle.dump()
with open(train_file, 'wb') as file:
    pickle.dump(train_dict, file)

# Open the file in binary write mode and save the dictionary using pickle.dump()
with open(test_file, 'wb') as file1:
    pickle.dump(val_test_dict, file1)

