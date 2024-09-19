# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import tensorflow as tf
from time import perf_counter
import copy 

# # own libs 
from biosignal_toolbox.eeg_lib import EEGData
from biosignal_toolbox.ML_lib import MLModel
import biosignal_toolbox.ML_pipelines_lib as pipeline

# models 
from biosignal_toolbox.models.CNNnets import EEGNet

# own model 
from biosignal_toolbox.models.MlpErp import MLP_Model, MLP_Model_reduced

print(tf.config.experimental.list_physical_devices('GPU'))

from userparams import * # --> parameters are stored and fully imported from this script 

# disable GPU for testing
#tf.config.set_visible_devices([], 'GPU') # disable now 


# init performance results list
perf_results_total_MLP = []
perf_results_total_EEGNet = []


# measure execution time
time_start = perf_counter()

# init early stopping 
early_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=0,patience=early_stopping_patience,verbose=0,mode="auto",baseline=None,restore_best_weights=True)

# *********************************************************************************
# ***************** Load train, test, val sets for every iteration ****************
# *********************************************************************************

#  loading and epoching for training   
#data_train = EEGData(format = "Brainvision", filenames = train_file_list, data_path = data_path)
EEG_data = EEGData(format = "Recorded_LSL_stream", filenames = train_file_LSL, data_path = data_path, f_samp = f_samp_eeg, channel_names = channel_names)

EEG_data.rereferencingEpoching(marker_number, error_number, channel_list, inverse_keep_channel = inverse_keep_channel, t1 = t1, t2= t2)

EEG_train, EEG_val = EEG_data.splitTrainTestEpochs(n_test_epochs=n_val_trials) # split in train and val_test 
#EEG_val, EEG_test = EEG_val_test.splitTrainTestEpochs(n_test_epochs=5) # split into val and test 


channel_names = EEG_data.getChannelNames()
print("channel names", channel_names)
print("channel length", len(channel_names))
print("")

# **********************************************************************************
# ********************* Preprocessing for data of both networks ********************
# **********************************************************************************


# first stage processing for both methods and train validation data 
EEG_data_train = pipeline.firstStageOnlinePreprocessing(EEG_train, window_size = window_size, window_step = window_step, windows_selected = train_windows) 
EEG_data_val = pipeline.firstStageOnlinePreprocessing(EEG_val, window_size = window_size, window_step = window_step, windows_selected = train_windows) 


# filter specifications 
sos_MLP = EEG_data_train.designFilter(f_low = 5.0, f_high = 0.3, order = 2, filter_type = "scipy_butter", return_type = "sos") #--> good one 
sos_EEGNet = EEG_data_train.designFilter(f_low = 40.0, f_high = 0.3, order = 2, filter_type = "scipy_butter", return_type = "sos")

# # train validation data splitting 
# EEG_train_MLP = EEG_train # time domain feates MLP
# EEG_train_freq_MLP = copy.deepcopy(EEG_train_MLP) # for frequency features of MLP
# EEG_train_EEGNet = copy.deepcopy(EEG_train_MLP) # for EEGNet

# EEG_val_MLP = EEG_val # time domain feates MLP
# EEG_val_freq_MLP = copy.deepcopy(EEG_val_MLP) # for frequency features of MLP
# EEG_val_EEGNet = copy.deepcopy(EEG_val_MLP) # for EEGNet


# get features by running processing pipeline 
x_train_MLP, y_train_MLP = pipeline.MLPProcessingOnline(copy.deepcopy(EEG_data_train), copy.deepcopy(EEG_data_train), window_labels_train, feature_indices_windows, None, xd_components= None, sos = sos_MLP)
x_val_MLP, y_val_MLP = pipeline.MLPProcessingOnline(copy.deepcopy(EEG_data_val), copy.deepcopy(EEG_data_val), window_labels_train, feature_indices_windows, None, xd_components = None, sos = sos_MLP)


# Load model with norm layer  
MLP = MLP_Model_reduced(x_train_MLP, use_norm_layer = use_norm_layer)
MLP_model = MLModel(model = MLP, type= "keras")
MLP_model.trainModel(save_trained_model = True, model_filename =data_path+subject+"_"+scenario_name+result_file_name+"_model_MLP_"+str(iteration), train_epochs= n_epochs, batch_size=n_batch_size_MLP, class_weights=None, x_train=x_train_MLP, y_train= y_train_MLP, x_val = x_val_MLP, y_val = y_val_MLP, callbacks=[early_callback], loss_fcn=loss_fcn, optimizer=optimizer,metrics=metrics)


# predict and get results 
print("predict MLP net")
MLP_model.predict(data = x_train_MLP, labels = y_train_MLP, encoding = "binary", show_results = True, show_pred_time = False, eval_type = "offline")
perf_results_MLP = MLP_model.getPerfResults()

# EEGNet processing pipeline 
x_train_EEGNet, y_train_EEGNet = pipeline.EEGNetProcessingOnline(copy.deepcopy(EEG_data_train), window_labels_train, num_classes, sos = sos_EEGNet)
x_val_EEGNet, y_val_EEGNet = pipeline.EEGNetProcessingOnline(copy.deepcopy(EEG_data_val), window_labels_train, num_classes, sos = sos_EEGNet)


#shape_input = EEG_train_EEGNet.getWindows().shape # get train data shape for network 
shape_input = x_train_EEGNet.shape
print("EEGNet Input shape", shape_input)
model_EEGNet = EEGNet(nb_classes=num_classes,Chans=shape_input[1], Samples=shape_input[2], dropoutRate=dropout_EEGNet, kernLength=kern_length_EEGNET, F1=F1, D=D, F2=F2,dropoutType='Dropout', x_train = x_train_EEGNet, use_norm_layer = use_norm_layer)
EEGNet_model = MLModel(model = model_EEGNet, type="keras")
EEGNet_model.trainModel(save_trained_model = True, model_filename =data_path+subject+"_"+scenario_name+result_file_name+"_model_EEGNet"+str(iteration), train_epochs= n_epochs, batch_size=n_batch_size_EEGNet, class_weights=None, x_train=x_train_EEGNet, y_train= y_train_EEGNet, x_val = x_val_EEGNet, y_val = y_val_EEGNet, callbacks=[early_callback], loss_fcn=loss_fcn, optimizer=optimizer,metrics=metrics)

# # predict and get results 
print("predict EEGNet")
EEGNet_model.predict(data = x_train_EEGNet, labels = y_train_EEGNet, encoding = "onehotencoding", show_results = True, show_pred_time = False, eval_type = "offline")
perf_results_EEGNet = EEGNet_model.getPerfResults()

# perf results 
perf_results_total_MLP.append(perf_results_MLP[0:3])
perf_results_total_EEGNet.append(perf_results_EEGNet[0:3])

# save res 
np.savetxt(results_path+result_file_name, perf_results_total_MLP, delimiter=",", fmt = "%1.8f", )
np.savetxt(results_path+result_file_name, perf_results_total_EEGNet, delimiter=",", fmt = "%1.8f", )


print("all done")
print("")
print("execution time: ")
print(perf_counter()-time_start)


