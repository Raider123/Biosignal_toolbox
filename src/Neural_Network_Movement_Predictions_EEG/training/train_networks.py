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
from biosignal_toolbox.models.autoencoderNet import FilterNet
# models 
from biosignal_toolbox.models.CNNnets import EEGNet
# own model 
from biosignal_toolbox.models.MlpErp import MLP_Model

print(tf.config.experimental.list_physical_devices('GPU'))

# disable GPU for testing
#tf.config.set_visible_devices([], 'GPU') # disable now 


# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************

# own libs
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox"

data_path = proj_path+"/data/"
results_path = proj_path+"/results/"

train_file_list = ["20211210_r_JV43_intentional_unilateral_set1.vhdr", "20211210_r_JV43_intentional_unilateral_set2.vhdr"]
val_test_file_list = ["20211210_r_JV43_intentional_unilateral_set3.vhdr"]


subject = "JV43"
# names for saving 
scenario_name = "intentional_unilateral"
result_file_name = "online_filter"


# *********************************************************************************
# ************** Network parameters ***********************************************
# *********************************************************************************

#machine learning params
num_classes = 2

# fcn model parameter 
n_epochs = 300 #300 training epochs (max since early stopping is used)
n_batch_size_EEGNet = 16 # 16 for EEGNet
n_batch_size_MLP = 64 #64 for MLP
weight_no_lrp_class = 0.5 # weight for the both classes for training (loss function weighting, has to sum to 1 !)
weight_lrp_class = 0.5
early_stopping_patience = 100 # 50 

#EEGNet-parameter
kern_length_EEGNET = 50 # 50 before 
F1 = 8 # 8 
D = 2 
F2 = 16 # 16 
dropout_EEGNet = 0.5

# training params 
loss_fcn =  "binary_crossentropy" #tf.keras.losses.Hinge()
optimizer  = "adam" # Nadam for MLP 
metrics = "accuracy"

# training windows and features
train_windows = ["bis-2500", "bis-1900", "bis-1500" ,"bis-1200", "bis-150", "bis-100", "bis-50", "bis0"]#, "bis-50", "bis0"] # alternatively 
window_labels_train = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]# alternative 

features = "fusion" # which features to be used for classification, "timepoints" or "meanfreqs" or "fusion" (combine both)
feature_indices_windows = np.arange(900, 1000, step = 2) # 900, 1000 numpy array with time feature indices, (950, 1000) means last 100 ms of a window are used 
use_norm_layer = True # use the input norm layer 

# validation trials used for performance evaluation 
n_test_trials = 20 

# window wise metric evaluation
window_size = 1100 # windowsize in ms (analog to pySPACE evaluation) + add 100 ms for cutting after filtering 
window_step = 50 # stepsize in ms (analog to pySPACE evaluation)

f_samp_eeg = 500.0 #sample Frequency of eeg
marker_number = 100 # onset markernumber (Qualisys)
error_number = 3 # number of the error marker 


# eeg channel that are kept (inverse_keep_channel = False) or dropped (inverse_keep_channel = True) for further evaluations, empty list meaning all channels are kept 
inverse_keep_channel = True # standard: True 
#channel_list = [] # do not drop channels
channel_list = ["F5", "F6", "x_dir", "y_dir", "z_dir", "FP1", "FP2", "F8", "T7", "T8", "TP9", "TP10", "P7", "P8", "PO9", "O1", "OZ", "O2", "PO10", "AF7", "AF3", "AF4", "AF8", "FT9", "FT7", "FT8", "FT10", "TP7", "TP8", "PO7", "PO3", "POZ", "PO4", "PO8", "F7"]


# just remap the parameters (need to be adapted)
t1 = -5.0
t2 = 0.05 # to cut this off later 

# filter settings 
f_highpass = 0.5
f_lowpass_MLP = 4.0
f_lowpass_EEGNet = 40.0

# switch between online and offline preprocessing 
use_offline_processing = False

# *********************************************************************************
# ***************** Main processing and classification loop ***********************
# *********************************************************************************


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
EEG_data_train = EEGData(format = "Brainvision", filenames = train_file_list, data_path = data_path)
EEG_data_val_test = EEGData(format = "Brainvision", filenames = val_test_file_list, data_path = data_path)


# when using offline filters for preprocessing 
if (use_offline_processing): 

     # pass a copy to the offline preprocessing pipeline 
    EEG_data_train_05_4Hz = pipeline.offlinePreprocessingAndFiltering(copy.deepcopy(EEG_data_train), f_highpass = f_highpass, f_lowpass = f_lowpass_MLP, marker_number = marker_number, error_number = error_number, channel_list = channel_list, t1 = t1, t2 = t2, windows_selected = train_windows, window_size = window_size, window_step = window_step)
    EEG_data_train_05_40Hz = pipeline.offlinePreprocessingAndFiltering(copy.deepcopy(EEG_data_train), f_highpass = f_highpass, f_lowpass = f_lowpass_EEGNet, marker_number = marker_number, error_number = error_number, channel_list = channel_list, t1 = t1, t2 = t2, windows_selected = train_windows, window_size = window_size, window_step = window_step)
    EEG_data_val_05_4Hz, EEG_data_test_05_4Hz  = pipeline.offlinePreprocessingAndFiltering(copy.deepcopy(EEG_data_val_test), f_highpass = f_highpass, f_lowpass = f_lowpass_MLP, marker_number = marker_number, error_number = error_number, channel_list = channel_list, t1 = t1, t2 = t2, windows_selected = train_windows, window_size = window_size, window_step = window_step, split_train_test_epochs = True, n_epochs = n_test_trials)
    EEG_data_val_05_40Hz, EEG_data_test_05_40Hz = pipeline.offlinePreprocessingAndFiltering(copy.deepcopy(EEG_data_val_test), f_highpass = f_highpass, f_lowpass = f_lowpass_EEGNet, marker_number = marker_number, error_number = error_number, channel_list = channel_list, t1 = t1, t2 = t2, windows_selected = train_windows, window_size = window_size, window_step = window_step, split_train_test_epochs = True, n_epochs = n_test_trials)
    
else: # no 
    # do rereferencing here since in real online case no epoching is done ! 
    EEG_data_train.rereferencingEpoching(marker_number, error_number, channel_list, inverse_keep_channel = inverse_keep_channel, t1 = t1, t2= t2)
    EEG_data_val_test.rereferencingEpoching(marker_number, error_number, channel_list, inverse_keep_channel = inverse_keep_channel, t1 = t1, t2= t2)
    EEG_data_val, EEG_data_test = EEG_data_val_test.splitTrainTestEpochs(n_test_epochs=n_test_trials) # split in train and val_test 

    # do first stage preprocessing 
    EEG_data_train = pipeline.firstStageOnlinePreprocessing(EEG_data_train, window_size = window_size, window_step = window_step, windows_selected = train_windows) 
    EEG_data_val = pipeline.firstStageOnlinePreprocessing(EEG_data_val, window_size = window_size, window_step = window_step, windows_selected = train_windows) 
    EEG_data_test = pipeline.firstStageOnlinePreprocessing(EEG_data_test, window_size = window_size, window_step = window_step, windows_selected = train_windows) 

# channel_names = EEG_data_train_05_4Hz.getChannelNames()
# print("channel names", channel_names)
# print("channel length", len(channel_names))
# print("")


# **********************************************************************************
# ********************* MLP net processing and training  ***************************
# **********************************************************************************

# get features by running processing pipeline 
    
if (use_offline_processing): 
    x_train_MLP, y_train_MLP = pipeline.MLPProcessingOffline(copy.deepcopy(EEG_data_train_05_4Hz), copy.deepcopy(EEG_data_train_05_40Hz), window_labels_train, feature_indices_windows)
    x_val_MLP, y_val_MLP = pipeline.MLPProcessingOffline(copy.deepcopy(EEG_data_val_05_4Hz), copy.deepcopy(EEG_data_val_05_40Hz), window_labels_train, feature_indices_windows)
    x_test_MLP, y_test_MLP = pipeline.MLPProcessingOffline(copy.deepcopy(EEG_data_test_05_4Hz), copy.deepcopy(EEG_data_test_05_40Hz), window_labels_train, feature_indices_windows)

else: 
    x_train_MLP, y_train_MLP = pipeline.MLPProcessingOnline(copy.deepcopy(EEG_data_train), copy.deepcopy(EEG_data_train), window_labels_train, feature_indices_windows)
    x_val_MLP, y_val_MLP = pipeline.MLPProcessingOnline(copy.deepcopy(EEG_data_val), copy.deepcopy(EEG_data_val), window_labels_train, feature_indices_windows)
    x_test_MLP, y_test_MLP = pipeline.MLPProcessingOnline(copy.deepcopy(EEG_data_test), copy.deepcopy(EEG_data_test), window_labels_train, feature_indices_windows)


# Load model with norm layer and train model  
MLP = MLP_Model(x_train_MLP, use_norm_layer = use_norm_layer)
MLP_model = MLModel(model = MLP, type= "keras")
MLP_model.trainModel(save_trained_model = False, model_filename =data_path+subject+"_"+scenario_name+result_file_name+"_model_MLP_", train_epochs= n_epochs, batch_size=n_batch_size_MLP, class_weights=None, x_train=x_train_MLP, y_train= y_train_MLP, x_val = x_val_MLP, y_val = y_val_MLP, callbacks=[early_callback], loss_fcn=loss_fcn, optimizer=optimizer,metrics=metrics)


# predict and get results 
print("predict MLP net")
MLP_model.predict(data = x_test_MLP, labels = y_test_MLP, encoding = "binary", show_results = True, show_pred_time = False, eval_type = "offline")
perf_results_MLP = MLP_model.getPerfResults()


# **********************************************************************************
# ********************* EEGnet processing and training  ***************************
# **********************************************************************************

# EEGNet processing pipeline 
if (use_offline_processing): 
    x_train_EEGNet, y_train_EEGNet = pipeline.EEGNetProcessingOffline(EEG_data_train_05_40Hz, window_labels_train, num_classes)
    x_val_EEGNet, y_val_EEGNet = pipeline.EEGNetProcessingOffline(EEG_data_val_05_40Hz, window_labels_train, num_classes)
    x_test_EEGNet, y_test_EEGNet = pipeline.EEGNetProcessingOffline(EEG_data_test_05_40Hz, window_labels_train, num_classes)
else: 
    x_train_EEGNet, y_train_EEGNet = pipeline.EEGNetProcessingOnline(copy.deepcopy(EEG_data_train), window_labels_train, num_classes)
    x_val_EEGNet, y_val_EEGNet = pipeline.EEGNetProcessingOnline(copy.deepcopy(EEG_data_val), window_labels_train, num_classes)
    x_test_EEGNet, y_test_EEGNet = pipeline.EEGNetProcessingOnline(copy.deepcopy(EEG_data_test), window_labels_train, num_classes)


shape_input = x_train_EEGNet.shape # get train data shape for network 

print("EEGNet Input shape", shape_input)
model_EEGNet = EEGNet(nb_classes=num_classes,Chans=shape_input[1], Samples=shape_input[2], dropoutRate=dropout_EEGNet, kernLength=kern_length_EEGNET, F1=F1, D=D, F2=F2,dropoutType='Dropout', x_train = x_train_EEGNet, use_norm_layer = use_norm_layer)
EEGNet_model = MLModel(model = model_EEGNet, type="keras")
EEGNet_model.trainModel(save_trained_model = False, model_filename =data_path+subject+"_"+scenario_name+result_file_name+"_model_EEGNet", train_epochs= n_epochs, batch_size=n_batch_size_EEGNet, class_weights=None, x_train=x_train_EEGNet, y_train= y_train_EEGNet, x_val = x_val_EEGNet, y_val = y_val_EEGNet, callbacks=[early_callback], loss_fcn=loss_fcn, optimizer=optimizer,metrics=metrics)

# # predict and get results 
print("predict EEGNet")
EEGNet_model.predict(data = x_test_EEGNet, labels = y_test_EEGNet, encoding = "onehotencoding", show_results = True, show_pred_time = False, eval_type = "offline")
perf_results_EEGNet = EEGNet_model.getPerfResults()


# **********************************************************************************
# ********************* Saving and storing performances  ***************************
# **********************************************************************************

# perf results 
perf_results_total_MLP.append(perf_results_MLP[0:3])
perf_results_total_EEGNet.append(perf_results_EEGNet[0:3])


# save res 
np.savetxt(results_path+subject+result_file_name+"_MLP", perf_results_total_MLP, delimiter=",", fmt = "%1.8f", )
np.savetxt(results_path+subject+result_file_name+"_EEGNet", perf_results_total_EEGNet, delimiter=",", fmt = "%1.8f", )


print("all done")
print("")
print("execution time: ")
print(perf_counter()-time_start)


if(use_offline_processing): 
    print("windows shape", EEG_data_train_05_4Hz.windows.shape) 
else: 
    print("windows shape", EEG_data_train.windows.shape) 
