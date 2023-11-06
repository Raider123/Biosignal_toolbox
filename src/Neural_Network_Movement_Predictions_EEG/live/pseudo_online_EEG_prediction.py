

# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import copy 
from pylsl import StreamInlet, resolve_stream
from time import perf_counter
import time
import serial


# # own libs 
from biosignal_toolbox.eeg_lib import EEGData, OnlineEEGUtils
from biosignal_toolbox.ML_lib import MLModel

# own libs
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox"
data_path = proj_path+"/data/"

# models 
from biosignal_toolbox.models.CNNnets import EEGNet

# own model 
from biosignal_toolbox.models.MlpErp import MLP_Model

print(tf.config.experimental.list_physical_devices('GPU'))

# disable GPU for testing
tf.config.set_visible_devices([], 'GPU') 


#************************************************************
# ********************** user params ************************
#************************************************************

# online params 
buffer_size = 500  # size of ringbuffer in samples, currently set to 2500 (5 sec data times 500 Hz sampling rate)
dt_read_buffer= 0.05 # time in seconds how often the buffer is read  (updated with new incoming chunks)
print_times = False
send_marker = False

# subject info 
subject = "Test"
scenario_name = "intentional_unilateral"
iteration = 0
result_file_name = "live_train_results"

# model names 
MLP_eval_name = "test_intentional_unilateraltest_recorder_LSL_transfer_model_MLP_0"
EEGNet_eval_name = "test_intentional_unilateraltest_recorder_LSL_transfer_model_EEGNet0"

# ML params 
decision_bound = 0.7
# MLP Net 
features = "fusion" # which features to be used for classification, "timepoints" or "meanfreqs" or "fusion" (combine both)
feature_indices_windows = np.arange(900, 1000, step = 2) # numpy array with time feature indices, (950, 1000) means last 100 ms of a window are used 

# marker params 
usb_port = '/dev/ttyUSB0'
Baudrate = 115200

# epoching params 
marker_number = 5 # onset markernumber (Qualisys)
onset_number = marker_number
error_number = 7 # number of the error marker 

# specifying the movement onset marker 
event_id_used = {"movement_onset": marker_number} 

# time selection for epoching of the data 
epoching_time_before_onset = -5.0 # time in seconds (start epoch)
epoching_time_after_onset = 0.0 # time in seconds (0 = movement onset)

# eeg channel that are kept (inverse_keep_channel = False) or dropped (inverse_keep_channel = True) for further evaluations, empty list meaning all channels are kept 
inverse_keep_channel = True # standard: True 
channel_list = [] # do not drop channels

# just remap the parameters (need to be adapted)
t1 = epoching_time_before_onset
t2 = epoching_time_after_onset


#************************************************************
#************************************************************
#************************************************************
# init serial markers
if(send_marker): 
    ser = serial.Serial(usb_port, Baudrate)
    time.sleep(3)

# load models

# load MLP model 
MLP_model = MLModel() 
MLP_model.loadModel(path =data_path, filename = MLP_eval_name)

# load EEGNet model 
model_EEGNet = MLModel()
model_EEGNet.loadModel(path =data_path, filename = EEGNet_eval_name)

# load data and epoching to cut out data 

#  loading and epoching for training  
data_list = ["BR60D_unilateral_LSL_set4_1.vhdr"] #"BR60D_unilateral_LSL_set4_1.vhdr"
data_raw = EEGData(format = "Brainvision", filenames = data_list, data_path = data_path)
# data_raw.rereferencingEpoching(marker_number, error_number, channel_list, inverse_keep_channel = inverse_keep_channel, event_id_used = event_id_used, t1 = t1, t2= t2)
# time_axis, eeg_epochs = data_raw.getEpochs() # trials, channels, sampels 
raw_obj = data_raw.getRawObject()
raw_data = raw_obj.get_data(units = "uV")
print(raw_data.shape)

#20857 # onset ind 
ind  = 48380
example_window = raw_data[:, ind-500:ind]
#example_window = eeg_epochs[15, :, 1900:2400]
example_window = np.expand_dims(example_window, 0) 
example_window = np.expand_dims(example_window, 3) # same shape as input 

print("")
print("some values about data")
print("mean", np.mean(example_window[0, 10, :, 0])) 
print("std", np.std(example_window[0, 10, :, 0])) 
print("type: ", example_window.dtype)
print("")

# # create online EEG utils Object  
EEGutils = OnlineEEGUtils(n_channels=32, n_samples=buffer_size, dt_process_data = dt_read_buffer) # use this normally stream_info.channel_count()

#inits 
channel_names = None
EEG_live = EEGData(format = "Live", f_samp = 500.0, channel_names = channel_names)


# run one cycle on window
EEG_live.windows = example_window

#print(np.mean(EEG_live.windows[0, 10, :, 0]))

if(print_times): 
    t1 = perf_counter()

# **********************************************
# *********** Model apply here *****************
# **********************************************

# copy data objects for different processing 
EEG_live_MLP = EEG_live # time domain feates MLP
EEG_live_freq_MLP = copy.deepcopy(EEG_live_MLP) # for frequency features of MLP
EEG_live_EEGNet = copy.deepcopy(EEG_live_MLP) # for EEGNet

    # ******** MLP processing *******************

# bandpass filter data 
EEG_live_MLP.FilterWindows(f_low = 5.0, f_high = 0.3, filter_type = "scipy_butter", order=2, show_response = False) # bandpass filter

# time domain features (MLP)
EEG_live_MLP.featureExtractionFromWindows(feature_type = "timepoints", feature_indices_windows = feature_indices_windows) # time dom features 
EEG_live_freq_MLP.featureExtractionFromWindows(feature_type = "freqBandPower") # freq domain features 

# feauture combination 
x_val_freq = EEG_live_freq_MLP.getFeatures() # get features of freq
EEG_live_MLP.addFeatures(x_val_freq) # add frequency domain features 

# input features network 
x_live_MLP = EEG_live_MLP.getFeatures()


# *********** EEGNet processing *******************

#EEG_live_EEGNet.FilterWindows(f_low = None, f_high = 0.1, filter_type = "scipy_butter", order=2, show_response = False) # try this ? 
EEG_live_EEGNet.FilterWindows(f_low = 40.0, f_high = 0.3, filter_type = "scipy_butter", order=2, show_response = False) # bandpass filter  

# get train windows 
x_live_EEGNet = EEG_live_EEGNet.getWindows()

# ********** make model prediction  ***********

# predict and get results 
MLP_model.predict(data = x_live_MLP, labels = None, encoding = "binary", show_results = False, show_pred_time = True, eval_type = "online")

# # predict and get results 
model_EEGNet.predict(data = x_live_EEGNet, labels = None, encoding = "onehotencoding", show_results = False, show_pred_time = True, eval_type = "online")

# postprocessing 
MLP_score =  MLP_model.prediction_scores[0] 
prod_score = MLP_score# final output score 

#print("hole score:", prod_score)
print("EEGNet", model_EEGNet.prediction_scores[1])
print("MLP", MLP_score)

if(prod_score > decision_bound): 
    print("onset detected")

    if(send_marker):
        ser.write(b's') # send marker when detected 
        
