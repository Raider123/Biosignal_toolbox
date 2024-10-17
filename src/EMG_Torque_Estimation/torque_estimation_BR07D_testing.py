# This script uses the recorded dataset for subject BR07D for training. Instead of concatenating all the data into a single array at the same time before pre-processing, each individual file is read, synchronised, pre-processed and windowed before proceeding with the next file in a loop.

# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
import time
from scipy.signal import butter

# # own libs 
from biosignal_toolbox.eeg_lib import EEGData
from biosignal_toolbox.emg_lib import OnlineEMG
from biosignal_toolbox.ML_lib import MLModel
import biosignal_toolbox.ML_pipelines_lib as pipeline

# models 
from biosignal_toolbox.models.AANModel import AAN_Model

# disable GPU for testing
#tf.config.set_visible_devices([], 'GPU') # disable now 
import warnings
warnings.filterwarnings('ignore')

# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************

# own libs
proj_path = "/home/dfki.uni-bremen.de/kschari/kc_ws/repos/biosignal_toolbox"

data_path = proj_path+"/data/m-rock_demo/"
# results_path = proj_path+"/results/"

#! Files for training
train_file_prefix = "emg_data/09092024_BR07D_"
target_file_prefix = ["quali_data/quali_torque_elbow_", "quali_data/quali_torque_front_", "quali_data/quali_torque_side_"]

#! Read Qualisys data param
# weights_d = ['0g', '1000g']
weights_d = ['0g']

mov_type_d=['complex', 'grasp']
# mov_type_d=['grasp']

set_num_d = ['3','4','5','6']
# set_num_d = ['5','6']

#! fcn model parameter 
n_epochs = 300
n_batch_size = 25

#! training params 
loss_fcn    =  "mse" 
optimizer   = "nadam"
metrics     = "mse"

#! Param for train/val data split
train_test_split_ratio = 0.9    #0.x means x% of data will be training data and rest val data
validation_split = 0.2          #0.x means x% of training data will be used as validation data

# init early stopping 
early_stop = False
if early_stop:
    early_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=0.01,patience=50,verbose=0,mode="auto",baseline=None,restore_best_weights=True)
else:
    early_callback = None

#! window wise metric evaluation
# Window params for EMG input data
window_size_x = 50              #in samples 
window_step_x = 50
# Window params for target torque values
window_size_y = window_size_x
window_step_y = window_step_x

#! Window params for feature extraction !
feature_size = 20
feature_sel = "end" # end or mid

if feature_sel == "end":
    ## Indices to extract features from the end of the window
    feature_indices_windows_x = np.arange(window_size_x-feature_size, window_size_x, step = 1) 
    feature_indices_windows_y = np.arange(window_size_y-1, window_size_y, step = 1)
elif feature_sel == "mid":
    ## Indices to extract features from the middle of the window
    feature_indices_windows_x = np.arange(round(window_size_x/2)-feature_size/2, round(window_size_x/2)+feature_size/2, step = 1)
    feature_indices_windows_y = np.arange(round(window_size_y/2)-1, round(window_size_y/2), step = 1)
else:
    print("Please select a valid feature selection type!")

#! Pre-processing parameters
f_samp = 500
f_cutoff_hpf = 15
f_cutoff_lpf = 10
var_filt_width = 20
mvc = 2.7579163508176626e-06
delay = 50
beta1 = 0.25
beta2 = 0.05
gamma = 0.7
A = -1.5
use_new_function = True     # if true, use calculateActivationForceFunctionCPPNew 

#! BPNN params
neurons_inp = 160
neurons_h1 = 40
neurons_h2 = 8

act_inp = 'relu'
act_h1 = 'relu'
act_h2 = 'linear'

#! plot folder
save_fig = False
plot_folder = "good9"

#! subject params 
subject = "BR07D"
if len(mov_type_d) >1:
    scenario_name = mov_type_d[0] + '_' + mov_type_d[1]
else:
    scenario_name = mov_type_d[0]
result_file_name = '_' + weights_d[0]

#! emg params
channel_names_i = ['BP1', 'BP2', 'BP3', 'BP4', 'BP5', 'BP6', 'BP7', 'BP8']

#! init performance results list
perf_results_total_MLP = []

#! Initialise arrays to append data
length_of_each_feature_window = feature_size * int(len(channel_names_i))
x_train_combined = np.empty(shape=[0,length_of_each_feature_window])
y_e_train_combined = np.empty(shape=[0,2])
y_f_train_combined = np.empty(shape=[0,2])
y_s_train_combined = np.empty(shape=[0,2])

x_test_combined = np.empty(shape=[0,length_of_each_feature_window])
y_e_test_combined = np.empty(shape=[0,2])
y_f_test_combined = np.empty(shape=[0,2])
y_s_test_combined = np.empty(shape=[0,2])

sos_hpf = butter(N=2, Wn=15, btype='highpass', analog=False, output='sos', fs=500)
sos_lpf = butter(N=2, Wn=10, btype='lowpass', analog=False, output='sos', fs=500)
# print(f"sos: {sos}")
sos_hpf_idx = 0
sos_lpf_idx = 0

# *********************************************************************************
# ***************** Load train, test, val sets for every iteration ****************
# *********************************************************************************

# Create online EMG object
EMG_live = OnlineEMG(stream_type = "data", n_channels=8, channel_names=channel_names_i, n_samples=500, f_samp=500)
print("Created EMG_live object!!")

EMG_live.startANTEegoStreaming(path_to_so_file="/home/dfki.uni-bremen.de/kschari/kc_ws/repos/eego-sdk-pybind11/build/python3")

while True:
    time.sleep(0.1)
    chunk = EMG_live.getChunk(return_chunk=True)
    print(np.array(chunk).shape)
    # update the ring buffer
    EMG_live.updateBuffer(show_data_shape = False, channel_indices=[0, 1, 2, 3, 4, 5, 6, 7])
    # high pass filter
    EMG_live.highPassFilterOnline(cutoff_freq=15, order=2, fs=500, type="butter", sos=sos_hpf, counter=sos_idx)
    sos_hpf_idx = 1
    # variance filter
    EMG_live.applyVarianceFilterOnline(ring_buffer=np.zeros(20), width=20, index=0)

    # normalisation
    EMG_live.normalizeContinuousDataOnline(mvc=mvc)

    # low pass filter
    EMG_live.lowPassFilterOnline(cutoff_freq=10, order=2, fs=500, type="butter", sos=sos_lpf, counter=sos_lpf_idx)
    sos_lpf_idx = 1

    # neural activation force
    EMG_live.calculateActivationForceOnline(d=50, b1=0.75, b2=0.05, g=0.1, nonlinear_shape_factor=-1.5)

        
    



