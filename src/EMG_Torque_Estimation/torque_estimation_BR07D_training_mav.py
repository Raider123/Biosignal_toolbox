# This script uses the recorded dataset for subject BR07D for training. Instead of concatenating all the data into a single array at the same time before pre-processing, each individual file is read, synchronised, pre-processed and windowed before proceeding with the next file in a loop.

# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib

# # own libs 
from biosignal_toolbox.eeg_lib import EEGData
from biosignal_toolbox.emg_lib import EMGData
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
weights_d = ['0g', '1000g']
# weights_d = ['0g']

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
f_cutoff_lpf = 4
var_filt_width = 20
mvc = 2.7579163508176626e-06
delay = 50
beta1 = 0.25
beta2 = 0.05
gamma = 0.7
A = -1.5
use_new_function = True     # if true, use calculateActivationForceFunctionCPPNew 

#! BPNN params
neurons_inp = 8
neurons_h1 = 5
neurons_h2 = 2

act_inp = 'relu'
act_h1 = 'relu'
act_h2 = 'linear'

#! plot folder
save_fig = False
plot_folder = "plot4_mav"

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
length_of_each_feature_window = int(len(channel_names_i))
x_train_combined = np.empty(shape=[0,length_of_each_feature_window])
y_e_train_combined = np.empty(shape=[0,2])
y_f_train_combined = np.empty(shape=[0,2])
y_s_train_combined = np.empty(shape=[0,2])

x_test_combined = np.empty(shape=[0,length_of_each_feature_window])
y_e_test_combined = np.empty(shape=[0,2])
y_f_test_combined = np.empty(shape=[0,2])
y_s_test_combined = np.empty(shape=[0,2])

# *********************************************************************************
# ***************** Load train, test, val sets for every iteration ****************
# *********************************************************************************
for typ_idx in range(len(mov_type_d)):
    for set_idx in range(len(set_num_d)):

        #! Check if the file exists
        if pathlib.Path(data_path + train_file_prefix + weights_d[0] + '_' + mov_type_d[typ_idx] + '_' + set_num_d[set_idx] + '.txt').is_file():

            #! Loading and epoching for training   
            # EMG_Data = EEGData(format = "Brainvision", filenames = [train_file_prefix + mov_type_d[typ_idx] + '_' + weights_d[wgt_idx] + 'g.vhdr'], data_path = data_path)
            EMG_Data = EMGData(format = "ANTmini", filenames = [train_file_prefix + weights_d[0] + '_' + mov_type_d[typ_idx] + '_' + set_num_d[set_idx] + '.txt'], data_path = data_path, f_samp=f_samp, channel_names=channel_names_i)

            #! Plotting the raw EMG data
            # plt.figure()
            # plt.plot(np.arange(0,EMG_Data.data[4,:].shape[0], 1)/f_samp,EMG_Data.data[4,:]*1e6)
            # plt.title("Raw EMG plot for Channel 5")
            # plt.grid()
            # plt.xlabel("Time in s")
            # plt.ylabel("Amplitude in uV")
            # plt.show()

            #! Loading the target values for the 3 joints
            channel_names_t = ['right', 'left', 'marker']
            print("Creating Quali Elbow object!!")
            Quali_Data_Elbow = EEGData(format = "Recorded_LSL_stream", filenames = [target_file_prefix[0] + weights_d[0] + '_' + mov_type_d[typ_idx] + '_set' + set_num_d[set_idx]], data_path = data_path, f_samp=f_samp, channel_names=channel_names_t, file_type='individual')
            print("Creating Quali Shoulder Front object!!")
            Quali_Data_Front = EEGData(format = "Recorded_LSL_stream", filenames = [target_file_prefix[1] + weights_d[0] + '_' + mov_type_d[typ_idx] + '_set' + set_num_d[set_idx]], data_path = data_path, f_samp=f_samp, channel_names=channel_names_t, file_type='individual')
            print("Creating Quali Shoulder Side object!!")
            Quali_Data_Side = EEGData(format = "Recorded_LSL_stream", filenames = [target_file_prefix[2] + weights_d[0] + '_' + mov_type_d[typ_idx] + '_set' + set_num_d[set_idx]], data_path = data_path, f_samp=f_samp, channel_names=channel_names_t, file_type='individual')


            channel_names = EMG_Data.getChannelNames()
            print("Channel Names: ", channel_names)
            print("Channel Length: ", len(channel_names))
            print("")

            # print(Quali_Data_Elbow.data.shape)
            # plt.figure()
            # plt.plot(np.arange(0,Quali_Data_Elbow.data[0,:].shape[0], 1)/f_samp,Quali_Data_Elbow.data[1,:])
            # plt.title("Elbow Torque Plot")
            # plt.grid()

            # **********************************************************************************
            # ***************************** Preprocessing of data ******************************
            # **********************************************************************************

            #! High pass filter
            EMG_Data.highPassFilter(cutoff_freq=f_cutoff_hpf, order=2, fs=f_samp, type="butter")

            #! Plotting HP filtered data
            # plt.figure()
            # plt.plot(np.arange(0,EMG_Data.filtered_data[4,:].shape[0], 1)/f_samp,EMG_Data.filtered_data[4,:]*1e6)
            # plt.title("High-Pass Filtered and Rectified EMG plot for Channel 5")
            # plt.grid()
            # plt.xlabel("Time in s")
            # plt.ylabel("Amplitude in uV")
            # plt.show()

            #! Apply Variance Filter from variance_tools_api
            print("Applying Variance filter ...")
            width           = var_filt_width
            ring_buffer     = np.zeros(width)
            index           = 0
            EMG_Data.applyVarianceFilterCPP(ring_buffer=ring_buffer, width=width, index=index)
            print("Variance Filter applied!!\n")

            #! Plot and print specific variance filtered windows 
            # var_filtered_window_x = EMG_Data.filtered_data
            # print(f"Shape of Variance filtered windows: {var_filtered_window_x.shape}")
            # print(f"Variance filtered windows: {var_filtered_window_x[4,:]}")

            # plt.figure()
            # plt.plot(np.arange(0,EMG_Data.filtered_data[4,:].shape[0], 1)/f_samp,var_filtered_window_x[4,:]*1e6)
            # plt.title("Variance Filtered EMG plot for Channel 5")
            # plt.grid()
            # plt.xlabel("Time in s")
            # plt.ylabel("Amplitude in uV")
            # plt.show()

            #! Normalisation
            print("Performing Normalization with Max Voluntary Contraction ...")
            EMG_Data.normalizeContinuousDataNew(mvc=mvc)
            print("Normalization with Max Voluntary Contraction performed !!\n")

            #! Low pass filter to smoothen the signal
            EMG_Data.lowPassFilter(cutoff_freq=f_cutoff_lpf, order=2, fs=f_samp, type="butter")

            #! Plot normalised and smoothened data
            # plt.figure()
            # plt.plot(np.arange(0,EMG_Data.filtered_data[4,:].shape[0], 1)/f_samp,EMG_Data.filtered_data[4,:])
            # plt.title("Normalised and Smoothened EMG plot for Channel 5")
            # plt.grid()
            # plt.xlabel("Time in s")
            # plt.ylabel("Amplitude in V")
            # plt.show()

            #! Calculate Neural Activation Force
            print("Replacing sample with its force activation value ...")
            if not use_new_function:
                EMG_Data.calculateActivationForceFunctionCPP(d=50, c1=0.5, c2=-0.5, nonlinear_shape_factor=-1.5)
            else: 
                EMG_Data.calculateActivationForceFunctionCPPNew(d=delay, b1=beta1, b2=beta2, g=gamma, nonlinear_shape_factor=A)
            print("Replaced each sample with its force activation value !!\n")

            #! Plot force activation data
            # plt.figure()
            # plt.plot(np.arange(0,EMG_Data.filtered_data[4,:].shape[0], 1)/f_samp,EMG_Data.filtered_data[4,:])
            # plt.title("Force Activated EMG plot for Channel 5")
            # plt.grid()
            # plt.xlabel("Time in s")
            # plt.ylabel("Amplitude in V")
            # plt.show()

            #! Windowing the filtered data
            _ = EMG_Data.windowContinuousData(startmarkernumber = 1, stopmarkernumber = 1, window_size = window_size_x, window_step = window_step_x, start_index_offset = 0, start_channel_pick=0, end_channel_pick=8,return_window_end_indices = True, choose_data="filtered_data")

            _ = Quali_Data_Elbow.windowContinuousData(startmarkernumber = 1, stopmarkernumber = 1, window_size = window_size_y, window_step = window_step_y, start_index_offset = 0, start_channel_pick=0, end_channel_pick=3,return_window_end_indices = True)

            _ = Quali_Data_Front.windowContinuousData(startmarkernumber = 1, stopmarkernumber = 1, window_size = window_size_y, window_step = window_step_y, start_index_offset = 0, start_channel_pick=0, end_channel_pick=3,return_window_end_indices = True)

            _ = Quali_Data_Side.windowContinuousData(startmarkernumber = 1, stopmarkernumber = 1, window_size = window_size_y, window_step = window_step_y, start_index_offset = 0, start_channel_pick=0, end_channel_pick=3,return_window_end_indices = True)
            # use the EMG_Data.windows if you want to access the windowed data 

            #! Plot specific filtered windows for debugging
            # plt.figure()
            # plt.plot(EMG_Data.getWindows()[0,2,:,18])
            # plt.show()

            # print(f"Window: {EMG_Data.getWindows()[0,0,:,0]}")

            # **********************************************************************************
            # ******************************* Feature Extraction *******************************
            # **********************************************************************************

            #! time domain feature extraction
            print("Extracting features from windowed data ...")
            EMG_Data.featureExtractionFromWindows(feature_type = "timepoints", feature_indices_windows = feature_indices_windows_x)
            Quali_Data_Elbow.featureExtractionFromWindows(feature_type = "timepoints", feature_indices_windows = feature_indices_windows_y)
            Quali_Data_Front.featureExtractionFromWindows(feature_type = "timepoints", feature_indices_windows = feature_indices_windows_y)
            Quali_Data_Side.featureExtractionFromWindows(feature_type = "timepoints", feature_indices_windows = feature_indices_windows_y)
            print("Feature extraction from windowed data completed !!\n")

            #! input features network 
            x   = EMG_Data.calculateMAVFromFeatures(len(channel_names))
            y_e = Quali_Data_Elbow.getFeatures()[:,0:2]
            y_f = Quali_Data_Front.getFeatures()[:,0:2]
            y_s = Quali_Data_Side.getFeatures()[:,0:2]

            #! Ensure same number of rows for imput and target features
            end_idx = x.shape[0] if x.shape[0] <= y_e.shape[0] else y_e.shape[0]
            x = x[0:end_idx,:]
            y_e = y_e[0:end_idx,0:2]
            y_f = y_f[0:end_idx,0:2]
            y_s = y_s[0:end_idx,0:2]
            print(f"Input Feature Dim: {x.shape}")
            print(f"Output Feature Dim: {y_e.shape}")
    
            #! Split data into train and test
            print("Splitting train and test data ...")
            #Loop over the data and split it
            for idx in range(x.shape[0]):
                if idx <= round(train_test_split_ratio*x.shape[0]):
                    x_train_combined = np.vstack((x_train_combined, x[idx,:]))
                    y_e_train_combined = np.vstack((y_e_train_combined, y_e[idx,:]))
                    y_f_train_combined = np.vstack((y_f_train_combined, y_f[idx,:]))
                    y_s_train_combined = np.vstack((y_s_train_combined, y_s[idx,:]))
                else:
                    x_test_combined = np.vstack((x_test_combined, x[idx,:]))
                    y_e_test_combined = np.vstack((y_e_test_combined, y_e[idx,:]))
                    y_f_test_combined = np.vstack((y_f_test_combined, y_f[idx,:]))
                    y_s_test_combined = np.vstack((y_s_test_combined, y_s[idx,:]))
            print("Train and test data generated !!\n")
        
        else:
            continue

# **********************************************************************************
# *************************** Train, load or test Model ****************************
# **********************************************************************************

#! Init model with norm layer
model_e = AAN_Model(neurons_inp=neurons_inp, act_inp=act_inp, neurons_h1=neurons_h1, act_h1=act_h1, neurons_h2=neurons_h2, act_h2=act_h2)
MLP_model_e = MLModel(model = model_e, type= "keras")

model_f = AAN_Model(neurons_inp=neurons_inp, act_inp=act_inp, neurons_h1=neurons_h1, act_h1=act_h1, neurons_h2=neurons_h2, act_h2=act_h2)
MLP_model_f = MLModel(model = model_f, type= "keras")

model_s = AAN_Model(neurons_inp=neurons_inp, act_inp=act_inp, neurons_h1=neurons_h1, act_h1=act_h1, neurons_h2=neurons_h2, act_h2=act_h2)
MLP_model_s = MLModel(model = model_s, type= "keras")

#! Train model
print("Training MLP model for elbow joint...") 
MLP_model_e.trainModel(save_trained_model = True, model_filename =data_path+"zdemo_ml_models/"+subject+"_"+scenario_name+result_file_name+"_AAN_model_elbow_mav", train_epochs= n_epochs, batch_size=n_batch_size, class_weights=None, x_train=x_train_combined, y_train= y_e_train_combined[:,0], validation_split=validation_split, loss_fcn=loss_fcn, optimizer=optimizer,metrics=metrics, show_train_results=False, callbacks=early_callback)
print("MLP training for elbow done !!\n")

print("Training MLP model for shoulder front joint...") 
MLP_model_f.trainModel(save_trained_model = True, model_filename =data_path+"zdemo_ml_models/"+subject+"_"+scenario_name+result_file_name+"_AAN_model_front_mav", train_epochs= n_epochs, batch_size=n_batch_size, class_weights=None, x_train=x_train_combined, y_train= y_f_train_combined[:,0], validation_split=validation_split, loss_fcn=loss_fcn, optimizer=optimizer,metrics=metrics, show_train_results=False, callbacks=early_callback)
print("MLP training for front done !!\n")

print("Training MLP model for shoulder side joint...") 
MLP_model_s.trainModel(save_trained_model = True, model_filename =data_path+"zdemo_ml_models/"+subject+"_"+scenario_name+result_file_name+"_AAN_model_side_mav", train_epochs= n_epochs, batch_size=n_batch_size, class_weights=None, x_train=x_train_combined, y_train= y_s_train_combined[:,0], validation_split=validation_split, loss_fcn=loss_fcn, optimizer=optimizer,metrics=metrics, show_train_results=False, callbacks=early_callback)
print("MLP training for side done !!\n")

#! Load saved model
# print("Loading saved MLP models ...")
# MLP_model_e.loadModel(filename=subject+"_"+scenario_name+result_file_name+"_AAN_model_elbow", path=data_path)
# MLP_model_f.loadModel(filename=subject+"_"+scenario_name+result_file_name+"_AAN_model_front", path=data_path)
# MLP_model_s.loadModel(filename=subject+"_"+scenario_name+result_file_name+"_AAN_model_side", path=data_path)
# print("Saved MLP models loaded !!\n")

#! Predict and get results 
print("Predicting joint torques ...")
MLP_model_e.predictTarget(data = x_test_combined, labels = y_e_test_combined[:,0], classification=False, show_results = False, show_pred_time = False, eval_type = "offline")
MLP_model_f.predictTarget(data = x_test_combined, labels = y_f_test_combined[:,0], classification=False, show_results = False, show_pred_time = False, eval_type = "offline")
MLP_model_s.predictTarget(data = x_test_combined, labels = y_s_test_combined[:,0], classification=False, show_results = False, show_pred_time = False, eval_type = "offline")

perf_results_MLP_e = MLP_model_e.getPredictionScores()
perf_results_MLP_f = MLP_model_f.getPredictionScores()
perf_results_MLP_s = MLP_model_s.getPredictionScores()
# print(perf_results_MLP_e)

# **********************************************************************************
# **************************** Post Prediction Filtering ***************************
# **********************************************************************************
filtered_perf_results_MLP_e = np.zeros(perf_results_MLP_e.shape)
filtered_perf_results_MLP_f = np.zeros(perf_results_MLP_f.shape)
filtered_perf_results_MLP_s = np.zeros(perf_results_MLP_s.shape)

filter_window_size = 3

for idx in range(len(perf_results_MLP_e)):
    if idx < filter_window_size:
        filtered_perf_results_MLP_e[idx] = perf_results_MLP_e[idx]
        filtered_perf_results_MLP_f[idx] = perf_results_MLP_f[idx]
        filtered_perf_results_MLP_s[idx] = perf_results_MLP_s[idx]
    else:
        filtered_perf_results_MLP_e[idx] = np.mean(perf_results_MLP_e[idx-filter_window_size+1:idx])
        filtered_perf_results_MLP_f[idx] = np.mean(perf_results_MLP_f[idx-filter_window_size+1:idx])
        filtered_perf_results_MLP_s[idx] = np.mean(perf_results_MLP_s[idx-filter_window_size:idx])

        # filtered_perf_results_MLP[idx] = np.mean(perf_results_MLP[idx-filter_window_size:idx])
        # filtered_perf_results_MLP[idx] = np.median(perf_results_MLP[idx-filter_window_size:idx])

#! Calculate the RMSE values on the filtered predicted values
rmse_elbow  = MLModel.calculateRMSE(y_e_test_combined[:,0], filtered_perf_results_MLP_e)
rmse_front  = MLModel.calculateRMSE(y_f_test_combined[:,0], filtered_perf_results_MLP_f)
rmse_side   = MLModel.calculateRMSE(y_s_test_combined[:,0], filtered_perf_results_MLP_s)

#! Calculate the RMSE values on the filtered predicted values
rmse_elbow_raw  = MLModel.calculateRMSE(y_e_test_combined[:,0], perf_results_MLP_e)
rmse_front_raw  = MLModel.calculateRMSE(y_f_test_combined[:,0], perf_results_MLP_f)
rmse_side_raw   = MLModel.calculateRMSE(y_s_test_combined[:,0], perf_results_MLP_s)

#! Check if the save dir exists. If not create one
dir_d = pathlib.Path("../../plots/m-rock_demo/" + plot_folder)
if save_fig and not dir_d.exists():
    dir_d.mkdir()
    print("New dir to save plots is created!")
elif save_fig and dir_d.exists():
    print("The saving dir exists. Adding a suffix to avoid overwrite!")
    plot_folder = plot_folder + '_new'
    dir_d = pathlib.Path("../../plots/m-rock_demo/" + plot_folder)
    dir_d.mkdir()

#! Create a readme.txt and include all parameters in it
if save_fig:
    readme_file = dir_d / "readme.txt"
    with readme_file.open("w") as f:
        f.write(f"Scenario: {weights_d, mov_type_d}\n") 
        f.write(f"Subject: {subject}\n")
        f.write(f"Batch Size: {n_batch_size}\n")
        f.write(f"Feature Selection: {feature_sel} \n")
        f.write(f"Window Size: {window_size_x}\n")
        f.write(f"Window Step: {window_step_x}\n")
        f.write(f"\n")
        f.write(f"HPF Filter: {f_cutoff_hpf}Hz\n")
        f.write(f"Variance Filter width: {var_filt_width}\n")
        f.write(f"MVC: {mvc}\n")
        f.write(f"LPF Filter: {f_cutoff_lpf}Hz\n")
        f.write(f"Force Activation: {use_new_function} -> {delay, beta1, beta2, gamma, A}\n")
        f.write(f"\n")
        f.write(f"BPNN Neurons: {neurons_inp, neurons_h1, neurons_h2}\n")
        f.write(f"BPNN Act functions: {act_inp, act_h1, act_h2}\n")
        f.write(f"Early Stopping: {early_stop} ({n_epochs} epochs)\n")
        f.write(f"Post Processing: Median filter size {filter_window_size-1}\n")

#! Plotting the filtered prediction results
plt.figure()
x_samples = np.arange(0, len(y_e_test_combined[:,0]),1)
plt.plot(x_samples, y_e_test_combined[:,0], ls="dashed", label='real torque')
plt.plot(x_samples, filtered_perf_results_MLP_e, label='predicted torque')
plt.title(f"Elbow Joint Median (RMSE {rmse_elbow} N-m)")
plt.legend()
plt.grid()
if save_fig:
    plt.savefig("../../plots/m-rock_demo/" + plot_folder +"/test_elbow_median.png")

plt.figure()
x_samples = np.arange(0, len(y_f_test_combined[:,0]),1)
plt.plot(x_samples, y_f_test_combined[:,0], ls="dashed", label='real torque')
plt.plot(x_samples, filtered_perf_results_MLP_f, label='predicted torque')
plt.title(f"Shoulder Front Joint Median (RMSE {rmse_front} N-m)")
plt.legend()
plt.grid()
if save_fig:
    plt.savefig("../../plots/m-rock_demo/" + plot_folder +"/test_front_median.png")

plt.figure()
x_samples = np.arange(0, len(y_s_test_combined[:,0]),1)
plt.plot(x_samples, y_s_test_combined[:,0], ls="dashed", label='real torque')
plt.plot(x_samples, filtered_perf_results_MLP_s, label='predicted torque')
plt.title(f"Shoulder Side Joint Median (RMSE {rmse_side} N-m)")
plt.legend()
plt.grid()
if save_fig:
    plt.savefig("../../plots/m-rock_demo/" + plot_folder +"/test_side_median.png")


#! Plotting the raw prediction results
plt.figure()
x_samples = np.arange(0, len(y_e_test_combined[:,0]),1)
plt.plot(x_samples, y_e_test_combined[:,0], ls="dashed", label='real torque')
plt.plot(x_samples, perf_results_MLP_e, label='predicted torque')
plt.title(f"Elbow Joint Raw (RMSE {rmse_elbow_raw} N-m)")
plt.legend()
plt.grid()
if save_fig:
    plt.savefig("../../plots/m-rock_demo/" + plot_folder +"/test_elbow_raw.png")

plt.figure()
x_samples = np.arange(0, len(y_f_test_combined[:,0]),1)
plt.plot(x_samples, y_f_test_combined[:,0], ls="dashed", label='real torque')
plt.plot(x_samples, perf_results_MLP_f, label='predicted torque')
plt.title(f"Shoulder Front Joint Raw (RMSE {rmse_front_raw} N-m)")
plt.legend()
plt.grid()
if save_fig:
    plt.savefig("../../plots/m-rock_demo/" + plot_folder +"/test_front_raw.png")

plt.figure()
x_samples = np.arange(0, len(y_s_test_combined[:,0]),1)
plt.plot(x_samples, y_s_test_combined[:,0], ls="dashed", label='real torque')
plt.plot(x_samples, perf_results_MLP_s, label='predicted torque')
plt.title(f"Shoulder Side Joint Raw (RMSE {rmse_side_raw} N-m)")
plt.legend()
plt.grid()
if save_fig:
    plt.savefig("../../plots/m-rock_demo/" + plot_folder +"/test_side_raw.png")

#! Showing the plots
plt.show()


