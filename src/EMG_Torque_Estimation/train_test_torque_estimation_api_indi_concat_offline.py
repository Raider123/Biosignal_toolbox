# This script uses the complete dataset for subject HW90 for training. Instead of concatenating all the data into a single array at the same time before pre-processing, each individual file is read, synchronised, pre-processed and windowed before proceeding with the next file in a loop.

# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import tensorflow as tf
from time import perf_counter
import copy 
import matplotlib.pyplot as plt

# # own libs 
from biosignal_toolbox.eeg_lib import EEGData
from biosignal_toolbox.ML_lib import MLModel
import biosignal_toolbox.ML_pipelines_lib as pipeline

# models 
from biosignal_toolbox.models.AANModel import AAN_Model

# disable GPU for testing
#tf.config.set_visible_devices([], 'GPU') # disable now 


# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************

# own libs
proj_path = "/home/dfki.uni-bremen.de/kschari/kc_ws/repos/biosignal_toolbox"

data_path = proj_path+"/data/"
results_path = proj_path+"/results/"

#! Files for training
train_file_prefix = "aan_emg_data/HW90/20170317_r_HW90_EMG_Assist_as_needed_"
target_file_prefix = ["aan_quali_data/quali_torque_elbow_", "aan_quali_data/quali_torque_front_", "aan_quali_data/quali_torque_side_"]

#! Read Qualisys data param
# weights_order_d=['0','500','1000','1500']
weights_order_d=['0','500']

# mov_type_order_d=['complex', 'curl','grasp','front','side']
mov_type_order_d=['grasp']

#! subject params 
subject = "HW90"
scenario_name = "grasp"
result_file_name = "_0g_500g_1000g"

#! fcn model parameter 
n_epochs = 1000
n_batch_size = 32

#! training params 
loss_fcn    =  "mse" 
optimizer   = "nadam"
metrics     = "mse"

#! window wise metric evaluation
# Window params for EMG input data
window_size_x = 100 # windowsize in samples 
window_step_x = 100
# Window params for target torque values
window_size_y = int(window_size_x/4)
window_step_y = int(window_step_x/4)

#! Window params for feature extraction !
feature_size = 10
## Indices to extract features from the end of the window
# feature_indices_windows_x = np.arange(window_size_x-feature_size, window_size_x, step = 1) 
# feature_indices_windows_y = np.arange(window_size_y-1, window_size_y, step = 1)

## Indices to extract features from the middle of the window
feature_indices_windows_x = np.arange(round(window_size_x/2)-feature_size/2, round(window_size_x/2)+feature_size/2, step = 1)
feature_indices_windows_y = np.arange(round(window_size_y/2)-1, round(window_size_y/2), step = 1)

#! Param for train/val data split
train_test_split_ratio = 0.9   # 0.x means x% of data will be training data and rest val data
validation_split = 0.2

#! init performance results list
perf_results_total_MLP = []

# init early stopping 
# early_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=0,patience=early_stopping_patience,verbose=0,mode="auto",baseline=None,restore_best_weights=True)

#! Initialise arrays to append data
length_of_each_feature_window = feature_size * 10
x_train_combined = np.empty(shape=[0,length_of_each_feature_window])
y_train_combined = np.empty(shape=[0,2])

x_test_combined = np.empty(shape=[0,length_of_each_feature_window])
y_test_combined = np.empty(shape=[0,2])
# *********************************************************************************
# ***************** Load train, test, val sets for every iteration ****************
# *********************************************************************************
for wgt_idx in range(len(weights_order_d)):
    for typ_idx in range(len(mov_type_order_d)):

        #! Loading and epoching for training   
        #data_train = EEGData(format = "Brainvision", filenames = train_file_list, data_path = data_path)
        EMG_Data = EEGData(format = "Brainvision", filenames = [train_file_prefix + mov_type_order_d[typ_idx] + '_' + weights_order_d[wgt_idx] + 'g.vhdr'], data_path = data_path)

        #! Plotting the raw EMG data
        # plt.figure()
        # plt.plot(np.arange(0,EMG_Data.data[3,:].shape[0], 1)/1000,EMG_Data.data[4,:]*1e6)
        # plt.title("Raw EMG plot for Channel 3")
        # plt.grid()
        # plt.xlabel("Time in s")
        # plt.ylabel("Amplitude in uV")
        # plt.show()

        #! Loading the target values for the 3 joints
        channel_names_t = ['right', 'left', 'marker']
        print("Creating Quali Elbow object!!")
        Quali_Data_Elbow = EEGData(format = "Recorded_LSL_stream", filenames = [target_file_prefix[0] + weights_order_d[wgt_idx] + 'g_' + mov_type_order_d[typ_idx]], data_path = data_path, f_samp=250, channel_names=channel_names_t, file_type='individual')
        # print("Creating Quali Shoulder Front object!!")
        # Quali_Data_Front = EEGData(format = "Recorded_LSL_stream", filenames = [target_file_prefix[1] + weights_order_d[wgt_idx] + 'g_' + mov_type_order_d[typ_idx]], data_path = data_path, f_samp=250, channel_names=channel_names_t, file_type='individual')
        # print("Creating Quali Shoulder Side object!!")
        # Quali_Data_Side = EEGData(format = "Recorded_LSL_stream", filenames = [target_file_prefix[2] + weights_order_d[wgt_idx] + 'g_' + mov_type_order_d[typ_idx]], data_path = data_path, f_samp=250, channel_names=channel_names_t, file_type='individual')


        channel_names = EMG_Data.getChannelNames()
        print("channel names", channel_names)
        print("channel length", len(channel_names))
        print("")

        # **********************************************************************************
        # ***************************** Preprocessing of data ******************************
        # **********************************************************************************

        #! High pass filter 20 Hz
        EMG_Data.highPassFilter(cutoff_freq=30, order=2, fs=1000, type="butter")

        #! Plotting HP filtered data
        # plt.figure()
        # plt.plot(np.arange(0,EMG_Data.filtered_data[4,:].shape[0], 1)/1000,EMG_Data.filtered_data[4,:]*1e6)
        # plt.title("High-Pass Filtered and Rectified EMG plot for Channel 4")
        # plt.grid()
        # plt.xlabel("Time in s")
        # plt.ylabel("Amplitude in uV")
        # plt.show()

        #! Apply Variance Filter from variance_tools_api
        print("Applying Variance filter ...")
        ring_buffer     = np.zeros(20)
        width           = 20
        index           = 0
        EMG_Data.applyVarianceFilterCPP(ring_buffer=ring_buffer, width=width, index=index)
        print("Variance Filter applied!!\n")

        #! Plot and print specific variance filtered windows 
        # var_filtered_window_x = EMG_Data.filtered_data
        # print(f"Shape of Variance filtered windows: {var_filtered_window_x.shape}")
        # print(f"Variance filtered windows: {var_filtered_window_x[18,:]}")
        # plt.figure()
        # plt.plot(np.arange(0,EMG_Data.filtered_data[18,:].shape[0], 1)/1000,var_filtered_window_x[18,:]*1e6)
        # plt.title("Variance Filtered EMG plot for Channel 18")
        # plt.grid()
        # plt.xlabel("Time in s")
        # plt.ylabel("Amplitude in uV")
        # plt.show()

        #! Normalisation
        print("Performing Normalization with Max Voluntary Contraction ...")
        EMG_Data.normalizeContinuousData()
        print("Normalization with Max Voluntary Contraction performed !!\n")

        #! Low pass filter 10 Hz to smoothen the signal
        EMG_Data.lowPassFilter(cutoff_freq=5, order=2, fs=1000, type="butter")

        #! Plot normalised and smoothened data
        # plt.figure()
        # plt.plot(np.arange(0,EMG_Data.filtered_data[18,:].shape[0], 1)/1000,EMG_Data.filtered_data[18,:])
        # plt.title("Normalised and Smoothened EMG plot for Channel 18")
        # plt.grid()
        # plt.xlabel("Time in s")
        # plt.ylabel("Amplitude in V")
        # plt.show()

        #! Calculate Neural Activation Force
        print("Replacing sample with its force activation value ...")
        EMG_Data.calculateActivationForceFunctionCPP(d=50, c1=0.5, c2=-0.5, nonlinear_shape_factor=-1.5)
        print("Replaced each sample with its force activation value !!\n")

        #! Windowing the filtered data
        window_end_indices_x = EMG_Data.windowContinuousData(startmarkernumber = 1, stopmarkernumber = 1, window_size = window_size_x, window_step = window_step_x, start_index_offset = 20, start_channel_pick=0, end_channel_pick=10,return_window_end_indices = True, choose_data="filtered_data")

        window_end_indices_y = Quali_Data_Elbow.windowContinuousData(startmarkernumber = 1, stopmarkernumber = 1, window_size = window_size_y, window_step = window_step_y, start_index_offset = 0, start_channel_pick=0, end_channel_pick=10,return_window_end_indices = True)
        # use the EMG_Data.windows if you want to access the windowed data 

        #! Plot specific filtered windows for debugging
        # plt.figure()
        # plt.plot(EMG_Data.getWindows()[0,2,:,18])
        # plt.show()

        # **********************************************************************************
        # ******************************* Feature Extraction *******************************
        # **********************************************************************************

        #! time domain feature extraction
        print("Extracting features from windowed data ...")
        EMG_Data.featureExtractionFromWindows(feature_type = "timepoints", feature_indices_windows = feature_indices_windows_x)
        Quali_Data_Elbow.featureExtractionFromWindows(feature_type = "timepoints", feature_indices_windows = feature_indices_windows_y)
        print("Feature extraction from windowed data completed !!\n")

        #! input features network 
        x = EMG_Data.getFeatures()
        y = Quali_Data_Elbow.getFeatures()[:,0:2]

        #! Ensure same number of rows for imput and target features
        end_idx = x.shape[0] if x.shape[0] <= y.shape[0] else y.shape[0]
        x = x[0:end_idx,:]
        y = y[0:end_idx,0:2]
        print(f"Input Feature Dim: {x.shape}")
        print(f"Output Feature Dim: {y.shape}")
 
        #! Split data into train and test
        print("Splitting train and test data ...")
        #Loop over the data and split it
        for idx in range(x.shape[0]):
            if idx <= round(train_test_split_ratio*x.shape[0]):
                x_train_combined = np.vstack((x_train_combined, x[idx,:]))
                y_train_combined = np.vstack((y_train_combined, y[idx,:]))
            else:
                x_test_combined = np.vstack((x_test_combined, x[idx,:]))
                y_test_combined = np.vstack((y_test_combined, y[idx,:]))
        print("Train and test data generated !!\n")

# **********************************************************************************
# *************************** Train, load or test Model ****************************
# **********************************************************************************

#! Init model with norm layer
model = AAN_Model()
MLP_model = MLModel(model = model, type= "keras")

#! Train model
print("Training MLP model ...") 
MLP_model.trainModel(save_trained_model = True, model_filename =data_path+subject+"_"+scenario_name+result_file_name+"_AAN_model", train_epochs= n_epochs, batch_size=n_batch_size, class_weights=None, x_train=x_train_combined, y_train= y_train_combined[:,0], validation_split=validation_split, loss_fcn=loss_fcn, optimizer=optimizer,metrics=metrics, show_train_results=True)
print("MLP training done !!\n")

#! Load saved model
print("Loading saved MLP model ...")
MLP_model.loadModel(filename=subject+"_"+scenario_name+result_file_name+"_AAN_model", path=data_path)
print("Saved MLP model loaded !!\n")

#! Predict and get results 
print("Predicting joint torques ...")
MLP_model.predictTarget(data = x_test_combined, labels = y_test_combined[:,0], classification=False, show_results = False, show_pred_time = False, eval_type = "offline")

perf_results_MLP = MLP_model.getPredictionScores()
# print(perf_results_MLP)

# **********************************************************************************
# **************************** Post Prediction Filtering ***************************
# **********************************************************************************
filtered_perf_results_MLP = np.zeros(perf_results_MLP.shape)
filter_window_size = 3
for idx in range(len(perf_results_MLP)):
    if idx < filter_window_size:
        filtered_perf_results_MLP[idx] = perf_results_MLP[idx]
    else:
        filtered_perf_results_MLP[idx] = np.median(perf_results_MLP[idx-filter_window_size:idx])
        # filtered_perf_results_MLP[idx] = np.mean(perf_results_MLP[idx-filter_window_size:idx])
        # filtered_perf_results_MLP[idx] = np.median(perf_results_MLP[idx-filter_window_size:idx])

#! Plotting the filtered prediction results
plt.figure()
x_samples = np.arange(0, len(y_test_combined[:,0]),1)
plt.plot(x_samples, y_test_combined[:,0], ls="dashed", label='real torque')
plt.plot(x_samples, filtered_perf_results_MLP, label='predicted torque')
plt.legend()
plt.grid()

#! Plotting the raw prediction results
plt.figure()
x_samples = np.arange(0, len(y_test_combined[:,0]),1)
plt.plot(x_samples, y_test_combined[:,0], ls="dashed", label='real torque')
plt.plot(x_samples, perf_results_MLP, label='predicted torque')
plt.legend()
plt.grid()

#! Showing the plots
plt.show()


# # # save res 
# # np.savetxt(results_path+result_file_name, perf_results_total_MLP, delimiter=",", fmt = "%1.8f", )
# # np.savetxt(results_path+result_file_name, perf_results_total_EEGNet, delimiter=",", fmt = "%1.8f", )


# print("all done")
# print("")
# print("execution time: ")
# print(perf_counter()-time_start)


