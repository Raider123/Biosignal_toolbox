# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import tensorflow as tf
from time import perf_counter
import copy 
import matplotlib.pyplot as plt
from glob import glob as g
from os import path

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
train_file = ["aan_emg_data/HW90/20170317_r_HW90_EMG_Assist_as_needed_complex_0g.vhdr",
              "aan_emg_data/HW90/20170317_r_HW90_EMG_Assist_as_needed_curl_0g.vhdr",
              "aan_emg_data/HW90/20170317_r_HW90_EMG_Assist_as_needed_grasp_0g.vhdr",
              "aan_emg_data/HW90/20170317_r_HW90_EMG_Assist_as_needed_front_0g.vhdr",
              "aan_emg_data/HW90/20170317_r_HW90_EMG_Assist_as_needed_side_0g.vhdr",
              "aan_emg_data/HW90/20170317_r_HW90_EMG_Assist_as_needed_complex_500g.vhdr",
              "aan_emg_data/HW90/20170317_r_HW90_EMG_Assist_as_needed_curl_500g.vhdr",
              "aan_emg_data/HW90/20170317_r_HW90_EMG_Assist_as_needed_grasp_500g.vhdr",
              "aan_emg_data/HW90/20170317_r_HW90_EMG_Assist_as_needed_front_500g.vhdr",
              "aan_emg_data/HW90/20170317_r_HW90_EMG_Assist_as_needed_side_500g.vhdr",
              "aan_emg_data/HW90/20170317_r_HW90_EMG_Assist_as_needed_complex_1000g.vhdr",
              "aan_emg_data/HW90/20170317_r_HW90_EMG_Assist_as_needed_curl_1000g.vhdr",
              "aan_emg_data/HW90/20170317_r_HW90_EMG_Assist_as_needed_grasp_1000g.vhdr",
              "aan_emg_data/HW90/20170317_r_HW90_EMG_Assist_as_needed_front_1000g.vhdr",
              "aan_emg_data/HW90/20170317_r_HW90_EMG_Assist_as_needed_side_1000g.vhdr",
              "aan_emg_data/HW90/20170317_r_HW90_EMG_Assist_as_needed_complex_1500g.vhdr",
              "aan_emg_data/HW90/20170317_r_HW90_EMG_Assist_as_needed_curl_1500g.vhdr",
              "aan_emg_data/HW90/20170317_r_HW90_EMG_Assist_as_needed_grasp_1500g.vhdr",
              "aan_emg_data/HW90/20170317_r_HW90_EMG_Assist_as_needed_front_1500g.vhdr",
              "aan_emg_data/HW90/20170317_r_HW90_EMG_Assist_as_needed_side_1500g.vhdr"]
target_file = ["aan_quali_data/old/HW90_quali_torque_elbow", "aan_quali_data/old/HW90_quali_torque_front", "aan_quali_data/old/HW90_quali_torque_side"]

#! Read Qualisys data param
weights_order_d=['0','500','1000','1500']
mov_type_order_d=['complex', 'curl','grasp','front','side']

#! subject params 
subject = "HW90"  # "JV43", "AV82", "UP28", "XP01", "ZS27", "JD68", "QS70"] # specify which subjects data should be evaluated
scenario_name = "all_combined"
result_file_name = "_allg"

#! fcn model parameter 
n_epochs = 20 #20 training epochs
n_batch_size = 8

#! training params 
loss_fcn =  "mse" #--> need to check 
optimizer  = "nadam" # Nadam for MLP 
metrics = "mse"

#! window wise metric evaluation
# Window params for EMG input data
window_size_x = 1000 # windowsize in samples 
window_step_x = 100
# Window params for target torque values
window_size_y = int(window_size_x/4)
window_step_y = int(window_step_x/4)

#! Window params for feature extraction !
feature_size = 20
feature_indices_windows_x = np.arange(window_size_x-feature_size, window_size_x, step = 1) #980,1000 means last 20 samples will be extracted from a window as a sample
feature_indices_windows_y = np.arange(window_size_y-1, window_size_y, step = 1)

#! Param for train/val data split
train_test_split_ratio = 0.9   # 0.x means x% of data will be training data and rest val data
validation_split = 0.2

# *********************************************************************************
# ***************** Main processing and classification loop ***********************
# *********************************************************************************


#! init performance results list
perf_results_total_MLP = []

# init early stopping 
# early_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=0,patience=early_stopping_patience,verbose=0,mode="auto",baseline=None,restore_best_weights=True)

# *********************************************************************************
# ***************** Load train, test, val sets for every iteration ****************
# *********************************************************************************

#! Loading and epoching for training   
#data_train = EEGData(format = "Brainvision", filenames = train_file_list, data_path = data_path)
EMG_Data = EEGData(format = "Brainvision", filenames = train_file, data_path = data_path)

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
Quali_Data_Elbow = EEGData(format = "Recorded_LSL_stream", filenames = [target_file[0]], data_path = data_path, f_samp=250, channel_names=channel_names_t, file_type='combined', outer_key_order_d=weights_order_d, inner_key_order_d=mov_type_order_d)
print("Creating Quali Shoulder Front object!!")
Quali_Data_Front = EEGData(format = "Recorded_LSL_stream", filenames = [target_file[1]], data_path = data_path, f_samp=250, channel_names=channel_names_t, file_type='combined', outer_key_order_d=weights_order_d, inner_key_order_d=mov_type_order_d)
print("Creating Quali Shoulder Side object!!")
Quali_Data_Side = EEGData(format = "Recorded_LSL_stream", filenames = [target_file[2]], data_path = data_path, f_samp=250, channel_names=channel_names_t, file_type='combined', outer_key_order_d=weights_order_d, inner_key_order_d=mov_type_order_d)


channel_names = EMG_Data.getChannelNames()
print("channel names", channel_names)
print("channel length", len(channel_names))
print("")

# **********************************************************************************
# ***************************** Preprocessing of data ******************************
# **********************************************************************************

window_end_indices_x = EMG_Data.windowContinuousData(startmarkernumber = 1, stopmarkernumber = 1, window_size = window_size_x, window_step = window_step_x, start_index_offset = 20, start_channel_pick=0, end_channel_pick=10,return_window_end_indices = True)

window_end_indices_y = Quali_Data_Elbow.windowContinuousData(startmarkernumber = 1, stopmarkernumber = 1, window_size = window_size_y, window_step = window_step_y, start_index_offset = 0, start_channel_pick=0, end_channel_pick=10,return_window_end_indices = True)
# use the EMG_Data.windows if you want to access the windowed data 

#! Plot specific unfiltered windows for debugging
# plt.figure()
# plt.plot(EMG_Data.getWindows()[0,2,:,18])
# plt.show()

#! Variance Filter
print("Applying Variance filter ...")
EMG_Data.applyVarianceFilter(n_var = 20, apply_to_structures="windows")
print("Variance Filter applied!!\n")


#! Plot and print specific variance filtered windows 
# var_filtered_window_x = EMG_Data.getWindows()
# print(f"Shape of Variance filtered windows: {var_filtered_window_x.shape}")
# print(f"Variance filtered windows: {var_filtered_window_x[0,2,:,18]}")
# plt.figure()
# plt.plot(var_filtered_window_x[0,2,:,18])
# plt.show()

#! Normalisation
print("Performing Normalization with Max Voluntary Contraction ...")
#Calculate the maximum value of the entire data (channel-wise)
EMG_Data.calcCalibStats()
#Get the max values of each channel from the flattened windows(overlapping)
_,_,_, data_maxima = EMG_Data.getCalibStats()
#Normalise the data
EMG_Data.windowStandardization(method='max_norm')
print("Normalization with Max Voluntary Contraction performed !!\n")

#! Calculate Neural Activation Force
print("Replacing sample with its force activation value ...")
EMG_Data.calculateActivationForceFunction(d=50, c1=0.5, c2=-0.5, nonlinear_shape_factor=-1.5)
print("Replaced each sample with its force activation value !!\n")

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
# y = EMG_Data.getLabels()
y = Quali_Data_Elbow.getFeatures()[:,0:2]

#! Ensure same number of rows for imput and target features
end_idx = x.shape[0] if x.shape[0] <= y.shape[0] else y.shape[0]
x = x[0:end_idx,:]
y = y[0:end_idx,0:2]
print(f"Input Feature Dim: {x.shape}")
print(f"Output Feature Dim: {y.shape}")

#! Split data into train and test
print("Splitting train and test data ...")
#Creating empty train and val arrays
x_train = np.empty(shape=[0,x.shape[1]])
x_test  = np.empty(shape=[0,x.shape[1]])
y_train = np.empty(shape=[0,y.shape[1]])
y_test  = np.empty(shape=[0,y.shape[1]])
#Loop over the data and split it
for idx in range(end_idx):
    if idx <= round(train_test_split_ratio*end_idx):
        x_train = np.vstack((x_train, x[idx,:]))
        y_train = np.vstack((y_train, y[idx,:]))
    else:
        x_test = np.vstack((x_test, x[idx,:]))
        y_test = np.vstack((y_test, y[idx,:]))
print("Train and test data generated !!\n")

#Compile model for right arm
# self.model_r[joint].compile(loss='mse', optimizer= self.optimizer) # optimizers: adamax, adam,adadelta, nadam with 10/5
# self.model_r[joint].fit(self.train_inp_r, self.train_out_r[joint].tolist(), epochs = self.np_epoch)

# **********************************************************************************
# *************************** Train, load or test Model ****************************
# **********************************************************************************

#! Init model with norm layer
model = AAN_Model()
MLP_model = MLModel(model = model, type= "keras")

#! Train model
print("Training MLP model ...") 
MLP_model.trainModel(save_trained_model = True, model_filename =data_path+subject+"_"+scenario_name+result_file_name+"_AAN_model", train_epochs= n_epochs, batch_size=n_batch_size, class_weights=None, x_train=x_train, y_train= y_train[:,0], validation_split=validation_split, loss_fcn=loss_fcn, optimizer=optimizer,metrics=metrics, show_train_results=True)
print("MLP training done !!\n")

#! Load saved model
print("Loading saved MLP model ...")
MLP_model.loadModel(filename=subject+"_"+scenario_name+result_file_name+"_AAN_model", path=data_path)
print("Saved MLP model loaded !!\n")

#! Predict and get results 
print("Predicting joint torques ...")
MLP_model.predictTarget(data = x_test, labels = y_test[:,0], classification=False, show_results = False, show_pred_time = False, eval_type = "offline")

perf_results_MLP = MLP_model.getPredictionScores()
# print(perf_results_MLP)

#! Plotting the prediction results
plt.figure()
x_samples = np.arange(0, len(y_test[:,0]),1)
plt.plot(x_samples, y_test[:,0], ls="dashed", label='real torque')
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


