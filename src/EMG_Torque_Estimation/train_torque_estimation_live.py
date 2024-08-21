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

# use LSL file recorded 
train_file = ["aan_emg_data/HW90/20170317_r_HW90_EMG_Assist_as_needed_complex_0g.vhdr"] #"BR60D_unilateral_live_2_data", "BR60D_intentional_unilateral_set8_data", ]
target_file = ["aan_quali_data/quali_torque_elbow", "aan_quali_data/quali_torque_front", "aan_quali_data/quali_torque_side"]


# subject params 
subject = "HW90"  # "JV43", "AV82", "UP28", "XP01", "ZS27", "JD68", "QS70"] # specify which subjects data should be evaluated
scenario_name = "complex"
result_file_name = "_0g"


# fcn model parameter 
n_epochs = 20 #20 training epochs
n_batch_size = 8

# training params 
loss_fcn =  "mse" #--> need to check 
optimizer  = "nadam" # Nadam for MLP 
metrics = "mse"


# # training windows and features
# train_windows = ["bis-2500", "bis-1900", "bis-1500" ,"bis-1200", "bis-150", "bis-100", "bis-50", "bis0"]#, "bis-50", "bis0"] # alternatively 

# window_target_values = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]# alternative 

# might be used for feature extraction !
feature_indices_windows_x = np.arange(980, 1000, step = 1) #980,1000 means last 20 samples will be extracted from a window as a sample
feature_indices_windows_y = np.arange(249, 250, step = 1)


#! window wise metric evaluation
# Window params for EMG input data
window_size_x = 1000 # windowsize in samples 
window_step_x = 100
# Window params for target torque values
window_size_y = int(1000/4)
window_step_y = int(window_step_x/4)

#! Param for train/val data split
split_ratio = 0.8   # 0.x means x% of data will be training data and rest val data

# *********************************************************************************
# ***************** Main processing and classification loop ***********************
# *********************************************************************************


#! init performance results list
perf_results_total_MLP = []
perf_results_total_EEGNet = []

# init early stopping 
# early_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=0,patience=early_stopping_patience,verbose=0,mode="auto",baseline=None,restore_best_weights=True)

# *********************************************************************************
# ***************** Load train, test, val sets for every iteration ****************
# *********************************************************************************

#! Loading and epoching for training   
#data_train = EEGData(format = "Brainvision", filenames = train_file_list, data_path = data_path)
EMG_Data = EEGData(format = "Brainvision", filenames = train_file, data_path = data_path)

#! Plotting the raw EMG data
# plt.plot(np.arange(0,EMG_Data.data[3,:].shape[0], 1)/1000,EMG_Data.data[4,:]*1e6)
# plt.title("Raw EMG plot for Channel 3")
# plt.grid()
# plt.xlabel("Time in s")
# plt.ylabel("Amplitude in uV")
# plt.show()

#! Loading the target values for the 3 joints
channel_names_t = ['right', 'left', 'marker']
print("Creating Quali Elbow object!!")
Quali_Data_Elbow = EEGData(format = "Recorded_LSL_stream", filenames = [target_file[0]], data_path = data_path, f_samp=250, channel_names=channel_names_t)
print("Creating Quali Shoulder Front object!!")
Quali_Data_Front = EEGData(format = "Recorded_LSL_stream", filenames = [target_file[1]], data_path = data_path, f_samp=250, channel_names=channel_names_t)
print("Creating Quali Shoulder Side object!!")
Quali_Data_Side = EEGData(format = "Recorded_LSL_stream", filenames = [target_file[2]], data_path = data_path, f_samp=250, channel_names=channel_names_t)


channel_names = EMG_Data.getChannelNames()
print("channel names", channel_names)
print("channel length", len(channel_names))
print("")

# **********************************************************************************
# ********************* Preprocessing for data of both networks ********************
# **********************************************************************************

window_end_indices_x = EMG_Data.windowContinousData(startmarkernumber = 1, stopmarkernumber = 1, window_size = window_size_x, window_step = window_step_x, start_index_offset = 20, start_channel_pick=0, end_channel_pick=10,return_window_end_indices = True)

window_end_indices_y = Quali_Data_Elbow.windowContinousData(startmarkernumber = 1, stopmarkernumber = 1, window_size = window_size_y, window_step = window_step_y, start_index_offset = 0, start_channel_pick=0, end_channel_pick=10,return_window_end_indices = True)
# use the EMG_Data.windows if you want to access the windowed data 

EMG_Data.applyVarianceFilter(n_var = 20, apply_to_structures="windows")

# here you could actually set the target values I think !
#EMG_Data.setWindowLabels(target_values_list)


#! time domain feature extraction
EMG_Data.featureExtractionFromWindows(feature_type = "timepoints", feature_indices_windows = feature_indices_windows_x)
Quali_Data_Elbow.featureExtractionFromWindows(feature_type = "timepoints", feature_indices_windows = feature_indices_windows_y)

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

#! Split data into training and validation
# Creating empty train and val arrays
x_train = np.empty(shape=[0,x.shape[1]])
x_val   = np.empty(shape=[0,x.shape[1]])
y_train = np.empty(shape=[0,y.shape[1]])
y_val   = np.empty(shape=[0,y.shape[1]])
# Loop over the data and split it
for idx in range(end_idx):
    if idx <= round(split_ratio*end_idx):
        x_train = np.vstack((x_train, x[idx,:]))
        y_train = np.vstack((y_train, y[idx,:]))
    else:
        x_val = np.vstack((x_val, x[idx,:]))
        y_val = np.vstack((y_val, y[idx,:]))


#Compile model for right arm
# self.model_r[joint].compile(loss='mse', optimizer= self.optimizer) # optimizers: adamax, adam,adadelta, nadam with 10/5
# self.model_r[joint].fit(self.train_inp_r, self.train_out_r[joint].tolist(), epochs = self.np_epoch)


#! Load model with norm layer  
model = AAN_Model()
MLP_model = MLModel(model = model, type= "keras")
MLP_model.trainModel(save_trained_model = True, model_filename =data_path+subject+"_"+scenario_name+result_file_name+"_AAN_model", train_epochs= n_epochs, batch_size=n_batch_size, class_weights=None, x_train=x_train, y_train= y_train[:,0], x_val = x_val, y_val = y_val[:,0], loss_fcn=loss_fcn, optimizer=optimizer,metrics=metrics, show_train_results=True)
print("all done")



#! Predict and get results 
# print("predict MLP net")
# MLP_model.predict(data = x_train_MLP, labels = y_train_MLP, encoding = "binary", show_results = True, show_pred_time = False, eval_type = "offline")
# perf_results_MLP = MLP_model.getPerfResults()



# # # save res 
# # np.savetxt(results_path+result_file_name, perf_results_total_MLP, delimiter=",", fmt = "%1.8f", )
# # np.savetxt(results_path+result_file_name, perf_results_total_EEGNet, delimiter=",", fmt = "%1.8f", )


# print("all done")
# print("")
# print("execution time: ")
# print(perf_counter()-time_start)


