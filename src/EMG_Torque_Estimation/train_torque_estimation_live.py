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


# subject params 
subject = "current"  # "JV43", "AV82", "UP28", "XP01", "ZS27", "JD68", "QS70"] # specify which subjects data should be evaluated
# scenario_name = "intentional_unilateral"
# result_file_name = "_live"


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
feature_indices_windows = np.arange(980, 1000, step = 1) #980,1000 means last 20 samples will be extracted from a window as a sample


# window wise metric evaluation
window_size = 1000 # windowsize in samples 
window_step = 100

# *********************************************************************************
# ***************** Main processing and classification loop ***********************
# *********************************************************************************


# init performance results list
perf_results_total_MLP = []
perf_results_total_EEGNet = []

# init early stopping 
# early_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=0,patience=early_stopping_patience,verbose=0,mode="auto",baseline=None,restore_best_weights=True)

# *********************************************************************************
# ***************** Load train, test, val sets for every iteration ****************
# *********************************************************************************

#  loading and epoching for training   
#data_train = EEGData(format = "Brainvision", filenames = train_file_list, data_path = data_path)
EMG_Data = EEGData(format = "Brainvision", filenames = train_file, data_path = data_path)


channel_names = EMG_Data.getChannelNames()
print("channel names", channel_names)
print("channel length", len(channel_names))
print("")

# **********************************************************************************
# ********************* Preprocessing for data of both networks ********************
# **********************************************************************************

window_end_indices = EMG_Data.windowContinousData(startmarkernumber = 1, stopmarkernumber = 1, window_size = window_size, window_step = window_step, start_index_offset = 0, start_channel_pick=0, end_channel_pick=10,return_window_end_indices = True)
# use the EMG_Data.windows if you want to access the windowed data 

EMG_Data.applyVarianceFilter(n_var = 20, apply_to_structures="windows")

# here you could actually set the target values I think !
#EMG_Data.setWindowLabels(target_values_list)


# time domain features 
EMG_Data.featureExtractionFromWindows(feature_type = "timepoints", feature_indices_windows = feature_indices_windows)

# input features network 
x = EMG_Data.getFeatures()
print(f"Input Feature Dim: {x.shape}")
# y = EMG_Data.getLabels()


#Compile model for right arm
# self.model_r[joint].compile(loss='mse', optimizer= self.optimizer) # optimizers: adamax, adam,adadelta, nadam with 10/5
# self.model_r[joint].fit(self.train_inp_r, self.train_out_r[joint].tolist(), epochs = self.np_epoch)


# # Load model with norm layer  
model = AAN_Model()
MLP_model = MLModel(model = model, type= "keras")
#MLP_model.trainModel(save_trained_model = True, model_filename =data_path+subject+"_"+scenario_name+result_file_name+"_AAN_model_", train_epochs= n_epochs, batch_size=n_batch_size, class_weights=None, x_train=x, y_train= y, x_val = None, y_val = None, loss_fcn=loss_fcn, optimizer=optimizer,metrics=metrics)
print("all done")



# # predict and get results 
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


