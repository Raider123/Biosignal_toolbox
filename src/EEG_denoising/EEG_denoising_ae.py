
# # own libs 
from biosignal_toolbox.eeg_lib import EEGData
from biosignal_toolbox.ML_lib import MLModel
import biosignal_toolbox.ML_pipelines_lib as pipeline
import matplotlib.pyplot as plt 
import copy 
import numpy as np 
import tensorflow as tf

# models 
from biosignal_toolbox.models.autoencoderNet import Autoencoder_Net, FilterNet, Conv2D2KernelLayer, MLPFilter, FilterNetV2

# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************

# own libs
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox"

data_path = proj_path+"/data/"
results_path = proj_path+"/results/"

# disable GPU for testing
#tf.config.set_visible_devices([], 'GPU')

# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************
data_path = proj_path+"/data/"
results_path = proj_path+"/results/"

train_file_list = ["20211210_r_JV43_intentional_unilateral_set1.vhdr","20211210_r_JV43_intentional_unilateral_set2.vhdr", "20211210_r_JV43_intentional_unilateral_set3.vhdr"]

train_windows = ["bis-26", "bis-2526"]

f_samp_eeg = 500 #sample Frequency of eeg

# window wise metric evaluation
window_size = 1024 #windowsize in ms 
window_step = 50 # stepsize in ms

n_epochs = 200
batch_size = 16 # 

# training params 
loss_fcn =  "mean_absolute_error" 
optimizer  = "adam" # Nadam for MLP 
metrics = "mean_absolute_error"

# paradigm params 
marker_number = 100
error_number = 3
channel_list = ["x_dir", "y_dir", "z_dir", "FP1", "FP2", "F8", "T7", "T8", "TP9", "TP10", "P7", "P8", "PO9", "O1", "OZ", "O2", "PO10", "AF7", "AF3", "AF4", "AF8", "FT9", "FT7", "FT8", "FT10", "TP7", "TP8", "PO7", "PO3", "POZ", "PO4", "PO8", "F7"]
inverse_keep_channel = True # standard: True 
# just remap the parameters (need to be adapted)
t1 = -5.0
t2 = 0.0

early_stopping_patience = 50

# *********************************************************************************
# ***************** Main processing and classification loop ***********************
# *********************************************************************************

EEG_data_raw = EEGData(format = "Brainvision", filenames = train_file_list, data_path = data_path)
EEG_data_processed = copy.deepcopy(EEG_data_raw) # make a copy before preprocessing 

# process as raw data 
EEG_data_raw.rereferencingEpoching(marker_number, error_number, channel_list, inverse_keep_channel = inverse_keep_channel, t1 = t1, t2= t2) 
# process with filtering 
EEG_data_processed.rereferencingEpoching(marker_number, error_number, channel_list, apply_filter=True, f_highpass = 0.5, f_lowpass= 4.0, inverse_keep_channel = inverse_keep_channel, t1 = t1, t2= t2) 

# windowing 
EEG_data_raw.windowEEGEpochs(window_size = window_size, window_step = window_step)
EEG_data_processed.windowEEGEpochs(window_size = window_size, window_step = window_step)

EEG_data_raw.windowSelection(train_windows)
EEG_data_processed.windowSelection(train_windows)

# reshape for nets 
EEG_data_raw.reshapeWindowsForCNNnets()
EEG_data_processed.reshapeWindowsForCNNnets()

print(EEG_data_raw.windows.shape)


# get data for training 
x_EEG_raw = EEG_data_raw.getWindows()
x_EEG_processed = EEG_data_processed.getWindows()

# init early stopping 
early_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=0,patience=early_stopping_patience,verbose=0,mode="auto",baseline=None,restore_best_weights=True)

AE_model = FilterNetV2() #FilterNet() # should simply be a bandpass filter 
ml_model = MLModel(model = AE_model, type="keras")
#ml_model.modelSummary()

# # # train it 
ml_model.trainModel(show_train_results=True, callbacks=early_callback, train_epochs=n_epochs, shuffle = True, x_train =  x_EEG_raw, y_train =x_EEG_processed, x_val = x_EEG_processed, y_val = x_EEG_processed, loss_fcn = loss_fcn, metrics = metrics, optimizer = optimizer) 

trial = 20
channel = 3

print(EEG_data_raw.windows.shape)
window_to_predict = np.zeros((1, EEG_data_raw.windows.shape[1], EEG_data_raw.windows.shape[2], 1))
window_to_predict[0, :, :, 0] = EEG_data_raw.windows[trial, :, :, 0]

ml_model.predict(data = window_to_predict, classification = False, show_pred_time = True)
output = ml_model.getPredictionScores()

print("output shape", output.shape)

# show the signals before training 
plt.figure()
plt.plot(EEG_data_raw.windows[trial, channel, :, 0])
plt.plot(EEG_data_processed.windows[trial, channel, :, 0])
plt.legend(["raw", "processed"])


# show the signals before training 
plt.figure()
plt.plot(EEG_data_raw.windows[trial, channel, :, 0])
plt.plot(output[0, 7, :, 0])
plt.legend(["raw", "filter model"])


# show the signals before training 
plt.figure()
plt.plot(EEG_data_processed.windows[trial, channel, :, 0])
plt.plot(output[0, channel, :, 0])
plt.legend(["processed", "filter model"])
plt.show()

# print layer names 
ml_model.printKerasModelLayerNames()
ml_model.printKerasModelLayerWeights(layer_name="time_distributed")

