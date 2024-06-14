
# # own libs 
from biosignal_toolbox.eeg_lib import EEGData
from biosignal_toolbox.ML_lib import MLModel
import biosignal_toolbox.ML_pipelines_lib as pipeline
import matplotlib.pyplot as plt 
import copy 
import numpy as np 
import tensorflow as tf

# models 
from biosignal_toolbox.models.autoencoderNet import FilterNet, FilterNetRNN

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

train_file_list = ["20211210_r_JV43_intentional_unilateral_set1.vhdr","20211210_r_JV43_intentional_unilateral_set2.vhdr", 
                   "20211216_r_RA12_intentional_unilateral_set1.vhdr", "20211216_r_RA12_intentional_unilateral_set2.vhdr", "20211216_r_RA12_intentional_unilateral_set3.vhdr", 
                   "20211220_r_AV82_intentional_unilateral_set1.vhdr", "20211220_r_AV82_intentional_unilateral_set2.vhdr", "20211220_r_AV82_intentional_unilateral_set3.vhdr", 
                   "20220107_r_QS70_intentional_unilateral_set1.vhdr", "20220107_r_QS70_intentional_unilateral_set2.vhdr", "20220107_r_QS70_intentional_unilateral_set3.vhdr", 
                   "20211222_r_XP01_intentional_unilateral_set1.vhdr", "20211222_r_XP01_intentional_unilateral_set2.vhdr", "20211222_r_XP01_intentional_unilateral_set3.vhdr", "20211222_r_XP01_intentional_unilateral_set4.vhdr", 
                   "20220107_r_QS70_intentional_unilateral_set1.vhdr", "20220107_r_QS70_intentional_unilateral_set2.vhdr", "20220107_r_QS70_intentional_unilateral_set3.vhdr", 
                   "20220104_r_ZS27_intentional_unilateral_set1.vhdr", "20220104_r_ZS27_intentional_unilateral_set2.vhdr", "20220104_r_ZS27_intentional_unilateral_set3.vhdr", 
                   "20211220_r_AV82_intentional_unilateral_set1.vhdr", "20211220_r_AV82_intentional_unilateral_set2.vhdr", "20211220_r_AV82_intentional_unilateral_set3.vhdr"]

val_test_file_list = ["20211210_r_JV43_intentional_unilateral_set3.vhdr"]
filename_model = "pooling_model_filterNet_05_040Hz_1000ms"

train_windows = ["bis-2500", "bis-2400", "bis-2300", "bis-2000", "bis-1900", "bis-1800", "bis-100", "bis-80", "bis-60", "bis-40", "bis-20", "bis0"]

f_samp_eeg = 500 #sample Frequency of eeg

# window wise metric evaluation
window_size = 1000 #windowsize in ms 
window_step = 20 # stepsize in ms

n_epochs = 100
batch_size = 16 # 

# training params 
loss_fcn =  "mean_absolute_error" 
optimizer  = "adam" # Nadam for MLP 
#optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
metrics = "mean_absolute_error"

# paradigm params 
marker_number = 100
error_number = 3
channel_list = ["F5", "F6", "x_dir", "y_dir", "z_dir", "FP1", "FP2", "F8", "T7", "T8", "TP9", "TP10", "P7", "P8", "PO9", "O1", "OZ", "O2", "PO10", "AF7", "AF3", "AF4", "AF8", "FT9", "FT7", "FT8", "FT10", "TP7", "TP8", "PO7", "PO3", "POZ", "PO4", "PO8", "F7"]
inverse_keep_channel = True # standard: True 
# just remap the parameters (need to be adapted)
t1 = -5.0
t2 = 0.0

early_stopping_patience = 50

target_flowpass = 40.0
target_fhighpass = 0.5


# *********************************************************************************
# ***************** Main processing and classification loop ***********************
# *********************************************************************************

EEG_data_raw = EEGData(format = "Brainvision", filenames = train_file_list, data_path = data_path)
EEG_data_processed = copy.deepcopy(EEG_data_raw) # make a copy before preprocessing 

# val test 
EEG_data_raw_val_test = EEGData(format = "Brainvision", filenames = val_test_file_list, data_path = data_path)
EEG_data_processed_val_test = copy.deepcopy(EEG_data_raw_val_test) # make a copy before preprocessing 

# process as raw data 
EEG_data_raw.rereferencingEpoching(marker_number, error_number, channel_list, inverse_keep_channel = inverse_keep_channel, t1 = t1, t2= t2) 
# process with filtering 
EEG_data_processed.rereferencingEpoching(marker_number, error_number, channel_list, apply_filter=True, f_highpass = target_fhighpass, f_lowpass= target_flowpass, inverse_keep_channel = inverse_keep_channel, t1 = t1, t2= t2) 

# for validation data 
# process as raw data 
EEG_data_raw_val_test.rereferencingEpoching(marker_number, error_number, channel_list, inverse_keep_channel = inverse_keep_channel, t1 = t1, t2= t2) 
# process with filtering 
EEG_data_processed_val_test.rereferencingEpoching(marker_number, error_number, channel_list, apply_filter=True, f_highpass = target_fhighpass, f_lowpass= target_flowpass, inverse_keep_channel = inverse_keep_channel, t1 = t1, t2= t2) 

# split train and test epochs 
EEG_data_raw_val, EEG_data_raw_test = EEG_data_raw_val_test.splitTrainTestEpochs(n_test_epochs=20) 
EEG_data_processed_val, EEG_data_processed_test = EEG_data_processed_val_test.splitTrainTestEpochs(n_test_epochs=20)

# windowing 
EEG_data_raw.windowEEGEpochs(window_size = window_size, window_step = window_step)
EEG_data_processed.windowEEGEpochs(window_size = window_size, window_step = window_step)

# val test windowing 
EEG_data_raw_val.windowEEGEpochs(window_size = window_size, window_step = window_step)
EEG_data_raw_test.windowEEGEpochs(window_size = window_size, window_step = window_step)

EEG_data_processed_val.windowEEGEpochs(window_size = window_size, window_step = window_step)
EEG_data_processed_test.windowEEGEpochs(window_size = window_size, window_step = window_step)

# do some preprocessing in advance ? 

# sos = EEG_data_raw.designFilter(f_low = None, f_high = 0.2, order = 2, filter_type = "scipy_butter", return_type = "sos")
# EEG_data_raw.filterWindows(sos = sos, apply_method = "zero_phase_sos") # bandpass filter (zero phase with padding)
# EEG_data_raw_val.filterWindows(sos = sos, apply_method = "zero_phase_sos") # bandpass filter (zero phase with padding)
# EEG_data_raw_test.filterWindows(sos = sos, apply_method = "zero_phase_sos") # bandpass filter (zero phase with padding)

print("********")
print("Window names: ", EEG_data_processed.getWindowNames())
print("********")

EEG_data_raw.windowSelection(train_windows)
EEG_data_processed.windowSelection(train_windows)

EEG_data_raw_val.windowSelection(train_windows)
EEG_data_raw_test.windowSelection(train_windows)
EEG_data_processed_val.windowSelection(train_windows)
EEG_data_processed_test.windowSelection(train_windows)


# reshape for nets 
EEG_data_raw.reshapeWindowsForCNNnets()
EEG_data_processed.reshapeWindowsForCNNnets()

EEG_data_raw_val.reshapeWindowsForCNNnets()
EEG_data_raw_test.reshapeWindowsForCNNnets()
EEG_data_processed_val.reshapeWindowsForCNNnets()
EEG_data_processed_test.reshapeWindowsForCNNnets()


# get data for training 
x_EEG_raw = EEG_data_raw.getWindows()
x_EEG_processed = EEG_data_processed.getWindows()

x_EEG_raw_val = EEG_data_raw_val.getWindows()
x_EEG_processed_val = EEG_data_processed_val.getWindows()
x_EEG_raw_test = EEG_data_raw_test.getWindows()
x_EEG_processed_test = EEG_data_processed_test.getWindows()


# init early stopping 
early_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=0,patience=early_stopping_patience,verbose=0,mode="auto",baseline=None,restore_best_weights=True)

AE_model = FilterNet(Samples = int(window_size/2), Chans = len(EEG_data_raw.getChannelNames())) #FilterNet() # should simply be a bandpass filter 
ml_model = MLModel(model = AE_model, type="keras")
ml_model.modelSummary()

# # # train it 
ml_model.trainModel(show_train_results=True, callbacks=early_callback, train_epochs=n_epochs, shuffle = True, x_train =  x_EEG_raw, y_train =x_EEG_processed, x_val = x_EEG_raw_val, y_val =x_EEG_processed_val, loss_fcn = loss_fcn, metrics = metrics, optimizer = optimizer,save_trained_model=True, model_filename=data_path+filename_model) 

trial = 6
channel = 3

# test windows 
window_to_predict = np.zeros((1, EEG_data_raw_test.windows.shape[1], EEG_data_raw_test.windows.shape[2], 1))
window_to_predict[0, :, :, 0] = EEG_data_raw_test.windows[trial, :, :, 0]

ml_model.predict(data = window_to_predict, classification = False, show_pred_time = True)
output = ml_model.getPredictionScores()

print("output shape", output.shape)

# # show the signals before training 
# plt.figure()
# plt.plot(EEG_data_raw_test.windows[trial, channel, :, 0])
# plt.plot(EEG_data_processed.windows[trial, channel, :, 0])
# plt.legend(["raw", "processed"])


# # show the signals before training 
# plt.figure()
# plt.plot(EEG_data_raw.windows[trial, channel, :, 0])
# plt.plot(output[0, 7, :, 0])
# plt.legend(["raw", "filter model"])


# show the signals before training 
plt.figure()
plt.plot(EEG_data_processed_test.windows[trial, channel, :, 0])
plt.plot(output[0, channel, :, 0])
plt.legend(["processed", "filter model"])

# plots for different trials 
trial = 10
channel = 3

window_to_predict = np.zeros((1, EEG_data_raw_test.windows.shape[1], EEG_data_raw_test.windows.shape[2], 1))
window_to_predict[0, :, :, 0] = EEG_data_raw_test.windows[trial, :, :, 0]

ml_model.predict(data = window_to_predict, classification = False, show_pred_time = True)
output = ml_model.getPredictionScores()

# show the signals before training 
plt.figure()
plt.plot(EEG_data_processed_test.windows[trial, channel, :, 0])
plt.plot(output[0, channel, :, 0])
plt.legend(["processed", "filter model"])



# plots for different trials 
trial = 3
channel = 3
print(EEG_data_raw_test.windows.shape)
window_to_predict = np.zeros((1, EEG_data_raw_test.windows.shape[1], EEG_data_raw_test.windows.shape[2], 1))
window_to_predict[0, :, :, 0] = EEG_data_raw_test.windows[trial, :, :, 0]

ml_model.predict(data = window_to_predict, classification = False, show_pred_time = True)
output = ml_model.getPredictionScores()

# show the signals before training 
plt.figure()
plt.plot(EEG_data_processed_test.windows[trial, channel, :, 0])
plt.plot(output[0, channel, :, 0])
plt.legend(["processed", "filter model"])
plt.show()



# print layer names 
# ml_model.printKerasModelLayerNames()
# ml_model.printKerasModelLayerWeights(layer_name="custom_scaling_layer")


# EEG_data_processed.windows[trial, channel, :, 0]
# output[0, channel, :, 0]

# #np.save("X.npy", output[0, channel, :, 0])
# np.save("X.npy", EEG_data_raw.windows[trial, channel, :, 0])
# np.save("Y.npy", EEG_data_processed.windows[trial, channel, :, 0])


# window_to_predict[0, :, :, 0] = EEG_data_raw.windows[trial+1, channel, :, 0]
# ml_model.predict(data = window_to_predict, classification = False, show_pred_time = True)
# output1 = ml_model.getPredictionScores()


# #np.save("X_test.npy", output1[0, channel, :, 0])

# np.save("X_test.npy", EEG_data_raw.windows[trial+1, channel, :, 0])
# np.save("Y_test.npy", EEG_data_processed.windows[trial+1, channel, :, 0])



