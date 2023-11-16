# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import matplotlib.pyplot as plt
import numpy as np
from dtw import *


# # own libs 
from biosignal_toolbox.eeg_lib import EEGData
from biosignal_toolbox.ML_lib import MLModel
import biosignal_toolbox.ML_pipelines_lib as pipeline
from mne.preprocessing import Xdawn
import copy

# own libs
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox"


# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************

data_path = proj_path+"/data/"
results_path = proj_path+"/results/"

marker_number = 22
error_number = 3
f_samp_eeg = 500.0

channel_names = ["F5", "F3", "F1", "FZ", "F2", "F4", "F6", "FC5", "FC3", "FC1", "FC2", "FC4", "FC6", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "P5", "P3", "P1", "PZ", "P2", "P4", "P6"]
inverse_keep_channel = True 
channel_list = []
t1 = -2.0
t2 = 0.0

window_size = 1100
window_step = 50
n_components = 1

# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************

#use LSL file recorded 
train_file_LSL = ["XY90_unilateral_set3_data"] #"BR60D_unilateral_live_2_data", "BR60D_intentional_unilateral_set8_data", ]
train_file_LSL_val = ["XY90_unilateral_set4_data"]

EEG_data = EEGData(format = "Recorded_LSL_stream", filenames = train_file_LSL, data_path = data_path, f_samp = f_samp_eeg, channel_names = channel_names)
EEG_data_val = EEGData(format = "Recorded_LSL_stream", filenames = train_file_LSL_val, data_path = data_path, f_samp = f_samp_eeg, channel_names = channel_names)

# x daWN fitting on training data 
EEG_data_xDAWN = copy.deepcopy(EEG_data)
EEG_data_xDAWN.rereferencingEpoching(marker_number, error_number, channel_list, apply_filter=True, f_highpass = 0.5, f_lowpass= 4.0, inverse_keep_channel = inverse_keep_channel, event_id_used = marker_number, t1 = t1, t2= t2)
#EEG_data_xDAWN.epoch_obj.filter(l_freq = 0.5, h_freq = 4.0, picks=None, method='fir', iir_params=None, phase='zero', filter_length = 1000, fir_design='firwin2')
# EEG_data_xDAWN.epochs = EEG_data_xDAWN.epoch_obj.get_data()

# estimate filter
xd_filt = EEG_data_xDAWN.xDAWNSpatialfilter(n_components = n_components, processing_type="fit", return_filter = True)

t1 = -5.1
EEG_data.rereferencingEpoching(marker_number, error_number, channel_list, apply_filter=False, f_highpass = None, f_lowpass= None, inverse_keep_channel = inverse_keep_channel, event_id_used = marker_number, t1 = t1, t2= t2)
EEG_data_val.rereferencingEpoching(marker_number, error_number, channel_list, apply_filter=False, f_highpass = None, f_lowpass= None, inverse_keep_channel = inverse_keep_channel, event_id_used = marker_number, t1 = t1, t2= t2)


# filtering pseudo online on epochs 
# EEG_data.epoch_obj.filter(l_freq = 0.5, h_freq = 4.0, picks=None, method='fir', iir_params=None, phase='zero', filter_length = 1000, fir_design='firwin2')
# EEG_data.epochs = EEG_data.epoch_obj.get_data()
# EEG_data_val.epoch_obj.filter(l_freq = 0.5, h_freq = 4.0, picks=None, method='fir', iir_params=None, phase='zero', filter_length = 1000, fir_design='firwin2')
# EEG_data_val.epochs = EEG_data_val.epoch_obj.get_data()


#Epochs | Evoked | ndarray, shape ([n_epochs, ]n_channels, n_times) --> trials, channel, sampels 
# train data 
EEG_data.windowEEGEpochs(window_size, window_step)
EEG_data.windowSelection(["bis-2000", "bis-2500", "bis-100", "bis0"]) # shape of wind trials, channel, sampels, windows 
EEG_data.setWindowLabels([0.0, 0.0, 1.0, 1.0])
EEG_data.FilterWindows(f_low = 5.0, f_high = 0.3, order = 2, filter_type = "scipy_butter")
EEG_data.windows = EEG_data.windows[:, :, 25:-25, :]
EEG_data.applyxDAWNToWindows(xd_filt, n_components = n_components)
#EEG_data.minMaxNormWindows()

# generate templates for matching algorithm 
windows_template = EEG_data.getWindows()
pos_class_temp = np.mean(windows_template[:, :, :, -1], axis = 0)
neg_class_temp = np.mean(windows_template[:, :, :, 0], axis = 0)


print("template shape", neg_class_temp.shape)
# pos_class_temp_norm = pos_class_temp
# neg_class_temp_norm = neg_class_temp

pos_class_temp_norm = np.zeros(pos_class_temp.shape)
pos_class_temp_norm[:] = pos_class_temp[:] - (np.median(pos_class_temp[0:50]))
#pos_class_temp_norm = pos_class_temp_norm/np.std(pos_class_temp_norm)

#pos_class_temp_norm[:] = pos_class_temp[:] + (-1 *np.min(pos_class_temp[:]))
#pos_class_temp_norm[:] = pos_class_temp_norm[:] / np.max(pos_class_temp_norm[:])
# pos_class_temp_norm[1, :] = pos_class_temp[1, :] + (-1 *np.min(pos_class_temp[1, :]))
# pos_class_temp_norm[1, :] = pos_class_temp_norm[1, :] / np.max(pos_class_temp_norm[1, :])

neg_class_temp_norm = np.zeros(neg_class_temp.shape)
neg_class_temp_norm[:] = neg_class_temp[:] - (np.median(neg_class_temp[0:50]))
#neg_class_temp_norm = neg_class_temp_norm/np.std(neg_class_temp_norm)

#neg_class_temp_norm[:] = neg_class_temp[:] + (-1 *np.min(neg_class_temp[:]))
#neg_class_temp_norm[:] = neg_class_temp_norm[:] / np.max(pos_class_temp_norm[:])
# neg_class_temp_norm[1, :] = neg_class_temp[1, :] + (-1 *np.min(neg_class_temp[1, :]))
# neg_class_temp_norm[1, :] = neg_class_temp_norm[1, :] / np.max(pos_class_temp_norm[1, :])


# validation set 
EEG_data_val.windowEEGEpochs(window_size, window_step)
# on all windows 
labels = np.zeros(81)
labels[-6:] = 1.0
EEG_data_val.setWindowLabels(labels)
EEG_data_val.FilterWindows(f_low = 5.0, f_high = 0.3, order = 2, filter_type = "scipy_butter")
EEG_data_val.windows = EEG_data_val.windows[:, :, 25:-25, :] # cut off artifact samples 
EEG_data_val.applyxDAWNToWindows(xd_filt, n_components = n_components)
EEG_data_val.windowStandardization()
#EEG_data_val.minMaxNormWindows()
EEG_data_val.WindowMedianCorrection()

plt.figure()
plt.plot(pos_class_temp_norm.T)

# plt.figure()
# plt.plot(EEG_data_val.windows[10, :, :, -1].T)
# plt.figure()
# plt.plot(EEG_data_val.windows[9, :, :, -1].T)
# plt.figure()
# plt.plot(EEG_data_val.windows[8, :, :, -1].T)
# plt.figure()
# plt.plot(EEG_data_val.windows[7, :, :, -1].T)
# plt.show()

plt.figure()
for window_idx in range(0, EEG_data_val.windows.shape[3] -5): 
    plt.plot(EEG_data_val.windows[10, :, :, window_idx].T)

plt.figure()
for window_idx in range(EEG_data_val.windows.shape[3] -5, EEG_data_val.windows.shape[3]): 
    plt.plot(EEG_data_val.windows[10, :, :, window_idx].T)
plt.show()


EEG_data.dtwFeatureVecWindows()
EEG_data_val.dtwFeatureVecWindows()

# get features 
x_train = EEG_data.getFeatures()
y_train = EEG_data.getLabels()
x_val = EEG_data_val.getFeatures()
y_val = EEG_data_val.getLabels()


dtw_model = MLModel(x_train=x_train, y_train= y_train, x_val = x_val, y_val = y_val, type = "dtw")

# predict and get results 
dtw_model.predict(data = x_val, labels = y_val, encoding = "distance_array", show_results = True, show_pred_time = False, eval_type = "offline", templates= [pos_class_temp_norm, neg_class_temp_norm])


# # perf_results_MLP = MLP_model.getPerfResults()







