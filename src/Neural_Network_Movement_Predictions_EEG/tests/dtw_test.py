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
EEG_data.WindowMeanCorrection()
EEG_data.minMaxNormWindows()

# generate templates for matching algorithm 
windows_template = EEG_data.getWindows()
pos_class_temp = np.mean(windows_template[:, :, :, -1], axis = 0)
neg_class_temp = np.mean(windows_template[:, :, :, 0], axis = 0)

# pos_class_temp = (windows_template[0, :, :, -1])
# neg_class_temp = (windows_template[0, :, :, 0])


print("template shape", neg_class_temp.shape)
# pos_class_temp_norm = pos_class_temp
# neg_class_temp_norm = neg_class_temp

pos_class_temp_norm = np.zeros(pos_class_temp.shape)
pos_class_temp_norm[:] = pos_class_temp[:] - (np.mean(pos_class_temp))

# pos_class_temp_norm[:] = pos_class_temp_norm[:] + (-1 *np.min(pos_class_temp_norm[:]))
# pos_class_temp_norm[:] = pos_class_temp_norm[:] / np.max(pos_class_temp_norm[:])

neg_class_temp_norm = np.zeros(neg_class_temp.shape)
neg_class_temp_norm[:] = neg_class_temp[:] -(np.mean(neg_class_temp))

# neg_class_temp_norm[:] = neg_class_temp_norm[:] + (-1 *np.min(neg_class_temp_norm[:]))
# neg_class_temp_norm[:] = neg_class_temp_norm[:] / np.max(pos_class_temp_norm[:])


# validation set 
EEG_data_val.windowEEGEpochs(window_size, window_step)
# on all windows 
labels = np.zeros(81)
labels[-4:] = 1.0
EEG_data_val.setWindowLabels(labels)
EEG_data_val.FilterWindows(f_low = 5.0, f_high = 0.3, order = 2, filter_type = "scipy_butter")
EEG_data_val.windows = EEG_data_val.windows[:, :, 25:-25, :] # cut off artifact samples 
EEG_data_val.applyxDAWNToWindows(xd_filt, n_components = n_components)
#EEG_data_val.windowStandardization()
#EEG_data_val.minMaxNormWindows()
EEG_data_val.WindowMeanCorrection()
EEG_data_val.minMaxNormWindows()


# plt.figure()
# plt.plot(pos_class_temp_norm.T)
# plt.plot(EEG_data_val.windows[12, :, :, -1].T)
# plt.plot(EEG_data_val.windows[10, :, :, -1].T)
# plt.legend(["template", "example1", "example2"])

# plt.figure()
# plt.plot(EEG_data_val.windows[10, :, :, -1].T)
# plt.figure()
# plt.plot(EEG_data_val.windows[9, :, :, -1].T)
# plt.figure()
# plt.plot(EEG_data_val.windows[8, :, :, -1].T)
# plt.figure()
# plt.plot(EEG_data_val.windows[7, :, :, -1].T)
#plt.show()

# plt.figure()
# for window_idx in range(0, EEG_data_val.windows.shape[3] -5): 
#     plt.plot(EEG_data_val.windows[12, :, :, window_idx].T)

# plt.figure()
# for window_idx in range(EEG_data_val.windows.shape[3] -5, EEG_data_val.windows.shape[3]): 
#     plt.plot(EEG_data_val.windows[12, :, :, window_idx].T)
#plt.show()


EEG_data.dtwFeatureVecWindows()
EEG_data_val.dtwFeatureVecWindows()

# get features 
x_train = EEG_data.getFeatures()
y_train = EEG_data.getLabels()
x_val = EEG_data_val.getFeatures()
y_val = EEG_data_val.getLabels()


dtw_model = MLModel(type = "dtw")

# predict and get results 
# dtw_model.predict(data = x_val, labels = y_val, encoding = "distance_array", show_results = True, show_pred_time = False, eval_type = "offline", templates= [pos_class_temp_norm, neg_class_temp_norm], treshold=80)


# # perf_results_MLP = MLP_model.getPerfResults()

# test this without dtw 

current_trial = 8
trial_windows = EEG_data_val.windows[current_trial, :, :, :]
trial_windows_last1 = EEG_data_val.windows[current_trial-1, :, :, -1]
trial_windows_last2 = EEG_data_val.windows[current_trial-2, :, :, -1]
trial_windows_last3 = EEG_data_val.windows[current_trial-3, :, :, -1]
trial_windows_last4 = EEG_data_val.windows[current_trial-4, :, :, -1]


plt.figure()
plt.plot(trial_windows[:, :, -1].T)
plt.plot(trial_windows_last1.T)
plt.plot(trial_windows_last2.T)
plt.plot(trial_windows_last3.T)
plt.plot(trial_windows_last4.T)
plt.legend(["target", "temp1", "temp2", "temp3", "temp4"])
plt.show()

start_index = 300

print(trial_windows_last2.shape)
print(trial_windows.shape)

print("trials wind  shape", trial_windows.shape)

dist_list = []
for window_idx in range(1, trial_windows.shape[2]):
    dtw_dist = dtw(trial_windows[0, start_index:, window_idx], y=trial_windows_last1[0, start_index:], dist_method="sqeuclidean", step_pattern='symmetric2', window_type="sakoechiba", window_args={"window_size" : 20}) 
    dtw_dist1 = dtw(trial_windows[0, start_index:, window_idx], y=trial_windows_last2[0, start_index:], dist_method="sqeuclidean", step_pattern='symmetric2', window_type="sakoechiba", window_args={"window_size" : 20}) 
    dtw_dist2 = dtw(trial_windows[0, start_index:, window_idx], y=trial_windows_last3[0, start_index:], dist_method="sqeuclidean", step_pattern='symmetric2', window_type="sakoechiba", window_args={"window_size" : 20})
    dtw_dist3 = dtw(trial_windows[0, start_index:, window_idx], y=trial_windows_last4[0, start_index:], dist_method="sqeuclidean", step_pattern='symmetric2', window_type="sakoechiba", window_args={"window_size" : 20})
#dist = np.sum(np.abs(trial_windows[:, :, window_idx] -pos_class_temp_norm))
    dist = dtw_dist.distance
    dist1 = dtw_dist1.distance
    dist2 = dtw_dist2.distance
    dist3 = dtw_dist3.distance

    dist = np.mean(np.array([dist, dist1, dist2, dist3]))
    dist_list.append(dist)

dist_arr = np.array(dist_list)

print("dist vals trial", dist_arr)

thresh = 30
print("labels:", (dist_arr < thresh).astype(float))








