# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import matplotlib.pyplot as plt
import numpy as np
from dtw import *
import copy 

# # own libs 
from biosignal_toolbox.eeg_lib import EEGData
import mne

# own libs
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox"


# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************

data_path = proj_path+"/data/"
results_path = proj_path+"/results/"

marker_number = 100
error_number = 3
f_samp_eeg = 500.0

channel_names = ["F5", "F3", "F1", "FZ", "F2", "F4", "F6", "FC5", "FC3", "FC1", "FC2", "FC4", "FC6", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "P5", "P3", "P1", "PZ", "P2", "P4", "P6"]
inverse_keep_channel = True 
channel_list = []
t1 = -1.0
t2 = 0.0

window_size = 1100
window_step = 50
n_components = 5

inverse_keep_channel = True 
channel_list = ["x_dir", "y_dir", "z_dir"]#, "FP1", "FP2", "F8", "T7", "T8", "TP9", "TP10", "P7","P8", "PO9", "O1", "OZ", "O2", "PO10", "AF7", "AF3", "AF4", "AF8", "FT9", "FT7", "FT8", "FT10", "TP7", "TP8", "PO7", "PO3", "POZ", "PO4", "PO8","F7"]
rename_channels = True 


# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************

#use LSL file recorded 
filenames = ["20211210_r_JV43_intentional_unilateral_set1.vhdr", "20211210_r_JV43_intentional_unilateral_set2.vhdr"] #"BR60D_unilateral_live_2_data", "BR60D_intentional_unilateral_set8_data", ]

#train_file_LSL_val = ["XY90_unilateral_set4_data"]
EEG_data = EEGData(format = "Brainvision", filenames = filenames, data_path = data_path)
#EEG_data = EEGData(format = "Recorded_LSL_stream", filenames = train_file_LSL, data_path = data_path, f_samp = f_samp_eeg, channel_names = channel_names)
#EEG_data_val = EEGData(format = "Recorded_LSL_stream", filenames = train_file_LSL_val, data_path = data_path, f_samp = f_samp_eeg, channel_names = channel_names)

EEG_data.rereferencingEpoching(marker_number = marker_number, error_number = error_number, reref_channels=["average"], f_highpass= 0.5, f_lowpass = 4.0, t1 = t1, t2 = t2, apply_filter = True, inverse_keep_channel = inverse_keep_channel, channel_list = channel_list) # setting to False will ignore the filtering )

#EEG_data.createActicapMontage(rename_channels = rename_channels, plot_montage = True)
ten_twenty = mne.channels.make_standard_montage('standard_1020', head_size=0.095)
ten_twenty.rename_channels({'Cz' : 'CZ','Pz' : 'PZ','Fz' : 'FZ','CPz' : 'CPZ', 'Fp1': 'FP1','Fp2': 'FP2','Oz': 'OZ','POz': 'POZ'}, allow_duplicates=False)
EEG_data.epoch_obj.set_montage(ten_twenty)
print(EEG_data.getChannelNames())


epochs_channels_C1 = EEG_data.epoch_obj.get_data(picks= ["C1", "C5", "C2", "F1", "P1"])
epochs_channels_FC1 = EEG_data.epoch_obj.get_data(picks= ["FC1", "FC2", "FC5", "CP1", "FZ"])

C1_diffs = np.array([epochs_channels_C1[:, 0, :] -epochs_channels_C1[:, 1, :], epochs_channels_C1[:, 0, :] -epochs_channels_C1[:, 2, :], epochs_channels_C1[:, 0, :] -epochs_channels_C1[:, 3, :], epochs_channels_C1[:, 0, :] -epochs_channels_C1[:, 4, :]])
C1_result = np.mean(C1_diffs, axis = 0)

FC1_diffs = np.array([epochs_channels_FC1[:, 0, :] -epochs_channels_FC1[:, 1, :], epochs_channels_FC1[:, 0, :] -epochs_channels_FC1[:, 2, :], epochs_channels_FC1[:, 0, :] -epochs_channels_FC1[:, 3, :], epochs_channels_FC1[:, 0, :] -epochs_channels_FC1[:, 4, :]])
FC1_result = np.mean(FC1_diffs, axis = 0)


# plt.figure() 
# plt.plot(np.mean(epochs_filtered, axis = 0)[0, :])
# plt.show()

# ica = mne.preprocessing.ICA(n_components=n_components, noise_cov=None, random_state=98, method='fastica', fit_params=None, max_iter='auto', allow_ref_meg=False, verbose=None)
# ica.fit(EEG_data.epoch_obj)
# ica.plot_components(inst = EEG_data.epoch_obj)


# EEG_data.epoch_obj = ica.apply(inst = copy.deepcopy(EEG_data.epoch_obj), include=[1]) 

# epochs_filtered = EEG_data.timeShiftingLinearSpatialFilter()
# print(epochs_filtered.shape)

# # print(epochs_arr.shape)

# plt.figure()

for trial_idx in range(0, C1_result.shape[0]-40): 
    plt.plot(C1_result[trial_idx, :] +FC1_result[trial_idx, :])
    #plt.plot(FC1_result[trial_idx, :])
    plt.legend(["sum"]) #C1", "FC1"])
    plt.show()


# plt.plot(np.mean(epochs_arr[:, 0, :], axis = 0))
# plt.show()

