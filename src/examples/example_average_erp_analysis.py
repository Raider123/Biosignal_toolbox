
# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import sys 
import matplotlib.pyplot as plt

# own libs 
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/mne_machine_learning"
sys.path.append(proj_path+"/lib") # path to lib folder 
import eeg_lib


# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************


data_path = proj_path+"/data/"

# Create an array with dataset file names 
#data_str_arr = np.array([data_path+"20211210_r_JV43_intentional_unilateral_set1.vhdr"])# data_path+"20211210_r_JV43_intentional_unilateral_set2.vhdr", data_path+"20211210_r_JV43_intentional_unilateral_set3.vhdr"])
data_str_arr = np.array([data_path+"221220_PP01_curls_with_pause_set2.vhdr"])


# Filtering Params for EEG data 
f_highpass = 0.5 # in Hz 
f_lowpass = 4.0 # in Hz 
apply_filter = True # setting to False will ignore the filtering 

#rereferencing (["average"] or [] for no reref (otherwise specify channel names))
reref_channel = ["average"]

f_samp_eeg = 500 #sample Frequency of eeg
marker_number = 20 # markernumber that should be used for e.g. epoching (e.g.  movement onset)
error_number = 3 # number of the error marker (trials will be excluded)

# specifying marker type and give it a name (event that is used for epoching)
event_id_used = {"movement_onset": marker_number} 

# time selection for epoching of the data 
epoching_time_before_onset = -1.5 # time in seconds (start epoch)
epoching_time_after_onset = 0 # time in seconds (0 = movement onset)

# should baseline correction be applied ? (standard -1.5 to -1 seconds)
apply_baseline_correction = True 
t0_baseline = -1.5
t1_baseline = -1

# topoplot params 
plot_montage = False 
min_val = -6e-06 # Voltage values for colour scale 
max_val = 6e-06

# topoplot params 
topo_start_time = 1000 # time in ms in relation to the event (e.g before movement onset to show)
topo_time_step = 100 # time resolution of the topopot in ms 
topoplot_title_str = "time to movement "

# select channel name for visualizing it 
channel_to_evaluate = "FC1"

# eeg channel that are kept (inverse_keep_channel = False) or dropped (inverse_keep_channel = True) for further evaluations, empty list meaning all channels are kept 
inverse_keep_channel = True 
channel_list = ["x_dir", "y_dir", "z_dir"]#"x_dir", "y_dir", "z_dir", "FP1", "FP2", "F8", "T7", "T8", "TP9", "TP10", "P7","P8", "PO9", "O1", "OZ", "O2", "PO10", "AF7", "AF3", "AF4", "AF8", "FT9", "FT7", "FT8", "FT10", "TP7", "TP8", "PO7", "PO3", "POZ", "PO4", "PO8","F7"]
rename_channels = False 

# *********************************************************************************
# ***************** Load and concatenate a dataset *********
# *********************************************************************************

raw= eeg_lib.loadBrainproductsData(data_str_arr) # read data in brainproducts format

# *********************************************************************************
# **************** Make EEG average analysis **************************************
# *********************************************************************************

# epoch the eeg data to trial length (for merged sets)
erp_epochs, erp_epochs_obj, time_axis_eeg_epochs, remaining_eeg_channel_names, raw_filtered = eeg_lib.rereferencingEpoching(raw, marker_number, error_number, channel_list, inverse_keep_channel, reref_channel, apply_filter, f_highpass, f_lowpass, event_id_used, epoching_time_before_onset, epoching_time_after_onset, f_samp_eeg, apply_baseline_correction,  t0_baseline, t1_baseline)
#erp_epochs has shape (trials, eeg-channels, sampels)

average_erp_epochs = np.mean(erp_epochs, axis = 0) # average the trials (average analysis), has now shape(channel, sampels) 

print(average_erp_epochs.shape)
# create an acticap montage 
acticap_montage = eeg_lib.createActicapMontage(plot_montage, rename_channels)
raw_filtered.set_montage(acticap_montage) # set created montage 


# # make a topoplot 
eeg_lib.topoplot(average_erp_epochs, time_axis_eeg_epochs, raw_filtered, topo_start_time, topo_time_step, topoplot_title_str, min_val, max_val, f_samp_eeg)


# # show average erp signal selected channel 
print(remaining_eeg_channel_names)
channel_index = remaining_eeg_channel_names.index(channel_to_evaluate)
average_eeg_selected_channel = average_erp_epochs[channel_index, :]

# Plot the selected channel (average)
plt.figure()
plt.plot(time_axis_eeg_epochs, average_eeg_selected_channel)
plt.title("Average eeg signal for channel "+channel_to_evaluate)
plt.xlabel("Time in seconds")
plt.ylabel("Voltage in V")
plt.show()

