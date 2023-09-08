
# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import sys 
import matplotlib.pyplot as plt
import os 


# project path settings 
current_path = os.path.dirname(os.path.abspath(__file__)) # project path 
project_path = os.path.split(os.path.split(current_path)[0])[0] # go up two folders to get the current path
data_path = os.path.join(project_path, 'data') # path where the data lays 

# # own libs 
from biosignal_toolbox.eeg_lib import EEGData

# proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/mne_machine_learning"
# sys.path.append(proj_path+"/lib") # path to lib folder 
#import eeg_lib


# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************

#filenames 
filenames = ["20032023_AF64D_unilateral_set1.vhdr","20032023_AF64D_unilateral_set2.vhdr", "20032023_AF64D_unilateral_set3.vhdr"]


# name pattern of current subject and paradigm 
subject_paradigm_name = "unilateral"

# Channelnumbers with EEG Data from Dataset
eeg_channel_start_number = 0 
eeg_channel_end_number   = 67 # 64 eeg and 3 axis accelerometer (automatically removed laterl on )

# Filtering Params for EEG data 
f_highpass = 0.5 # in Hz 
f_lowpass = 4.0 # in Hz 
apply_filter = True # setting to False will ignore the filtering 


#rereferencing (["average"] or [] for no reref (otherwise specify channel names))
reref_channel = ["average"]

marker_number = 22 # markernumber that should be used for e.g. epoching (e.g.  movement onset)
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
topoplot_times =  [-1000, -500, -200, -100, 0] # times in ms to the event after epoching 
topoplot_title_str = "time to movement "

# select channel name for visualizing it 
channel_to_evaluate = "C1"

# eeg channel that are kept (inverse_keep_channel = False) or dropped (inverse_keep_channel = True) for further evaluations, empty list meaning all channels are kept 
inverse_keep_channel = True 
channel_list = ["x_dir", "y_dir", "z_dir"]#"x_dir", "y_dir", "z_dir", "FP1", "FP2", "F8", "T7", "T8", "TP9", "TP10", "P7","P8", "PO9", "O1", "OZ", "O2", "PO10", "AF7", "AF3", "AF4", "AF8", "FT9", "FT7", "FT8", "FT10", "TP7", "TP8", "PO7", "PO3", "POZ", "PO4", "PO8","F7"]
rename_channels = True 


# *********************************************************************************
# ***************** Load and concatenate a dataset *********
# *********************************************************************************


# *********************************************************************************
# **************** Make EEG average analysis **************************************
# *********************************************************************************


#create EEGData object
data = EEGData(format = "Brainvision", filenames = filenames, data_path = data_path)
data.rereferencingEpoching(marker_number, error_number,channel_list, inverse_keep_channel, reref_channel, apply_filter, f_highpass, f_lowpass, event_id_used, epoching_time_before_onset, epoching_time_after_onset, apply_baseline_correction,  t0_baseline, t1_baseline)

# create an acticap montage 
data.createActicapMontage(False, rename_channels)

# # make a topoplot 
data.topoplot(topoplot_times, topoplot_title_str, min_val, max_val)

one_channel, time_axis = data.getDataOneChannel(channel_to_evaluate)

# show average erp signal selected channel 

# Plot the selected channel (average)
plt.figure()
plt.plot(time_axis, one_channel)
plt.title("Average eeg signal for channel "+channel_to_evaluate)
plt.xlabel("Time in seconds")
plt.ylabel("Voltage in V")
plt.show()

