
# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import sys 
import matplotlib.pyplot as plt
import os 

import mne 

# project path settings 
current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.split(os.path.split(current_path)[0])[0] # go up two folders to get the current path
data_path = os.path.join(project_path, 'data') # path where the data lays 
lib_path = os.path.join(project_path, 'lib') # path were the additional library is located 
sys.path.append(lib_path) # append own libs to path 


# own libs 
import eeg_lib

# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************


#filenames 
#filenames = ["00052720230421_AQ59D_orthosisErrorIjcai_multi_set7", "00052720230421_AQ59D_orthosisErrorIjcai_multi_set8", "00052720230421_AQ59D_orthosisErrorIjcai_multi_set9", "00052720230421_AQ59D_orthosisErrorIjcai_multi_set10"]

filenames = ["20230426_AJ05D_orthosisErrorIjcai_multi_set7"] 
# filenames = ["20211210_r_JV43_intentional_unilateral_set1", "20211210_r_JV43_intentional_unilateral_set2", "20211210_r_JV43_intentional_unilateral_set3", 
#              "20211216_r_RA12_intentional_unilateral_set1", "20211216_r_RA12_intentional_unilateral_set2", "20211216_r_RA12_intentional_unilateral_set3", 
#              "20211220_r_AV82_intentional_unilateral_set1", "20211220_r_AV82_intentional_unilateral_set2", "20211220_r_AV82_intentional_unilateral_set3", 
#              "20211223_r_UP28_intentional_unilateral_set1", "20211223_r_UP28_intentional_unilateral_set2", "20211223_r_UP28_intentional_unilateral_set3", 
#              "20211222_r_XP01_intentional_unilateral_set1","20211222_r_XP01_intentional_unilateral_set2","20211222_r_XP01_intentional_unilateral_set3","20211222_r_XP01_intentional_unilateral_set4",
#              "20220104_r_ZS27_intentional_unilateral_set1", "20220104_r_ZS27_intentional_unilateral_set2", "20220104_r_ZS27_intentional_unilateral_set3", 
#              "20220105_r_JD68_intentional_unilateral_set1", "20220105_r_JD68_intentional_unilateral_set2", "20220105_r_JD68_intentional_unilateral_set3", 
#              "20220107_r_QS70_intentional_unilateral_set1", "20220107_r_QS70_intentional_unilateral_set2", "20220107_r_QS70_intentional_unilateral_set3"
#              ]


# name pattern of current subject and paradigm 
subject_paradigm_name = "Errp"

# Channelnumbers with EEG Data from Dataset
eeg_channel_start_number = 0 
eeg_channel_end_number   = 67 # 64 eeg and 3 axis accelerometer (automatically removed laterl on )

# Filtering Params for EEG data 
f_highpass = 0.1 # in Hz 
f_lowpass = 4.0 # in Hz 
apply_filter = True # setting to False will ignore the filtering 


#rereferencing (["average"] or [] for no reref (otherwise specify channel names))
reref_channel = ["average"]

marker_number = 80 # markernumber that should be used for e.g. epoching (e.g.  movement onset)
error_number = 3 # number of the error marker (trials will be excluded)

# specifying marker type and give it a name (event that is used for epoching)
event_id_used = {"movement_onset": marker_number} 

# time selection for epoching of the data 
epoching_time_before_onset = -0.3 # time in seconds (start epoch)
epoching_time_after_onset = 1.0 # time in seconds (0 = movement onset)

# should baseline correction be applied ? (standard -1.5 to -1 seconds)
apply_baseline_correction = True 
t0_baseline = -0.25
t1_baseline = 0.0

# topoplot params 
plot_montage = False 
min_val = -6e-06 # Voltage values for colour scale 
max_val = 6e-06

# topoplot params 
topoplot_times =  [-200, 0, 500, 1000] # times in ms to the event after epoching 
topoplot_title_str = "time to movement "

# select channel name for visualizing it 
channel_to_evaluate = "FZ"

# eeg channel that are kept (inverse_keep_channel = False) or dropped (inverse_keep_channel = True) for further evaluations, empty list meaning all channels are kept 
inverse_keep_channel = True 
channel_list = ["x_dir", "y_dir", "z_dir"]#"x_dir", "y_dir", "z_dir", "FP1", "FP2", "F8", "T7", "T8", "TP9", "TP10", "P7","P8", "PO9", "O1", "OZ", "O2", "PO10", "AF7", "AF3", "AF4", "AF8", "FT9", "FT7", "FT8", "FT10", "TP7", "TP8", "PO7", "PO3", "POZ", "PO4", "PO8","F7"]
rename_channels = True 


# *********************************************************************************
# ***************** Load and concatenate a dataset *********
# *********************************************************************************

#create numpy array with file names 
data_str_arr = []
for files_str in filenames: 
    data_str_arr.append(os.path.join(data_path, files_str)) 


raw= eeg_lib.loadBrainproductsData(data_str_arr) # read data in brainproducts format
raw.filter(0.1, 15) # bandpass
f_samp_eeg = raw.info['sfreq'] # get sampling rate 


scalings = dict(mag=1e-12, grad=4e-11, eeg=50e-6, eog=150e-6, ecg=5e-4,
     emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1,
     resp=1, chpi=1e-4, whitened=1e2)

fig = raw.plot(scalings = scalings)

plt.show()
fig.savefig("pic.png")




