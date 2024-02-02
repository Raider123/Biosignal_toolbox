
# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************
import matplotlib.pyplot as plt 
import os 


# project path settings 
# current_path = os.path.dirname(os.path.abspath(__file__)) # project path 
# project_path = os.path.split(os.path.split(current_path)[0])[0] # go up two folders to get the current path
#data_path = os.path.join(project_path, 'data') # path where the data lays 

# # own libs 
from biosignal_toolbox.eeg_lib import EEGData


# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************

#proj path 
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox"
data_path = os.path.join(proj_path, 'data') # path where the data lays 


#filenames = ["test3.vhdr", "test4.vhdr"]
filenames = ["20220105_r_JD68_intentional_unilateral_set1.vhdr", "20220105_r_JD68_intentional_unilateral_set2.vhdr", "20220105_r_JD68_intentional_unilateral_set1.vhdr"]

# name pattern of current subject and paradigm 
subject_paradigm_name = "unilateral"

# Filtering Params for EEG data 
f_highpass = 0.5 # in Hz 
f_lowpass = 4.0 # in Hz 
apply_filter = True # setting to False will ignore the filtering 

f_samp_eeg = 500.0
#rereferencing (["average"] or [] for no reref (otherwise specify channel names))
reref_channel = ["average"]

marker_number = 100 # markernumber that should be used for e.g. epoching (e.g.  movement onset)
error_number = 3 # number of the error marker (trials will be excluded)

# specifying marker type and give it a name (event that is used for epoching)
event_id_used = {"movement_onset": marker_number} 

# time selection for epoching of the data 
epoching_time_before_onset = -1.5 # time in seconds (start epoch)
epoching_time_after_onset = 0.0 # time in seconds (0 = movement onset)

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
channels_to_evaluate = ["CP1", "C2", "C1"]

# channel_names = ["F5", "F3", "F1", "FZ", "F2", "F4", "F6", "FC5", "FC3", "FC1", "FC2", "FC4", "FC6", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "P5", "P3", "P1", "PZ", "P2"]#, "P4", "P6"]

# eeg channel that are kept (inverse_keep_channel = False) or dropped (inverse_keep_channel = True) for further evaluations, empty list meaning all channels are kept 
inverse_keep_channel = True 
channel_list = ["x_dir", "y_dir", "z_dir"]# "FP1", "FP2", "F8", "T7", "T8", "TP9", "TP10", "P7","P8", "PO9", "O1", "OZ", "O2", "PO10", "AF7", "AF3", "AF4", "AF8", "FT9", "FT7", "FT8", "FT10", "TP7", "TP8", "PO7", "PO3", "POZ", "PO4", "PO8","F7"]
rename_channels = True 


# *********************************************************************************
# **************** Make EEG average analysis **************************************
# *********************************************************************************

#create EEGData object
#data = EEGData(format = "Brainvision", filenames = filenames, data_path = data_path)

data = EEGData(format = "Brainvision", filenames = filenames, data_path = data_path)
#data = EEGData(format = "Recorded_LSL_stream", filenames = filenames, data_path = data_path, f_samp = f_samp_eeg, channel_names = channel_names)


data.rereferencingEpoching(marker_number, error_number, channel_list, apply_filter=apply_filter,reref_channels =reref_channel, f_highpass = f_highpass, f_lowpass= f_lowpass, inverse_keep_channel = inverse_keep_channel, t1 =epoching_time_before_onset, t2= epoching_time_after_onset)

# if ica should be applied 
#data.simpleICAFiltering(exclude_components=[0, 1]) # apply ica 

# create an acticap montage 
data.createActicapMontage(rename_channels = rename_channels)

# if special designed spatial filter should be used 
#resulting_channel_epochs = data.timeShiftingLinearSpatialFilter(replace_epochs = True)

selected_eeg_channels_filter = data.getDataFromChannels(channels_to_evaluate, average=True, epoched_data = True)
time_axis = data.getTimeAxisEpochs()

# make a topoplot 
data.topoplot(topoplot_times, topoplot_title_str, min_val, max_val)

# # show average erp signal selected channel 

# Plot the selected channel (average)
plt.figure()
plt.plot(time_axis, selected_eeg_channels_filter[0, :])
plt.plot(time_axis, selected_eeg_channels_filter[1, :])
plt.plot(time_axis, selected_eeg_channels_filter[2, :])
plt.title("Average eeg signal for channels") 
plt.legend(channels_to_evaluate)
plt.xlabel("Time in seconds")
plt.ylabel("Voltage in V")
plt.show()



