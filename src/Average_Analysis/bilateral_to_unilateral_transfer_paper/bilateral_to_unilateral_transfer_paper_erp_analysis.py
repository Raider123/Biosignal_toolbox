
# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************
import matplotlib.pyplot as plt 

# project path settings 
# # own libs 
from biosignal_toolbox.eeg_lib import EEGData
from user_params_bilateral_to_unilateral_paper import * # import the parameters used here 

import numpy as np

# *********************************************************************************
# **************** Make EEG average analysis **************************************
# *********************************************************************************

#create EEGData object
data = EEGData(format = "Brainvision", filenames = filenames, data_path = data_path)
#data = EEGData(format = "Recorded_LSL_stream", filenames = filenames, data_path = data_path, f_samp = f_samp_eeg, channel_names = channel_names)


# rereferencing and epoching of data 
data.rereferencingEpoching(marker_number, error_number, channel_list, apply_filter=apply_filter,reref_channels =reref_channel, f_highpass = f_highpass, f_lowpass= f_lowpass, inverse_keep_channel = inverse_keep_channel, t1 =epoching_time_before_onset, t2= epoching_time_after_onset, apply_baseline_correction = False)
# create an acticap montage 
data.createActicapMontage(rename_channels = rename_channels)


# if ica should be applied 
if(apply_ica): 
    data.simpleICAFilteringEpochs(n_components=n_ica_comp, exclude_components = exclude_ica_comp) # apply ica 

# show the ica components 
data.showICAcomponents() 

# apply baseline correction after ica 
data.applyBaselineCorrectionToEpochs(t0_baseline = t0_baseline, t1_baseline = t1_baseline) 


# if special designed spatial filter should be used 
#resulting_channel_epochs = data.timeShiftingLinearSpatialFilter(replace_epochs = True)

#selected_eeg_channels_filter = data.getDataFromChannels(channels_to_evaluate, average=True, epoched_data = True)
#time_axis = data.getTimeAxisEpochs()

# make a topoplot 
data.topoplot(topoplot_times, topoplot_title_str, min_val, max_val, save_figure = True, filename = subject+evalname, path = figure_path)

np.save(data_path+"/"+subject+evalname+".npy", data.getEpochs())

# show average erp signal selected channel 

# Plot the selected channel (average)
# plt.figure()
# plt.plot(time_axis, selected_eeg_channels_filter[0, :])
# plt.plot(time_axis, selected_eeg_channels_filter[1, :])
# plt.title("Average eeg signal for channels") 
# plt.legend(channels_to_evaluate)
# plt.xlabel("Time in seconds")
# plt.ylabel("Voltage in V")
# plt.show()



