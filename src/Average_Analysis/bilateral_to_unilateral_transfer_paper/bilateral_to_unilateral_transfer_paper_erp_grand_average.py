
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
data = EEGData(format = "Brainvision", filenames = ["20211210_r_JV43_intentional_bilateral_set1.vhdr"], data_path = data_path) # just an example file for getting correct params 


# rereferencing and epoching of data  --> just for getting the same parameters 
data.rereferencingEpoching(marker_number, error_number, channel_list, apply_filter=apply_filter,reref_channels =reref_channel, f_highpass = f_highpass, f_lowpass= f_lowpass, inverse_keep_channel = inverse_keep_channel, t1 =epoching_time_before_onset, t2= epoching_time_after_onset, apply_baseline_correction = True, t0_baseline=t0_baseline, t1_baseline=t1_baseline)
# create an acticap montage 
data.createActicapMontage(rename_channels = rename_channels)


# set the averaged numpy values for generating the average plot 
XP01_epochs = np.load(data_path+"/XP01_unilateral_01_4Hz_ICA.npy")
JV43_epochs = np.load(data_path+"/JV43_unilateral_01_4Hz_ICA.npy")
RA12_epochs = np.load(data_path+"/RA12_unilateral_01_4Hz_ICA.npy")
UP28_epochs = np.load(data_path+"/UP28_unilateral_01_4Hz_ICA.npy")
ZS27_epochs = np.load(data_path+"/ZS27_unilateral_01_4Hz_ICA.npy")
JD68_epochs = np.load(data_path+"/JD68_unilateral_01_4Hz_ICA.npy")
QS70_epochs = np.load(data_path+"/QS70_unilateral_01_4Hz_ICA.npy")
AV82_epochs = np.load(data_path+"/AV82_unilateral_01_4Hz_ICA.npy")

grand_epochs = np.concatenate((XP01_epochs, JV43_epochs, RA12_epochs, UP28_epochs, ZS27_epochs, JD68_epochs, QS70_epochs, AV82_epochs)) 
print("shape: ", grand_epochs.shape)

data.epochs = grand_epochs

# make a topoplot 
data.topoplot(topoplot_times, topoplot_title_str, min_val, max_val, save_figure = True, filename = "grand_average_"+evalname +".svg", path = figure_path)

time_axis = data.getTimeAxisEpochs()
selected_eeg_channels_filter = data.getDataFromChannels(channels_to_evaluate, average=True, epoched_data = True)

# show average erp signal selected channel 

# Plot the selected channel (average)
fig = plt.figure()
plt.plot(time_axis, selected_eeg_channels_filter[0, :])
plt.plot(time_axis, selected_eeg_channels_filter[1, :])
plt.plot(time_axis, selected_eeg_channels_filter[1, :] -selected_eeg_channels_filter[0, :])
plt.axhline(y=0.0, color='black', linestyle='--')
#plt.title("Bilateral movements: Grand average EEG signals") 
channels_to_evaluate.append("C1-C2") # manually append the difference 
print(channels_to_evaluate) 
plt.legend(channels_to_evaluate, fontsize=14)
plt.xlabel("Time in to movement onset seconds", fontsize=14)
plt.ylabel("Voltage in V", fontsize=14)
plt.show()

fig.savefig(figure_path+"channelsPlots_"+evalname+".svg")



