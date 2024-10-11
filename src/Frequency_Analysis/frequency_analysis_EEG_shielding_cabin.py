# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import matplotlib.pyplot as plt 
import mne

# # own libs 
from biosignal_toolbox.eeg_lib import EEGData

# import all user parameters for the evaluation
from userparams import * # --> parameters are stored and fully imported from this script 

# *********************************************************************************
# **************************** Main section****************************************
# *********************************************************************************

eeg_data = EEGData(format = "Brainvision", filenames = filename, data_path = data_path)

# eeg_data.filterRaw(f_highpass = 20, f_lowpass = None) not used anymore 

eeg_data.mneRawMethod(method_name = "filter", l_freq = f_highpass, h_freq = None)

#eeg_data.mneRawMethod(method_name = "drop_channels", ch_names =["x_dir", "y_dir", "z_dir"])

f_samp = eeg_data.getSamplingRate()
time_axis = np.arange(0,eeg_data.data.shape[1]/f_samp, 1/f_samp)

ch_names = eeg_data.getChannelNames()
print(f"channel names: {ch_names}")

n_chan = len(ch_names)
n_plots = int(n_chan/4) 
print("num channels", n_chan)

plt.figure()
plt.plot(eeg_data.data[0, :])
plt.title("first channel data :" )
plt.show()


# fig 1 
fig = plt.figure(figsize=(20, 10))
#fig.subplots_adjust(hspace=0.6)

for index in range(0, n_channels_to_show):
    axs = plt.subplot(int(n_channels_to_show/2), 2, index+1)
    plt.plot(time_axis, eeg_data.data[index,:])
    plt.legend([ch_names[index]])
    plt.ylim([-1*scales_plot, scales_plot])
    plt.ylabel("Mag. in V", fontsize = 8)
    plt.xlabel("Time in s", fontsize = 8)
    axs.tick_params(axis='both', which='major', labelsize=8)

fig.suptitle(title+" time domain", fontsize=12)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

# save the figure
#fig.savefig(data_path+title+"test.png", dpi = 500)


# fig 1 Frequency domain 
fig = plt.figure(figsize=(20, 10))
#fig.subplots_adjust(hspace=0.6)

for index in range(0, n_channels_to_show): 
    axs = plt.subplot(int(n_channels_to_show/2), 2, index+1)
    xf, yf = eeg_data.OnechannelFFT(eeg_data.data[index, :])
    xf = xf[0:(int(len(xf)/4))]
    yf = yf[0:(int(len(yf)/4))]
    plt.plot(xf, yf)
    plt.legend([ch_names[index]])
    plt.ylim([0, scales_plot_freq])    
    plt.ylabel("|Hf|", fontsize = 8)
    plt.xlabel("Freq. in Hz", fontsize = 8)  
    plt.xticks(np.arange(0, 150, 10))
    axs.tick_params(axis='both', which='major', labelsize=8)
fig.suptitle(title+" freq domain", fontsize=12)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()
#fig.savefig(data_path+title+"_freq_domain_1.png", dpi = 500)



