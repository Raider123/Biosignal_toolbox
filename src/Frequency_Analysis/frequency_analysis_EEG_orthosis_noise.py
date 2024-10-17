# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import matplotlib.pyplot as plt 
import mne

# # own libs 
from biosignal_toolbox.eeg_lib import EEGData

# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************

# own libs
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox"
data_path = proj_path+"/data/20240626_r______Test_Shielding_cabine_outside_shielding_cabin_resting/Set2"


filename = ["20240626_r______Test_Shielding_cabine_outside_shielding_cabin_resting_Set2.vhdr"]
# channels_to_evaluate = ["FC6"]
title = filename[0]
#indizes = [0, -1] # indices of the window to evaluate 
scales_plot = 0.002 # how to scale the axis 
scales_plot_freq = 0.0001

# *********************************************************************************
# **************************** Main section****************************************
# *********************************************************************************

eeg_data = EEGData(format = "Brainvision", filenames = filename, data_path = data_path)

# eeg_data.filterRaw(f_highpass = 20, f_lowpass = None) not used anymore 

eeg_data.mneRawMethod(method_name = "filter", l_freq = 0.5, h_freq = None)

#eeg_data.mneRawMethod(method_name = "drop_channels", ch_names =["x_dir", "y_dir", "z_dir"])

f_samp = eeg_data.getSamplingRate()
time_axis = np.arange(0,eeg_data.data.shape[1]/f_samp, 1/f_samp)

ch_names = eeg_data.getChannelNames()
n_chan = len(ch_names)
n_plots = int(n_chan/4) 
print("num channels", n_chan)

plt.figure()
plt.plot(eeg_data.data[0, :])
plt.title("first channel")
plt.show()


# fig 1 
fig = plt.figure(figsize=(20, 10))
fig.subplots_adjust(hspace=0.6)

for index in range(0, int(n_chan/4)):
    axs = plt.subplot(int(n_chan/8), 2, index+1)
    plt.plot(time_axis, eeg_data.data[index,:])
    plt.legend([ch_names[index]])
    plt.ylim([-1*scales_plot, scales_plot])
    plt.ylabel("Mag. in V", fontsize = 8)
    plt.xlabel("Time in s", fontsize = 8)
    axs.tick_params(axis='both', which='major', labelsize=8)

fig.suptitle(title+" time domain 1", fontsize=12)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

# save the figure
#fig.savefig(data_path+title+"test.png", dpi = 500)


# fig 1 Frequency domain 
fig = plt.figure(figsize=(20, 10))
fig.subplots_adjust(hspace=0.6)

for index in range(0, int(n_chan/4)): 
    axs = plt.subplot(int(n_chan/8), 2, index+1)
    xf, yf = eeg_data.OnechannelFFT(eeg_data.data[index, :])
    plt.plot(xf, yf)
    plt.legend([ch_names[index]])
    plt.ylim([-1*scales_plot_freq, scales_plot_freq])    
    plt.ylabel("|Hf|", fontsize = 8)
    plt.xlabel("Freq. in Hz", fontsize = 8)  
    plt.xticks(np.arange(0, 250, 25))
    axs.tick_params(axis='both', which='major', labelsize=8)
fig.suptitle(title+" freq domain 1", fontsize=12)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()
#fig.savefig(data_path+title+"_freq_domain_1.png", dpi = 500)


# # fig 2 time dome
# fig = plt.figure(figsize=(20, 10))
# fig.subplots_adjust(hspace=0.6)

# for index in range(0, int(n_chan/4)): 
#     axs = plt.subplot(int(n_chan/8), 2, index+1)
#     plt.plot(time_axis, eeg_data.data[index+n_plots, indizes[0]:indizes[1]])
#     plt.legend([ch_names[index+n_plots]])
#     plt.ylim([-1*scales_plot, scales_plot])    
#     plt.ylabel("Mag. in V", fontsize = 8)
#     plt.xlabel("Time in s", fontsize = 8)  
#     axs.tick_params(axis='both', which='major', labelsize=8)
# fig.suptitle(title+" time domain 2", fontsize=12)
# figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()
# # plt.show()
# fig.savefig(data_path+title+"_time_domain_2.png", dpi = 500)

# # fig 2 Frequency domain 
# fig = plt.figure(figsize=(20, 10))
# fig.subplots_adjust(hspace=0.6)

# for index in range(0, int(n_chan/4)): 
#     axs = plt.subplot(int(n_chan/8), 2, index+1)
#     xf, yf = eeg_data.OnechannelFFT(eeg_data.data[index+n_plots, indizes[0]:indizes[1]])
#     print(np.max(yf))
#     plt.plot(xf, yf)
#     plt.legend([ch_names[index+n_plots]])
#     plt.ylim([-1*scales_plot_freq, scales_plot_freq])    
#     plt.ylabel("|Hf|", fontsize = 8)
#     plt.xlabel("Freq. in Hz", fontsize = 8)  
#     plt.xticks(np.arange(0, 250, 25))
#     axs.tick_params(axis='both', which='major', labelsize=8)
# fig.suptitle(title+" freq domain 2", fontsize=12)
# figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()
# # plt.show()
# fig.savefig(data_path+title+"_freq_domain_2.png", dpi = 500)

# # fig 3 
# fig = plt.figure(figsize=(20, 10))
# fig.subplots_adjust(hspace=0.6)
# for index in range(0, int(n_chan/4)): 
#     axs = plt.subplot(int(n_chan/8), 2, index+1)
#     plt.plot(time_axis, eeg_data.data[index+2*n_plots, indizes[0]:indizes[1]])
#     plt.legend([ch_names[index+2*n_plots]])
#     plt.ylim([-1*scales_plot, scales_plot])    
#     plt.ylabel("Mag. in V", fontsize = 8)
#     plt.xlabel("Time in s", fontsize = 8) 
#     axs.tick_params(axis='both', which='major', labelsize=8)
# fig.suptitle(title+" time domain 3", fontsize=12)
# figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()
# # plt.show()
# fig.savefig(data_path+title+"_time_domain_3.png", dpi = 500)

# # fig3 Frequency domain 
# fig = plt.figure(figsize=(20, 10))
# fig.subplots_adjust(hspace=0.6)

# for index in range(0, int(n_chan/4)): 
#     axs = plt.subplot(int(n_chan/8), 2, index+1)
#     xf, yf = eeg_data.OnechannelFFT(eeg_data.data[index+2*n_plots, indizes[0]:indizes[1]])
#     print(np.max(yf))
#     plt.plot(xf, yf)
#     plt.legend([ch_names[index+2*n_plots]])
#     plt.ylim([-1*scales_plot_freq, scales_plot_freq])    
#     plt.ylabel("|Hf|", fontsize = 8)
#     plt.xlabel("Freq. in Hz", fontsize = 8)  
#     plt.xticks(np.arange(0, 250, 25))
#     axs.tick_params(axis='both', which='major', labelsize=8)
# fig.suptitle(title+" freq domain 3", fontsize=12)
# figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()
# # plt.show()
# fig.savefig(data_path+title+"_freq_domain_3.png", dpi = 500)

# # fig 4
# fig = plt.figure(figsize=(20, 10))
# fig.subplots_adjust(hspace=0.6)
# for index in range(0, int(n_chan/4)): 
#     axs = plt.subplot(int(n_chan/8), 2, index+1)
#     plt.plot(time_axis, eeg_data.data[index+3*n_plots, indizes[0]:indizes[1]])
#     plt.legend([ch_names[index+3*n_plots]])
#     plt.ylim([-1*scales_plot, scales_plot])    
#     plt.ylabel("Mag. in V", fontsize = 8)
#     plt.xlabel("Time in s", fontsize = 8)  
#     axs.tick_params(axis='both', which='major', labelsize=8)
# fig.suptitle(title+" time domain 4", fontsize=12)
# figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()
# # plt.show()
# fig.savefig(data_path+title+"_time_domain_4.png", dpi = 500)


# # fig 4 Frequency domain 
# fig = plt.figure(figsize=(20, 10))
# fig.subplots_adjust(hspace=0.6)

# for index in range(0, int(n_chan/4)): 
#     axs = plt.subplot(int(n_chan/8), 2, index+1)
#     xf, yf = eeg_data.OnechannelFFT(eeg_data.data[index+3*n_plots, indizes[0]:indizes[1]])
#     print(np.max(yf))
#     plt.plot(xf, yf)
#     plt.legend([ch_names[index+3*n_plots]])
#     plt.ylim([-1*scales_plot_freq, scales_plot_freq])    
#     plt.ylabel("|Hf|", fontsize = 8)
#     plt.xlabel("Freq. in Hz", fontsize = 8)  
#     plt.xticks(np.arange(0, 250, 25))
#     axs.tick_params(axis='both', which='major', labelsize=8)
# fig.suptitle(title+" freq domain 4", fontsize=12)
# figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()
# # plt.show()
# fig.savefig(data_path+title+"_freq_domain_4.png", dpi = 500)


# # eeg_data.OnechannelFFT(no_motor_noise, plot = True, window = None, beta = None, title = "no motor touching")
# eeg_data.OnechannelFFT(motor_noise, plot = True, window = None, beta = None, title = "motor touching"+ filename[0])