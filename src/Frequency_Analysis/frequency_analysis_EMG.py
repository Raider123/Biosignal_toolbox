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
data_path = proj_path+"/data/EMG_orthosis_noise_test"


filename = ["20240108_r_BB77D_ANT_orthosis_noise_Set1.vhdr"]
# channels_to_evaluate = ["FC6"]
title = filename[0]
indizes = [12000, 14000] # indices of the window to evaluate 
scales_plot = 0.0001 # how to scale the axis 
scales_plot_freq = 0.0001
channel_names_emg = ["tric. right 1", "tric. right 2", "tric. left", "tric. parallel right", "tric. long dist right"]


# *********************************************************************************
# **************************** Main section****************************************
# *********************************************************************************

emg_data = EEGData(format = "Brainvision", filenames = filename, data_path = data_path)
emg_data.setChannelNames(channel_names_emg)

# eeg_data.filterRaw(f_highpass = 20, f_lowpass = None) not used anymore 

emg_data.mneRawMethod(method_name = "filter", l_freq = 20.0, h_freq = None)
#eeg_data.mneRawMethod(method_name = "drop_channels", ch_names =["x_dir", "y_dir", "z_dir"])

f_samp = emg_data.getSamplingRate()
time_axis = np.arange(0, (indizes[1]- indizes[0])/f_samp, 1/f_samp)

ch_names = emg_data.getChannelNames()
print(ch_names)
n_chan = len(ch_names)
# n_plots = int(n_chan/4) 
print("num channels", n_chan)

fig = plt.figure(figsize=(20, 10))
fig.subplots_adjust(hspace=0.5)

for index in range(0, int(n_chan)):
    axs = plt.subplot(int(n_chan), 2, index+1)
    plt.plot(emg_data.data[index, 1000:])
    plt.legend([ch_names[index]])
    plt.ylim([-1*scales_plot, scales_plot])
    plt.ylabel("Mag. in V", fontsize = 8)
    plt.xlabel("Time in s", fontsize = 8)
    axs.tick_params(axis='both', which='major', labelsize=8)
plt.show()
fig.savefig(data_path+title+"_time_domain_ANT.png", dpi = 500)





# plt.figure()
# plt.plot(emg_data.data[0, :])
# plt.plot(emg_data.data[1, :])
# plt.plot(emg_data.data[2, :])
# plt.plot(emg_data.data[3, :])
# plt.title("first channel")
# plt.show()


# # fig 1 
# fig = plt.figure(figsize=(20, 10))
# fig.subplots_adjust(hspace=0.6)

# for index in range(0, int(n_chan/4)):
#     axs = plt.subplot(int(n_chan/8), 2, index+1)
#     plt.plot(time_axis, eeg_data.data[index, indizes[0]:indizes[1]])
#     plt.legend([ch_names[index]])
#     plt.ylim([-1*scales_plot, scales_plot])
#     plt.ylabel("Mag. in V", fontsize = 8)
#     plt.xlabel("Time in s", fontsize = 8)
#     axs.tick_params(axis='both', which='major', labelsize=8)

# fig.suptitle(title+" time domain 1", fontsize=12)
# figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()

# # save the figure
# fig.savefig(data_path+title+"_time_domain_1.png", dpi = 500)


# # fig 1 Frequency domain 
# fig = plt.figure(figsize=(20, 10))
# fig.subplots_adjust(hspace=0.6)

# for index in range(0, int(n_chan/4)): 
#     axs = plt.subplot(int(n_chan/8), 2, index+1)
#     xf, yf = eeg_data.OnechannelFFT(eeg_data.data[index, indizes[0]:indizes[1]])
#     plt.plot(xf, yf)
#     plt.legend([ch_names[index]])
#     plt.ylim([-1*scales_plot_freq, scales_plot_freq])    
#     plt.ylabel("|Hf|", fontsize = 8)
#     plt.xlabel("Freq. in Hz", fontsize = 8)  
#     plt.xticks(np.arange(0, 250, 25))
#     axs.tick_params(axis='both', which='major', labelsize=8)
# fig.suptitle(title+" freq domain 1", fontsize=12)
# figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()
# # plt.show()
# fig.savefig(data_path+title+"_freq_domain_1.png", dpi = 500)


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