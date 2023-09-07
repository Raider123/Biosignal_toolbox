from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt
import mne
import os 
import sys 

# own libs 
# project path settings 
current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.split(os.path.split(current_path)[0])[0] # go up two folders to get the current path
data_path = os.path.join(project_path, 'data') # path where the data lays 
lib_path = os.path.join(project_path, 'lib') # path were the additional library is located 
sys.path.append(lib_path) # append own libs to path 


import eeg_lib
import emg_lib


# files to load 
#filenames = ["Test_Orthosis/20230623_Test_Orthosis.vhdr"] 
filenames = ["Test_Orthosis/20230425_AY63D_orthosisErrorIjcai_multi_set3.vhdr"]
evaluation = "old_supply_AQ59D"

channel_to_evaluate = "CP1"

# start_ind = 38576
# stop_ind = 108885


#create numpy array with file names 
data_str_arr = []
for files_str in filenames: 
    data_str_arr.append(os.path.join(data_path, files_str)) 


raw= eeg_lib.loadBrainproductsData(data_str_arr) # read data in brainproducts format
raw.filter(49.9, 50.1, method="iir")

f_samp_eeg = raw.info['sfreq'] # get sampling rate 

EEG_data = raw.get_data()

ch_names = list(raw.ch_names) 


channel_index = ch_names.index(channel_to_evaluate)
print("Channel: ", ch_names[channel_index])

# specify time range 
EEG_ch_selected = EEG_data[channel_index, :]

i = 0

#for one_ch_data in EEG_ch_selected: 
# Number of sample points

N = len(EEG_ch_selected)

# sample spacing

dT = 1.0/f_samp_eeg

x = np.linspace(0.0, N*dT, N, endpoint=False)

y = EEG_ch_selected

yf = fft(y)

xf = fftfreq(N, dT)[:N//2]

yf = 2.0/N * np.abs(yf[0:N//2])



fig, axs = plt.subplots(2, 1)
plt.subplots_adjust(hspace = 0.5)
axs[0].plot(xf, yf)

axs[0].grid()
axs[0].set_xlabel("Frequencies in Hz")
axs[0].set_ylabel("|H|")
axs[0].set_title(evaluation+" FFT of channel:" +ch_names[channel_index])

#for one_ch_data in EEG_ch_selected: 
# Number of sample points


axs[1].plot(x, y)

axs[1].set_xlabel("Time in seconds")
axs[1].set_ylabel("Magnitude")
axs[1].set_title(evaluation+" time domain:" +ch_names[channel_index])

fig.savefig(evaluation+ch_names[channel_index])

plt.show()

index_50Hz = np.where(xf.astype(int) == 50)[0]
val_50Hz = np.max(yf[index_50Hz]) # get max values of indices around 50 Hz 

print("max value 50 Hz: ", val_50Hz)

# raw.plot()
# plt.show()

# EMG 

# f_samp_emg = 1000 

# # for ANT Systems 
# emg_ch_names = ["EMG1", "EMG2", "EMG3", "EMG4", "EMG5", "EMG6", "EMG7", "EMG8"]


#  # EMG data loading and processing 
# filename_EMG = "20230426_AJ05D_orthosisErrorIjcai_multi_set7.txt"


# file_str_emg = os.path.join(data_path, filename_EMG)
# emg_data, emg_time_axis = emg_lib.loadMiniANTEMGData(file_str_emg, f_samp_emg)

# emg_data_filtered = emg_lib.applyBPFilterRectifying(f_samp_emg, 20, 450, emg_data)


# i = 0
# for one_ch_data in emg_data_filtered.T: 
# # Number of sample points

#     N = len(one_ch_data)

#     # sample spacing

#     dT = 1.0/f_samp_emg

#     x = np.linspace(0.0, N*dT, N, endpoint=False)

#     y = one_ch_data

#     yf = fft(y)

#     xf = fftfreq(N, dT)[:N//2]


#     plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))

#     plt.grid()
#     plt.xlabel("Frequencies in Hz")
#     plt.ylabel("|H|")
#     plt.title("FFT of channel:" +str(emg_ch_names[i]))

#     plt.show()
#     i = i+1