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
# filenames = ["20230426_AJ05D_orthosisErrorIjcai_multi_set4"] 

# #create numpy array with file names 
# data_str_arr = []
# for files_str in filenames: 
#     data_str_arr.append(os.path.join(data_path, files_str)) 


# raw= eeg_lib.loadBrainproductsData(data_str_arr) # read data in brainproducts format
# f_samp_eeg = raw.info['sfreq'] # get sampling rate 

# data = raw.get_data()
# print(data.shape)

# one_ch_data = data[10, :]

# #print(raw.ch_names)

# names = list(raw.ch_names) 

# print(len(names))

# i = 0

# for one_ch_data in data: 
# # Number of sample points

#     N = len(one_ch_data)

#     # sample spacing

#     dT = 1.0/f_samp_eeg

#     x = np.linspace(0.0, N*dT, N, endpoint=False)

#     y = one_ch_data

#     yf = fft(y)

#     xf = fftfreq(N, dT)[:N//2]


#     plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))

#     plt.grid()
#     plt.xlabel("Frequencies in Hz")
#     plt.ylabel("|H|")
#     plt.title("FFT of channel:" +str(names[i]))

#     plt.show()
#     i = i+1



# EMG 

f_samp_emg = 1000 

# for ANT Systems 
emg_ch_names = ["EMG1", "EMG2", "EMG3", "EMG4", "EMG5", "EMG6", "EMG7", "EMG8"]


 # EMG data loading and processing 
filename_EMG = "20230426_AJ05D_orthosisErrorIjcai_multi_set7.txt"



file_str_emg = os.path.join(data_path, filename_EMG)
emg_data, emg_time_axis = emg_lib.loadMiniANTEMGData(file_str_emg, f_samp_emg)

emg_data_filtered = emg_lib.applyBPFilterRectifying(f_samp_emg, 20, 450, emg_data)


i = 0
for one_ch_data in emg_data_filtered.T: 
# Number of sample points

    N = len(one_ch_data)

    # sample spacing

    dT = 1.0/f_samp_emg

    x = np.linspace(0.0, N*dT, N, endpoint=False)

    y = one_ch_data

    yf = fft(y)

    xf = fftfreq(N, dT)[:N//2]


    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))

    plt.grid()
    plt.xlabel("Frequencies in Hz")
    plt.ylabel("|H|")
    plt.title("FFT of channel:" +str(emg_ch_names[i]))

    plt.show()
    i = i+1