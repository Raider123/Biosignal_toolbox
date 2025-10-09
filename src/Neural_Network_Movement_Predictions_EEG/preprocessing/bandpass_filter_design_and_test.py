
# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import matplotlib.pyplot as plt 
import copy
import mne 
from scipy import signal as sig
from time import perf_counter
from PyEMD import EMD, EEMD

# set paths
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox"
data_path = proj_path+"/data/"

# # own libs 
from biosignal_toolbox.eeg_lib import EEGData


# methods 
def getFilterCharakteristicsFreq(b, a, f_samp): 
    w, h = sig.freqz(b, a, worN=4048)
    x = (w/np.pi)*(f_samp/2)
    db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
    w_group, gd = sig.group_delay((b, a), w=4048)

    return w, h, x, db, gd


# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************

f_samp_eeg = 500.0
numtaps = 43 
# window wise metric evaluation
window_size = 5000 # windowsize in ms (analog to pySPACE evaluation) + add 100 ms for cutting after filtering 
window_step = 20 # stepsize in ms (analog to pySPACE evaluation)


f_samp_eeg = 500.0 #sample Frequency of eeg
marker_number = 22 # onset markernumber (Qualisys)
error_number = 3 # number of the error marker 


# eeg channel that are kept (inverse_keep_channel = False) or dropped (inverse_keep_channel = True) for further evaluations, empty list meaning all channels are kept 
inverse_keep_channel = True # standard: True 
#channel_list = [] # do not drop channels
channel_list = ["F5", "F6", "x_dir", "y_dir", "z_dir", "FP1", "FP2", "F8", "T7", "T8", "TP9", "TP10", "P7", "P8", "PO9", "O1", "OZ", "O2", "PO10", "AF7", "AF3", "AF4", "AF8", "FT9", "FT7", "FT8", "FT10", "TP7", "TP8", "PO7", "PO3", "POZ", "PO4", "PO8", "F7"]


# just remap the parameters (need to be adapted)
t1 = -10.0
t2 = 0.0 # to cut this off later 

# filter settings 
f_highpass = None
f_lowpass_MLP = None
f_lowpass_EEGNet = None


#taps = mne.filter.create_filter(data = None, sfreq = f_samp_eeg, l_freq = None, h_freq = 6.0, filter_length=100, l_trans_bandwidth='auto', h_trans_bandwidth='auto', method='fir', iir_params=None, phase='zero', fir_window='hamming', fir_design='firwin2', verbose="DEBUG")
#b1 = mne.filter.create_filter(data = None, sfreq = f_samp_eeg, l_freq = 0.5, h_freq = 4.0, filter_length='auto', l_trans_bandwidth='auto', h_trans_bandwidth='auto', method='fir', iir_params=None, phase='zero', fir_window='blackman', fir_design='firwin', verbose="DEBUG")

# # params for firls method
# desired = (1, 1, 0, 0) # bandpass 
# bands = (0, 6.0, 6.0, 250)

# # dc removal filter 
# alpha = 0.998 # 0.998 is good 
# a = [1, -1 * alpha]
# b = [0.95, -1]

# a = [1.0]

# n_moving_ave = 70
# b_ave = np.ones(n_moving_ave) / n_moving_ave

# # remez zero phas filter 1 
# cutoff = 6.0    # Desired cutoff frequency, Hz
# trans_width = 3  # Width of transition from pass to stop, Hz    # Size of the FIR filter.

# # filter design methods (coefficient calculation)
# b_ls = sig.firls(numtaps, bands, desired, fs=f_samp_eeg)
# b_window_hamming = sig.firwin(numtaps, 4.0, width=None, window='hamming', pass_zero='lowpass', scale=True, nyq=None, fs=f_samp_eeg)
# b_window_blackman = sig.firwin(numtaps, 5.0, width=None, window='flattop', pass_zero='lowpass', scale=True, nyq=None, fs=f_samp_eeg)
# # b = sig.remez(numtaps, [0, cutoff, cutoff + trans_width, 0.5*f_samp_eeg], [0.825, 0], fs=f_samp_eeg) # 0.825


# # making minimum phase versions 
# b_wind_min = sig.minimum_phase(b_window_hamming, method='hilbert', n_fft=8192)
# b_wind_min_1 = sig.minimum_phase(b_window_blackman, method='hilbert', n_fft=8192)
# b_ls_min = sig.minimum_phase(b_ls, method='hilbert', n_fft=8192)

# eeg channel that are kept (inverse_keep_channel = False) or dropped (inverse_keep_channel = True) for further evaluations, empty list meaning all channels are kept 
inverse_keep_channel = True # standard: True 
#channel_list = [] # do not drop channels
channel_list = ["C1", "FC1"]



# filter settings 
f_highpass = None
f_lowpass_MLP = None


# # # test with data 
# train_file_LSL = ["XY90_unilateral_set3_data"] #"BR60D_unilateral_live_2_data", "BR60D_intentional_unilateral_set8_data", ]
# channel_names = ['F3', 'F1', 'FZ', 'F2', 'F4', 'FFC1h', 'FC5', 'FC3', 'FC1', 'FC2', 'FCC3h', 'FCC1h', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'FCC2h', 'CCP3h', 'CCP1h', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'P3', 'P1', 'PZ', 'P2', 'P4', 'FF3', "FF4"]

# generate some example sine waves for testing the filters 
f = 3.0
w = 2. * np.pi * f
time_interval = 1.5
samples = 550
t = np.linspace(0, time_interval, samples)
y = np.sin(w * t - w*t*3/4)

f1= 15.0
w1 = 2. * np.pi * f1
y1 = np.sin(w1 * t - w1*t*3/4)

#y = y+1
sinewave = copy.deepcopy(y) # to be processed 
sinewave_1 = copy.deepcopy(y1) # to be processed 
sinewave_res = sinewave+sinewave_1


#EEG_data= EEGData(format = "Recorded_LSL_stream", filenames = train_file_LSL, data_path = data_path, f_samp = f_samp_eeg, channel_names = channel_names)

EEG_data = EEGData(format = "Brainvision", filenames = ["31102023_BR60D_unilateral_set1.vhdr"], data_path = data_path)
EEG_data_offline_filter = copy.deepcopy(EEG_data)
EEG_data.rereferencingEpoching(marker_number, error_number, channel_list, inverse_keep_channel = inverse_keep_channel, t1 = t1, t2= t2)


b, a = EEG_data_offline_filter.designFilter(f_low = 5.0, f_high = 0.3, order = 2, filter_type = "scipy_butter", return_type = "ba")
EEG_data_offline_filter.filterRawData(b= b, a = a, apply_method = "zero_phase_ba", padtype = "even")
EEG_data_offline_filter.rereferencingEpoching(marker_number, error_number, channel_list, inverse_keep_channel = inverse_keep_channel, t1 = t1, t2= t2, f_lowpass=4.0, f_highpass = 0.5, apply_filter = False)


EEG_data.windowEEGEpochs(window_size, window_step)
EEG_data_offline_filter.windowEEGEpochs(window_size, window_step)
EEG_data_offline_filter.windowSelection(["bis0"])

print("window names", EEG_data.getWindowNames())

# use sine waves for testing 
# EEG_data.windows = np.zeros((1, 2, len(y), 1))
# EEG_data.windows[0, 0, :, 0] = sinewave_res # just used dummy sin wave as windows 
# EEG_data.windows[0, 1, :, 0] = sinewave_res

EEG_data_raw = copy.deepcopy(EEG_data)
EEG_data_filter1 = copy.deepcopy(EEG_data)
EEG_data_filter_var = copy.deepcopy(EEG_data)

#EEG_data_raw.cutWindows()

# b_extend = [b[0]]*50
# b_new = np.concatenate((b_extend, b))

# #EEG_data.FilterWindows(f_low = 20, f_high = 0.3, order = 2, filter_type = "scipy_butter", apply_method="padding")
# #EEG_data.cutWindows()EEG_data.FilterWindows(f_low = None, f_high = 0.3, order = 2, filter_type = "scipy_butter", apply_method="padding")

# wind = sig.windows.kaiser_bessel_derived(M=1000, beta = 600, sym=True)
# wind  = wind*wind*wind
# wind_band = wind[225:-225]
# wind_band[0] = 0 # ensure zeros at ends 
# wind_band[-1]  = 0 # endsure zeros at ends 
# EEG_data.windows[0, 0, :, 0] = EEG_data.windows[0, 0, :, 0] *wind_band


#EEG_data.reverseWindows()
#sos = EEG_data.designFilter(f_low = 9.0, f_high = 0.2, order = 1, filter_type = "scipy_butter", return_type = "sos")
#b, a = EEG_data.designFilter(f_low = 5.0, f_high = 0.3, order = 2, filter_type = "scipy_butter", return_type = "ba")
# b1 = [-0.5, 1, -0.5]
# a1= [1]



EEG_data.windowSelection(["bis0"])
#EEG_data.reverseWindows()

#EEG_data.filterWindows(b = b1, a = a1, apply_method = "forward_ba_filter") # bandpass filter (zero phase with padding) 
#EEG_data.filterWindows(b = b, a = a, apply_method = "gustav") # bandpass filter (zero phase with padding) 

#b, a = EEG_data.designFilter(f_low = 4.0, f_high = None, order = 39, filter_type = "fir_hann", return_type = "ba")
b, a = EEG_data.designFilter(f_low = 5.0, f_high = 0.3, order = 2, filter_type = "scipy_butter", return_type = "ba")


EEG_data.WindowMeanCorrection()

EEG_data.filterWindows(b = b, a = a, apply_method = "gustav", padtype = "own") # bandpass filter (zero phase with padding) 
# EEG_data.cutWindows(2000, 2000)
#EEG_data.detrendWindows()

# EEG_data.filterWindows(sos = sos, apply_method = "forward_sos_filter")
# EEG_data.filterWindows(b = b, a = a, apply_method = "forward_ba_filter")

# emd = EMD(max_imf=10)

# IMF = emd.emd(EEG_data.windows[3, 4, :, 0])
# print("shape IMF:", IMF.shape)

#EEG_data.reverseWindows()

#EEG_data.reverseWindows() # reverse window back 

#EEG_data.windows[2, 4, :, 0] = IMF[-1, :]

# apply fir filter after reversing array to compensate  for the delay 
#EEG_data.applyMovingAverageFilter(n = 19, apply_to_structures = "windows")

#b, a  = EEG_data.designFilter(f_low = 5.0, f_high = 0.3, order = 4, filter_type = "scipy_butter", return_type = "ba")

#sos1 = EEG_data.designFilter(f_low = 5.0, f_high = 0.4, order = 4, filter_type = "scipy_butter", return_type = "sos", rp = 0.05, rs = 80.0)

#b, a = EEG_data_filter1.designFilter(f_low = 5.0, f_high = 0.3, order = 2, filter_type = "scipy_butter", return_type = "ba")
#EEG_data_filter1.filterWindows(sos = sos1, apply_method = "zero_phase_sos") # bandpass filter (zero phase with padding)

#EEG_data.filterWindows(sos = sos, apply_method = "zero_phase_sos") # bandpass filter (zero phase with padding)
#EEG_data.filterWindows(b = b, a = a, apply_method = "gustav") # bandpass filter
#EEG_data_filter_var.applyMovingAverageFilter(apply_to_structures="windows", n = n_moving_ave)


plt.figure()
#plt.plot(EEG_data_raw.windows[0, 0, :, 0], linewidth = 1)
#plt.plot(EEG_data_filter1.windows[0, 0, :, 0], linewidth = 3)
plt.plot(EEG_data.windows[3, 4, :, 0], linewidth = 3)
#plt.plot(EEG_data.windows[2, 4, :, 0], linewidth = 3)#
plt.plot(EEG_data_offline_filter.windows[3, 4, :, 0], linewidth = 3)


print("shape old_online wind", EEG_data.windows.shape)
print("shape offline wind", EEG_data_offline_filter.windows.shape)

# plt.plot(EEG_data_filter_var.windows[0, 0, :, 0], linewidth = 3)
plt.legend(["Gustav", "offline filt"])
#plt.legend(["sine 1", "sine 2", "raw", "filter zero"])

window_diffs =  EEG_data_offline_filter.windows -EEG_data.windows

for i in range(0, window_diffs.shape[0]): 
    plt.figure()
    plt.plot(window_diffs[i, 4, :, 0])

# plt.figure()
# plt.plot(IMF[-1, :], linewidth = 3)


# plt.figure()
# #plt.plot(EEG_data_raw.windows[0, 1, :, 0], linewidth = 1)
# plt.plot(EEG_data_offline_filter.windows[0, 1, :, 0], linewidth = 3)
# # plt.plot(EEG_data_filter1.windows[0, 1, :, 0], linewidth = 3)
# plt.plot(EEG_data.windows[0, 1, :, 0], linewidth = 3)
# # plt.plot(EEG_data_filter_var.windows[0, 1, :, 0], linewidth = 3)
# plt.legend(["raw", "offline filt", "gustav", "padding", "moving ave filter"])
# #plt.legend(["sine 1", "sine 2", "raw", "filter zero"])


# trial = 10
# plt.figure()
# #plt.plot(EEG_data_raw.windows[trial, 0, :, 0], linewidth = 1)
# plt.plot(EEG_data_offline_filter.windows[trial, 0, :, 0], linewidth = 3)
# # plt.plot(EEG_data_filter1.windows[trial, 0, :, 0], linewidth = 3)
# plt.plot(EEG_data.windows[trial, 0, :, 0], linewidth = 3)
# # plt.plot(EEG_data_filter_var.windows[trial, 0, :, 0], linewidth = 3)
# plt.legend(["raw", "offline filt", "gustav", "padding", "moving ave filter"])
# #plt.legend(["sine 1", "sine 2", "raw", "filter zero"])


# plt.figure()
# #plt.plot(EEG_data_raw.windows[trial, 1, :, 0], linewidth = 1)
# plt.plot(EEG_data_offline_filter.windows[trial, 1, :, 0], linewidth = 3)
# # plt.plot(EEG_data_filter1.windows[trial, 1, :, 0], linewidth = 3)
# plt.plot(EEG_data.windows[trial, 1, :, 0], linewidth = 3)
# # plt.plot(EEG_data_filter_var.windows[trial, 1, :, 0], linewidth = 3)
# plt.legend(["raw", "offline filt", "gustav", "padding", "moving ave filter"])
# #plt.legend(["sine 1", "sine 2", "raw", "filter zero"])


# trial = 5
# plt.figure()
# plt.plot(EEG_data_raw.windows[trial, 0, :, 0], linewidth = 1)
# plt.plot(EEG_data_offline_filter.windows[trial, 0, :, 0], linewidth = 3)
# plt.plot(EEG_data_filter1.windows[trial, 0, :, 0], linewidth = 3)
# plt.plot(EEG_data_filter_var.windows[trial, 0, :, 0], linewidth = 3)
# plt.legend(["raw", "offline filt", "gustav", "padding", "variance filter"])
# #plt.legend(["sine 1", "sine 2", "raw", "filter zero"])


# plt.figure()
# plt.plot(EEG_data_raw.windows[trial, 1, :, 0], linewidth = 1)
# plt.plot(EEG_data_offline_filter.windows[trial, 1, :, 0], linewidth = 3)
# plt.plot(EEG_data_filter1.windows[trial, 1, :, 0], linewidth = 3)
# plt.plot(EEG_data.windows[trial, 1, :, 0], linewidth = 3)
# plt.plot(EEG_data_filter_var.windows[trial, 1, :, 0], linewidth = 3)
# plt.legend(["raw", "offline filt", "gustav", "padding", "variance filter"])
# #plt.legend(["sine 1", "sine 2", "raw", "filter zero"])


# trial = 15
# plt.figure()
# plt.plot(EEG_data_raw.windows[trial, 0, :, 0], linewidth = 1)
# plt.plot(EEG_data_offline_filter.windows[trial, 0, :, 0], linewidth = 3)
# plt.plot(EEG_data_filter1.windows[trial, 0, :, 0], linewidth = 3)
# plt.plot(EEG_data.windows[trial, 0, :, 0], linewidth = 3)
# plt.plot(EEG_data_filter_var.windows[trial, 0, :, 0], linewidth = 3)
# plt.legend(["raw", "offline filt", "gustav", "padding", "variance filter"])
# #plt.legend(["sine 1", "sine 2", "raw", "filter zero"])


# plt.figure()
# plt.plot(EEG_data_raw.windows[trial, 1, :, 0], linewidth = 1)
# plt.plot(EEG_data_offline_filter.windows[trial, 1, :, 0], linewidth = 3)
# plt.plot(EEG_data_filter1.windows[trial, 1, :, 0], linewidth = 3)
# plt.plot(EEG_data.windows[trial, 1, :, 0], linewidth = 3)
# plt.plot(EEG_data_filter_var.windows[trial, 1, :, 0], linewidth = 3)
# plt.legend(["raw", "offline filt", "gustav", "padding", "variance filter"])
# #plt.legend(["sine 1", "sine 2", "raw", "filter zero"])




plt.show() #plt.plot(filtered_sinewave, linewidth = 3)

# plt.figure()
# # plt.plot(sinewave)
# # plt.plot(sinewave_1)
# #plt.plot(sinewave_res, linewidth = 2)
# plt.plot(EEG_data_raw.windows[0, 10, :, 0], linewidth = 1)
# plt.plot(EEG_data_filter1.windows[0, 10, :, 0], linewidth = 3)
# plt.plot(EEG_data.windows[0, 10, :, 0], linewidth = 3)
# plt.legend(["raw", "gustav", "padding"])
# #plt.legend(["sine 1", "sine 2", "raw", "filter zero"])
# plt.show() #plt.plot(filtered_sinewave, linewidth = 3)


# plt.figure()
# # plt.plot(sinewave)
# # plt.plot(sinewave_1)
# # plt.plot(sinewave_res, linewidth = 2)
# plt.plot(EEG_data_raw.windows[0, 0, :, 0], linewidth = 1)
# plt.plot(filtered_sinewave, linewidth = 3)
# plt.legend(["sine 1", "sine 2", "raw", "fir filter"])
# plt.show() #plt.plot(filtered_sinewave, linewidth = 3)


# plt.figure()
# plt.plot(wind_band)
# #plt.show()

