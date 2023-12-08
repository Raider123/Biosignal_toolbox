
# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import matplotlib.pyplot as plt 
import copy
import mne 
from scipy import signal as sig
from time import perf_counter

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

#taps = mne.filter.create_filter(data = None, sfreq = f_samp_eeg, l_freq = None, h_freq = 6.0, filter_length=100, l_trans_bandwidth='auto', h_trans_bandwidth='auto', method='fir', iir_params=None, phase='zero', fir_window='hamming', fir_design='firwin2', verbose="DEBUG")
#b1 = mne.filter.create_filter(data = None, sfreq = f_samp_eeg, l_freq = 0.5, h_freq = 4.0, filter_length='auto', l_trans_bandwidth='auto', h_trans_bandwidth='auto', method='fir', iir_params=None, phase='zero', fir_window='blackman', fir_design='firwin', verbose="DEBUG")

# params for firls method
desired = (1, 1, 0, 0) # bandpass 
bands = (0, 6.0, 6.0, 250)

# dc removal filter 
alpha = 0.998 # 0.998 is good 
a = [1, -1 * alpha]
b = [1, -1]

a = [1.0]
# remez zero phas filter 1 
cutoff = 6.0    # Desired cutoff frequency, Hz
trans_width = 3  # Width of transition from pass to stop, Hz    # Size of the FIR filter.

# filter design methods (coefficient calculation)
b_ls = sig.firls(numtaps, bands, desired, fs=f_samp_eeg)
b_window_hamming = sig.firwin(numtaps, 4.0, width=None, window='hamming', pass_zero='lowpass', scale=True, nyq=None, fs=f_samp_eeg)
b_window_blackman = sig.firwin(numtaps, 5.0, width=None, window='flattop', pass_zero='lowpass', scale=True, nyq=None, fs=f_samp_eeg)
# b = sig.remez(numtaps, [0, cutoff, cutoff + trans_width, 0.5*f_samp_eeg], [0.825, 0], fs=f_samp_eeg) # 0.825


# making minimum phase versions 
b_wind_min = sig.minimum_phase(b_window_hamming, method='hilbert', n_fft=8192)
b_wind_min_1 = sig.minimum_phase(b_window_blackman, method='hilbert', n_fft=8192)
b_ls_min = sig.minimum_phase(b_ls, method='hilbert', n_fft=8192)

# get filter characteristic for plot 
w_ham, h_ham, x_ham, db_ham, gd_ham = getFilterCharakteristicsFreq(b_window_hamming, a, f_samp_eeg)
w_black, h_black, x_black, db_black, gd_black = getFilterCharakteristicsFreq(b_window_blackman, a, f_samp_eeg)
w_ls, h_ls, x_ls, db_ls, gd_ls = getFilterCharakteristicsFreq(b_ls, a, f_samp_eeg)

plt.figure()
plt.plot(x_ham, db_ham)
plt.plot(x_black, db_black)
plt.plot(x_ls, db_ls)
plt.legend(["wind hamming", "wind blackman", "ls"] ) #"fir remez", "fir least squares", 
plt.ylim(-75, 5)
plt.grid(True)
plt.title("fir methods comparison")
plt.yticks([0, -20, -40, -60])
plt.ylabel('Gain [dB]')
plt.title('Frequency Response')
plt.show()


# # test with data 
train_file_LSL = ["XY90_unilateral_set3_data"] #"BR60D_unilateral_live_2_data", "BR60D_intentional_unilateral_set8_data", ]
channel_names = ['F3', 'F1', 'FZ', 'F2', 'F4', 'FFC1h', 'FC5', 'FC3', 'FC1', 'FC2', 'FCC3h', 'FCC1h', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'FCC2h', 'CCP3h', 'CCP1h', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'P3', 'P1', 'PZ', 'P2', 'P4', 'FF3', "FF4"]

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


EEG_data= EEGData(format = "Recorded_LSL_stream", filenames = train_file_LSL, data_path = data_path, f_samp = f_samp_eeg, channel_names = channel_names)
EEG_data.windows = np.zeros((1, 2, len(y), 1))
EEG_data.windows[0, 0, :, 0] = sinewave_res # just used dummy sin wave as windows 
EEG_data.windows[0, 1, :, 0] = sinewave_res

EEG_data_raw = copy.deepcopy(EEG_data)
#EEG_data_raw.cutWindows()

b_extend = [b[0]]*50
b_new = np.concatenate((b_extend, b))

#EEG_data.FilterWindows(f_low = 20, f_high = 0.3, order = 2, filter_type = "scipy_butter", apply_method="padding")
#EEG_data.cutWindows()EEG_data.FilterWindows(f_low = None, f_high = 0.3, order = 2, filter_type = "scipy_butter", apply_method="padding")

wind = sig.windows.kaiser_bessel_derived(M=1000, beta = 600, sym=True)
wind  = wind*wind*wind
wind_band = wind[225:-225]
wind_band[0] = 0 # ensure zeros at ends 
wind_band[-1]  = 0 # endsure zeros at ends 
EEG_data.windows[0, 0, :, 0] = EEG_data.windows[0, 0, :, 0] *wind_band

EEG_data.FilterWindows(f_low = 6.0, f_high = 0.3, order = 2, filter_type = "scipy_butter", apply_method="padding")

filtered_sinewave = sig.lfilter(b_window_hamming, [1.0], sinewave_res)


plt.figure()
plt.plot(sinewave)
plt.plot(sinewave_1)
plt.plot(sinewave_res, linewidth = 2)
plt.plot(EEG_data.windows[0, 0, :, 0], linewidth = 3)
plt.legend(["sine 1", "sine 2", "raw", "filter zero"])
plt.show() #plt.plot(filtered_sinewave, linewidth = 3)


plt.figure()
plt.plot(sinewave)
plt.plot(sinewave_1)
plt.plot(sinewave_res, linewidth = 2)
plt.plot(filtered_sinewave, linewidth = 3)
plt.legend(["sine 1", "sine 2", "raw", "fir filter"])
plt.show() #plt.plot(filtered_sinewave, linewidth = 3)


# plt.figure()
# plt.plot(wind_band)
# #plt.show()

