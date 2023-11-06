
# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import matplotlib.pyplot as plt 
import copy
import mne 
from scipy import signal as sig

# set paths
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/mne_machine_learning"
data_path = proj_path+"/data/"

# # own libs 
from biosignal_toolbox.eeg_lib import EEGData

# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************

f_samp_eeg = 500 #sample Frequency of eeg

# subject data 
subject = "JV43"
scenario_name = "intentional_unilateral"
preprocessed_data_filename_end_raw = "34ch_raw_no_scale"
preprocessed_data_filename_end_filtered = "34ch_05Hz"
iteration = 0 # first train test condition 

# window wise metric evaluation
window_size = 1000 #windowsize in ms (analog to pySPACE evaluation)
window_step = 50 # stepsize in ms (analog to pySPACE evaluation)

train_windows = ["bis-2500", "bis-2050", "bis-2200", "bis-1800", "bis-150", "bis-100", "bis-50", "bis0"]

# *********************************************************************************
# ****************** Load and preprocess data   ***********************************
# *********************************************************************************

# load numpy epochs 
# epochs_train_scaled = np.load(data_path+subject+"_"+scenario_name+preprocessed_data_filename_end_raw+"_train_"+str(iteration)+".npy")
# epochs_train_scaled_filterd = np.load(data_path+subject+"_"+scenario_name+preprocessed_data_filename_end_filtered+"_train_"+str(iteration)+".npy")


# ch_names = np.load(data_path+"remaining_eeg_channel_names.npy")

# # train data instance 
# EEG_train_raw = EEGData(format = "NumpyEpochs", epochs = epochs_train_scaled, f_samp = f_samp_eeg)
# EEG_train_raw.setChannelNames(ch_names)

# EEG_train_filtered = EEGData(format = "NumpyEpochs", epochs = epochs_train_scaled_filterd, f_samp = f_samp_eeg)
# EEG_train_filtered.setChannelNames(ch_names)

# # window EEG epochs 
# EEG_train_raw.windowEEGEpochs(window_size, window_step)
# EEG_train_filtered.windowEEGEpochs(window_size, window_step)

# # window selection 
# EEG_train_raw.windowSelection(train_windows)
# EEG_train_filtered.windowSelection(train_windows)

# # filter the data 
# EEG_train_filter_off = copy.deepcopy(EEG_train_raw)
# # EEG_train_filter_low = copy.deepcopy(EEG_train_raw)


# # filter data 
# #DC notch 
# EEG_train_filter_off.FilterWindows(f_low = None, f_high = 0.3, filter_type = "scipy_iir_highpass", order=5,  Q = None, show_response = False) # 0.3 order 5 most similar to offline FIR
# # fir lowpass 
# # EEG_train_filter_off.FilterWindows(f_low = 4.0, f_high = None, filter_type = "fir", order=50)
# # EEG_train_filter_low.FilterWindows(f_low = 4.0, f_high = None, filter_type = "fir", order=50)


# EEG_one_channel_filter_off = EEG_train_filter_off.getDataOneChannel("C1", average = False, is_windowed = True)
# EEG_one_channel_filter_on = EEG_train_filtered.getDataOneChannel("C1", average = False, is_windowed = True)
# EEG_one_channel_filter_low = EEG_train_filter_low.getDataOneChannel("C1", average = False, is_windowed = True)

# for trial_idx in range(0, 10): 
#     plt.figure()
#     plt.plot(EEG_one_channel_filter_off[trial_idx, :, 2])
#     plt.plot(EEG_one_channel_filter_on[trial_idx, :, 2])
#     #plt.plot(EEG_one_channel_filter_low[trial_idx, :, 7])
#     plt.legend(["offline filt", "online filt"])

#     plt.figure()
#     plt.plot(EEG_one_channel_filter_off[trial_idx, :, 7])
#     plt.plot(EEG_one_channel_filter_on[trial_idx, :, 7])

#     plt.legend(["offline filt", "online filt"])
#     plt.show()


# mne filter design 

#b = sig.firwin(3500, 0.5, window='hamming')#('kaiser', 8)) 

b = mne.filter.create_filter(data = None, sfreq = f_samp_eeg, l_freq = 0.0, h_freq = 4.0, filter_length='auto', l_trans_bandwidth='auto', h_trans_bandwidth='auto', method='fir', iir_params=None, phase='zero', fir_window='hamming', fir_design='firwin2', verbose=None)
b1, a1 = sig.iirfilter(2, [0.3, 5.0], btype='bandpass', ftype='butter', output='ba', fs=f_samp_eeg)
b2, a2 = sig.iirfilter(1, [0.1, 7.0], btype='bandpass', ftype='bessel', output='ba', fs=f_samp_eeg)


print("length off filter: ", len(b))

w, h = sig.freqz(b, worN=4048)
w1, h1 = sig.freqz(b1, a1,  worN=4048)
w2, h2 = sig.freqz(b2, a2,  worN=4048)


x = (w/np.pi)*(f_samp_eeg/2)
x1 = (w1/np.pi)*(f_samp_eeg/2)
x2 = (w2/np.pi)*(f_samp_eeg/2)

db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
db1 = 20*np.log10(np.maximum(np.abs(h1), 1e-5))
db2 = 20*np.log10(np.maximum(np.abs(h2), 1e-5))

w2, gd2 = sig.group_delay((b2, a2), w=4048)


plt.figure()
plt.plot(x[0:1000], db[0:1000])
plt.plot(x1[0:1000], db1[0:1000])
plt.plot(x2[0:1000], db2[0:1000])
plt.legend(["off", "butter", "bessel"])
plt.ylim(-75, 5)
plt.grid(True)
plt.yticks([0, -20, -40, -60])
plt.ylabel('Gain [dB]')
plt.title('Frequency Response')
plt.show()


plt.plot(x2[0:1000], (np.angle(h2))[0:1000])
plt.grid(True)
plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],[r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
plt.ylabel('Phase [rad]')
plt.xlabel('Frequency[Hz]')
plt.show()

plt.figure()
plt.title('Digital filter group delay')
plt.plot(x2, gd2)
plt.ylabel('Group delay [samples]')
plt.xlabel('Frequency in Hz')
plt.show()


# res = 0.01
# window = np.zeros(int(500/res))
# offset = 0

# n = 700
# bessel = sig.windows.kaiser_bessel_derived(n, 250)
# bessel = bessel[140:]
# window[0:len(bessel)] = bessel


# plt.figure()
# plt.plot(np.arange(0, 500, step = res),  window)
# plt.show()