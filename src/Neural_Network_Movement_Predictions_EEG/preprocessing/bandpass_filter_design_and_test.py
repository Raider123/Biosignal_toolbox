
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

# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************
f_samp_eeg = 500.0

# mne filter design 

#b = sig.firwin2(1000, [0.5, 40.0], gain = 0, window='hamming', fs = f_samp_eeg)#('kaiser', 8)) 

b = mne.filter.create_filter(data = None, sfreq = f_samp_eeg, l_freq = 0.5, h_freq = 40.0, filter_length=1000, l_trans_bandwidth='auto', h_trans_bandwidth='auto', method='fir', iir_params=None, phase='minimum', fir_window='hamming', fir_design='firwin2', verbose=None)
#b1, a1 = sig.iirfilter(2, [0.3, 5.0], btype='bandpass', ftype='butter', output='ba', fs=f_samp_eeg)
sos= sig.iirfilter(2, [0.5, 40.0], btype='bandpass', ftype='butter', output='sos', fs=f_samp_eeg)



print("length off filter: ", len(b))

w, h = sig.freqz(b, worN=4048)
# w1, h1 = sig.freqz(b1, a1,  worN=4048)
# w2, h2 = sig.freqz(b2, a2,  worN=4048)

x = (w/np.pi)*(f_samp_eeg/2)
# x1 = (w1/np.pi)*(f_samp_eeg/2)
# x2 = (w2/np.pi)*(f_samp_eeg/2)

db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
# db1 = 20*np.log10(np.maximum(np.abs(h1), 1e-5))
# db2 = 20*np.log10(np.maximum(np.abs(h2), 1e-5))

# w2, gd2 = sig.group_delay((b2, a2), w=4048)


plt.figure()
plt.plot(x[0:1000], db[0:1000])
#plt.plot(x1[0:1000], db1[0:1000])
#plt.plot(x2[0:1000], db2[0:1000])
plt.legend(["off", "butter", "bessel"])
plt.ylim(-75, 5)
plt.grid(True)
plt.yticks([0, -20, -40, -60])
plt.ylabel('Gain [dB]')
plt.title('Frequency Response')
plt.show()


# plt.plot(x2[0:1000], (np.angle(h2))[0:1000])
# plt.grid(True)
# plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],[r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
# plt.ylabel('Phase [rad]')
# plt.xlabel('Frequency[Hz]')
# plt.show()

# plt.figure()
# plt.title('Digital filter group delay')
# plt.plot(x2, gd2)
# plt.ylabel('Group delay [samples]')
# plt.xlabel('Frequency in Hz')
# plt.show()


# test with data 
train_file_LSL = ["XY90_unilateral_set3_data"] #"BR60D_unilateral_live_2_data", "BR60D_intentional_unilateral_set8_data", ]
channel_names = ['F3', 'F1', 'FZ', 'F2', 'F4', 'FFC1h', 'FC5', 'FC3', 'FC1', 'FC2', 'FCC3h', 'FCC1h', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'FCC2h', 'CCP3h', 'CCP1h', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'P3', 'P1', 'PZ', 'P2', 'P4', 'FF3', "FF4"]
inverse_keep_channel = True
marker_number = 22 
error_number = 3
t1 = -3.0
t2 = 0.0
window_size = 1000
window_step = 50


EEG_data = EEGData(format = "Recorded_LSL_stream", filenames = train_file_LSL, data_path = data_path, f_samp = f_samp_eeg, channel_names = channel_names)
EEG_data.rereferencingEpoching(marker_number, error_number, [], apply_filter=False, f_highpass = None, inverse_keep_channel = inverse_keep_channel, event_id_used = marker_number, t1 = t1, t2= t2)
EEG_data.windowEEGEpochs(window_size, window_step)


time_axis, epochs = EEG_data.getEpochs()
mean_epochs = np.mean(epochs, axis = 0)
epochs_filter = copy.deepcopy(epochs)

signal = mean_epochs[10, :]

EEG_data.windows = EEG_data.windows[0:2, :, :, -2:]
windows = EEG_data.getWindows()
EEG_data_filter = copy.deepcopy(EEG_data)

t1 = perf_counter()*1000
EEG_data_filter.FilterWindows(f_low = 40.0, f_high = 0.3, order = 2, filter_type = "scipy_butter")
#EEG_data_filter.FilterWindows(f_low =40.0, f_high = 0.5, order = 'auto', filter_type = "mne_iir")
t2 = perf_counter()*1000

print("process time: ", t2-t1)

windows_filtered = EEG_data_filter.getWindows()

#epochs_filtered = mne.filter.filter_data(data =signal, sfreq = f_samp_eeg, l_freq = 0.5, h_freq = 4.0, picks=None, method ="iir") #filter_length=3000, fir_design='firwin2'

#epochs_filtered = sig.sosfilt(sos, signal)

# plt.figure()
# plt.plot(time_axis, mean_epochs[10, :])
# plt.show()

# plt.figure()
# plt.plot(time_axis, epochs_filtered)
# plt.show()

print(windows_filtered.shape)


plt.figure()
plt.plot(windows[0, 9, :, -1])
plt.show()

plt.plot(windows_filtered[0, 9, :, -1])
plt.show()
