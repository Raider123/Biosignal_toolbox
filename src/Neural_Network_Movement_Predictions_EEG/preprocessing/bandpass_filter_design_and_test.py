
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

#taps = sig.firwin2(100, [0.5, 40.0], gain = 0, window='hamming', fs = f_samp_eeg)#('kaiser', 8)) 


#taps = mne.filter.create_filter(data = None, sfreq = f_samp_eeg, l_freq = None, h_freq = 6.0, filter_length=100, l_trans_bandwidth='auto', h_trans_bandwidth='auto', method='fir', iir_params=None, phase='zero', fir_window='hamming', fir_design='firwin2', verbose="DEBUG")
b1 = mne.filter.create_filter(data = None, sfreq = f_samp_eeg, l_freq = 0.5, h_freq = 4.0, filter_length='auto', l_trans_bandwidth='auto', h_trans_bandwidth='auto', method='fir', iir_params=None, phase='zero', fir_window='blackman', fir_design='firwin', verbose="DEBUG")

#b = sig.firwin(499, [0.5, 4.0], width=None, window='blackman', pass_zero=False, scale=True, nyq=None, fs=f_samp_eeg)

# dc removal filter 
alpha = 0.998 # 0.998 is good 
a = [1, -1 * alpha]
b = [1, -1]


# scale =1
# b2 = [ 1 *scale, -alpha *scale] 
# a2 = [-alpha*scale, 1*scale]


# bandpass remeze 
# band = [0.5, 4.0]  # Desired pass band, Hz
# trans_width = 0.5    # Width of transition from pass to stop, Hz
# numtaps = 100        # Size of the FIR filter.

cutoff = 6.0    # Desired cutoff frequency, Hz
trans_width = 2  # Width of transition from pass to stop, Hz
numtaps = 99      # Size of the FIR filter.

taps = sig.remez(numtaps, [0, cutoff, cutoff + trans_width, 0.5*f_samp_eeg], [0.825, 0], fs=f_samp_eeg) # 0.825

# edges = [0, band[0] - trans_width, band[0], band[1], band[1] + trans_width, 0.5*f_samp_eeg]
# taps = sig.remez(numtaps, edges, [0, 1, 0], fs=f_samp_eeg)


#b1, a1 = sig.iirfilter(2, [0.3, 5.0], btype='bandpass', ftype='butter', output='ba', fs=f_samp_eeg)
sos= sig.iirfilter(2, [0.5, 4.0], btype='bandpass', ftype='butter', output='sos', fs=f_samp_eeg)


print("length off filter: ", len(b))

w, h = sig.freqz(b, a, worN=4048)
w1, h1 = sig.freqz(b1,  worN=4048)
w2, h2 = sig.freqz(taps, [1],  worN=4048)

x = (w/np.pi)*(f_samp_eeg/2)
x1 = (w1/np.pi)*(f_samp_eeg/2)
x2 = (w2/np.pi)*(f_samp_eeg/2)

db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
db = db

db1 = 20*np.log10(np.maximum(np.abs(h1), 1e-5))
db2 = 20*np.log10(np.maximum(np.abs(h2), 1e-5))

# w2, gd2 = sig.group_delay((b2, a2), w=4048)
w, gd = sig.group_delay((b, a), w=4048)

plt.figure()
plt.plot(x[0:1000], db[0:1000])
plt.plot(x1[0:1000], db1[0:1000])
plt.plot(x2[0:1000], db2[0:1000])
#plt.plot(x2[0:1000], db2[0:1000])
plt.legend(["dc removal","offline fir", "lowpass fir N = 100"])
plt.ylim(-75, 5)
plt.grid(True)
plt.yticks([0, -20, -40, -60])
plt.ylabel('Gain [dB]')
plt.title('Frequency Response')
plt.show()


plt.plot(x[0:1000], (np.angle(h))[0:1000])
# plt.plot(x2[0:1000], (np.angle(h2))[0:1000])
plt.grid(True)
plt.legend(["dc removal", "allpass"])
#plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],[r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
plt.ylabel('Phase [rad]')
plt.xlabel('Frequency[Hz]')
plt.show()

plt.figure()
plt.title('Digital filter group delay')
plt.plot(x, gd)
# plt.plot(x2, gd2)
plt.ylabel('Group delay [samples]')
plt.xlabel('Frequency in Hz')
plt.show()


# test with data 
train_file_LSL = ["XY90_unilateral_set3_data"] #"BR60D_unilateral_live_2_data", "BR60D_intentional_unilateral_set8_data", ]
channel_names = ['F3', 'F1', 'FZ', 'F2', 'F4', 'FFC1h', 'FC5', 'FC3', 'FC1', 'FC2', 'FCC3h', 'FCC1h', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'FCC2h', 'CCP3h', 'CCP1h', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'P3', 'P1', 'PZ', 'P2', 'P4', 'FF3', "FF4"]
inverse_keep_channel = True
marker_number = 22 
error_number = 3
t1 = -5.0
t2 = 0.0
window_size = 1100
window_step = 50


EEG_data_offline = EEGData(format = "Recorded_LSL_stream", filenames = train_file_LSL, data_path = data_path, f_samp = f_samp_eeg, channel_names = channel_names)
EEG_data_online = copy.deepcopy(EEG_data_offline)

# online vs. offline filter comparison 
EEG_data_offline.rereferencingEpoching(marker_number, error_number, [], apply_filter=True, f_highpass = 0.5, f_lowpass=4.0, inverse_keep_channel = inverse_keep_channel, event_id_used = marker_number, t1 = t1, t2= t2)
EEG_data_online.rereferencingEpoching(marker_number, error_number, [], apply_filter=False, f_highpass = None, f_lowpass=None, inverse_keep_channel = inverse_keep_channel, event_id_used = marker_number, t1 = t1, t2= t2)

EEG_data_offline.windowEEGEpochs(window_size = window_size)
EEG_data_online.windowEEGEpochs(window_size = window_size)

print(EEG_data_online.getWindowNames())
print("len names", len(EEG_data_online.getWindowNames()))


EEG_data_offline.windowSelection(["bis-2000", "bis0"])
EEG_data_online.windowSelection(["bis-2000", "bis0"])


#EEG_data_online.FilterWindows(filter_type="dc_removal", alpha = 0.998)
#EEG_data_online.FilterWindows(filter_type="allpass", alpha = 0.998)
EEG_data_online.FilterWindows(f_low = 5.0, f_high = 0.3, order = 2, filter_type = "scipy_butter") # trials channels, sampels, windows 
print("window shape: ", EEG_data_online.windows.shape)
EEG_data_online.windows = EEG_data_online.windows[:, :, 25:-25, :]
print("window shape: ", EEG_data_online.windows.shape)

trial = 10
channel = 10 

plt.figure()
plt.plot(EEG_data_offline.getWindows()[trial, channel, :, -1])
plt.plot(EEG_data_online.getWindows()[trial, channel, :, -1])
plt.legend(["offline filter", "online filter"])
plt.show()


