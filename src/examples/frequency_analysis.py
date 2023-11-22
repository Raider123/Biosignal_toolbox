# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import matplotlib.pyplot as plt 

# # own libs 
from biosignal_toolbox.eeg_lib import EEGData

# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************

# own libs
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox"
data_path = proj_path+"/data/"


filename = ["20032023_AF64D_speech_set1.vhdr"]
channels_to_evaluate = ["F3"]


# *********************************************************************************
# **************************** Main section****************************************
# *********************************************************************************

eeg_data = EEGData(format = "Brainvision", filenames = filename, data_path = data_path)
eeg_data.FilterRaw(f_highpass = 20, f_lowpass = None, picks=channels_to_evaluate) 

first_channel_eeg = eeg_data.getDataFromChannels(channels_to_evaluate)[0, :]

no_motor_noise = first_channel_eeg[2000:6000]
motor_noise = first_channel_eeg[51000:55000]

# unfiltered raw signale of channel 
plt.figure()
plt.plot(first_channel_eeg)
plt.xlabel("samples")
plt.ylabel("voltage in V")
plt.title("raw data EMG from EEG system")
plt.show()


eeg_data.OnechannelFFT(no_motor_noise, plot = True, window = None, beta = None, title = "no motor touching")
eeg_data.OnechannelFFT(motor_noise, plot = True, window = None, beta = None, title = "motor touching")