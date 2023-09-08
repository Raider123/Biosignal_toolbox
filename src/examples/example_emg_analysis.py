
# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import sys 
import matplotlib.pyplot as plt
import os 
from biosignal_toolbox.emg_lib import EMGData

# project path settings 
current_path = os.path.dirname(os.path.abspath(__file__)) # project path 
project_path = os.path.split(os.path.split(current_path)[0])[0] # go up two folders to get the current path
data_path = os.path.join(project_path, 'data') # path where the data lays 


# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************


# Create an string with dataset file name (ANT and Cometa)
file_str = "01092022_BO42D_B_S0_H_1.txt"


# *** channel selection params (comment in for channel selection) ***
#selected_channels =  []# ['\tR.Biceps Br.(uV)', '\tR.Ant.Deltoid(uV)', '\tR.Mid Delt.(uV)']
inverse = False # define True for channel selection


# for ANT Systems 
emg_ch_names = ["EMG1", "EMG2", "EMG3", "EMG4", "EMG5", "EMG6", "EMG7", "EMG8"]
selected_channels = ["EMG1", "EMG2"]


# EMG sampling rate (Cometa default)
#fsamp_emg = 2000 # in Hz

fsamp_emg = 500 # for ANT

# *** processing parameters (if used comment in) *** 
target_frequency = 500 # target frequency after downsampling (Cometa)

# variance filter length 
n_var = 170 # calc length of variance filter (default 170, tested for 500 Hz sampling rate)

# filter settings
f_high = 20 # in Hz
f_low = 245 # in Hz

# *********************************************************************************
# ************************* Load and process EMG data  ****************************
# *********************************************************************************

# define the processing params
 
# create EMGData Object
EMG = EMGData(data_path = data_path, filename = file_str, f_samp = 500, channel_names = emg_ch_names)

#EMG.channelSelection(selected_channels, inverse)

#data, time = EMG.getEMGData()

EMG.applyBPFilter(f_high, f_low)

#data, time = EMG.getEMGData()

EMG.showEMGData()

