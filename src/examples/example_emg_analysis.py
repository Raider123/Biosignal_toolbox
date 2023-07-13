
# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import sys 
import matplotlib.pyplot as plt
import os 
from ...lib.emg_lib import EMGData


# project path settings 
current_path = os.path.dirname(os.path.abspath(__file__)) # project path 
project_path = os.path.split(os.path.split(current_path)[0])[0] # go up two folders to get the current path
data_path = os.path.join(project_path, 'data') # path where the data lays 
lib_path = os.path.join(project_path, 'lib') # path were the additional library is located 
sys.path.append(lib_path) # append own libs to path 


# own libs 
from emg_lib import EMGData
from timeseries_data import TimeseriesData


# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************


# Create an string with dataset file name (ANT and Cometa)
file_str = "channel_null.txt"


# *** channel selection params (comment in for channel selection) ***
#selected_channels =  []# ['\tR.Biceps Br.(uV)', '\tR.Ant.Deltoid(uV)', '\tR.Mid Delt.(uV)']
#inverse = True


# for ANT Systems 
#emg_ch_names = ["EMG1", "EMG2", "EMG3", "EMG4", "EMG5", "EMG6", "EMG7", "EMG8"]
#selected_channels = ["EMG2"]


# EMG sampling rate (Cometa default)
fsamp_emg = 2000 # in Hz

#fsamp_emg = 500 # for ANT

# *** processing parameters (if used comment in) *** 
#target_frequency = 500 # target frequency after downsampling (Cometa)

# variance filter length 
#n_var = 170 # calc length of variance filter (default 170, tested for 500 Hz sampling rate)

# *********************************************************************************
# ************************* Load EMG data  ****************************************
# *********************************************************************************

# loading emg data (Cometa)
file_str = os.path.join(data_path, file_str)
emg_data, emg_time_axis, emg_ch_names = TimeseriesData.loadCometaEMGData(file_str)
# (data, channels)

# load EMG data ANT
#emg_data, emg_time_axis = TimeseriesData.loadMiniANTEMGData(file_str, fsamp_emg)

# *********************************************************************************
# ************************* Process EMG data  *************************************
# *********************************************************************************

# *** Just comment in what should be used for processing *** 

# channel selection, if inverse = False all channels specified are kept 
#emg_data_select, emg_ch_names_select = EMGData.channelSelection(emg_data, emg_ch_names, selected_channels, inverse)


# apply bandpass filter 
emg_data_filtered = EMGData.applyBPFilterRectifying(fsamp_emg, 20, 200, emg_data)

# show all EMG channels
EMGData.showEMGData(emg_data, emg_time_axis, emg_ch_names)

# show selected EMG channels
#EMGData.showEMGData(emg_data_filtered, emg_time_axis, emg_ch_names_select)



# downsampling to target frequency 
#emg_data_down, time_axis_down = EMGData.decimateEMGData(emg_data_select, emg_time_axis, target_frequency, fsamp_emg)

#EMGData.showEMGData(emg_data_down, time_axis_down, emg_ch_names_select)

#apply variance filter to EMG signals  
#emg_data_filtered = EMGData.applyVarianceFilter(emg_data_down, n_var)

# show EMG signals in plots 
#EMGData.showEMGData(emg_data_filtered, time_axis_down, emg_ch_names_select)

