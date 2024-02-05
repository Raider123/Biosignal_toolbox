
# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************
import os 
import matplotlib.pyplot as plt 
import numpy as np

# project path settings 
#proj path 
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox"
data_path = os.path.join(proj_path, 'data') # path where the data lays 


# # own libs 
from biosignal_toolbox.emg_lib import EMGData


# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************


# Create an string with dataset file name (ANT and Cometa)
file_str = "/miniEMG_test/miniTest_6_1.txt"


# *********************************************************************************
# ************************* Load and process EMG data  ****************************
# *********************************************************************************

# define the processing params

# create EMGData Object
emg_data = EMGData(data_path = data_path, format = "cometa", filename = file_str)


# check cometa sample loss 
loss_arr = emg_data.calcSampleLossFromSameSamples()


plt.figure()
plt.plot(loss_arr[0, :])
plt.plot(loss_arr[1, :])
plt.plot(loss_arr[2, :])
plt.plot(loss_arr[3, :])
plt.legend(["1", "2", "3", "4"])


# show data 
emg_data.showEMGData()
