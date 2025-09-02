# This script calculates the maximum voluntary contraction of subject BR07D by reading all the EMG data and choosing the maximum value.

# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pathlib

# # own libs 
from biosignal_toolbox.emg_lib import EMGData

# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************

# own libs
proj_path = "F:/SMT_MASTERPROJEKT/biosignal_toolbox/"

data_path = proj_path+"/data/m-rock_demo/"
#data_path = proj_path+"/data/JTE/"

#! Files for training
train_file_prefix = "emg/24092024_FW28D_"
#train_file_prefix = "emg/BU62D/24072025_BU62D_"

#! Read Qualisys data param
weights_d = ['0g']
mov_type_d=['complex', 'grasp']
set_num_d = ['1','2','3']

#! emg params
channel_names_i = ['BP1', 'BP2', 'BP3', 'BP4', 'BP5', 'BP6', 'BP7', 'BP8']

#! Variables to store the maximum value of each scenario/set/weight
max_value_arr = np.empty((0,1))
mvc = 0

# *********************************************************************************
# ***************** Load train, test, val sets for every iteration ****************
# *********************************************************************************
for wgt_idx in range(len(weights_d)):
    for typ_idx in range(len(mov_type_d)):
        for set_idx in range(len(set_num_d)):
            
            #! Check if the file exists
            if pathlib.Path(data_path + train_file_prefix + weights_d[wgt_idx] + '_' + mov_type_d[typ_idx] + '_' + set_num_d[set_idx] + '.txt').is_file():

                EMG_Data = EMGData(format = "ANTmini", filenames = [train_file_prefix + weights_d[wgt_idx] + '_' + mov_type_d[typ_idx] + '_' + set_num_d[set_idx] + '.txt'], data_path = data_path, f_samp=500, channel_names=channel_names_i)

                #! High pass filter 25 Hz
                EMG_Data.highPassFilter(cutoff_freq=15, order=2, fs=500, filter_type="butter")

                #! Apply Variance Filter from variance_tools_api
                print("Applying Variance filter ...")
                ring_buffer     = np.zeros(20)
                width           = 20
                index           = 0
                EMG_Data.applyVarianceFilterCPP(ring_buffer=ring_buffer, width=width, index=index)
                print("Variance Filter applied!!\n")

                #! Extract max values of each channel
                flattened_arr = EMG_Data.data[:-1,:].flatten()
                max_value_arr = np.append(max_value_arr, np.max(flattened_arr))
                print(f"Current appending array: {max_value_arr}")
            else:
                continue

#! Choose the maximum value over all the datasets
#mvc = np.max(max_value_arr)
#print(f"Maximum Voluntary Contraction over all the channels and datasets: {mvc}")

import matplotlib.pyplot as plt
import numpy as np

# Deine Werte
werte = np.array(max_value_arr)

# x-Achse als Indizes
x = np.arange(len(werte))

# Plotten
plt.plot(x, werte, marker="o")
plt.title("Plot deiner Werte")
plt.xlabel("Index")
plt.ylabel("Wert")
plt.grid(True)
plt.show()


