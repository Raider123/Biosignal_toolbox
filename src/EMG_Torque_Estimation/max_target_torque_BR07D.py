# This script calculates the maximum voluntary contraction of subject BR07D by reading all the EMG data and choosing the maximum value.

# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pathlib
import matplotlib.pyplot as plt 

# # own libs 
from biosignal_toolbox.eeg_lib import EEGData



# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************

# own libs
proj_path = "/home/dfki.uni-bremen.de/kschari/kc_ws/repos/biosignal_toolbox"

data_path = proj_path+"/data/m-rock_demo/"

#! Files for training
target_file_prefix = ["quali_data/FW28D/quali_torque_elbow_", "quali_data/FW28D/quali_torque_front_", "quali_data/FW28D/quali_torque_side_"]
# target_file_prefix = ["quali_data/quali_torque_elbow_"]

#! Read Qualisys data param
weights_d = ['0g']
mov_type_d=['complex', 'grasp']
set_num_d = ['1','2','3']

#! emg params
channel_names_i = ['BP1', 'BP2', 'BP3', 'BP4', 'BP5', 'BP6', 'BP7', 'BP8']

#! Variables to store the maximum value of each scenario/set/weight
max_value_arr_elbow = np.empty((0,1))
max_value_arr_front = np.empty((0,1))
max_value_arr_side = np.empty((0,1))
mvc_elbow = 0
mvc_front = 0
mvc_side = 0

# *********************************************************************************
# ***************** Load train, test, val sets for every iteration ****************
# *********************************************************************************
for wgt_idx in range(len(weights_d)):
    for typ_idx in range(len(mov_type_d)):
        for set_idx in range(len(set_num_d)):

            #! Check if the file exists
            if pathlib.Path(data_path + target_file_prefix[0] + weights_d[wgt_idx] + '_' + mov_type_d[typ_idx] + '_set' + set_num_d[set_idx] + '.npy').is_file():
                # print(f"Current filename: {target_file_prefix[0] + weights_d[wgt_idx] + '_' + mov_type_d[typ_idx] + '_' + set_num_d[set_idx] + '.npy'}")
                #! Loading and epoching for training   
                channel_names_t = ['right', 'left', 'marker']
                print("Creating Quali Elbow object!!")
                Quali_Data_Elbow = EEGData(format = "Recorded_LSL_stream", filenames = [target_file_prefix[0] + weights_d[0] + '_' + mov_type_d[typ_idx] + '_set' + set_num_d[set_idx]], data_path = data_path, f_samp=500, channel_names=channel_names_t, file_type='individual')
                print("Creating Quali Shoulder Front object!!")
                Quali_Data_Front = EEGData(format = "Recorded_LSL_stream", filenames = [target_file_prefix[1] + weights_d[0] + '_' + mov_type_d[typ_idx] + '_set' + set_num_d[set_idx]], data_path = data_path, f_samp=500, channel_names=channel_names_t, file_type='individual')
                print("Creating Quali Shoulder Side object!!")
                Quali_Data_Side = EEGData(format = "Recorded_LSL_stream", filenames = [target_file_prefix[2] + weights_d[0] + '_' + mov_type_d[typ_idx] + '_set' + set_num_d[set_idx]], data_path = data_path, f_samp=500, channel_names=channel_names_t, file_type='individual')

                #! Extract max values of each channel
                # flattened_arr = EMG_Data.filtered_data[:-1,:].flatten()
                max_value_arr_elbow = np.append(max_value_arr_elbow, np.max(np.abs(Quali_Data_Elbow.data[:,0])))
                max_value_arr_front = np.append(max_value_arr_front, np.max(np.abs(Quali_Data_Front.data[:,0])))
                max_value_arr_side = np.append(max_value_arr_side, np.max(np.abs(Quali_Data_Side.data[:,0])))
                print(f"Current appending array: {max_value_arr_elbow}")
                # print(f"Quali data shape: {Quali_Data_Elbow.data}")
                plt.plot(Quali_Data_Side.data[:,0])
                plt.show()
            else:
                continue

#! Choose the maximum value over all the datasets
mvc_elbow = np.max(max_value_arr_elbow)
print(f"Maximum absolute torque for right elbow: {mvc_elbow}")

mvc_front = np.max(max_value_arr_front)
print(f"Maximum absolute torque for right front: {mvc_front}")

mvc_side = np.max(max_value_arr_side)
print(f"Maximum absolute torque for right side: {mvc_side}")


