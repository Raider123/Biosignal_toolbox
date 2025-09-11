# This script calculates the maximum voluntary contraction by reading all the EMG data and choosing the maximum value.

#! ************************************************
#! Imports
#! ************************************************

import numpy as np
import warnings
import itertools

# # own libs 
from biosignal_toolbox.emg_lib import EMGData
from biosignal_toolbox.utils import customWarningFormat, loadConfig, getAbsolutePath

warnings.formatwarning = customWarningFormat

#! ************************************************
#! User Parameters and Data Collection
#! ************************************************

#? load config file
config_filename = 'emg_torque_estimation_jte.yaml'
cfg = loadConfig(filename=config_filename)

#? Variables to store the maximum value of each scenario/set/weight
data_appended = []
max_value_arr = []
mvc = 0

#! ************************************************
#! Calculate MVC for each channel
#! ************************************************

for mov_idx, wgt_idx, set_idx in itertools.product(cfg.data_param.mov_type, cfg.data_param.weights, cfg.data_param.set_num):
            
    file_pattern = f"{cfg.filepath.emg_file_prefix}_{wgt_idx}_{mov_idx}_{set_idx}.txt"
    #? Check if the file exists
    matched_files = list(getAbsolutePath(cfg.filepath.data_path + cfg.filepath.emg_path).glob(file_pattern))
    if not matched_files:
        continue
    for file in matched_files:
        #? Loading and epoching for training   
        EMG_Data = EMGData(format="ANTmini", filenames=[cfg.filepath.emg_path + file.name], data_path=cfg.filepath.data_path, f_samp=cfg.preprocess_param.f_samp, channel_names=cfg.preprocess_param.channel_names_emg)

        #! ************************************************
        #! Data Pre-processing
        #! ************************************************

        #? High pass filter
        #* design the highpass filter
        sos_hp = EMG_Data.designFilter(f_high=cfg.preprocess_param.f_cutoff_hpf,
                                        f_low=cfg.preprocess_param.f_cutoff_lpf,
                                        order=cfg.preprocess_param.filter_order,
                                        filter_type="scipy_butter",
                                        return_type="sos")
        #* apply hpf filter 
        EMG_Data.filterData_offline(filter_method=cfg.preprocess_param.filter_method, 
                                    sos=sos_hp)
        #? Plotting HP filtered data
        if cfg.plot_param.is_plot_hpf:
            EMG_Data.plotEMG(data=EMG_Data.data[4,:], 
                                unit="uV", 
                                title="High-Pass Filtered plot for Channel 5", 
                                xlabel="Time in s", 
                                ylabel="Voltage in uV", 
                                is_grid_on=True)
            
        #? Apply Variance Filter from variance_tools_api
        print("Applying Variance filter ...")
        width           = cfg.preprocess_param.var_filter_width
        ring_buffer     = np.zeros(width)
        index           = 0
        EMG_Data.applyVarianceFilter_data(ring_buffer=ring_buffer, 
                                        width=width, 
                                        index=index)
        print("Variance Filter applied!!\n")

        if cfg.plot_param.is_plot_var_filter:
            EMG_Data.plotEMG(data=EMG_Data.data[4,:], 
                                unit="uV", 
                                title="Variance Filtered EMG plot for Channel 5", 
                                xlabel="Time in s", 
                                ylabel="Voltage in uV", 
                                is_grid_on=True)
        
        # Extract max values of each channel
        data_appended.append(EMG_Data.data[:-1,:])

data_appended = np.concatenate(data_appended, axis=1)
print(f"Concatenated array shape: {data_appended.shape}")

max_value_arr = np.max(data_appended, axis=1)
print(f"MVC channel-wise: {max_value_arr}")

mvc = np.max(max_value_arr)
print(f"MVC overall: {mvc}")