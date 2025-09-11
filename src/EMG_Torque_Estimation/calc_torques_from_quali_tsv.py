
#* This file calculates and saves elbow and shoulder torques from Qualisys motion data.

<<<<<<< HEAD
import itertools
import re
=======
import numpy as np
import matplotlib.pyplot as plt
import itertools
import re

from biosignal_toolbox.utils import getAbsolutePath, loadConfig, customWarningFormat
from biosignal_toolbox.motion_lib import MotionData

import warnings
warnings.formatwarning = customWarningFormat

#? load config file
config_filename = 'emg_torque_estimation_jte.yaml'
cfg = loadConfig(filename=config_filename)

for mov_idx, wgt_idx, set_idx in itertools.product(cfg.data_param.mov_type, 
                                                   cfg.data_param.weights, 
                                                   cfg.data_param.set_num):
    file_pattern = f"{cfg.filepath.quali_tsv_prefix}_{wgt_idx}_{mov_idx}_{set_idx}.tsv"
    matched_files = list(getAbsolutePath(cfg.filepath.data_path + cfg.filepath.quali_tsv_path).glob(file_pattern))
    if not matched_files:
        warnings.warn("No files match the pattern :(")
    for file in matched_files:
        #? create a quali motion data object
        qualisys_data = MotionData(data_path= cfg.filepath.data_path, 
                                   filename=cfg.filepath.quali_tsv_path+file.name)
        #? calculate the torques from quali .tsv
        qualisys_data.calculateTorque(body_weight_kg=cfg.data_param.subject_weight, 
                                      obj_weight_g=int(re.search(r"\d+", wgt_idx).group()), 
                                      subject_biological_sex=cfg.data_param.subject_bio_sex, 
                                      subject_hand_length_mm=cfg.data_param.subject_hand_len, 
                                      method="com")
        #? save the torques into individual .npy files
        qualisys_data.saveTorques_npy(save_torques=True, save_dir=cfg.filepath.quali_torques_save_path, joints_to_save=['all'])

