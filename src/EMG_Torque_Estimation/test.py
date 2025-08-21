from biosignal_toolbox.utils import loadConfig, convertDictToNamespace

# cfg_param = loadConfig(filename='emg_torque_estimation_mav.yaml')
cfg = convertDictToNamespace(loadConfig(filename='emg_torque_estimation_mav.yaml'))

print(cfg.data_param.subject_code)
