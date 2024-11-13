# This script is the general training script for joint torque estimation using sEMG signals offline. It loads all the hyper-parameters from a yaml_config file.

# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

# # own libs 
from biosignal_toolbox.eeg_lib import EEGData
from biosignal_toolbox.emg_lib import EMGData
from biosignal_toolbox.ML_lib import MLModel
from biosignal_toolbox.models.AANModel import AAN_Model
from biosignal_toolbox.utils import loadConfig, checkCreateDir, createReadme, plotResults

import warnings
warnings.filterwarnings('ignore')

# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************
#! load config file
config_filename = 'emg_torque_estimation_mav.yaml'
config_param = loadConfig(filename=config_filename)

#! init early stopping 
if config_param['model_param']['is_early_stop']:
    early_callback = tf.keras.callbacks.EarlyStopping(monitor=config_param['model_param']['monitor'], 
                                                      min_delta=config_param['model_param']['min_delta'], 
                                                      patience=config_param['model_param']['patience'], 
                                                      verbose=config_param['model_param']['verbose'], 
                                                      baseline=config_param['model_param']['baseline'], 
                                                      restore_best_weights=config_param['model_param']['restore_best_weights'], 
                                                      start_from_epoch=config_param['model_param']['start_from_epoch'])
else:
    early_callback = None

#! feature extraction window
if config_param['preprocess_param']['feature_select'] == "end":
    ## Indices to extract features from the end of the window
    feature_indices_windows_x = np.arange(config_param['preprocess_param']['window_size_x']-config_param['preprocess_param']['feature_size'], config_param['preprocess_param']['window_size_x'], step=1) 
    feature_indices_windows_y = np.arange(config_param['preprocess_param']['window_size_y']-1, config_param['preprocess_param']['window_size_y'], step=1)
elif config_param['preprocess_param']['feature_select'] == "mid":
    ## Indices to extract features from the middle of the window
    feature_indices_windows_x = np.arange(round(config_param['preprocess_param']['window_size_x']/2)-config_param['preprocess_param']['feature_size']/2, round(config_param['preprocess_param']['window_size_x']/2)+config_param['preprocess_param']['feature_size']/2, step=1)
    feature_indices_windows_y = np.arange(round(config_param['preprocess_param']['window_size_y']/2)-1, round(config_param['preprocess_param']['window_size_y']/2), step=1)
else:
    print("Please select a valid feature selection type!")

#! subject params 
if len(config_param['data_param']['mov_type']) >1:
    
    scenario_name = config_param['data_param']['mov_type'][0] + '_' + config_param['data_param']['mov_type'][1]
else:
    scenario_name = config_param['data_param']['mov_type'][0]

result_file_name = '_' + config_param['data_param']['weights'][0]

#! init performance results list
perf_results_total_MLP = []

#! Initialise arrays to append data
length_of_each_feature_window = int(len(config_param['preprocess_param']['channel_names_emg']))
x_train_combined = np.empty(shape=[0,length_of_each_feature_window])
y_e_train_combined = np.empty(shape=[0,2])
y_f_train_combined = np.empty(shape=[0,2])
y_s_train_combined = np.empty(shape=[0,2])

x_test_combined = np.empty(shape=[0,length_of_each_feature_window])
y_e_test_combined = np.empty(shape=[0,2])
y_f_test_combined = np.empty(shape=[0,2])
y_s_test_combined = np.empty(shape=[0,2])

#! Choose number of input neurons
if config_param['preprocess_param']['feature_type'] == "mav":
    neurons_inp = len(config_param['preprocess_param']['channel_names_emg'])
elif config_param['preprocess_param']['feature_type'] == "default":
    neurons_inp = len(config_param['preprocess_param']['channel_names_emg']) * config_param['preprocess_param']['feature_size']
else:
    print("ERROR!! Please enter the correct feature type in yaml file!")

# *********************************************************************************
# ***************** Load train, test, val sets for every iteration ****************
# *********************************************************************************
for typ_idx in range(len(config_param['data_param']['mov_type'])):
    for wgt_idx in range(len(config_param['data_param']['weights'])):
        for set_idx in range(len(config_param['data_param']['set_num'])):

            #! Check if the file exists
            if Path(config_param['filepath']['project_path']+config_param['filepath']['data_path']+config_param['filepath']['emg_path']+config_param['data_param']['weights'][wgt_idx]+'_'+config_param['data_param']['mov_type'][typ_idx]+'_'+config_param['data_param']['set_num'][set_idx]+'.txt').is_file():

                #! Loading and epoching for training   
                EMG_Data = EMGData(format="ANTmini", filenames=[config_param['filepath']['emg_path']+config_param['data_param']['weights'][wgt_idx]+'_'+config_param['data_param']['mov_type'][typ_idx]+'_'+config_param['data_param']['set_num'][set_idx]+'.txt'], data_path=config_param['filepath']['project_path']+config_param['filepath']['data_path'], f_samp=config_param['preprocess_param']['f_samp'], channel_names=config_param['preprocess_param']['channel_names_emg'])

                #! Plotting the raw EMG data
                if config_param['plot_param']['is_plot_raw']:
                    EMG_Data.plotEMG(data=EMG_Data.data[4,:], 
                                     unit="uV", 
                                     title="Raw EMG plot for Channel 5", 
                                     xlabel="Time in s", 
                                     ylabel="Voltage in uV", 
                                     is_grid_on=True)

                #! Loading the target values for the 3 joints
                print("Creating Quali Elbow object!!")
                Quali_Data_Elbow = EEGData(format="NumpyQualisys", 
                                           filenames=[config_param['filepath']['quali_path'][0]+config_param['data_param']['weights'][wgt_idx]+'_'+config_param['data_param']['mov_type'][typ_idx]+'_set'+config_param['data_param']['set_num'][set_idx]],
                                           data_path=config_param['filepath']['project_path']+config_param['filepath']['data_path'], 
                                           f_samp=config_param['preprocess_param']['f_samp'], 
                                           channel_names=config_param['preprocess_param']['channel_names_quali'], 
                                           file_type='individual', 
                                           add_marker_channel=True)

                print("Creating Quali Shoulder Front object!!")
                Quali_Data_Front = EEGData(format="NumpyQualisys", 
                                           filenames=[config_param['filepath']['quali_path'][1]+config_param['data_param']['weights'][wgt_idx]+'_'+config_param['data_param']['mov_type'][typ_idx]+'_set'+config_param['data_param']['set_num'][set_idx]],
                                           data_path=config_param['filepath']['project_path']+config_param['filepath']['data_path'], 
                                           f_samp=config_param['preprocess_param']['f_samp'], 
                                           channel_names=config_param['preprocess_param']['channel_names_quali'], 
                                           file_type='individual', 
                                           add_marker_channel=True)

                print("Creating Quali Shoulder Side object!!")
                Quali_Data_Side = EEGData(format="NumpyQualisys", 
                                          filenames=[config_param['filepath']['quali_path'][2]+config_param['data_param']['weights'][wgt_idx]+'_'+config_param['data_param']['mov_type'][typ_idx]+'_set'+config_param['data_param']['set_num'][set_idx]], 
                                          data_path =config_param['filepath']['project_path']+config_param['filepath']['data_path'], 
                                          f_samp=config_param['preprocess_param']['f_samp'], 
                                          channel_names=config_param['preprocess_param']['channel_names_quali'], 
                                          file_type='individual', 
                                          add_marker_channel=True)


                channel_names = EMG_Data.getChannelNames()
                print("Channel Names: ", channel_names)
                print("Channel Length: ", len(channel_names))
                print("")

                # print(Quali_Data_Elbow.data.shape)
                if config_param['plot_param']['is_plot_quali']:
                    plt.figure()
                    plt.plot(np.arange(0,Quali_Data_Elbow.data[0,:].shape[0], 1)/Quali_Data_Elbow.f_samp,Quali_Data_Elbow.data[1,:])
                    plt.title("Elbow Torque plot for right arm")
                    plt.xlabel("Time in s")
                    plt.ylabel("Torque in N-m")
                    plt.grid()
                    plt.show()

                # **********************************************************************************
                # ***************************** Preprocessing of data ******************************
                # **********************************************************************************

                #! High pass filter
                EMG_Data.highPassFilter(cutoff_freq=config_param['preprocess_param']['f_cutoff_hpf'], 
                                        order=2, 
                                        fs=config_param['preprocess_param']['f_samp'], 
                                        filter_type="butter")

                #! Plotting HP filtered data
                if config_param['plot_param']['is_plot_hpf']:
                    EMG_Data.plotEMG(data=EMG_Data.data[4,:], 
                                     unit="uV", 
                                     title="High-Pass Filtered and Rectified EMG plot for Channel 5", 
                                     xlabel="Time in s", 
                                     ylabel="Voltage in uV", 
                                     is_grid_on=True)

                #! Apply Variance Filter from variance_tools_api
                print("Applying Variance filter ...")
                width           = config_param['preprocess_param']['var_filter_width']
                ring_buffer     = np.zeros(width)
                index           = 0
                EMG_Data.applyVarianceFilterCPP(ring_buffer=ring_buffer, 
                                                width=width, 
                                                index=index)
                print("Variance Filter applied!!\n")

                #! Plot and print specific variance filtered windows 
                # var_filtered_window_x = EMG_Data.filtered_data
                # print(f"Shape of Variance filtered windows: {var_filtered_window_x.shape}")
                # print(f"Variance filtered windows: {var_filtered_window_x[4,:]}")

                if config_param['plot_param']['is_plot_var_filter']:
                    EMG_Data.plotEMG(data=EMG_Data.data[4,:], 
                                     unit="uV", 
                                     title="Variance Filtered EMG plot for Channel 5", 
                                     xlabel="Time in s", 
                                     ylabel="Voltage in uV", 
                                     is_grid_on=True)

                #! Normalisation
                print("Performing Normalization with Max Voluntary Contraction ...")
                EMG_Data.normalizeContinuousData(mvc=config_param['preprocess_param']['mvc'])
                print("Normalization with Max Voluntary Contraction performed !!\n")

                #! Low pass filter to smoothen the signal
                EMG_Data.lowPassFilter(cutoff_freq=config_param['preprocess_param']['f_cutoff_lpf'], 
                                       order=2, 
                                       fs=config_param['preprocess_param']['f_samp'], 
                                       filter_type="butter")

                #! Plot normalised and smoothened data
                if config_param['plot_param']['is_plot_norm']:
                    EMG_Data.plotEMG(data=EMG_Data.data[4,:], 
                                     unit="V", 
                                     title="Normalised and Smoothened EMG plot for Channel 5", 
                                     xlabel="Time in s", 
                                     ylabel="Voltage in V", 
                                     is_grid_on=True)

                #! Calculate Neural Activation Force
                print("Replacing sample with its force activation value ...")
                EMG_Data.calculateActivationForceFunction(d=config_param['preprocess_param']['act_delay'], 
                                                          b1=config_param['preprocess_param']['act_beta1'], 
                                                          b2=config_param['preprocess_param']['act_beta2'], 
                                                          g=config_param['preprocess_param']['act_gamma'], 
                                                          nonlinear_shape_factor=config_param['preprocess_param']['act_A'])
                print("Replaced each sample with its force activation value !!\n")

                #! Plot force activation data
                if config_param['plot_param']['is_plot_act']:
                    EMG_Data.plotEMG(data=EMG_Data.data[4,:], 
                                     unit="V", 
                                     title="Force Activated EMG plot for Channel 5", 
                                     xlabel="Time in s", 
                                     ylabel="Voltage in V", 
                                     is_grid_on=True)

                #! Windowing the filtered data
                _ = EMG_Data.windowContinuousData(startmarkernumber=1, 
                                                  stopmarkernumber=1, 
                                                  window_size=config_param['preprocess_param']['window_size_x'], 
                                                  window_step=config_param['preprocess_param']['window_step'], 
                                                  start_index_offset=0, 
                                                  start_channel_pick=0, 
                                                  end_channel_pick=8, 
                                                  return_window_end_indices=True)

                _ = Quali_Data_Elbow.windowContinuousData(startmarkernumber=1, 
                                                          stopmarkernumber=1, window_size=config_param['preprocess_param']['window_size_y'], 
                                                          window_step=config_param['preprocess_param']['window_step'], 
                                                          start_index_offset=0, 
                                                          start_channel_pick=0, 
                                                          end_channel_pick=3, 
                                                          return_window_end_indices=True)

                _ = Quali_Data_Front.windowContinuousData(startmarkernumber=1, 
                                                          stopmarkernumber=1, 
                                                          window_size=config_param['preprocess_param']['window_size_y'], 
                                                          window_step=config_param['preprocess_param']['window_step'], 
                                                          start_index_offset=0, 
                                                          start_channel_pick=0, 
                                                          end_channel_pick=3,
                                                          return_window_end_indices=True)

                _ = Quali_Data_Side.windowContinuousData(startmarkernumber=1, 
                                                         stopmarkernumber=1, 
                                                         window_size=config_param['preprocess_param']['window_size_y'], 
                                                         window_step=config_param['preprocess_param']['window_step'], 
                                                         start_index_offset=0, 
                                                         start_channel_pick=0, 
                                                         end_channel_pick=3,
                                                         return_window_end_indices=True)
                # use the EMG_Data.windows if you want to access the windowed data 

                #! Plot specific filtered windows for debugging
                if config_param['plot_param']['is_plot_filt_win']:
                    EMG_Data.plotEMG(data=EMG_Data.getWindows()[0,2,:,18], #[trl,chn,smpl,wnd]
                                     n_samples= EMG_Data.getWindows().shape[2],
                                     unit="V", 
                                     title="Force Activated EMG plot for Channel 5", 
                                     xlabel="Time in s", 
                                     ylabel="Voltage in V", 
                                     is_grid_on=True)
                    
                # **********************************************************************************
                # ******************************* Feature Extraction *******************************
                # **********************************************************************************

                #! time domain feature extraction
                print("Extracting features from windowed data ...")
                EMG_Data.featureExtractionFromWindows(feature_type="timepoints",
                                                      feature_indices_windows=feature_indices_windows_x)
                Quali_Data_Elbow.featureExtractionFromWindows(feature_type="timepoints", 
                                                              feature_indices_windows=feature_indices_windows_y)
                Quali_Data_Front.featureExtractionFromWindows(feature_type="timepoints", 
                                                              feature_indices_windows=feature_indices_windows_y)
                Quali_Data_Side.featureExtractionFromWindows(feature_type="timepoints", 
                                                             feature_indices_windows=feature_indices_windows_y)
                print("Feature extraction from windowed data completed !!\n")

                #! input features network 
                if config_param['preprocess_param']['feature_type'] == "mav":
                    x = EMG_Data.calculateMAVFromFeatures(len(channel_names))
                    neurons_inp = len(channel_names)
                elif config_param['preprocess_param']['feature_type'] == "default":
                    x = EMG_Data.getFeatures()
                    neurons_inp = len(channel_names) * config_param['preprocess_param']['feature_size']
                else:
                    print("ERROR!! Please enter the correct feature type in yaml file!")

                y_e = Quali_Data_Elbow.getFeatures()[:,0:2]
                y_f = Quali_Data_Front.getFeatures()[:,0:2]
                y_s = Quali_Data_Side.getFeatures()[:,0:2]

                #! Ensure same number of rows for imput and target features
                end_idx = x.shape[0] if x.shape[0] <= y_e.shape[0] else y_e.shape[0]
                x = x[0:end_idx,:]
                y_e = y_e[0:end_idx,0:2]
                y_f = y_f[0:end_idx,0:2]
                y_s = y_s[0:end_idx,0:2]
                print(f"Input Feature Dim: {x.shape}")
                print(f"Output Feature Dim: {y_e.shape}")
        
                #! Split data into train and test
                print("Splitting train and test data ...")
                #Loop over the data and split it
                for idx in range(x.shape[0]):
                    if idx <= round(config_param['model_param']['train_test_split']*x.shape[0]):
                        x_train_combined = np.vstack((x_train_combined, x[idx,:]))
                        y_e_train_combined = np.vstack((y_e_train_combined, y_e[idx,:]))
                        y_f_train_combined = np.vstack((y_f_train_combined, y_f[idx,:]))
                        y_s_train_combined = np.vstack((y_s_train_combined, y_s[idx,:]))
                    else:
                        x_test_combined = np.vstack((x_test_combined, x[idx,:]))
                        y_e_test_combined = np.vstack((y_e_test_combined, y_e[idx,:]))
                        y_f_test_combined = np.vstack((y_f_test_combined, y_f[idx,:]))
                        y_s_test_combined = np.vstack((y_s_test_combined, y_s[idx,:]))
                print("Train and test data generated !!\n")
            
            else:
                continue

# **********************************************************************************
# *************************** Train, load or test Model ****************************
# **********************************************************************************

#! Init model with norm layer
model_e = AAN_Model(neurons_inp=neurons_inp, 
                    act_inp=config_param['model_param']['act_inp'], 
                    neurons_h1=config_param['model_param']['neurons_h1'], 
                    act_h1=config_param['model_param']['act_h1'], 
                    neurons_h2=config_param['model_param']['neurons_h2'], 
                    act_h2=config_param['model_param']['act_h2'])
MLP_model_e = MLModel(model = model_e, type= "keras")

model_f = AAN_Model(neurons_inp=neurons_inp, 
                    act_inp=config_param['model_param']['act_inp'], 
                    neurons_h1=config_param['model_param']['neurons_h1'], 
                    act_h1=config_param['model_param']['act_h1'], 
                    neurons_h2=config_param['model_param']['neurons_h2'], 
                    act_h2=config_param['model_param']['act_h2'])
MLP_model_f = MLModel(model = model_f, type= "keras")

model_s = AAN_Model(neurons_inp=neurons_inp, 
                    act_inp=config_param['model_param']['act_inp'], 
                    neurons_h1=config_param['model_param']['neurons_h1'], 
                    act_h1=config_param['model_param']['act_h1'], 
                    neurons_h2=config_param['model_param']['neurons_h2'], 
                    act_h2=config_param['model_param']['act_h2'])
MLP_model_s = MLModel(model = model_s, type= "keras")

#! Train model
print("Training MLP model for elbow joint...") 
MLP_model_e.trainModel(save_trained_model=config_param['model_param']['is_save_model'], 
                       model_filename=config_param['filepath']['project_path']+config_param['filepath']['data_path']+"zdemo_ml_models/"+config_param['data_param']['subject_code']+"_"+scenario_name+result_file_name+"_AAN_model_elbow_mav", 
                       train_epochs=config_param['model_param']['n_epochs'], 
                       batch_size=config_param['model_param']['batch_size'], 
                       class_weights=None, 
                       x_train=x_train_combined, 
                       y_train=y_e_train_combined[:,0], 
                       validation_split=config_param['model_param']['validation_split'], 
                       loss_fcn=config_param['model_param']['loss_fcn'], 
                       optimizer=config_param['model_param']['optimizer'], 
                       metrics=config_param['model_param']['metrics'], 
                       show_train_results=False, 
                       callbacks=early_callback)
print("MLP training for elbow done !!\n")

print("Training MLP model for shoulder front joint...") 
MLP_model_f.trainModel(save_trained_model=config_param['model_param']['is_save_model'], 
                       model_filename=config_param['filepath']['project_path']+config_param['filepath']['data_path']+"zdemo_ml_models/"+config_param['data_param']['subject_code']+"_"+scenario_name+result_file_name+"_AAN_model_front_mav", 
                       train_epochs=config_param['model_param']['n_epochs'], 
                       batch_size=config_param['model_param']['batch_size'], 
                       class_weights=None, 
                       x_train=x_train_combined, 
                       y_train=y_f_train_combined[:,0], 
                       validation_split=config_param['model_param']['validation_split'], 
                       loss_fcn=config_param['model_param']['loss_fcn'], 
                       optimizer=config_param['model_param']['optimizer'],
                       metrics=config_param['model_param']['metrics'], 
                       show_train_results=False, 
                       callbacks=early_callback)
print("MLP training for front done !!\n")

print("Training MLP model for shoulder side joint...") 
MLP_model_s.trainModel(save_trained_model=config_param['model_param']['is_save_model'], 
                       model_filename=config_param['filepath']['project_path']+config_param['filepath']['data_path']+"zdemo_ml_models/"+config_param['data_param']['subject_code']+"_"+scenario_name+result_file_name+"_AAN_model_side_mav", 
                       train_epochs=config_param['model_param']['n_epochs'], 
                       batch_size=config_param['model_param']['batch_size'], 
                       class_weights=None, 
                       x_train=x_train_combined, 
                       y_train= y_s_train_combined[:,0], 
                       validation_split=config_param['model_param']['validation_split'], 
                       loss_fcn=config_param['model_param']['loss_fcn'], 
                       optimizer=config_param['model_param']['optimizer'],
                       metrics=config_param['model_param']['metrics'], 
                       show_train_results=False, 
                       callbacks=early_callback)
print("MLP training for side done !!\n")

#! Load saved model
# print("Loading saved MLP models ...")
# MLP_model_e.loadModel(filename=subject+"_"+scenario_name+result_file_name+"_AAN_model_elbow", path=data_path)
# MLP_model_f.loadModel(filename=subject+"_"+scenario_name+result_file_name+"_AAN_model_front", path=data_path)
# MLP_model_s.loadModel(filename=subject+"_"+scenario_name+result_file_name+"_AAN_model_side", path=data_path)
# print("Saved MLP models loaded !!\n")

#! Predict and get results 
print("Predicting joint torques ...")
MLP_model_e.predictTarget(data=x_test_combined, 
                          labels=y_e_test_combined[:,0], 
                          classification=False, 
                          show_results=False, 
                          show_pred_time=False, 
                          eval_type=config_param['post_train_param']['eval_type'])

MLP_model_f.predictTarget(data=x_test_combined, 
                          labels=y_f_test_combined[:,0], 
                          classification=False, 
                          show_results=False, 
                          show_pred_time=False, 
                          eval_type=config_param['post_train_param']['eval_type'])

MLP_model_s.predictTarget(data=x_test_combined, 
                          labels=y_s_test_combined[:,0], 
                          classification=False, 
                          show_results=False, 
                          show_pred_time=False, 
                          eval_type=config_param['post_train_param']['eval_type'])

perf_results_MLP_e = MLP_model_e.getPredictionScores()
perf_results_MLP_f = MLP_model_f.getPredictionScores()
perf_results_MLP_s = MLP_model_s.getPredictionScores()
# print(perf_results_MLP_e)

# **********************************************************************************
# **************************** Post Prediction Filtering ***************************
# **********************************************************************************
filtered_perf_results_MLP_e = np.zeros(perf_results_MLP_e.shape)
filtered_perf_results_MLP_f = np.zeros(perf_results_MLP_f.shape)
filtered_perf_results_MLP_s = np.zeros(perf_results_MLP_s.shape)

filter_window_size = config_param['post_train_param']['filter_size']

for idx in range(len(perf_results_MLP_e)):
    if idx < filter_window_size:
        filtered_perf_results_MLP_e[idx] = perf_results_MLP_e[idx]
        filtered_perf_results_MLP_f[idx] = perf_results_MLP_f[idx]
        filtered_perf_results_MLP_s[idx] = perf_results_MLP_s[idx]
    else:
        if config_param['post_train_param']['filter_type'] == 'mean':
            filtered_perf_results_MLP_e[idx] = np.mean(perf_results_MLP_e[idx-filter_window_size+1:idx])
            filtered_perf_results_MLP_f[idx] = np.mean(perf_results_MLP_f[idx-filter_window_size+1:idx])
            filtered_perf_results_MLP_s[idx] = np.mean(perf_results_MLP_s[idx-filter_window_size:idx])
        elif config_param['post_train_param']['filter_type'] == 'median':
            filtered_perf_results_MLP_e[idx] = np.median(perf_results_MLP_e[idx-filter_window_size+1:idx])
            filtered_perf_results_MLP_f[idx] = np.median(perf_results_MLP_f[idx-filter_window_size+1:idx])
            filtered_perf_results_MLP_s[idx] = np.median(perf_results_MLP_s[idx-filter_window_size:idx])
        else:
            print("ERROR!! Please enter the correct post training filter type!")

#! Calculate the RMSE values on the filtered predicted values
rmse_elbow  = MLModel.calculateRMSE(y_e_test_combined[:,0], filtered_perf_results_MLP_e)
rmse_front  = MLModel.calculateRMSE(y_f_test_combined[:,0], filtered_perf_results_MLP_f)
rmse_side   = MLModel.calculateRMSE(y_s_test_combined[:,0], filtered_perf_results_MLP_s)

#! Calculate the RMSE values on the filtered predicted values
rmse_elbow_raw  = MLModel.calculateRMSE(y_e_test_combined[:,0], perf_results_MLP_e)
rmse_front_raw  = MLModel.calculateRMSE(y_f_test_combined[:,0], perf_results_MLP_f)
rmse_side_raw   = MLModel.calculateRMSE(y_s_test_combined[:,0], perf_results_MLP_s)

#! Check if the save dir exists. If not create one
#! Create a readme.txt and include all parameters in it
if config_param['post_train_param']['is_save_plot']:
    dir_path = checkCreateDir(param_obj=config_param)
    createReadme(param_obj=config_param,
                  dir_path=dir_path)

#! Plotting the filtered prediction results
plotResults(data_ref=y_e_test_combined[:,0],
            label_ref="real torque", 
            data_out=filtered_perf_results_MLP_e, 
            label_out="predicted torque", 
            title=f"Elbow Joint Filtered; RMSE: {rmse_elbow} N-m", 
            ylabel="Torque in N-m", 
            is_grid_on=True)
if config_param['post_train_param']['is_save_plot']:
    plt.savefig(dir_path / Path(f"/test_elbow_{config_param['post_train_param']['filter_type']}.png"))

plotResults(data_ref=y_f_test_combined[:,0],
            label_ref="real torque", 
            data_out=filtered_perf_results_MLP_f, 
            label_out="predicted torque", 
            title=f"Shoulder Front Joint Filtered; RMSE: {rmse_front} N-m", 
            ylabel="Torque in N-m", 
            is_grid_on=True)
if config_param['post_train_param']['is_save_plot']:
    plt.savefig(dir_path / Path(f"/test_front_{config_param['post_train_param']['filter_type']}.png"))

plotResults(data_ref=y_s_test_combined[:,0],
            label_ref="real torque", 
            data_out=filtered_perf_results_MLP_s, 
            label_out="predicted torque", 
            title=f"Shoulder Side Joint Filtered; RMSE: {rmse_side} N-m", 
            ylabel="Torque in N-m", 
            is_grid_on=True)
if config_param['post_train_param']['is_save_plot']:
    plt.savefig(dir_path / Path(f"/test_side_{config_param['post_train_param']['filter_type']}.png"))


#! Plotting the raw prediction results
plotResults(data_ref=y_e_test_combined[:,0],
            label_ref="real torque", 
            data_out=perf_results_MLP_e, 
            label_out="predicted torque", 
            title=f"Elbow Joint; RMSE: {rmse_elbow_raw} N-m", 
            ylabel="Torque in N-m", 
            is_grid_on=True)
if config_param['post_train_param']['is_save_plot']:
    plt.savefig(dir_path / Path(f"/test_elbow_raw.png"))

plotResults(data_ref=y_f_test_combined[:,0],
            label_ref="real torque", 
            data_out=perf_results_MLP_f, 
            label_out="predicted torque", 
            title=f"Shoulder Front Joint; RMSE: {rmse_front_raw} N-m", 
            ylabel="Torque in N-m", 
            is_grid_on=True)
if config_param['post_train_param']['is_save_plot']:
    plt.savefig(dir_path / Path(f"/test_front_raw.png"))

plotResults(data_ref=y_s_test_combined[:,0],
            label_ref="real torque", 
            data_out=perf_results_MLP_s, 
            label_out="predicted torque", 
            title=f"Shoulder Side Joint; RMSE: {rmse_side_raw} N-m", 
            ylabel="Torque in N-m", 
            is_grid_on=True)
if config_param['post_train_param']['is_save_plot']:
    plt.savefig(dir_path / Path(f"/test_side_raw.png"))

#! Showing the plots
plt.show()


