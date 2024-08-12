# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import tensorflow as tf
from time import perf_counter
import copy 
import datetime 

# # own libs 
from biosignal_toolbox.eeg_lib import EEGData
from biosignal_toolbox.ML_lib import MLModel
import biosignal_toolbox.ML_pipelines_lib as pipeline
# models 
from biosignal_toolbox.models.CNNnets import EEGNet
# own model 
from biosignal_toolbox.models.MlpErp import MLP_Model, MLP_Model_reduced

# import all user parameters for the evaluation
from userparams import * # --> parameters are stored and fully imported from this script 

print(tf.config.experimental.list_physical_devices('GPU'))

from mne.time_frequency import tfr_array_morlet
# disable GPU for testing
#tf.config.set_visible_devices([], 'GPU') # disable now 

from scipy.signal import lfilter_zi, lfilter


# *********************************************************************************
# ***************** Main processing and classification loop ***********************
# *********************************************************************************



# init performance results list
perf_results_total_MLP = []
perf_results_total_EEGNet = []

# measure execution time
time_start = perf_counter()

# init early stopping 
early_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=0,patience=early_stopping_patience,verbose=0,mode="auto",baseline=None,restore_best_weights=True)
# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=reduce_patients, min_lr=1e-6, verbose=1)

# *********************************************************************************
# ***************** Load train, test, val sets for every iteration ****************
# *********************************************************************************

# load ML model for preprocessing (one for all subjects )
filter_model_05_4Hz = MLModel(type = "keras") 
filter_model_05_4Hz.loadModel(filename=filter_model_name_05_4Hz, path=data_path)
filter_model_05_40Hz = MLModel(type = "keras") 
filter_model_05_40Hz.loadModel(filename=filter_model_name_05_40Hz, path=data_path)

# do this for all subjects: 

for current_condition_idx in range(0, len(train_test_conditions)): 
    
    # get the current train test conditions from the mapping (list of dict)
    train_file_list = train_test_conditions[current_condition_idx]["train"] 
    val_test_file_list = train_test_conditions[current_condition_idx]["test"] 

    subject = train_file_list[0].split("_")[2] # extract the subject name from the filename

    print(f"")
    print(f"current condition: {current_condition_idx}")
    print(f"subject: ", subject)
    print(f"train sets {train_file_list}")
    print(f"val_test set(s) {val_test_file_list}")
    print(f"")


    #  loading and epoching for training   
    EEG_data_train = EEGData(format = "Brainvision", filenames = train_file_list, data_path = data_path)
    #EEG_data_train_xDAWN = copy.deepcopy(EEG_data_train)
    EEG_data_val_test = EEGData(format = "Brainvision", filenames = val_test_file_list, data_path = data_path)

    

#     # Filter settings 
#     b, a = EEG_data_train.designFilter(f_low = 5.0, f_high = 0.3, order = 2, filter_type = "scipy_butter", return_type = "ba")
#     # zi1 = lfilter_zi(b, a)
#     # zi2 = lfilter_zi(b, a)

#     chunk_size = 500

#     # zi1_arr = zi1*EEG_data_train.data[:, 0]
#     # zi2_arr = zi2*EEG_data_train.data[:, 0]
#     # Initialize filter states for each channel
#     num_channels = EEG_data_train.data.shape[0]
#     zi1_arr = np.zeros((num_channels, len(b) - 1))
#     zi2_arr = np.zeros((num_channels, len(b) - 1))

#     for channel_idx in range(num_channels):
#         channel_data = EEG_data_train.data[channel_idx, :]

#         filtered_signal = []
#         zi1 = lfilter_zi(b, a) * channel_data[0]  # Initial filter state for the forward pass

#         # Process the data in chunks
#         for start in range(0, len(channel_data), chunk_size):
#             end = min(start + chunk_size, len(channel_data))
#             chunk = channel_data[start:end]
            
#             # Forward filter
#             y, zi1 = lfilter(b, a, chunk, zi=zi1)
            
#             # Reverse the signal
#             y_rev = y[::-1]
            
#             # Backward filter with reversed signal
#             y_rev_filtered = lfilter(b, a, y_rev)
            
#             # Reverse again to get the final filtered signal chunk
#             y_final = y_rev_filtered[::-1]
            
#             # Append the filtered chunk to the output signal
#             filtered_signal.extend(y_final)
        
#         # Convert the filtered signal to a NumPy array
#         filtered_signal = np.array(filtered_signal)
        
#         # Ensure the filtered signal has the same length as the original data
#         if len(filtered_signal) > len(channel_data):
#             filtered_signal = filtered_signal[:len(channel_data)]
#         elif len(filtered_signal) < len(channel_data):
#             filtered_signal = np.pad(filtered_signal, (0, len(channel_data) - len(filtered_signal)), 'constant')

#         # Update the EEG data with the filtered signal
#         EEG_data_train.data[channel_idx, :] = filtered_signal
#         EEG_data_train.raw_obj._data = EEG_data_train.data

# # ***************************************************************

#     num_channels = EEG_data_val_test.data.shape[0]
#     zi1_arr = np.zeros((num_channels, len(b) - 1))
#     zi2_arr = np.zeros((num_channels, len(b) - 1))

#     for channel_idx in range(num_channels):
#         channel_data = EEG_data_val_test.data[channel_idx, :]

#         filtered_signal = []
#         zi1 = lfilter_zi(b, a) * channel_data[0]  # Initial filter state for the forward pass

#         # Process the data in chunks
#         for start in range(0, len(channel_data), chunk_size):
#             end = min(start + chunk_size, len(channel_data))
#             chunk = channel_data[start:end]
            
#             # Forward filter
#             y, zi1 = lfilter(b, a, chunk, zi=zi1)
            
#             # Reverse the signal
#             y_rev = y[::-1]
            
#             # Backward filter with reversed signal
#             y_rev_filtered = lfilter(b, a, y_rev)
            
#             # Reverse again to get the final filtered signal chunk
#             y_final = y_rev_filtered[::-1]
            
#             # Append the filtered chunk to the output signal
#             filtered_signal.extend(y_final)

#         # Convert the filtered signal to a NumPy array
#         filtered_signal = np.array(filtered_signal)
        
#         # Ensure the filtered signal has the same length as the original data
#         if len(filtered_signal) > len(channel_data):
#             filtered_signal = filtered_signal[:len(channel_data)]
#         elif len(filtered_signal) < len(channel_data):
#             filtered_signal = np.pad(filtered_signal, (0, len(channel_data) - len(filtered_signal)), 'constant')

#         # Update the EEG data with the filtered signal
#         EEG_data_val_test.data[channel_idx, :] = filtered_signal
#         EEG_data_val_test.raw_obj._data = EEG_data_val_test.data
        

    #sos = EEG_data_train.designFilter(f_low = 10.0, f_high = None, order = 3, filter_type = "scipy_bessel", return_type = "sos")
    
    # # dc removal 
    # b = [1, -1]
    # a = [1, -0.9885]
    
    # # # #b, a = EEG_MLP.designFilter(f_low = 4.0, f_high = None, order = 21, filter_type = "fir_hann", return_type = "ba")
    # # b, a = EEG_data_train.designFilter(f_low = 4.0, f_high = None, order = 21, filter_type = "fir_kaiser", return_type = "ba", beta = 5.3)
    # # b1, a1 = EEG_data_train.designFilter(f_low = 4.0, f_high = None, order = 21, filter_type = "fir_kaiser", return_type = "ba", beta = 2.0)

    # # # online/offline Filtering
    # EEG_data_train.filterRawData(sos = sos, apply_method = "forward_sos_filter")
    # EEG_data_train.filterRawData(b= b, a = a, apply_method = "forward_ba_filter")
    # # EEG_data_train.filterRawData(b= b, a = a, apply_method = "zero_phase_ba", padtype = "even")
    # # EEG_data_train.filterRawData(b= b1, a = a1, apply_method = "zero_phase_ba", padtype = "even")

    # # print("mean of data", print(np.mean(EEG_data_train.data[0, :])))
    # EEG_data_val_test.filterRawData(sos = sos, apply_method = "forward_sos_filter")
    # EEG_data_val_test.filterRawData(b= b, a = a, apply_method = "forward_ba_filter")

    # # online/offline Filtering
    # EEG_data_val_test.filterRawData(sos = sos, apply_method = "zero_phase_sos", padtype = "even")
    # EEG_data_val_test.filterRawData(b= b, a = a, apply_method = "zero_phase_ba", padtype = "even")
    # EEG_data_val_test.filterRawData(b= b1, a = a1, apply_method = "zero_phase_ba", padtype = "even")
    
    
    # when using offline filters for preprocessing 
    if (use_offline_processing): 
        print(f"using offline preprocessing")
        # pass a copy to the offline preprocessing pipeline 
        EEG_data_train_05_4Hz = pipeline.offlinePreprocessingAndFiltering(copy.deepcopy(EEG_data_train), f_highpass = f_highpass, f_lowpass = f_lowpass_MLP, marker_number = marker_number, error_number = error_number, channel_list = channel_list, t1 = t1, t2 = t2, windows_selected = train_windows, window_size = window_size, window_step = window_step)
        EEG_data_train_05_40Hz = pipeline.offlinePreprocessingAndFiltering(copy.deepcopy(EEG_data_train), f_highpass = f_highpass, f_lowpass = f_lowpass_EEGNet, marker_number = marker_number, error_number = error_number, channel_list = channel_list, t1 = t1, t2 = t2, windows_selected = train_windows, window_size = window_size, window_step = window_step)
        EEG_data_val_05_4Hz, EEG_data_test_05_4Hz  = pipeline.offlinePreprocessingAndFiltering(copy.deepcopy(EEG_data_val_test), f_highpass = f_highpass, f_lowpass = f_lowpass_MLP, marker_number = marker_number, error_number = error_number, channel_list = channel_list, t1 = t1, t2 = t2, windows_selected = train_windows, window_size = window_size, window_step = window_step, split_train_test_epochs = True, n_epochs = n_test_trials)
        EEG_data_val_05_40Hz, EEG_data_test_05_40Hz = pipeline.offlinePreprocessingAndFiltering(copy.deepcopy(EEG_data_val_test), f_highpass = f_highpass, f_lowpass = f_lowpass_EEGNet, marker_number = marker_number, error_number = error_number, channel_list = channel_list, t1 = t1, t2 = t2, windows_selected = train_windows, window_size = window_size, window_step = window_step, split_train_test_epochs = True, n_epochs = n_test_trials)
        
        
    else: # no 
        print(f"using online preprocessing")
        # do rereferencing here since in real online case no epoching is done ! 
        # if(current_condition_idx >1): 
        #     EEG_data_train.rereferencingEpoching(marker_number, error_number, channel_list, inverse_keep_channel = inverse_keep_channel, t1 = t1, t2= t2)
        # else: 
        #     # do not do extra stuff currently 
        #    EEG_data_train.rereferencingEpoching(marker_number, error_number, channel_list = [], inverse_keep_channel = inverse_keep_channel, t1 = t1, t2= t2)
        EEG_data_train.rereferencingEpoching(marker_number, error_number, channel_list, inverse_keep_channel = inverse_keep_channel, t1 = t1, t2= t2)
        
            

        # freqs = np.arange(0.5, 4, 0.5) # at 2 Hz
        # tfr = tfr_array_morlet(EEG_data_train.epochs, sfreq = EEG_data_train.getSamplingRate(), freqs = freqs, n_cycles = 1)
        # phase_data = np.angle(tfr)
        #print(f"*** phase data", phase_data.shape)
        #print(phase_data)
        
        
        # using xDAWN 
        # EEG_data_train_05_4Hz_xDAWN = copy.deepcopy(EEG_data_train)
        # EEG_data_train_05_4Hz_xDAWN.rereferencingEpoching(marker_number, error_number, channel_list, inverse_keep_channel = inverse_keep_channel, t1 = -1.0, t2= t2, apply_filter = True, f_lowpass=f_lowpass_MLP, f_highpass = f_highpass)
        # xd_trained = EEG_data_train_05_4Hz_xDAWN.xDAWNSpatialfilter(n_components = n_xDAWN, markernumber = marker_number, processing_type="fit", return_filter = True)

        #EEG_data_val_test.rereferencingEpoching(marker_number, error_number, channel_list, inverse_keep_channel = inverse_keep_channel, t1 = t1, t2= t2)
        if(current_condition_idx >1): 
            EEG_data_val_test.rereferencingEpoching(marker_number, error_number, channel_list, inverse_keep_channel = inverse_keep_channel, t1 = t1, t2= t2)
        else: 
            EEG_data_val_test.rereferencingEpoching(marker_number, error_number, channel_list = [], inverse_keep_channel = inverse_keep_channel, t1 = t1, t2= t2)

        
        EEG_data_val, EEG_data_test = EEG_data_val_test.splitTrainTestEpochs(n_test_epochs=n_test_trials) # split in train and val_test 

        

        # do first stage preprocessing 
        EEG_data_train = pipeline.firstStageOnlinePreprocessing(EEG_data_train, window_size = window_size, window_step = window_step, windows_selected = train_windows) 
        EEG_data_val = pipeline.firstStageOnlinePreprocessing(EEG_data_val, window_size = window_size, window_step = window_step, windows_selected = train_windows) 
        EEG_data_test = pipeline.firstStageOnlinePreprocessing(EEG_data_test, window_size = window_size, window_step = window_step, windows_selected = train_windows) 


    # **********************************************************************************
    # ********************* MLP net processing and training  ***************************
    # **********************************************************************************
    
    # get features by running processing pipeline 
    
    if (use_offline_processing): 
        print(f"using Offline processing and filtering")
        x_train_MLP, y_train_MLP = pipeline.MLPProcessingOffline(copy.deepcopy(EEG_data_train_05_4Hz), copy.deepcopy(EEG_data_train_05_40Hz), window_labels_train, feature_indices_windows)
        x_val_MLP, y_val_MLP = pipeline.MLPProcessingOffline(copy.deepcopy(EEG_data_val_05_4Hz), copy.deepcopy(EEG_data_val_05_40Hz), window_labels_train, feature_indices_windows)
        x_test_MLP, y_test_MLP = pipeline.MLPProcessingOffline(copy.deepcopy(EEG_data_test_05_4Hz), copy.deepcopy(EEG_data_test_05_40Hz), window_labels_train, feature_indices_windows)
        
    else: 
        if(not use_net): 
            print(f"using Online processing with iir processing")
            x_train_MLP, y_train_MLP = pipeline.MLPProcessingOnline(copy.deepcopy(EEG_data_train), copy.deepcopy(EEG_data_train), window_labels_train, feature_indices_windows, None, xd_components= None)
            x_val_MLP, y_val_MLP = pipeline.MLPProcessingOnline(copy.deepcopy(EEG_data_val), copy.deepcopy(EEG_data_val), window_labels_train, feature_indices_windows, None, xd_components = None)
            x_test_MLP, y_test_MLP = pipeline.MLPProcessingOnline(copy.deepcopy(EEG_data_test), copy.deepcopy(EEG_data_test), window_labels_train, feature_indices_windows, None, xd_components = None)
        else: 
            print(f"using Online processing with filterNet processing")
            x_train_MLP, y_train_MLP = pipeline.MLPProcessingOnlineFilterNet(copy.deepcopy(EEG_data_train), copy.deepcopy(EEG_data_train), window_labels_train, feature_indices_windows, filter_model = filter_model_05_4Hz)
            x_val_MLP, y_val_MLP = pipeline.MLPProcessingOnlineFilterNet(copy.deepcopy(EEG_data_val), copy.deepcopy(EEG_data_val), window_labels_train, feature_indices_windows, filter_model = filter_model_05_4Hz)
            x_test_MLP, y_test_MLP = pipeline.MLPProcessingOnlineFilterNet(copy.deepcopy(EEG_data_test), copy.deepcopy(EEG_data_test), window_labels_train, feature_indices_windows, filter_model = filter_model_05_4Hz)


    # Load model with norm layer and train model  
    #MLP = MLP_Huge(x_train_MLP, use_norm_layer = use_norm_layer)
    MLP = MLP_Model_reduced(x_train_MLP, use_norm_layer = use_norm_layer)
    # from tensorflow.keras.utils import plot_model
    
    # # Plot and save the model architecture to a file
    # plot_model(MLP, to_file='MLP_architecture.png', show_shapes=True, show_layer_names=True)


    MLP_model = MLModel(model = MLP, type= "keras", model_summary = False)
    print(f"shape of train data {x_train_MLP.shape}")
    MLP_model.trainModel(save_trained_model = False, model_filename =data_path+subject+"_"+scenario_name+result_file_name+"_model_MLP_", train_epochs= n_epochs, batch_size=n_batch_size_MLP, class_weights=None, x_train=x_train_MLP, y_train= y_train_MLP, x_val = x_val_MLP, y_val = y_val_MLP, callbacks=[early_callback], loss_fcn=loss_fcn, optimizer=optimizer,metrics=metrics, show_train_results=show_train_results)


    # predict and get results 
    print("predict MLP net")
    MLP_model.predict(data = x_test_MLP, labels = y_test_MLP, encoding = "binary", show_results = True, show_pred_time = False, eval_type = "offline")
    perf_results_MLP = MLP_model.getPerfResults()


    # **********************************************************************************
    # ********************* EEGnet processing and training  ***************************
    # **********************************************************************************

    # # EEGNet processing pipeline 
    # if (use_offline_processing): 
    #     x_train_EEGNet, y_train_EEGNet = pipeline.EEGNetProcessingOffline(EEG_data_train_05_40Hz, window_labels_train, num_classes)
    #     x_val_EEGNet, y_val_EEGNet = pipeline.EEGNetProcessingOffline(EEG_data_val_05_40Hz, window_labels_train, num_classes)
    #     x_test_EEGNet, y_test_EEGNet = pipeline.EEGNetProcessingOffline(EEG_data_test_05_40Hz, window_labels_train, num_classes)
    # else: 
    #     if(not use_net): 
    #         x_train_EEGNet, y_train_EEGNet = pipeline.EEGNetProcessingOnline(copy.deepcopy(EEG_data_train), window_labels_train, num_classes)
    #         x_val_EEGNet, y_val_EEGNet = pipeline.EEGNetProcessingOnline(copy.deepcopy(EEG_data_val), window_labels_train, num_classes)
    #         x_test_EEGNet, y_test_EEGNet = pipeline.EEGNetProcessingOnline(copy.deepcopy(EEG_data_test), window_labels_train, num_classes)

    #     else: 
    #         #  use filterNet 
    #         x_train_EEGNet, y_train_EEGNet = pipeline.EEGNetProcessingOnlineFilterNet(copy.deepcopy(EEG_data_train), window_labels_train, num_classes, filter_model=filter_model_05_40Hz)
    #         x_val_EEGNet, y_val_EEGNet = pipeline.EEGNetProcessingOnlineFilterNet(copy.deepcopy(EEG_data_val), window_labels_train, num_classes, filter_model=filter_model_05_40Hz)
    #         x_test_EEGNet, y_test_EEGNet = pipeline.EEGNetProcessingOnlineFilterNet(copy.deepcopy(EEG_data_test), window_labels_train, num_classes, filter_model=filter_model_05_40Hz)


    # shape_input = x_train_EEGNet.shape # get train data shape for network 

    # print("EEGNet Input shape", shape_input)
    # model_EEGNet = EEGNet(nb_classes=num_classes,Chans=shape_input[1], Samples=shape_input[2], dropoutRate=dropout_EEGNet, kernLength=kern_length_EEGNET, F1=F1, D=D, F2=F2,dropoutType='Dropout', x_train = x_train_EEGNet, use_norm_layer = use_norm_layer)
    # EEGNet_model = MLModel(model = model_EEGNet, type="keras")
    # EEGNet_model.trainModel(save_trained_model = False, show_train_results=show_train_results, model_filename =data_path+subject+"_"+scenario_name+result_file_name+"_model_EEGNet", train_epochs= n_epochs, batch_size=n_batch_size_EEGNet, class_weights=None, x_train=x_train_EEGNet, y_train= y_train_EEGNet, x_val = x_val_EEGNet, y_val = y_val_EEGNet, callbacks=[early_callback], loss_fcn=loss_fcn, optimizer=optimizer,metrics=metrics)


    # # predict and get results 
    # print("predict EEGNet")
    # EEGNet_model.predict(data = x_test_EEGNet, labels = y_test_EEGNet, encoding = "onehotencoding", show_results = True, show_pred_time = False, eval_type = "offline")
    # perf_results_EEGNet = EEGNet_model.getPerfResults()


    # **********************************************************************************
    # ********************* Saving and storing performances  ***************************
    # **********************************************************************************
    
    # perf results 
    perf_results_total_MLP.append(perf_results_MLP[0:3])
    #perf_results_total_EEGNet.append(perf_results_EEGNet[0:3])


# Get the current date and time for saving the results 
current_datetime = datetime.datetime.now()
# Format the datetime as a string
datetime_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

# save res 
np.savetxt(results_path+result_file_name+"_MLP_"+datetime_string, perf_results_total_MLP, delimiter=",", fmt = "%1.8f", )
#np.savetxt(results_path+result_file_name+"_EEGNet_"+datetime_string, perf_results_total_EEGNet, delimiter=",", fmt = "%1.8f", )


print("all done")
print("")
print("execution time: ")
print(perf_counter()-time_start)


# save the userparams for the evaluation 
with open('userparams.py', 'r') as python_file:
    # Read the contents of the Python file
    python_code = python_file.read()

# Open a new text file in write mode
with open(results_path+"params_"+result_file_name+"_"+datetime_string+".txt", 'w') as text_file:
    # Write the Python code to the text file
    text_file.write(python_code)

