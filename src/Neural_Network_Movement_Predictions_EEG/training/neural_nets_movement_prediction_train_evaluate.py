# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from time import perf_counter
import sys
import copy 

# # own libs 
from biosignal_toolbox.eeg_lib import EEGData
from biosignal_toolbox.ML_lib import MLModel


# own libs
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox"
#sys.path.append(proj_path+"/lib/biosignal_toolbox") # path to lib folder

# models 
from biosignal_toolbox.models.CNNnets import EEGNet

# own model 
from biosignal_toolbox.models.MlpErp import MLP_Model

print(tf.config.experimental.list_physical_devices('GPU'))

# disable GPU for testing
#tf.config.set_visible_devices([], 'GPU')


# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************

data_path = proj_path+"/data/"
results_path = proj_path+"/results/"
subject_names = ["JV43","RA12"]# "JV43", "AV82", "UP28", "XP01", "ZS27", "JD68", "QS70"] # specify which subjects data should be evaluated
interations = [0] # the evaluation numbers which train test permutations are used
scenario_name = "intentional_unilateral"
result_file_name = "fcn_network_results_34ch_EEGNet_denoise_test_1"
preprocessed_data_filename_end = "34ch_denoising"  #"34ch_raw_no_scale" # TODO: implement online filter and normalization  
#eval_name = "fcn_network_results_34ch_MLP_scalings_test"

# fine_tune = False
# pre_trained_model = "EEGNet_pretrained_b128"


f_samp_eeg = 500 #sample Frequency of eeg

#machine learning params

# model selection 
used_model = "EEGNet"
num_classes = 2


# fcn model parameter 
n_epochs = 300 #300 training epochs (max since early stopping is used)
n_batch_size = 16 # 16 for EEGNet, 64 for MLP
weight_no_lrp_class = 0.5 # weight for the both classes for training (loss function weighting, has to sum to 1 !)
weight_lrp_class = 0.5
early_stopping_patience = 50 # 100 


#EEGNet-parameter
kern_length_EEGNET = 50 # 50 before 
F1 = 16 # 8 
D = 2 
F2 = 32 # 16 
dropout_EEGNet = 0.5


# training params 
loss_fcn =  "binary_crossentropy" #tf.keras.losses.Hinge()
optimizer  = "adam" # Nadam for MLP 

# optimizer = tf.keras.optimizers.SGD( learning_rate=0.005, # choose slow learning rate 
#     momentum=0.0,
#     nesterov=False,
# )

metrics = "accuracy"

# training windows and features
train_windows = ["bis-2700", "bis-2500", "bis-2300" ,"bis-2100", "bis-1900", "bis-1700", "bis-1500", "bis-1300" ,"bis-1000", "bis-150", "bis-100", "bis-50", "bis0"] #["bis-2500", "bis-2050", "bis-2200", "bis-1800", "bis-150", "bis-100", "bis-50", "bis0"]
test_windows = ["bis-2500", "bis-2050", "bis-100", "bis0"] # ["bis-2500", "bis-2050", "bis-100", "bis0"]
window_labels_train = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0] # [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
window_labels_test = [0.0, 0.0, 1.0, 1.0] #  [0.0, 0.0, 1.0, 1.0]

used_trials_training = 80 # trials to use for training 

features = "fusion" # which features to be used for classification, "timepoints" or "meanfreqs" or "fusion" (combine both)
feature_indices_windows = np.arange(900, 1000, step = 1) # 900, 1000 numpy array with time feature indices, (950, 1000) means last 100 ms of a window are used 

# is using neighbour features from channels specify the neighbouring channels 
# neighbours_list = [("P6", "P4"), ("P4", "P2"), ("P2", "PZ"), ("PZ", "P1"), ("P3", "P5"), ("CP6", "CP4"), ("CP4", "CP2"), ("CP2", "CPZ"), ("CPZ", "CP1"), ("CP1", "CP3"), ("CP3", "CP5"), 
#                    ("C6", "C4"), ("C4", "C2"), ("C2", "CZ"), ("CZ", "C1"), ("C1", "C3"), ("CZ", "C1"), ("C3", "C5"), ("FC6", "C4"), ("FC4", "FC2"), ("FC2", "FC1"), ("FC1", "FC3"), 
#                    ("FC3", "FC5"), ("F6", "F4"), ("F4", "F2"), ("F2", "FZ"), ("F4", "F2"), ("F2", "FZ"), ("F1", "F3"), ("F3", "F5")]


# window wise metric evaluation
window_size = 1000 #windowsize in ms (analog to pySPACE evaluation)
window_step = 50 # stepsize in ms (analog to pySPACE evaluation)


# *********************************************************************************
# ***************** Main processing and classification loop ***********************
# *********************************************************************************


# init performance results list
perf_results_total = []
max_val_indices = []


# load the time axis of the epoched data
time_axis_eeg_batch = np.load(data_path+"time_axis_eeg_epochs.npy")

# measure execution time
time_start = perf_counter()


# init early stopping 
early_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=early_stopping_patience,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)


for subject in subject_names:
    for iteration in interations:
        
        # *********************************************************************************
        # ***************** Load train, test, val sets for every iteration ****************
        # *********************************************************************************
        
        # load each individual train, val and test sets (preprocessed)
        lrp_epochs_train_scaled = np.load(data_path+subject+"_"+scenario_name+preprocessed_data_filename_end+"_train_"+str(iteration)+".npy")
        lrp_epochs_train_scaled = lrp_epochs_train_scaled[0:used_trials_training,:, :] # limit the number of trials used for training 

        lrp_epochs_val_scaled = np.load(data_path+subject+"_"+scenario_name+preprocessed_data_filename_end+"_test_"+str(iteration)+".npy")
        channel_names = np.load(data_path+"remaining_eeg_channel_names"+".npy")


        # **********************************************************************************
        # ********************* Preprocessing for data of both networks ********************
        # **********************************************************************************

        EEG_train = EEGData(format = "NumpyEpochs", epochs = lrp_epochs_train_scaled, f_samp = f_samp_eeg, channel_names = list(channel_names))
        EEG_val = EEGData(format = "NumpyEpochs", epochs = lrp_epochs_val_scaled, f_samp = f_samp_eeg, channel_names = list(channel_names))

        # window EEG epochs 
        EEG_train.windowEEGEpochs(window_size, window_step)
        EEG_val.windowEEGEpochs(window_size, window_step)


        # window selection 
        EEG_train.windowSelection(train_windows)
        EEG_val.windowSelection(test_windows) 
        
        # split up obj for frequency features (other preprocessing)
        if(features == "fusion"): 
            EEG_train_freq = copy.deepcopy(EEG_train)
            EEG_val_freq = copy.deepcopy(EEG_val)

        
        # filtering windows 
        # bandpass iir 
        if(used_model == "MLP"): # different filtering for different nets 
            f_low = 5.0 
            f_high = 0.3 # 0.3 
        else: 
            f_low = 40.0 # check filter design again ! 
            f_high = 0.3 # 0.3 
        
        # bandpass filter data 
        EEG_train.FilterWindows(f_low = f_low, f_high = f_high, filter_type = "scipy_butter", order=2, show_response = False) 
        EEG_val.FilterWindows(f_low = f_low, f_high = f_high, filter_type = "scipy_butter", order=2, show_response = False) 


        # # specify the window labels 
        EEG_train.setWindowLabels(window_labels_train)
        EEG_val.setWindowLabels(window_labels_test)

        
        # **********************************************************************************
        # ********************* Model selection and training *******************************
        # **********************************************************************************
        
        # MLP Net training pipeline 
        if not (used_model =="EEGNet"): 
            
            # time domain features 
            EEG_train.featureExtractionFromWindows(feature_type = "timepoints", feature_indices_windows = feature_indices_windows)
            EEG_val.featureExtractionFromWindows(feature_type = "timepoints", feature_indices_windows = feature_indices_windows)

            # frequency domain features 
            if(features == "fusion"): 
                # EEG_train_freq.featureExtractionFromWindows(feature_type = "meanfreqs")
                # EEG_val_freq.featureExtractionFromWindows(feature_type = "meanfreqs") 

                EEG_train_freq.featureExtractionFromWindows(feature_type = "freqBandPower")
                EEG_val_freq.featureExtractionFromWindows(feature_type = "freqBandPower")
                

                x_train_freq = EEG_train_freq.getFeatures()
                x_val_freq = EEG_val_freq.getFeatures()

                # add the frequency features to the time domain features 
                EEG_train.addFeatures(x_train_freq)
                EEG_val.addFeatures(x_val_freq)


            # print feature shape 
            EEG_train.printFeatureShape()

            # get train data from EEG instances 
            x_train = EEG_train.getFeatures()
            y_train = EEG_train.getTrainLabels()
            x_val = EEG_val.getFeatures()
            y_val = EEG_val.getTrainLabels()


            # create MLP model 
            print("Use MLP model")

            # Load model with norm layer  
            MLP = MLP_Model(x_train, use_norm_layer = True)

            MLP_model = MLModel(model = MLP, train_epochs= n_epochs, batch_size=n_batch_size, class_weights=None, x_train=x_train, y_train= y_train, x_val = x_val, y_val = y_val, callbacks=[early_callback], loss_fcn=loss_fcn, optimizer=optimizer,metrics=metrics)
            MLP_model.trainModel(save_trained_model = True, model_filename =subject+"_"+scenario_name+result_file_name+"_model_"+str(iteration))
            
            # predict and get results 
            MLP_model.predict(data = x_val, labels = y_val, encoding = "binary", show_results = True)
            perf_results = MLP_model.getPerfResults()


        # **** EEGNet training pipeline **** 
        else: 
            
            # reshape the windows to fit to input shape of EEGNet 
            EEG_train.reshapeWindowsForCNNnets()
            EEG_val.reshapeWindowsForCNNnets()
        
            # EEGNet label conversion (to one hot encodings)
            EEG_train.labelsToCategorical(num_classes = num_classes)
            EEG_val.labelsToCategorical(num_classes = num_classes)


            # get train data from EEG instances 
            x_train = EEG_train.getWindows()
            y_train = EEG_train.getTrainLabels()
            x_val = EEG_val.getWindows()
            y_val = EEG_val.getTrainLabels()

            # model setup and training 

            # create EEGNet model for training 
            print("Use EEGNet")
            train_wind_shape = EEG_train.getWindows().shape # get train data shape for network 
            model_EEGNet = EEGNet(nb_classes=num_classes, Chans=train_wind_shape[1], Samples=train_wind_shape[2], dropoutRate=dropout_EEGNet, kernLength=kern_length_EEGNET, F1=F1, D=D, F2=F2,dropoutType='Dropout', x_train = x_train, use_norm_layer = True)
            EEGNet_model = MLModel(model = model_EEGNet, train_epochs= n_epochs, batch_size=n_batch_size, class_weights=None, x_train=x_train, y_train= y_train, x_val = x_val, y_val = y_val, callbacks=[early_callback], loss_fcn=loss_fcn, optimizer=optimizer,metrics=metrics)
            EEGNet_model.trainModel(save_trained_model = True, model_filename =subject+"_"+scenario_name+result_file_name+"_model_"+str(iteration))
            
            # predict and get results 
            EEGNet_model.predict(data = x_val, labels = y_val, encoding = "onehotencoding", show_results = True)
            perf_results = EEGNet_model.getPerfResults()
            
            
        perf_results_total.append(perf_results[0:3])


#save results
perf_results_total = np.array(perf_results_total)
np.savetxt(results_path+result_file_name, perf_results_total, delimiter=",", fmt = "%1.8f", )


print("all done")
print("")
print("execution time: ")
print(perf_counter()-time_start)



