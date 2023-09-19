# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from time import perf_counter
import sys

# # own libs 
from biosignal_toolbox.eeg_lib import EEGData
from biosignal_toolbox.ML_lib import MLModel


# own libs
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/mne_machine_learning"
sys.path.append(proj_path+"/lib/biosignal_toolbox") # path to lib folder

# models 
from EEGModels import EEGNet

# own model 
from MlpErp import MLP_Model

print(tf.config.experimental.list_physical_devices('GPU'))

# disable GPU for testing
#tf.config.set_visible_devices([], 'GPU')


# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************
data_path = proj_path+"/data/"
results_path = proj_path+"/results/"
subject_names = ["JV43","RA12", "JV43", "AV82", "UP28", "XP01", "ZS27", "JD68", "QS70"] # specify which subjects data should be evaluated
interations = [0, 1, 2] # the evaluation numbers which train test permutations are used
scenario_name = "intentional_unilateral"
result_file_name = "fcn_network_results_34ch_MLP_offline"
preprocessed_data_filename_end = "34ch_05_4Hz_1" # TODO: implement bp filter on windows for using raw data with MLP net 
eval_name = "fcn_network_results_34ch_MLP_offline"

fine_tune = False
pre_trained_model = "EEGNet_pretrained_b128"

f_samp_eeg = 500 #sample Frequency of eeg

#machine learning params

# model selection 
used_model = "MLP"
num_classes = 2

# fcn model parameter 
n_epochs = 300 #300 training epochs (max since early stopping is used)
n_batch_size = 64 # 16 for EEGNet, 64 for MLP
weight_no_lrp_class = 0.5 # weight for the both classes for training (loss function weighting, has to sum to 1 !)
weight_lrp_class = 0.5
show_training_results = False
#multiprocessing_cpus = 14
early_stopping_patience = 100 # 100 


#EEGNet-parameter
kern_length_EEGNET = 50 # 100 before 
F1 = 16 # 8 
D = 2 
F2 = 32 # 16 
dropout_EEGNet = 0.5


# training params 
loss_fcn =  "binary_crossentropy" #tf.keras.losses.Hinge()
optimizer  = "Nadam" 

# tf.keras.optimizers.SGD( learning_rate=0.005, # choose slow learning rate 
#     momentum=0.0,
#     nesterov=False,
# )

metrics = "accuracy"

# training windows and features
train_windows = ["bis-2500", "bis-2050", "bis-2200", "bis-1800", "bis-150", "bis-100", "bis-50", "bis0"]
test_windows = ["bis-2500", "bis-2050", "bis-100", "bis0"]
window_labels_train = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
window_labels_test = [0.0, 0.0, 1.0, 1.0] #np.zeros((81)) 

used_trials_training = 80 # trials to use for training 

features = "timepoints" # which features to be used for classification, "timepoints" or "meanfreqs" or "fusion" (combine both)
feature_times_windows = (900, 1000) # (900, 1000) seems to work well, time inside the windows to be used as features (last 100 ms)

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
     

        # **********************************************************************************
        # ********************* Preprocessing for data of both networks ********************
        # **********************************************************************************

        EEG_train = EEGData(format = "NumpyEpochs", epochs = lrp_epochs_train_scaled, f_samp = f_samp_eeg)
        EEG_val = EEGData(format = "NumpyEpochs", epochs = lrp_epochs_val_scaled, f_samp = f_samp_eeg)

        # window EEG epochs 
        EEG_train.windowEEGEpochs(window_size, window_step)
        EEG_val.windowEEGEpochs(window_size, window_step)

        # window selection 
        EEG_train.windowSelection(train_windows)
        EEG_val.windowSelection(test_windows)

        # specify the window labels 
        EEG_train.setWindowLabels(window_labels_train)
        EEG_val.setWindowLabels(window_labels_test)


        # **********************************************************************************
        # ********************* Model selection and training *******************************
        # **********************************************************************************

        # MLP Net training pipeline 
        if not (used_model =="EEGNet"): 

            EEG_train.featureExtractionFromWindows(feature_type = features, feature_times_windows = feature_times_windows, apply_lp_filter = False, f_lowpass = 4)
            EEG_val.featureExtractionFromWindows(feature_type = features, feature_times_windows = feature_times_windows, apply_lp_filter = False, f_lowpass = 4)
            #EEG_test.featureExtractionFromWindows(feature_type = features, feature_times_windows = feature_times_windows)

            # get train data from EEG instances 
            x_train = EEG_train.getFeatures()
            y_train = EEG_train.getTrainLabels()
            x_val = EEG_val.getFeatures()
            y_val = EEG_val.getTrainLabels()

            print(x_train.shape)
            print(y_train.shape)

            # create MLP model 
            print("Use MLP model")
            MLP = MLP_Model(x_train)

            MLP_model = MLModel(model = MLP, train_epochs= n_epochs, batch_size=n_batch_size, class_weights={0: weight_no_lrp_class, 1: weight_lrp_class}, x_train=x_train, y_train= y_train, x_val = x_val, y_val = y_val, callbacks=[early_callback], loss_fcn=loss_fcn, optimizer=optimizer,metrics=metrics)
            MLP_model.trainModel()
            
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
            x_train = EEG_train.getTrainWindows()
            y_train = EEG_train.getTrainLabels()
            x_val = EEG_val.getTrainWindows()
            y_val = EEG_val.getTrainLabels()

            # model setup and training 

            # create EEGNet model for training 
            print("Use EEGNet")
            train_wind_shape = EEG_train.getTrainWindows().shape # get train data shape for network 
            model_EEGNet = EEGNet(nb_classes=num_classes, Chans=train_wind_shape[1], Samples=train_wind_shape[2], dropoutRate=dropout_EEGNet, kernLength=kern_length_EEGNET, F1=F1, D=D, F2=F2,dropoutType='Dropout')
            EEGNet_model = MLModel(model = model_EEGNet, train_epochs= n_epochs, batch_size=n_batch_size, class_weights={0: weight_no_lrp_class, 1: weight_lrp_class}, x_train=x_train, y_train= y_train, x_val = x_val, y_val = y_val, callbacks=[early_callback], loss_fcn=loss_fcn, optimizer=optimizer,metrics=metrics)
            EEGNet_model.trainModel()
            
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



