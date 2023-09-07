# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from time import perf_counter
import sys
from tensorflow.keras.models import save_model, load_model
from keras_adabound import AdaBound

# own libs
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/mne_machine_learning"
sys.path.append(proj_path+"/lib") # path to lib folder
import eeg_lib

# models 
from EEGModels import EEGNet, ShallowConvNet

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
subject_names = ["AV82"] #["JV43"]#,"RA12", "JV43", "AV82", "UP28", "XP01", "ZS27", "JD68", "QS70"] # specify which subjects data should be evaluated
interations = [0] # the evaluation numbers which train test permutations are used
scenario_name = "intentional_unilateral"
result_file_name = "pretrained_EEGnet_for_AV82"
preprocessed_data_filename_end = "_34ch_05_40Hz_pool" 
preprocessed_data_filename_end_f = "_34ch_05_40Hz_pool" # _34ch_05_40Hz_pool_trainTune
eval_name = "pretrained_EEGnet_for_AV82"

f_samp_eeg = 500 #sample Frequency of eeg

#machine learning params

# model selection 
used_model = "EEGNet"
num_classes = 2


# fcn model parameter 
n_epochs = 300 #300 training epochs (max since early stopping is used)
n_batch_size = 16 # 16 for EEGNet, 64 for MLP
shuffle_data = True # shuffle all data for training, validation and testing
weight_no_lrp_class = 0.5 # weight for the both classes for training (loss function weighting, has to sum to 1 !)
weight_lrp_class = 0.5
show_training_results = True
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
optimizer = tf.keras.optimizers.SGD( learning_rate=0.005, # choose slow learning rate 
    momentum=0.0,
    nesterov=False,
)

# tf.keras.optimizers.SGD(
#     learning_rate=0.005,
#     momentum=0.0,
#     nesterov=False,
# )

metrics = "accuracy"


# training windows and features
train_windows = ["bis-2500", "bis-2050", "bis-2200", "bis-1800", "bis-150", "bis-100", "bis-50", "bis0"]
test_windows = ["bis-2500", "bis-2050", "bis-100", "bis0"]
window_labels_train = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
window_labels_test = [0.0, 0.0, 1.0, 1.0] #np.zeros((81)) 
#window_labels_test[76:] = 1.0 


feature_times_windows = (900, 1000) # (900, 1000) seems to work well# time inside the windows to be used as features (last 200 ms)
# feature types used for generating training data 
feature_type1 = "timepoints" 
feature_type2 = "meanfreqs"

features_used = "fusion" # which features to be used for classification  # fusion 

# window wise metric evaluation
window_size = 1000 #windowsize in ms (analog to pySPACE evaluation)
window_step = 50 # stepsize in ms (analog to pySPACE evaluation)

# specify params for metric evaluation with method "relabelling", otherwise the parameters are not relevant if use_relabelling = False
searching_bounds = [61, 81] # boundaries where the "label change point" is determined, values are the numbers of the windows (see wind_names param for which windows are selected as bounds)


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

        print(lrp_epochs_train_scaled.shape)


        # *********************************************************************************
        # ********************* Prepare data for networks **********************************
        # *********************************************************************************

    
        # windowing of the data 
        train_windows_EEG, num_of_windows, wind_names = eeg_lib.windowEEGEpochs(lrp_epochs_train_scaled, f_samp_eeg, window_size, window_step)

        #test_windows = wind_names

        
        # train and test window selection 
        train_windows_EEG_select = eeg_lib.windowSelection(train_windows_EEG, wind_names, train_windows)


        # set window labels 
        y_train = eeg_lib.setWindowLabels(train_windows_EEG_select, window_labels_train)


        if not(used_model == "EEGNet"):

            # feature extraction from windows
            # time domain 
            x_train_t = eeg_lib.featureExtractionFromWindows(train_windows_EEG_select, feature_type1, f_samp_eeg, feature_times_windows)


        # ***************************************************************************
        # ************************ Frequency preprocessing **************************
        # ***************************************************************************
        # for now use this with different processing steps for frequency features 


        # load each individual train, val and test sets (preprocessed)
        lrp_epochs_train_scaled_f = np.load(data_path+subject+"_"+scenario_name+preprocessed_data_filename_end_f+"_train_"+str(iteration)+".npy")


        # windowing of the data 
        train_windows_EEG_f, num_of_windows, wind_names = eeg_lib.windowEEGEpochs(lrp_epochs_train_scaled_f, f_samp_eeg, window_size, window_step)


        # train and test window selection 
        train_windows_EEG_select_f = eeg_lib.windowSelection(train_windows_EEG_f, wind_names, train_windows)


        #frequency domain 
        x_train_f = eeg_lib.featureExtractionFromWindows(train_windows_EEG_select_f, feature_type2, f_samp_eeg, None)


        if not(used_model == "EEGNet"): 

            # which train features should be used 
            if(features_used == "fusion"): 
                x_train = np.concatenate((x_train_t, x_train_f), axis = 1)

            elif(features_used == "frequency_domain"): 
                x_train = x_train_f

            else: 
                x_train = x_train_t

        
        # *********************************************************************************
        # ********************* Build ML model in keras  ***********************************
        # *********************************************************************************


        if not(used_model == "EEGNet"): 
            # MLP setup
            model = MLP_Model(x_train)

        else: 
            # preprocess the windows 
            x_train_EEG_net = eeg_lib.reshapeWindowsForCNNnets(train_windows_EEG_select_f)
            

            # # EEGNet setup 
            y_train = to_categorical(y_train, num_classes)


            # EEG Net 
            #model_EEGNet = EEGNet(nb_classes=num_classes, Chans=x_train_EEG_net.shape[1], Samples=x_train_EEG_net.shape[2], dropoutRate=dropout_EEGNet, dropoutType='Dropout')
            model_EEGNet = EEGNet(nb_classes=num_classes, Chans=x_train_EEG_net.shape[1], Samples=x_train_EEG_net.shape[2], dropoutRate=dropout_EEGNet, kernLength=kern_length_EEGNET, F1=F1, D=D, F2=F2,dropoutType='Dropout')


            
        # *********************************************************************************
        # ********************* Compile and train model  ***********************************
        # *********************************************************************************
        
        # early stopping callback
        #callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=early_stopping_patience, verbose=0, mode="auto", baseline=None, restore_best_weights=True)
        
        if not (used_model == "EEGNet"): 
            model.compile(loss=loss_fcn, optimizer=optimizer, metrics=metrics)
            history = model.fit(x_train,
                                y_train,
                                epochs  = n_epochs,
                                batch_size= n_batch_size,
                                shuffle = shuffle_data,
                                #workers=multiprocessing_cpus,
                                class_weight={0: weight_no_lrp_class, 1: weight_lrp_class},
                                #use_multiprocessing=True,
                                validation_split=0.2, 
                                callbacks = [early_callback])
            
        else: 
            model_EEGNet.compile(loss=loss_fcn, optimizer=optimizer, metrics=['accuracy']) 
            print("train shape:", x_train_EEG_net.shape)

            history = model_EEGNet.fit(x_train_EEG_net,
                                    y_train,
                                    epochs  = n_epochs,
                                    batch_size= n_batch_size,
                                    shuffle = shuffle_data,
                                    #workers=multiprocessing_cpus,
                                    class_weight={0: weight_no_lrp_class, 1: weight_lrp_class},
                                    #use_multiprocessing=True,
                                    validation_split=0.2, 
                                    callbacks = [early_callback])


#       
        #model.load_weights(results_path+'model_file.h5') # load best model weights 
        
        print("*********************************")
        print("Training basic model done")
        print("*********************************")

        
        # *********************************************************************************
        # ********************* Get and show training results   ***************************
        # *********************************************************************************

        # history of training process
        history_dict = history.history
        loss_values = history_dict["loss"]
        val_loss_values = history_dict["val_loss"]
        num_of_epochs = range(1, len(loss_values)+1)

        if(show_training_results):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize =(6, 8))
            fig.subplots_adjust(hspace = 0.3)
            ax1.plot(num_of_epochs, loss_values, "bo", label="Training loss")
            ax1.plot(num_of_epochs, val_loss_values, "b", label="Validation loss")
            ax1.set_xlabel("Epochs")
            ax1.set_ylabel("Loss")
            ax1.legend()
            ax1.set_title("Loss values over trained epochs")

            acc_values = history_dict["accuracy"]
            val_acc_values = history_dict["val_accuracy"]

            ax2.plot(num_of_epochs, acc_values, "bo", label="Training accuracy")
            ax2.plot(num_of_epochs, val_acc_values, "b", label="Validation accuracy")
            ax2.set_xlabel("Epochs")
            ax2.set_ylabel("Accuracy")
            ax2.legend()
            ax2.set_title("Accuracy over trained epochs")

            plt.show()

        val_acc_values = history_dict["val_accuracy"]
        #max_val_idx = np.argmax(val_acc_values)
        #max_val_indices.append(max_val_idx)


        # *********************************************************************************
        # *********************Single trial predictions   *********************************
        # *********************************************************************************

        
        if not (used_model =="EEGNet"): 

            save_model(model, subject+"_MLP_"+eval_name+str(iteration)+".h5") # save 
            print("saved MLP net model")

        else: 
            
            # save model 
            #model_EEGNet.save(subject+"_EEGNet_"+eval_name, overwrite=False, save_format="keras")
            save_model(model_EEGNet, subject+"_EEGNet_"+eval_name+".h5") # save 

            print("saved EEGNet model")
            

       
print("all done")
print("")
print("execution time: ")
print(perf_counter()-time_start)


#show model
print(model_EEGNet.summary())

