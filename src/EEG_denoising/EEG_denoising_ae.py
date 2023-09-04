# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow as tf
import sys
from tensorflow.keras.datasets import mnist
import matplotlib 


# own libs
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/mne_machine_learning"
sys.path.append(proj_path+"/lib") # path to lib folder
import eeg_lib

# disable GPU for testing
#tf.config.set_visible_devices([], 'GPU')

# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************
data_path = proj_path+"/data/"
results_path = proj_path+"/results/"
subject_names = ["JV43", "RA12", "JV43", "AV82", "UP28", "XP01", "ZS27", "JD68", "QS70"] # specify which subjects data should be evaluated
interations = [0, 1, 2] # the evaluation numbers which train test permutations are used
scenario_name = "intentional_unilateral"
preprocessed_data_filename_end = "32ch_05_4Hz"

train_windows = ["bis-2500", "bis-2050", "bis-2200", "bis-100", "bis-50", "bis0"]


f_samp_eeg = 500 #sample Frequency of eeg

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
ch_names = np.load(data_path+"remaining_eeg_channel_names.npy")

print( ch_names)
print(list(ch_names).index("C1"))
# params for now 
iteration = interations[0]
subject = subject_names[1]

# load each individual train, val and test sets (preprocessed)
lrp_epochs_train_scaled = np.load(data_path+subject+"_"+scenario_name+preprocessed_data_filename_end+"_train_"+str(iteration)+".npy")+5
lrp_epochs_val_scaled = np.load(data_path+subject+"_"+scenario_name+preprocessed_data_filename_end+"_test_"+str(iteration)+".npy")+5
lrp_epochs_test_scaled = np.load(data_path+subject+"_"+scenario_name+preprocessed_data_filename_end+"_val_"+str(iteration)+".npy")+5


avg_epoch_train = np.mean(lrp_epochs_train_scaled, axis = 0)


#create target windows 
target_lrp_epochs_train_scaled = np.zeros(lrp_epochs_train_scaled.shape)
for trial_idx in range(0, target_lrp_epochs_train_scaled.shape[0]): 
    target_lrp_epochs_train_scaled[trial_idx, :, :] = avg_epoch_train # just use the average as target for every single trial 

target_lrp_epochs_val_scaled = np.zeros(lrp_epochs_val_scaled.shape)
for trial_idx in range(0, target_lrp_epochs_val_scaled.shape[0]): 
    target_lrp_epochs_val_scaled[trial_idx, :, :] = avg_epoch_train # is this correct ? # just use the average as target for every single trial 

target_lrp_epochs_test_scaled = np.zeros(lrp_epochs_test_scaled.shape)
for trial_idx in range(0, target_lrp_epochs_test_scaled.shape[0]): 
    target_lrp_epochs_test_scaled[trial_idx, :, :] = avg_epoch_train # is this correct ? # just use the average as target for every single trial 


# preprocessing 
train_windows_EEG, num_of_windows, wind_names = eeg_lib.windowEEGEpochs(lrp_epochs_train_scaled, f_samp_eeg, window_size, window_step)
val_windows_EEG, num_of_windows, wind_names = eeg_lib.windowEEGEpochs(lrp_epochs_val_scaled, f_samp_eeg, window_size, window_step)
test_windows_EEG, num_of_windows, wind_names = eeg_lib.windowEEGEpochs(lrp_epochs_test_scaled, f_samp_eeg, window_size, window_step)

train_windows_EEG = eeg_lib.windowSelection(train_windows_EEG, wind_names, train_windows)
val_windows_EEG = eeg_lib.windowSelection(val_windows_EEG, wind_names, train_windows)
test_windows_EEG = eeg_lib.windowSelection(test_windows_EEG, wind_names, train_windows)

#targets 
target_train_windows_EEG, num_of_windows, wind_names = eeg_lib.windowEEGEpochs(target_lrp_epochs_train_scaled, f_samp_eeg, window_size, window_step)
target_val_windows_EEG, num_of_windows, wind_names = eeg_lib.windowEEGEpochs(target_lrp_epochs_val_scaled, f_samp_eeg, window_size, window_step)
target_test_windows_EEG, num_of_windows, wind_names = eeg_lib.windowEEGEpochs(target_lrp_epochs_test_scaled, f_samp_eeg, window_size, window_step)

target_train_windows_EEG = eeg_lib.windowSelection(target_train_windows_EEG, wind_names, train_windows)
target_val_windows_EEG = eeg_lib.windowSelection(target_val_windows_EEG, wind_names, train_windows)
target_test_windows_EEG = eeg_lib.windowSelection(target_test_windows_EEG, wind_names, train_windows)

# reshape to train input 
x_train = np.reshape(train_windows_EEG, (train_windows_EEG.shape[0]*train_windows_EEG.shape[3], train_windows_EEG.shape[1], train_windows_EEG.shape[2],1))
x_val = np.reshape(val_windows_EEG, (val_windows_EEG.shape[0]*val_windows_EEG.shape[3], val_windows_EEG.shape[1], val_windows_EEG.shape[2],1))
x_test = np.reshape(test_windows_EEG, (test_windows_EEG.shape[0]*test_windows_EEG.shape[3], test_windows_EEG.shape[1], test_windows_EEG.shape[2], 1))

x_train_target = np.reshape(target_train_windows_EEG, (target_train_windows_EEG.shape[0]*target_train_windows_EEG.shape[3], target_train_windows_EEG.shape[1], target_train_windows_EEG.shape[2],1))
x_val_target = np.reshape(target_val_windows_EEG, (target_val_windows_EEG.shape[0]*target_val_windows_EEG.shape[3], target_val_windows_EEG.shape[1], target_val_windows_EEG.shape[2],1))
x_test_target = np.reshape(target_test_windows_EEG, (target_test_windows_EEG.shape[0]*target_test_windows_EEG.shape[3], target_test_windows_EEG.shape[1], target_test_windows_EEG.shape[2],1))

print(x_train.shape)
print(x_train_target.shape)
print(x_val.shape)
print(x_val_target.shape)

# #generate model 
input_wind = layers.Input(shape=(32, 500, 1))

x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_wind)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
# x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
# x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)


autoencoder = Model(input_wind, decoded)
autoencoder.compile(optimizer='adam', loss = tf.keras.losses.MeanSquaredError() )#loss='binary_crossentropy')
autoencoder.summary()

history = autoencoder.fit(
    x=x_train,
    y=x_train_target,
    epochs=50,
    batch_size=64,
    shuffle=True,
    validation_data=(x_val, x_val_target),
)

# history of training process
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
num_of_epochs = range(1, len(loss_values)+1)


show_training_results = True
if(show_training_results):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize =(6, 8))
    fig.subplots_adjust(hspace = 0.3)
    ax1.plot(num_of_epochs, loss_values, "bo", label="Training loss")
    ax1.plot(num_of_epochs, val_loss_values, "b", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Loss values over trained epochs")

    # acc_values = history_dict["accuracy"]
    # val_acc_values = history_dict["val_accuracy"]

    # ax2.plot(num_of_epochs, acc_values, "bo", label="Training accuracy")
    # ax2.plot(num_of_epochs, val_acc_values, "b", label="Validation accuracy")
    # ax2.set_xlabel("Epochs")
    # ax2.set_ylabel("Accuracy")
    # ax2.legend()
    # ax2.set_title("Accuracy over trained epochs")

    plt.show()



# show stuff 

# average
plt.figure()
plt.imshow(avg_epoch_train, aspect="auto", cmap='gray')
plt.show()

# some single trials 
plt.figure()
plt.imshow(train_windows_EEG[10, :, :, -1], aspect="auto", cmap='gray', norm = "linear")
plt.show()

pred_denoise = autoencoder.predict(np.reshape( train_windows_EEG[10, :, :, -1], (1, train_windows_EEG[10, :, :, -1].shape[0], train_windows_EEG[10, :, :, -1].shape[1], 1)))

plt.figure()
plt.imshow(pred_denoise[0, :, :,0], aspect="auto", cmap='gray', norm = "linear")
plt.show()



plt.figure()
plt.imshow(train_windows_EEG[20, :, :, -1], aspect="auto", cmap='gray', norm = "linear")
plt.show()

pred_denoise = autoencoder.predict(np.reshape(train_windows_EEG[20, :, :, -1], (1, train_windows_EEG[20, :, :, -1].shape[0], train_windows_EEG[20, :, :, -1].shape[1], 1)))


plt.figure()
plt.imshow(pred_denoise[0, :, :,0], aspect="auto", cmap='gray', norm = "linear")
plt.show()


plt.figure()
plt.plot(pred_denoise[0, 24, :,0])
plt.show()




