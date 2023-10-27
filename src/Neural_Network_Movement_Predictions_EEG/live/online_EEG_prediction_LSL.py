

# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import copy 
from pylsl import StreamInlet, resolve_stream
from time import perf_counter 

# # own libs 
from biosignal_toolbox.eeg_lib import EEGData, OnlineEEGUtils
from biosignal_toolbox.ML_lib import MLModel

# own libs
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox"
data_path = proj_path+"/data/"

# models 
from biosignal_toolbox.models.CNNnets import EEGNet

# own model 
from biosignal_toolbox.models.MlpErp import MLP_Model

print(tf.config.experimental.list_physical_devices('GPU'))

# disable GPU for testing
tf.config.set_visible_devices([], 'GPU') 

#************************************************************
# ********************** user params ************************
#************************************************************

# online params 
buffer_size = 500  # size of ringbuffer in samples, currently set to 2500 (5 sec data times 500 Hz sampling rate)
dt_read_buffer= 0.05 # time in seconds how often the buffer is read  (updated with new incoming chunks)
print_times = True

# subject info 
subject = "JV43"
scenario_name = "intentional_unilateral"
iteration = 0

# model names 
MLP_eval_name = "fcn_network_results_34ch_MLP_online_pre_test_reduced_net"
EEGNet_eval_name = "fcn_network_results_34ch_EEGNet_online_pre_test_reduced_net"

# ML params 
decision_bound = 0.7
# MLP Net 
features = "fusion" # which features to be used for classification, "timepoints" or "meanfreqs" or "fusion" (combine both)
feature_indices_windows = np.arange(900, 1000, step = 2) # numpy array with time feature indices, (950, 1000) means last 100 ms of a window are used 


#************************************************************
#************************************************************
#************************************************************

# load models

# load MLP model 
MLP_model = MLModel() 
MLP_model.loadModel(path =data_path, filename = subject+"_"+scenario_name+MLP_eval_name+"_model_"+str(iteration))

# load EEGNet model 
model_EEGNet = MLModel()
model_EEGNet.loadModel(path =data_path, filename = subject+"_"+scenario_name+EEGNet_eval_name+"_model_"+str(iteration))


# first resolve an EEG stream on the lab network
print("looking for an LSL EEG stream...")
streams = resolve_stream('type', 'EEG') # create data stream 

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0]) 
stream_info = inlet.info()

# create online EEG utils Object 
EEGutils = OnlineEEGUtils(n_channels=stream_info.channel_count(), n_samples=buffer_size, dt_process_data = dt_read_buffer)

EEGutils.printStreamMetadata(stream_info) # print stream info 

#inits 
channel_names = []
EEG_live = EEGData(format = "Live", f_samp = stream_info.nominal_srate(), channel_names = channel_names)


# run continiously 
running = True

#counter = 0
while running:
    
    chunk, timestamps = inlet.pull_chunk() # get a new data chunk

    if(chunk): # if list not empty (new data)
        
        # get the most recent buffer_size amount of values with a rate of dt_read_buffer, logs all important values for some time
        EEG_live.windows = EEGutils.updateBuffer(chunk)  
        
        if(print_times): 
            t1 = perf_counter()

        # **********************************************
        # *********** Model apply here *****************
        # **********************************************

        # copy data objects for different processing 
        EEG_live_MLP = EEG_live # time domain feates MLP
        EEG_live_freq_MLP = copy.deepcopy(EEG_live_MLP) # for frequency features of MLP
        EEG_live_EEGNet = copy.deepcopy(EEG_live_MLP) # for EEGNet

         # ******** MLP processing *******************
    
        # bandpass filter data 
        EEG_live_MLP.FilterWindows(f_low = 5.0, f_high = 0.3, filter_type = "scipy_butter", order=2, show_response = False) # bandpass filter
        
        # time domain features (MLP)
        EEG_live_MLP.featureExtractionFromWindows(feature_type = "timepoints", feature_indices_windows = feature_indices_windows) # time dom features 
        EEG_live_freq_MLP.featureExtractionFromWindows(feature_type = "freqBandPower") # freq domain features 
        
        # feauture combination 
        x_val_freq = EEG_live_freq_MLP.getFeatures() # get features of freq
        EEG_live_MLP.addFeatures(x_val_freq) # add frequency domain features 
        
        # input features network 
        x_live_MLP = EEG_live_MLP.getFeatures()

        
        # *********** EEGNet processing *******************

        EEG_live_EEGNet.FilterWindows(f_low = 40.0, f_high = 0.3, filter_type = "scipy_butter", order=2, show_response = False) # bandpass filter  

        # get train windows 
        x_live_EEGNet = EEG_live_EEGNet.getWindows()

        # ********** make model prediction  ***********
        
        # predict and get results 
        MLP_model.predict(data = x_live_MLP, labels = None, encoding = "binary", show_results = False, show_pred_time = True, eval_type = "online")

        # # predict and get results 
        model_EEGNet.predict(data = x_live_EEGNet, labels = None, encoding = "onehotencoding", show_results = False, show_pred_time = True, eval_type = "online")

        # postprocessing 
        prod_score = model_EEGNet.prediction_scores[1] *MLP_model.prediction_scores[0] # final output score 
        
        if(prod_score > decision_bound): 
            print("onset detected")


        # *****************************************************
        # *********** End processing section  *****************
        # *****************************************************

        if(print_times): 
            print("model time(ms): ",(perf_counter()-t1)*1000)


        # wait for some time to ensure a "fixed" frequency to read new data from buffer 
        while((perf_counter()-old_time) < dt_read_buffer): 
            pass
        
        if(print_times): 
            print("loop time(ms):  ", (perf_counter() - old_time)*1000)
        

        # uncomment to record ALL data received (not required for participants)
        #data_arr = data_arr+chunk
        #time_stamp_arr = time_stamp_arr + timestamps


    old_time = perf_counter()


# print("shape window", data_buffer.shape)
# plt.figure()
# plt.plot(data_buffer[0, 10, :, 0])
# plt.show()

