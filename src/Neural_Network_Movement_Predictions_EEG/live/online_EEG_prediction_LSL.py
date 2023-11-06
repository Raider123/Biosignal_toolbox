

# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import copy 
from pylsl import StreamInlet, resolve_stream
from time import perf_counter
import time
import serial
import multiprocessing as mp 
import SharedArray as sa
from matplotlib import animation


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
#********************** visualization ***********************
#************************************************************

def updateDataViz(i, data_chunk, inlet_viz, EEG_live_viz, EEGutils_live): 

    chunk, timestamps = inlet_viz.pull_chunk() # get a new data chunk

    if(chunk): # if list not empty (new data)
        
        # get the most recent buffer_size amount of values with a rate of dt_read_buffer, logs all important values for some time
        EEG_live_viz.windows = EEGutils_live.updateBuffer(chunk)  #list(channel_indices)
        #EEG_live_viz.FilterWindows(f_low = 245, f_high = 20, filter_type = "scipy_butter", order=2, show_response = False)  # changed
        data_chunk = EEG_live_viz.windows[0, :, :, 0]

    plt.cla() # clear the previous image
    plt.plot(data_chunk.T)
    #plt.hlines(onset_value, 0, len(emg_norm[param_viz.start_index:]), color = 'g')  
    

def dataVisualization(names, dt_read_buffer = 0.05, n_channels = 32, buffer_size = 500): 

    # from biosignal_toolbox.eeg_lib import EEGData, OnlineEEGUtils
    # from matplotlib import animation

    # first resolve an EEG stream on the lab network
    print("looking for an LSL EEG stream  visualization...")
    streams_viz = resolve_stream('type', 'EEG') # create data stream

    # create a new inlet to read from the stream
    inlet_viz = StreamInlet(streams_viz[0]) 
    stream_info_viz = inlet_viz.info()

    # create online EEG utils Object  
    EEGutils_live = OnlineEEGUtils(n_channels=n_channels, n_samples=buffer_size, dt_process_data = dt_read_buffer) # use this normally stream_info.channel_count()
    EEGutils_live.printStreamMetadata(stream_info_viz) # print stream info 

    EEG_live_viz = EEGData(format = "Live", f_samp = stream_info_viz.nominal_srate())

    data_chunk = np.zeros((n_channels, buffer_size))

    fig = plt.figure()
    plt.ylabel("data")
    plt.xlim([0, data_chunk.shape[1]]) # fix the x axis
    plt.legend(names)
    plt.title("raw EEG data")
    dt_read_buffer_ms = dt_read_buffer*1000
    # update the create plot (once created) in a loop this given interval dt and plot the values 
    anim = animation.FuncAnimation(fig, updateDataViz, frames = None, interval = dt_read_buffer_ms, blit = False, fargs = (data_chunk, inlet_viz, EEG_live_viz, EEGutils_live))
    plt.show()


def updateScoreViz(i, scores, names, buffersize): 

    plt.cla() # clear the previous image
    plt.plot(scores)
    plt.ylabel("prediction scores")
    plt.xlim([0, buffersize]) # fix the x axis
    plt.legend(names)
    plt.title("Prediction scores")

def predictionScoreVisualization(names, sa_name, dt_read_buffer = 0.05, buffersize = 500): 
    
    scores = sa.attach(sa_name)
    fig1 = plt.figure()
    dt_read_buffer_ms = dt_read_buffer*1000
    anim = animation.FuncAnimation(fig1, updateScoreViz, frames = None, interval = dt_read_buffer_ms, blit = False, fargs = (scores, names, buffersize), save_count=buffersize)
    plt.show()


#************************************************************
# ********************** user params ************************
#************************************************************
if __name__ == "__main__":

    # online params 
    buffer_size = 500  # size of ringbuffer in samples, currently set to 2500 (5 sec data times 500 Hz sampling rate)
    dt_read_buffer= 0.05 # time in seconds how often the buffer is read  (updated with new incoming chunks)
    print_times = False
    do_class_pred = True
    send_marker = False

    # subject info 
    subject = "Test"
    scenario_name = "intentional_unilateral"
    iteration = 0
    result_file_name = "live_train_results"

    # model names 
    MLP_eval_name = "test_intentional_unilateraltest_recorder_LSL_transfer_model_MLP_0"
    EEGNet_eval_name = "test_intentional_unilateraltest_recorder_LSL_transfer_model_EEGNet0"
    
    # ML params 
    decision_bound = 0.7
    # MLP Net 
    #features = "fusion" # which features to be used for classification, "timepoints" or "meanfreqs" or "fusion" (combine both)
    feature_indices_windows = np.arange(900, 1000, step = 2) # numpy array with time feature indices, (950, 1000) means last 100 ms of a window are used 
    
    # marker params 
    usb_port = '/dev/ttyUSB0'
    Baudrate = 115200

    # eeg stream params 
    channel_names = ['F3', 'F1', 'FZ', 'F2', 'F4', 'FFC1h', 'FC5', 'FC3', 'FC1', 'FC2', 'FCC3h', 'FCC1h', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'FCC2h', 'CCP3h', 'CCP1h', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'P3', 'P1', 'PZ', 'P2', 'P4']

    # shared array params for scores 
    sa_name1 = "shm://scores"
    sa_size1 = buffer_size
    score_names = ["MLP"]

    # EEG data params 
    n_channels = 32
    f_samp_eeg = 500.0
    
    #************************************************************
    # ********************** user params end ********************
    #************************************************************

    # init serial markers
    if(send_marker): 
        ser = serial.Serial(usb_port, Baudrate)
        time.sleep(3)

    # load models

    # load MLP model 
    MLP_model = MLModel() 
    MLP_model.loadModel(path =data_path, filename = MLP_eval_name)

    # load EEGNet model 
    model_EEGNet = MLModel()
    model_EEGNet.loadModel(path =data_path, filename = EEGNet_eval_name)

    # first resolve an EEG stream on the lab network
    print("looking for an LSL EEG stream...")
    streams = resolve_stream('type', 'EEG') # create data stream 

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0]) 
    stream_info = inlet.info()
    
    # create online EEG utils Object  
    EEGutils = OnlineEEGUtils(n_channels=n_channels, n_samples=buffer_size, dt_process_data = dt_read_buffer) # use this normally stream_info.channel_count()

    EEGutils.printStreamMetadata(stream_info) # print stream info 

    #inits 
    EEG_live = EEGData(format = "Live", f_samp = f_samp_eeg, channel_names = channel_names)

    # start visualization of data 
    viz_data_process = mp.Process(target=dataVisualization, args=(channel_names,))
    viz_data_process.start()

    viz_scores_process = mp.Process(target=predictionScoreVisualization, args=(score_names, sa_name1))
    viz_scores_process.start()

    # create shared array for passing prediction scores to visualization 
    # Create an array in shared memory.

    # Deleting old SharedArrays
    if len(sa.list()) != 0:
        sa.delete(sa_name1)

    scores_memory1 = sa.create(sa_name1, sa_size1) # len one for only one score 
    scores_buffer = np.zeros(buffer_size)

    # run continiously 
    running = True

    #counter = 0
    while running:
        
        chunk, timestamps = inlet.pull_chunk() # get a new data chunk
        
        if(chunk):#chunk # if list not empty (new data)
            
            # get the most recent buffer_size amount of values with a rate of dt_read_buffer, logs all important values for some time
            EEG_live.windows = EEGutils.updateBuffer(chunk, data_scale_factor=1)  #list(channel_indices)
            #print("windows type: ", EEG_live.windows.dtype)
            

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
            EEG_live_MLP.FilterWindows(f_low = 5.0, f_high = 0.3, filter_type = "scipy_butter", order=2, show_response = False)  # changed
            
            # time domain features (MLP)
            EEG_live_MLP.featureExtractionFromWindows(feature_type = "timepoints", feature_indices_windows = feature_indices_windows) # time dom features 
            EEG_live_freq_MLP.featureExtractionFromWindows(feature_type = "freqBandPower") # freq domain features 
            
            # feauture combination 
            x_val_freq = EEG_live_freq_MLP.getFeatures() # get features of freq
            EEG_live_MLP.addFeatures(x_val_freq) # add frequency domain features 
            
            # input features network 
            x_live_MLP = EEG_live_MLP.getFeatures()

            #print("feature type", x_live_MLP.dtype)
            

            # *********** EEGNet processing *******************
            
            #EEG_live_EEGNet.FilterWindows(f_low = None, f_high = 0.1, filter_type = "scipy_butter", order=2, show_response = False) # try this ? 
            EEG_live_EEGNet.FilterWindows(f_low = 40.0, f_high = 0.3, filter_type = "scipy_butter", order=2, show_response = False) # bandpass filter  
            


            # get train windows 
            x_live_EEGNet = EEG_live_EEGNet.getWindows()

            # ********** make model prediction  ***********
            
            # predict and get results 
            MLP_model.predict(data = x_live_MLP, labels = None, encoding = "binary", show_results = False, show_pred_time = True, eval_type = "online")

            # # predict and get results 
            model_EEGNet.predict(data = x_live_EEGNet, labels = None, encoding = "onehotencoding", show_results = False, show_pred_time = True, eval_type = "online")

            # postprocessing 
            MLP_score =  MLP_model.prediction_scores[0] 
            prod_score = MLP_score# final output score 
            

            if(do_class_pred): 
                print("hole score:", prod_score)
                print("EEGNet", model_EEGNet.prediction_scores[1])
                print("MLP", MLP_score)
                
                scores_buffer = np.roll(scores_buffer, shift = int(-1), axis = 0) 
                scores_buffer[-1] = MLP_score
                scores_memory1 = scores_buffer # write to shared memory 

            if(do_class_pred): 
                if(prod_score > decision_bound): 
                    print("onset detected")

                    if(send_marker):
                        ser.write(b's') # send marker when detected 
            
            
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

