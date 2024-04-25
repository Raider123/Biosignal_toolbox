

# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import tensorflow as tf
import copy 
from pylsl import StreamInlet, resolve_stream
from time import perf_counter
import time
import serial
import os 

# # own libs 
from biosignal_toolbox.eeg_lib import  OnlineEEG
from biosignal_toolbox.ML_lib import MLModel
import biosignal_toolbox.ML_pipelines_lib as pipeline

# own libs
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox"
data_path = proj_path+"/data/"

# models 
from biosignal_toolbox.models.CNNnets import EEGNet

# own model 
from biosignal_toolbox.models.MlpErp import MLP_Model

print(tf.config.experimental.list_physical_devices('GPU'))

# disable GPU for testing
#tf.config.set_visible_devices([], 'GPU') 


#************************************************************
# ********************** user params ************************
#************************************************************
if __name__ == "__main__":
    
    # online params 
    buffer_size = 550  # size of ringbuffer in samples, currently set to 2500 (5 sec data times 500 Hz sampling rate)
    dt_read_buffer= 0.05 # time in seconds how often the buffer is read  (updated with new incoming chunks)
    print_times = False
    do_class_pred = True
    send_marker = False
    deadtime = 1 # in seconds 
    # viz data and score 
    visualization_on = False
    
    # subject info 
    subject = "Test"
    scenario_name = "intentional_unilateral"
    iteration = 0
    result_file_name = "live_train_results"

    # model names 
    MLP_eval_name = "current_intentional_unilateral_live_model_MLP_0"
    EEGNet_eval_name = "current_intentional_unilateral_live_model_EEGNet0"
    
    # ML params 
    decision_bound = 0.7
    n_count_positives = 2 # how many window have to be positive 
    num_classes = 2

    # MLP Net 
    #features = "fusion" # which features to be used for classification, "timepoints" or "meanfreqs" or "fusion" (combine both)
    feature_indices_windows = np.arange(900, 1000, step = 2) # numpy array with time feature indices, (950, 1000) means last 100 ms of a window are used 
    
    # marker params 
    usb_port = '/dev/ttyUSB0'
    Baudrate = 115200

    # eeg stream params 
    channel_names = ["F5", "F3", "F1", "FZ", "F2", "F4", "F6", "FC5", "FC3", "FC1", "FC2", "FC4", "FC6", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "P5", "P3", "P1", "PZ", "P2", "P4", "P6"]
    
    # shared array params for scores 
    sa_name1 = "shm://scores"
    sa_size1 = buffer_size
    score_names = ["product"]

    # EEG data params 
    n_channels = 37
    f_samp_eeg = 500.0
    ignore_n_last_channels = 3 # markerchannels and indices 

    # zmq stuff 
    zmq_port = "34761"
    zmq_topic = b"10"

    #************************************************************
    # ********************** user params end ********************
    #************************************************************

    # dead time 
    dead_n_samples = deadtime/dt_read_buffer

    clear = lambda: os.system('clear')

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
    
    # create online EEG object 
    EEG_live = OnlineEEG(f_samp_eeg = f_samp_eeg, channel_names = channel_names, n_channels=n_channels, n_samples=buffer_size, dt_process_data = dt_read_buffer)
    EEG_live.printStreamMetadata(stream_info) # print stream info 

    # init serial markers
    if(send_marker): 
        ser = serial.Serial(usb_port, Baudrate)
        my_socket = EEG_live.establishZMQ(zmq_port)
        time.sleep(3)


    # init values 
    running = True    # run continiously 
    trajectory_done = False
    movement_start = False
    pos_last_prediction = True
    pos_prediction_count = 0 
    onset_detected = False
    counter = 0
    old_send_time = perf_counter()*1000


    while running:
        
        chunk, timestamps = inlet.pull_chunk() # get a new data chunk
        
        if(chunk):#chunk # if list not empty (new data)
            
            # get the most recent buffer_size amount of values with a rate of dt_read_buffer, logs all important values for some time
            EEG_live.updateBuffer(chunk, check_sample_loss = False)  #list(channel_indices)
            EEG_live.BufferToWindows(num_non_data_channels = ignore_n_last_channels) 
            
            # here with samples counter 
            markers = EEG_live.data_buffer[0, -1, :, 0].astype(int)
            #print(markers)
            for marker in markers[-50:]: 
                #print(marker)
                if(marker == 22): 
                    #print("****")
                    print("onset")
                    #print("****")
                elif(marker == 23): 
                    print("movement done")


            if(print_times): 
                t1 = perf_counter()

            # **********************************************
            # *********** Model apply here *****************
            # **********************************************

            # copy data objects for different processing 
            EEG_live_MLP = copy.deepcopy(EEG_live)# time domain feates MLP
            EEG_live_freq_MLP = copy.deepcopy(EEG_live) # for frequency features of MLP
            EEG_live_EEGNet = copy.deepcopy(EEG_live) # for EEGNet


            # ******** MLP processing *******************

            x_live_MLP, y_live_MLP = pipeline.MLPProcessing(EEG_live_MLP, EEG_live_freq_MLP, [0.5], feature_indices_windows)
            
            
            # *********** EEGNet processing *******************
            
            x_live_EEGNet, y_live_EEGNet = pipeline.EEGNetProcessing(EEG_live_EEGNet, [0.5], num_classes)

            # ********** make model prediction  ***********
            
            # predict and get results 
            MLP_model.predict(data = x_live_MLP, labels = None, encoding = "binary", show_results = False, show_pred_time = False, eval_type = "online")

            # # predict and get results 
            model_EEGNet.predict(data = x_live_EEGNet, labels = None, encoding = "onehotencoding", show_results = False, show_pred_time = False, eval_type = "online")

            # postprocessing 
            #MLP_score =  MLP_model.prediction_scores[0] 
            prod_score = MLP_model.prediction_scores[0] * model_EEGNet.prediction_scores[1] #MLP_score#* model_EEGNet.prediction_scores[1] # final output score 

            
            #if(do_class_pred): 
            #print("hole score:", prod_score)
            #print("EEGNet", model_EEGNet.prediction_scores[1])
            #print("MLP", MLP_score)
                

            if(do_class_pred):
                
                # model output count positives 
                if(prod_score > decision_bound): 
                    #print("onset detected")
                    pos_prediction_count = pos_prediction_count +1
                    if(pos_prediction_count >= n_count_positives): # onset detected after counting positives 
                        pos_prediction_count = 0
                        pos_last_prediction = False
                        onset_detected = True
                else: 
                    pos_prediction_count = 0
                    #print("resting")

                # send command to move  
                if(onset_detected):
                        #ser.write(b's') # send marker when detected 
                        #writer.write(sample)
                        onset_detected = False 

                        #print("*****")
                        print("move !!!!")
                        #print("*****")

                if (visualization_on): # write this continously 
                    my_socket.send(zmq_topic+str(prod_score).encode())

                
            #*****************************************************
            #*********** End processing section  *****************
            #*****************************************************

            if(print_times): 
                print("model time(ms): ",(perf_counter()-t1)*1000)


            # wait for some time to ensure a "fixed" frequency to read new data from buffer 
            while((perf_counter()-old_time) < dt_read_buffer): 
                pass
            
            if(print_times): 
                print("loop time(ms):  ", (perf_counter() - old_time)*1000)
            


        old_time = perf_counter()

