
# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import tensorflow as tf
import copy 
from pylsl import StreamInlet, resolve_stream
from time import perf_counter
import time
#import serial
import warnings
import pyrock

# # own libs 
from biosignal_toolbox.eeg_lib import OnlineEEG
from biosignal_toolbox.ML_lib import MLModel
import biosignal_toolbox.ML_pipelines_lib as pipeline
import json
import zmq

# # models 
# from biosignal_toolbox.models.CNNnets import EEGNet


#own libs
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox"
data_path = proj_path+"/data/"

# old_online params
buffer_size = 500  # size of ringbuffer in samples, currently set to 2500 (5 sec data times 500 Hz sampling rate)
dt_read_buffer= 0.05 # time in seconds how often the buffer is read  (updated with new incoming chunks)
num_classes = 2
print_times = False
do_class_pred = True
send_score = True
deadtime = 1 # in seconds after exo movement done 

# subject info 
subject = "Test"
scenario_name = "intentional_unilateral"
iteration = 0
result_file_name = "live_train_results"

# model names --> always the newest model thingi 
MLP_eval_name = "current_intentional_unilateral_live_model_MLP_0"
EEGNet_eval_name = "current_intentional_unilateral_live_model_EEGNet0"

# ML params 
decision_bound = 0.6
n_count_positives = 2 # how many window have to be positive 

# MLP Net 
#features = "fusion" # which features to be used for classification, "timepoints" or "meanfreqs" or "fusion" (combine both)
#feature_indices_windows = np.arange(900, 1000, step = 2) # numpy array with time feature indices, (950, 1000) means last 100 ms of a window are used 
feature_indices_windows = np.array([900, 920, 940, 960, 965, 970, 975, 980, 985, 990, 995, 999])


# eeg stream params 
#channel_names = ["F5", "F3", "F1", "FZ", "F2", "F4", "F6", "FC5", "FC3", "FC1", "FC2", "FC4", "FC6", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "P5", "P3", "P1", "PZ", "P2", "P4", "P6"]
channel_names = ["FC3", "FC1", "C3", "C1", "CZ", "CP3", "CP1", "CPZ", "CCP1h", "FCC1h", "CCP3h", "FCC3h", "marker"]
num_non_data_channels = 1 # or 3 ? 

# shared array params for scores 
#sa_name1 = "shm://scores"
#sa_size1 = buffer_size
score_names = ["product"]

# EEG data params 
n_channels = len(channel_names)
f_samp_eeg = 500.0

# ip of logic script 
ip_logic = "tcp://134.91.100.14:7001"

# # zmq stuff 
zmq_port = "34761"
zmq_topic = b"10"
# ip_rock = "10.250.3.15"


# # own model 
# from biosignal_toolbox.models.MlpErp import MLP_Model, MLP_Model_reduced

print(tf.config.experimental.list_physical_devices('GPU'))

#from Neural_Network_Movement_Predictions_EEG.live.userparams_prediction import * # --> parameters are stored and fully imported from this script 

# disable GPU for testing
tf.config.set_visible_devices([], 'GPU') 


#************************************************************
# ********************** user params ************************
#************************************************************
if __name__ == "__main__":
    

    # # pyrock command setup 
    # ns = pyrock.NameService(ip_rock)
    # task = ns.get_task_context("pyspace")
    # writer = task.writer("prediction_vector1")
    # sample = writer.get_sample()
    # sample['label'] = ["Right"]

    # task = ns.get_task_context("trajectory_from_fileTask")
    # reader = task.reader("state", type=pyrock.RTT.CBuffer, size=50)


    #************************************************************
    # ********************** user params end ********************
    #************************************************************
    
    # dead time 
    dead_n_samples = deadtime/dt_read_buffer


    # load MLP model 
    MLP_model = MLModel(type="keras") 
    MLP_model.loadModel(path =data_path, filename = MLP_eval_name)

    # load EEGNet model 
    model_EEGNet = MLModel(type="keras")
    model_EEGNet.loadModel(path =data_path, filename = EEGNet_eval_name)

    # first resolve an EEG stream on the lab network
    print("looking for an LSL EEG stream...")
    streams = resolve_stream('type', 'EEG') # create data stream 

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0]) 
    stream_info = inlet.info()
    
    # create old_online EEG utils Object
    EEG_live =OnlineEEG(stream_type = "data", channel_names=channel_names, dt_process_data=dt_read_buffer,f_samp=f_samp_eeg)

    EEG_live.printStreamMetadata(stream_info) # print stream info 


    # init serial markers
    if(send_score): 
        #ser = serial.Serial(usb_port, Baudrate)
        my_socket = EEG_live.startZMQServer(zmq_port)
        time.sleep(3)


    move_json_message = json.dumps({'moveCmd' : "right_arm"}) # move when speech offset detected 
    context = zmq.Context() 
    client = context.socket(zmq.PUB) 
    #client.connect("tcp://10.250.5.249:7001") # connect to logic ? with zmq 
    client.connect(ip_logic) # connect to logic ? with zmq 
    

    # init values 
    running = True    # run continiously 
    # trajectory_done = False
    # movement_start = False
    enable_onset_detection = True
    pos_last_prediction = True
    pos_prediction_count = 0 
    onset_detected = False
    counter = 0
    old_send_time = perf_counter()*1000

    # filter design before old_online setting
    sos_MLP = EEG_live.designFilter(f_low = 5.0, f_high = 0.3, order = 2, filter_type = "scipy_butter", return_type = "sos") #--> good one
    sos_EEGNet = EEG_live.designFilter(f_low = 40.0, f_high = 0.3, order = 2, filter_type = "scipy_butter", return_type = "sos")


    while running:
        
        chunk, timestamps = inlet.pull_chunk() # get a new data chunk
        
        if(chunk):#chunk # if list not empty (new data)
            
            #print(np.array(chunk).shape)

            t_start = perf_counter()
            # get the most recent buffer_size amount of values with a rate of dt_read_buffer, logs all important values for some time
            EEG_live.setChunk(chunk = chunk, chunk_type = "list") # data chunk is list from 
            EEG_live.updateBuffer(channel_indices = None, show_data_shape = False)#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], show_data_shape = False)
            EEG_live.bufferToWindows(num_non_data_channels=num_non_data_channels)
            
            
            if(print_times): 
                t1 = perf_counter()

            # **********************************************
            # *********** Model apply here *****************
            # **********************************************

            # copy data objects for different processing steps
            EEG_live_MLP = copy.deepcopy(EEG_live)# time domain feates MLP
            EEG_live_freq_MLP = copy.deepcopy(EEG_live) # for frequency features of MLP
            EEG_live_EEGNet = copy.deepcopy(EEG_live) # for EEGNet

            
            # ******** MLP processing *******************
            
            x_live_MLP, y_live_MLP = pipeline.MLPProcessingOnline(EEG_live_MLP, EEG_live_freq_MLP, [0.5], feature_indices_windows, sos_MLP)
            
            
            # *********** EEGNet processing *******************
            
            x_live_EEGNet, y_live_EEGNet = pipeline.EEGNetProcessingOnline(EEG_live_EEGNet, [0.5], num_classes, sos_EEGNet)

            # ********** make model prediction  ***********
            
            # predict and get results 
            MLP_model.predict(data = x_live_MLP, labels = None, encoding = "binary", show_results = False, show_pred_time = False, eval_type = "old_online")

            # # predict and get results 
            model_EEGNet.predict(data = x_live_EEGNet, labels = None, encoding = "onehotencoding", show_results = False, show_pred_time = False, eval_type = "old_online")

            # postprocessing 
            #MLP_score =  MLP_model.prediction_scores[0] 
            # change here if both scores should be multiplied or only one model used 
            prod_score = MLP_model.prediction_scores[0] *model_EEGNet.prediction_scores[1]# final output score

            
            # uncomment for showing scores 
            #print("hole score:", prod_score)
            #print("EEGNet", model_EEGNet.prediction_scores[1])
            #print("MLP", MLP_score)

            if(do_class_pred):

                # # exo states 
                # status = pyrock.RTT.CNewData
                # while status == pyrock.RTT.CNewData:
                #     #print(f"{status=}")
                #     status, state = reader.read(return_status=True)
                #     if state == 7:
                #         trajectory_done = True
                #         #print("exo state 7")
                #     elif state == 5: 
                #         movement_start = True
                #         #print("exo state 5")
                
                # # check if exo moving 
                # if (trajectory_done == False and movement_start == True): # in movement  
                #     print("movement ongoing")
                #     prod_score = 0.0 # output is zero from model 

                # elif ((trajectory_done == True) and (movement_start == True)): # after movement 
                
                #print("movement done ")
                
                # if onset detection disabled 
                if (enable_onset_detection == False): 

                    if (counter > dead_n_samples): # done waiting  
                        counter = 0
                        enable_onset_detection = True

                    else: 
                        print(f"waiting")
                        counter = counter +1  # waiting 
                        prod_score = 0.0  # no output from model 


                # model output count positives 
                if(prod_score > decision_bound and enable_onset_detection): 
                    print("onset detected")

                    pos_prediction_count = pos_prediction_count +1
                    if(pos_prediction_count >= n_count_positives): # onset detected after counting positives 
                        pos_prediction_count = 0
                        #pos_last_prediction = False
                        onset_detected = True
                else: 
                    pos_prediction_count = 0
                    # if(movement_start == False and trajectory_done == False): 
                    #print("resting")

                # send command to move  
                if(onset_detected):
                    
                    # send move command to logic 
                    client.send_string(move_json_message) # send 

                    onset_detected = False 
                    enable_onset_detection = False
                    print("*****")
                    print("move !!!!")
                    print("*****")
                    
                
                if (send_score): # write this continously 
                    my_socket.send(zmq_topic+str(prod_score).encode())
                    # print("send time:", perf_counter()*1000 -old_send_time)
                    # old_send_time = perf_counter()*1000

            
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

