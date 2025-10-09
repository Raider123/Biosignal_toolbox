

# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import matplotlib.pyplot as plt
import numpy as np
import time
from mne_lsl.lsl import (StreamInfo, StreamInlet, StreamOutlet, local_clock, resolve_streams)
import json 
import zmq

# # own libs 
from biosignal_toolbox.emg_lib import OnlineEMG
#from marker_sync.zmq_lib import ZMQEvents
from time import perf_counter


# own libs
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox"
data_path = proj_path+"/data/"
path_to_so_file = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox/clients"


#************************************************************
# ********************** user params ************************
#************************************************************
if __name__ == "__main__":
    

    # old_online params
    buffer_size = 500  # size of ringbuffer in samples, currently set to 2500 (5 sec data times 500 Hz sampling rate)
    dt_read_buffer= 0.05 # time in seconds how often the buffer is read  (updated with new incoming chunks)
    print_times = False
    f_samp = 2000.0

    
    got_baseline_values = False
    # emg stream params 
    channel_names = ['1', '2', '3', '4']#,'3','4','5','6','7','8','9','10','11','12','13','14','15','16']

    # E;G data params 
    #n_channels = 2
    
    thresh_value = 50
    deadtimecounter = 0
    deadtimecounter_max = 150
    enable_arm_onset = True
    channel_index_arm = 2 # channel 3 EMG
    channel_index_hand = 3 # channel 4 EMG
    factor = 10 # factor for calculation the threshold (5(fast response) -50 (slow response)) 
    factor2 = 30

    # zmq stuff 
    close_hand_msg = json.dumps({'handcmd' : "close"}) # move when speech offset detected 
    open_hand_msg = json.dumps({'handcmd' : "open"}) # move when speech offset detected 
    move_json_message = json.dumps({'moveCmd' : "right_arm"}) # move when speech offset detected 

    zmq_ip_hand_state = "tcp://134.91.100.13:7004"
    zmq_ip_logic = "tcp://134.91.100.14:7001"

    #************************************************************
    # ********************** user params end ********************
    #************************************************************

    # connect to zmq 
    context = zmq.Context() 
    zmq_client = context.socket(zmq.PUB) 
    zmq_client.connect(zmq_ip_hand_state)

    context = zmq.Context() 
    client_logic = context.socket(zmq.PUB) 
    client_logic.connect(zmq_ip_logic) # connect to logic ? with zmq 


    target_stream_name = "ExampleStream"

    print("Looking for an LSL stream...")
    streams = resolve_streams()  # You can modify 'EEG' to the type of stream you are interested in.
    print("")
    print("streams: ", streams)
    print("")
    stream_count = 0
    stream_index = None

    # find the correct stream 
    for stream in streams: 
        if(stream.name == target_stream_name): 
            stream_index = stream_count
        stream_count = stream_count +1

    inlet = StreamInlet(streams[stream_index]) # use selected stream


    #sfreq = 2000  # Sampling frequency of the LSL stream (change accordingly)
    ch_names = channel_names # Example channel names
    ch_types = ['emg'] * len(ch_names)  # EEG channels
    #info = inlet.info()
    #info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # n_channels = len(ch_names)
    
    # create old_online EEG utils Object
    EMG_live = OnlineEMG(stream_type = "data", channel_names=channel_names) # use this normally stream_info.channel_count()

    old_time = perf_counter()
    
    count = 0 
    send_counter = 10 

    hand_state = "open"
    
    running = True
    while running:
        
        # get a new data chunk 
        #chunk = EMG_live.getChunk(return_chunk=True)
        chunk, timestamps = inlet.pull_chunk(timeout = 0.01, max_samples=500) 
        EMG_live.setChunk(chunk = chunk)

        #print("shape chunk: ", chunk.shape)
        
        # read out buffer 
        EMG_live.updateBuffer(show_data_shape = False, channel_indices=[0, 1, 2, 3])#, channel_indices=[0])
        EMG_live.bufferToWindows() # use only first channel
        #print("EMG live ", EMG_live.windows[0, 1, :, 0])

        #print("EMG windows: " ,EMG_live.getWindows().shape) 
        EMG_live.applyVarianceFilter(apply_to_structures = "windows", n_var=10)
        

        # calc threshold 
        if (not got_baseline_values and count >= 100): 
            #thresh_value = np.mean(np.abs(EMG_live.windows[0, channel_index_hand, :, 0]))
            #thresh_value = 50
            print("tresh", thresh_value)
            got_baseline_values = True
            count = 100
            
        count = count+1

        #print("thresh ratio", np.mean(np.abs(EMG_live.windows[0, channel_index_hand, :, 0]))) 
        
        # check for and state based on forearm muscle and toggle it 
        if(np.mean(np.abs(EMG_live.windows[0, channel_index_hand, :, 0]) > thresh_value * factor) and got_baseline_values): 
            if(got_baseline_values): 
                #print("grasping")
                #zmq_publisher.sendMessage(topic=pub_topic, message=pub_msg)
                #send_counter = 10

                if(hand_state == "open"): 
                    print("grasped")
                    zmq_client.send_string(close_hand_msg)
                    zmq_client.send_string(close_hand_msg)
                    zmq_client.send_string(close_hand_msg)
                    hand_state = "closed"
                    factor = 5
                    
        else: 
            if(got_baseline_values): 
                if(hand_state == "closed"): 
                    print("opened")
                    zmq_client.send_string(open_hand_msg)
                    zmq_client.send_string(open_hand_msg)
                    zmq_client.send_string(open_hand_msg)
                    hand_state = "open"
                    factor = 10
            pass
        
        
        # onset from EMG of the arm 
        if(np.mean(np.abs(EMG_live.windows[0, channel_index_arm, :, 0]) > thresh_value * factor2) and got_baseline_values and enable_arm_onset): 
            #client_logic.send_string(move_json_message) # send 
            enable_arm_onset = False
            print("arm movement onset")
        
        # deadtime for arm onset 
        if(enable_arm_onset == False): 
            deadtimecounter = deadtimecounter +1 
            if(deadtimecounter >deadtimecounter_max): 
                enable_arm_onset = True
                deadtimecounter = 0
                print("deadtime done ...")

            
        
        # # states updating 
        # if(old_hand_state == "open" and np.mean(np.abs(EMG_live.windows[0, channel_index_hand, :, 0]) > thresh_value * factor) and got_baseline_values): 
        #     print("grasp")
        #     hand_state ="closed"
        # elif(old_hand_state == "closed" and np.mean(np.abs(EMG_live.windows[0, channel_index_hand, :, 0]) < thresh_value * factor *0.8) and got_baseline_values): 
        #     print("opened")
        #     hand_state ="open"

        # old_hand_state = hand_state

        
        # send_counter = send_counter -1
        # if (send_counter <= 0 ): 
        #     send_counter = 0
        
        
        if(print_times): 
            t1 = perf_counter()
        
        # *********** EMG processing and onset detection *****
        


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
        

            # uncomment to record ALL data received (not required for participants)
            #data_arr = data_arr+chunk
            #time_stamp_arr = time_stamp_arr + timestamps

    old_time = perf_counter()

        
