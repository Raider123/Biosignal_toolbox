

# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import matplotlib.pyplot as plt
import numpy as np
import time
from mne_lsl.lsl import (StreamInfo, StreamInlet, StreamOutlet, local_clock, resolve_streams)


import warnings 
# # own libs 
from biosignal_toolbox.emg_lib import OnlineEMG
from marker_sync.zmq_lib import ZMQEvents
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
    send_onset = True
    f_samp = 1000.0
    
    got_baseline_values = False
    # emg stream params 
    channel_names = ['biceps', 'other']#, '2', '3', '4', '5', '6', '7', '8']

    # E;G data params 
    n_channels = 2
    
    thresh_value = 0 
    factor = 20 # factor for calculation the threshold (5(fast response) -50 (slow response)) 
    
    # marker sync params 
    zmq_port = "7000"
    pub_topic   = "1"       # enter the topic as a string
    pub_msg     = "1"       # enter the msg as a string

    #************************************************************
    # ********************** user params end ********************
    #************************************************************

    # init zmq 
    zmq_publisher = ZMQEvents(type = "publisher", port = zmq_port)

    # mne_LSL stuff 
    sinfo = StreamInfo(name="my-stream",stype="eeg",n_channels=n_channels,sfreq=f_samp,dtype="float32",source_id="myid")
    sinfo.set_channel_names(channel_names)
    sinfo.set_channel_types("emg")
    sinfo.set_channel_units("millivolts")

    stream_outlet = StreamOutlet(sinfo)
    

    # LSL stuff 
    # info = StreamInfo('ANT', 'EEG', channel_count = 1, nominal_srate=f_samp,  source_id="AntEEGO")
    # print("created info")

    # next make an outlet
    # stream_outlet = StreamOutlet(info, chunk_size=20)


    # create old_online EEG utils Object
    EMG_live = OnlineEMG(stream_type = "data", n_channels=n_channels, channel_names=channel_names) # use this normally stream_info.channel_count()
    
    EMG_live.startANTEegoStreaming(path_to_so_file=path_to_so_file)

    # Eegoclient = AntEego(path_to_so_file=path_to_so_file)
    # Eegoclient.init_amp(fsamp=f_samp)

    old_time = perf_counter()

    count = 0 
    send_counter = 10
    
    running = True
    while running:
        
        # get a new data chunk 
        chunk = EMG_live.getChunk(return_chunk=True)
        
        # data visualization 
        if(np.any(chunk)): # ifdata is there 
            # send data via lsl for visualization 
            stream_outlet.push_chunk(chunk[:, 0:2].astype(np.float32)*1000000) # shape sampels, channels

        
        # read out buffer 
        EMG_live.updateBuffer(show_data_shape = False, channel_indices=[0, 1])#, channel_indices=[0])
        EMG_live.bufferToWindows() # use only first channel
        #print("EMG windows: " ,EMG_live.getWindows()[0, 0, :, 0]) 
        EMG_live.applyVarianceFilter(apply_to_structures = "windows", n_var=20)
        
     
        # calc threshold 
        if (not got_baseline_values and count >= 100): 
            thresh_value = np.mean(np.abs(EMG_live.windows[0, 0, :, 0]))
            print("tresh", thresh_value)
            got_baseline_values = True
            count = 100
            
        count = count+1

        #print("thresh ratio", np.mean(np.abs(EMG_live.windows[0, 0, :, 0])/thresh_value)) 
        
        # check for onset detection 
        if(np.mean(np.abs(EMG_live.windows[0, 0, :, 0]) > thresh_value * factor) and got_baseline_values and send_counter == 0): 
            print("onset detected")
            zmq_publisher.sendMessage(topic=pub_topic, message=pub_msg)
            send_counter = 10
            
        else: 
            print("resting")
            pass


        send_counter = send_counter -1
        if (send_counter <= 0 ): 
            send_counter = 0
        
        
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

        