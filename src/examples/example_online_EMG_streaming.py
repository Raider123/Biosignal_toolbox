
from biosignal_toolbox.emg_lib import OnlineEMG

# user params 
stream_type = "data" # stream should get data 
print_times = False 
channel_names = ["1", "2"]
buffer_size = 500 # in samples 
dt_process_data = 0.05 # in seconds 
f_samp = 1000.0

# path of the .so file of an SDK 
path_to_so_file = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox/lib/biosignal_toolbox/clients"


# create online object 
online_emg = OnlineEMG(stream_type=stream_type, channel_names=channel_names, n_channels=len(channel_names), n_samples=buffer_size, dt_process_data=dt_process_data, f_samp=f_samp)

online_emg.startANTEegoStreaming(path_to_so_file = path_to_so_file)

while True:
            
    #*****************************************************
    #*********** EMG processing section  *****************
    #*****************************************************
    
    # do processing here 
    online_emg.getChunk(return_chunk=False)
    online_emg.updateBuffer(channel_indices=None, show_data_shape=False)
    online_emg.bufferToWindows(num_non_data_channels=0)

    # process data windows here 

    
    #*****************************************************
    #*********** End processing section  *****************
    #*****************************************************

    # call this only after the 
    online_emg.ensureLoopFrequency(print_loop_time=False)
    


