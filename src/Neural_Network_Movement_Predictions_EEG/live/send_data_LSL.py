


from pylsl import StreamInlet, resolve_stream, StreamOutlet, StreamInfo
import numpy as np 
import time 
from mne_lsl.lsl import (StreamInfo, StreamInlet, StreamOutlet, local_clock, resolve_streams)


def main():

    #************************************************************
    # ********************** user params ************************
    #************************************************************

    f_samp_eeg = 500.0
    dt = 0.04
    chunksize = 20

    # set paths 
    proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox"
    data_path = proj_path+"/data/"


    #  loading and epoching for training  
    data_list = ["XY90_unilateral_set4_data"] #"BR60D_unilateral_LSL_set4_1.vhdr"
    # eeg stream params 
    channel_names = ["1", "2","3", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "FC5", "FC3", "FC1", "FC2", "FC4", "FC6", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "P5", "P3", "P1", "PZ", "P2", "P4", "P6"]

    #************************************************************
    # ***********************************************************
    #************************************************************

    # mne_LSL stuff 
    sinfo = StreamInfo(name="mystream",stype="eeg",n_channels=len(channel_names),sfreq=f_samp_eeg,dtype="float32",source_id="myid")
    sinfo.set_channel_names(channel_names)
    sinfo.set_channel_types("eeg")
    sinfo.set_channel_units("microvolts")

    stream_outlet = StreamOutlet(sinfo)
    
    print("n_channels", len(channel_names))
    
    
    # data_train = EEGData(format = "Recorded_LSL_stream", filenames = data_list, data_path = data_path, f_samp = f_samp_eeg, channel_names = channel_names)  
    # raw_obj = data_train.getRawObject()
    # raw_data = raw_obj.get_data()
    #print("raw data shape", raw_data.shape)

    raw_data = np.load(data_path+"XY90_unilateral_set3_data.npy")
    raw_data = raw_data.T
    print("raw data shape", raw_data.shape)


    # info = StreamInfo('Liveamp1', 'EEG', channel_count = n_channel, nominal_srate=f_samp_eeg)
    # print("created info")

    # next make an outlet
    # outlet = StreamOutlet(info)

    start = 0
    for i in range(0, raw_data.shape[1] -chunksize): 

        print(i)

        print("now sending data...")
    
        # make a new random 8-channel sample; this is converted into a
        # pylsl.vectorf (the data type that is expected by push_sample)

        print(raw_data.shape)
        
        window_chunk = raw_data[:, i*chunksize +start:i*chunksize+chunksize+start]
        
        window_chunk = window_chunk.T
        print("chunk shape", window_chunk.shape) # has shape 
        # now send it and wait for a bit
        stream_outlet.push_chunk(window_chunk.astype('float32')) # 


        time.sleep(dt)



if __name__ == '__main__':
    main()
