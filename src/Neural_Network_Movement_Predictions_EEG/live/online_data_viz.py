#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from matplotlib import animation
import zmq
from time import sleep

from pylsl import StreamInlet, resolve_stream
#from biosignal_toolbox.eeg_lib import EEGData, OnlineEEGUtils


def establishZMQ(zmq_server_ip, zmq_topic):
    """
    This function sets up the ZMQ Server client connection
    """
    my_context      = zmq.Context()
    my_socket       = my_context.socket(zmq.SUB)
    my_socket.connect("tcp://"+zmq_server_ip)
    my_socket.setsockopt(zmq.SUBSCRIBE, zmq_topic)

    return my_socket


def updateScoreViz(i, scores_arr, my_zmq_socket,buffer_size, y_lim_arr, n_ticks, prob_thr): 
    try:
        undecoded_msg = my_zmq_socket.recv(flags=zmq.NOBLOCK)
        zmq_msg = float(undecoded_msg[2:])
        scores_arr[:] = np.roll(scores_arr, shift = -1)
        scores_arr[-1] = zmq_msg
        # print(zmq_msg)
    except zmq.Again as e:
        # print("No message received yet")
        plt.cla() # clear the previous image
        lines = plt.plot(scores_arr)
        plt.xlabel("Recent Samples")
        plt.ylabel("Prediction Score")
        plt.xlim([0, buffer_size]) # fix the x axis
        plt.xticks(np.linspace(0,buffer_size,n_ticks))
        plt.ylim(y_lim_arr)
        plt.title("Online Movement Onset Prediction Scores")
        plt.axhline(y=prob_thr, color='r', linestyle='-.')
        
        plt.legend(['Ensemble model score', 'Threshold'])
        

def runZMQ():

    my_zmq_socket  = establishZMQ("localhost:34761", b"10")
    sleep(1)
    print("ZMQ Process Ready!")

    while True:
        
        dt_read_buffer_ms = 40
        buffer_size = 150
        prob_thr = 0.6
        y_lim_arr = [-0.5, 1.5]
        tick_res = 10
        n_ticks = int(buffer_size/tick_res) + 1
        scores_arr = np.zeros(buffer_size)
        inp_buf = []
        
        fig1 = plt.figure()
        fig1.set_size_inches(8.0, 4.8, forward=True)
        anim = animation.FuncAnimation(fig1, updateScoreViz, frames = None, interval = dt_read_buffer_ms, blit = False,\
                fargs = (scores_arr, my_zmq_socket,buffer_size, y_lim_arr, n_ticks, prob_thr), save_count=buffer_size)

        plt.show()


def normalise_raw_data(inp_arr):
    """
    This function normalises the input raw EEG data by subtracting all the elements in the buffer 
    from its first element value
    """
    buf_len = len(inp_arr)
    first_element = len(inp_arr[0])
    # create an empty list
    normalised_arr = []
    # append by substracting the first element
    [normalised_arr.append(inp_arr[i][j]-inp_arr[i][0]) for i in range(buf_len) for j in range(first_element)]

    return np.array(normalised_arr).reshape(buf_len,first_element)


def updateDataViz(i, data_chunk, inlet_viz, EEG_live_viz, EEGutils_live, legend_names): 

    chunk, timestamps = inlet_viz.pull_chunk() # get a new data chunk
    # print(chunk)
    if(chunk): # if list not empty (new data)
        EEG_live_viz.windows = EEGutils_live.updateBuffer(chunk)  #list(channel_indices)
        # EEG_live_viz.FilterWindows(filter_type="dc_removal")
        data_chunk = EEG_live_viz.windows[0, :, :, 0] # channels, sampels
        data_chunk = np.array(normalise_raw_data(data_chunk))
        # 34 500
        for channel_idx in range(0, data_chunk.shape[0]): 
            data_chunk[channel_idx, :] = data_chunk[channel_idx, :]+channel_idx*10

    plt.cla() # clear the previous image

    plt.xlim([0, data_chunk.shape[1]]) # fix the x axis
    plt.ylim([-50,200])
    plt.title("Raw EEG data")
    plt.xlabel("Data Samples")
    plt.ylabel("EEG Voltage (uV)")
    lines = plt.plot(data_chunk[:, :].T)
    plt.legend(lines,legend_names,loc='center left', bbox_to_anchor=(1, 0.5))


def dataVisualization(names, dt_read_buffer = 0.05, n_channels = 37, buffer_size = 500): 

    # first resolve an EEG stream on the lab network
    print("looking for an LSL EEG stream for Visualization...")
    streams_viz = resolve_stream('type', 'EEG') # create data stream

    # create a new inlet to read from the stream
    inlet_viz = StreamInlet(streams_viz[0]) 
    stream_info_viz = inlet_viz.info()

    # create old_online EEG utils Object
    EEGutils_live = OnlineEEGUtils(n_channels=n_channels, n_samples=buffer_size, dt_process_data = dt_read_buffer) # use this normally stream_info.channel_count()
    EEGutils_live.printStreamMetadata(stream_info_viz) # print stream info 
    EEG_live_viz = EEGData(format = "Live", f_samp = stream_info_viz.nominal_srate())

    data_chunk = np.zeros((n_channels, buffer_size))

    fig = plt.figure()
    fig.set_size_inches(9.6, 7.4, forward=True)
    plt.ylabel("data")
    plt.xlim([0, data_chunk.shape[1]]) # fix the x axis
    plt.title("raw EEG data")
    dt_read_buffer_ms = dt_read_buffer*1000
    # update the create plot (once created) in a loop this given interval dt and plot the values 
    anim = animation.FuncAnimation(fig, updateDataViz, frames = 500, interval = dt_read_buffer_ms, blit = False, fargs = (data_chunk, inlet_viz, EEG_live_viz, EEGutils_live, names))
    plt.show()


def main():
    # eeg stream params 
    channel_names = ['F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'P5','P3', 'P1', 'PZ', 'P2', 'P4', 'P6']


    #pr_dataViz = mp.Process(target=dataVisualization, args=(channel_names,))
    pr_scoreViz = mp.Process(target=runZMQ)

    # start the processes
    #pr_dataViz.start()
    pr_scoreViz.start()



if __name__=="__main__":
    main()
