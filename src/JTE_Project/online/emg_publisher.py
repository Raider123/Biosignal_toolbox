import zmq
import time
import pandas as pd
from biosignal_toolbox.utils import customWarningFormat, loadConfig, getAbsolutePath


emg_file = getAbsolutePath("data/jte/emg/BU62D/publisher.txt") #24072025_BU62D_complex_1100g_set1

df = pd.read_csv(emg_file, sep=" ", header=None)
df = df.drop(df.columns[0], axis=1)
max_rows = df.shape[0] - 1
idx = 0 # 47120 (max)

# ZeroMQ Kontext erstellen
context = zmq.Context()

# PUB-Socket erstellen
socket = context.socket(zmq.PUB)
socket.bind("tcp://127.0.0.1:5555")

print("Publisher is active")

start_time = time.time()

# Nachrichten versenden
while True:
    passed_time = time.time() - start_time
    if passed_time > 0.002: # Every 20ms output one EMG value (mimic EMG device with 500Hz)
        start_time = time.time()

        current_line = df.iloc[idx].to_frame().T.to_string(header=False, index=False)
        #print(current_line)
        socket.send_string(current_line)

        if idx == max_rows:
            idx = 0
        else:
            idx += 1

