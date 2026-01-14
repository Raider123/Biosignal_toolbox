import zmq
import time
import pandas as pd
from biosignal_toolbox.utils import customWarningFormat, loadConfig, getAbsolutePath

current_weight = '0g'
current_move = 'grasp'
set_num = '2'

# Load the emg file (just for length of the file)
print(f"Session Config: Weight={current_weight}, Move={current_move}")
filepath = f"data/jte/emg/BU62D/backup/24072025_BU62D_{current_weight}_{current_move}_{set_num}.txt"
print(f"Using following {filepath}")
emg_file = getAbsolutePath(filepath)

df = pd.read_csv(emg_file, sep=" ", header=None)
df = df.drop(df.columns[0], axis=1)
max_rows = df.shape[0] - 1
idx = 0

# ZeroMQ Kontext erstellen
context = zmq.Context()

# PUB-Socket erstellen
socket = context.socket(zmq.PUB)
socket.bind("tcp://127.0.0.1:5555")

print("Publisher is active")

start_time = time.perf_counter()

running = True
# Nachrichten versenden
while running:
    passed_time = time.perf_counter() - start_time
    if passed_time > 0.001: # Every 20ms output one EMG value (mimic EMG device with 500Hz) - 0.002
        start_time = time.perf_counter()

        current_line = df.iloc[idx].to_frame().T.to_string(header=False, index=False)
        socket.send_string(current_line)

        if idx == max_rows:
            # With this the script repeats at index 0
            #idx = 0
            # With this the script ends
            running = False
        else:
            idx += 1

