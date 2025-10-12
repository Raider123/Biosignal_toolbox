import zmq
import time
import pandas as pd

emg_file = "F:/SMT_MASTERPROJEKT/biosignal_toolbox/data/jte/emg/BU62D/inactive_24072025_BU62D_0g_complex_1.txt"

df = pd.read_csv(emg_file, sep=" ", header=None)
df = df.drop(df.columns[0], axis=1)
max_rows = df.shape[0]
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
    if passed_time > 0.02:
        start_time = time.time()

        current_line = df.iloc[idx].to_frame().T.to_string(header=False, index=False)

        #print(current_line)
        socket.send_string(current_line)

        #socket.send_string('0.002229 0.001478 0.002173 -0.001169 0.002038 0.004714 -0.001207 -0.001882 0.0 62930.0')

        if idx == max_rows:
            idx = 0
        else:
            idx += 1

