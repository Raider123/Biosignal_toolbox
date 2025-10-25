import numpy as np
import time
import os

# Parameter
n_channels = 4
buffer_len = 4          # Anzahl der gespeicherten letzten Werte
update_interval = 3.0   # Sekunden zwischen Updates

# Datenpuffer initialisieren
data_buffer = np.ones((n_channels, buffer_len)) * 10

print("Starte Ringpuffer... (alle 3 Sekunden Update)\n")

while True:
    # Neue Messwerte (ein neuer Wert pro Kanal)
    new_values = 10

    # Älteste Werte entfernen (nach links schieben)
    data_buffer[:, :-1] = data_buffer[:, 1:]

    # Neue Werte ans Ende schreiben
    data_buffer[:, -1] = new_values

    data_buffer[:,-1] /= 2

    print("---------------------------------")
    # Ausgabe der Kanäle untereinander
    for i in range(n_channels):
        vals = " ".join(f"{int(v):3d}" for v in data_buffer[i])
        print(f"Kanal {i+1}: {vals}")

    # 3 Sekunden warten
    time.sleep(update_interval)
