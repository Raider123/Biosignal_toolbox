import numpy as np
from biosignal_toolbox.utils import getAbsolutePath
import matplotlib.pyplot as plt

offline_res_raw = np.load(getAbsolutePath("src/JTE_Project/offline/filter_tests/highpass_offline.npy"))
online_res_raw = np.load(getAbsolutePath("src/JTE_Project/offline/filter_tests/highpass_online.npy"))

offline_res = offline_res_raw[4,:500]
online_res = online_res_raw[4,:]

print(offline_res.shape, " ", online_res.shape)



x = np.linspace(0,1, offline_res.shape[0])
y1 = offline_res
y2 = online_res
# Plot erstellen
plt.figure(figsize=(12, 6))

plt.plot(x, y1, label=f'Offline Bp', linestyle='-')
plt.plot(x, y2, label='Online Bp', linestyle='--')

#plt.ylim(-1, 200)
# Plot konfigurieren
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

corr = np.correlate(offline_res - np.mean(offline_res),
                    online_res - np.mean(online_res),
                    mode='full')

# Maximale Übereinstimmung → Verzögerung in Samples
delay_samples = np.argmax(corr) - (len(offline_res) - 1)

# Ein Sample = 20 ms
delay_ms = delay_samples * 20.0

print(f"Filterverzögerung: {delay_samples} Samples ({delay_ms:.1f} ms)")



