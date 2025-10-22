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

################################
variance_raw_offline = np.load(getAbsolutePath("src/JTE_Project/offline/filter_tests/variance_offline.npy"))
variance_raw_online = np.load(getAbsolutePath("src/JTE_Project/offline/filter_tests/variance_online.npy"))

variance_online = variance_raw_online[4,:]
variance_offline = variance_raw_offline[4,:variance_raw_online.shape[1]]

print(variance_offline.shape, " ", variance_online.shape)

x = np.linspace(0,1, variance_offline.shape[0])
y3 = variance_offline
y4 = variance_online
# Plot erstellen
plt.figure(figsize=(12, 6))

plt.plot(x, y3, label='Offline Variance', linestyle='-')
plt.plot(x, y4, label='Online Variance', linestyle='-')

# Plot konfigurieren
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

################################
mvc_raw_offline = np.load(getAbsolutePath("src/JTE_Project/offline/filter_tests/normalization_offline.npy"))
mvc_raw_online = np.load(getAbsolutePath("src/JTE_Project/offline/filter_tests/normalization_online.npy"))

mvc_online = mvc_raw_online[4,:]
mvc_offline = mvc_raw_offline[4,:mvc_raw_online.shape[1]]


print(mvc_offline.shape, " ", mvc_online.shape)

x = np.linspace(0,1, mvc_online.shape[0])
y5 = mvc_online
y6 = mvc_offline
# Plot erstellen
plt.figure(figsize=(12, 6))

#plt.plot(x, y5, label='MVC Online', linestyle='-')
plt.plot(x, y6, label='MVC Offline', linestyle='-')

# Plot konfigurieren
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()