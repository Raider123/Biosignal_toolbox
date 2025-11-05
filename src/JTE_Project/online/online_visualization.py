import numpy as np
from biosignal_toolbox.utils import getAbsolutePath
import matplotlib.pyplot as plt

all_preds = np.load(getAbsolutePath("src/JTE_Project/offline/saved_online_models/test/all_predictions.npy"))
all_times = np.load(getAbsolutePath("src/JTE_Project/offline/saved_online_models/test/all_times.npy"))
all_torques = np.load(getAbsolutePath("src/JTE_Project/offline/saved_online_models/test/all_torques.npy"))

print(all_preds.shape, " ", all_times.shape, ' ', all_torques.shape)

section = 600

x = np.linspace(0,1,section)
y1 = all_preds[:section,:]
y2 = all_torques[:section,:]
# Plot erstellen
plt.figure(figsize=(12, 6))

plt.plot(x, y1[:, 2], label=f'y1', linestyle='-')
plt.plot(x, y2[:, 2], label=f'y2', linestyle='--')

#plt.ylim(-1, 200)
# Plot konfigurieren
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()