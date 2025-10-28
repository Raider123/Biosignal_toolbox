import numpy as np
from biosignal_toolbox.utils import getAbsolutePath
import matplotlib.pyplot as plt

all_preds = np.load(getAbsolutePath("src/JTE_Project/offline/saved_online_models/test/all_predictions.npy"))
all_times = np.load(getAbsolutePath("src/JTE_Project/offline/saved_online_models/test/all_times.npy"))
all_torques = np.load(getAbsolutePath("src/JTE_Project/offline/saved_online_models/test/all_torques.npy"))

print(all_preds.shape, " ", all_times.shape, ' ', all_torques.shape)

x = np.linspace(0,10,919)
y1 = all_preds
y2 = all_torques
# Plot erstellen
plt.figure(figsize=(12, 6))

plt.plot(x, y1[:, 0], label=f'y1', linestyle='-')
plt.plot(x, y2[:, 1], label=f'y2', linestyle='--')

#plt.ylim(-1, 200)
# Plot konfigurieren
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()