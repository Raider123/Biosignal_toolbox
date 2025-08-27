import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from biosignal_toolbox.utils import loadConfig, resolvePath, plotResults
from biosignal_toolbox.motion_lib import MotionData

#? get the path of the project root
project_root = Path(__file__).resolve().parent.parent.parent

#? create a qualisys motion data object
qualisys_data = MotionData(data_path= "data/test/", filename='complex_0g_set6.tsv')
#? calculate the torques from qualisys .tsv
qualisys_data.calculateTorque()
#? save the torques into individual .npy files
qualisys_data.saveTorques_npy(save_torques=True, save_dir="data/test", joint_to_save=['all'])

#? plot front shoulder torque profile
numpy_file_path = resolvePath("data/test/quali_torque_shoulder_front_complex_0g_set6.npy", project_root)
data = np.load(numpy_file_path)
plt.plot(qualisys_data.time_axis, data)
plt.grid(linestyle=':', linewidth=1.5)
plt.title('Front Shoulder Torque')
plt.xlabel('time (sec.)')
plt.ylabel('Joint Torque (N-m)')
plt.minorticks_on()
plt.show()
