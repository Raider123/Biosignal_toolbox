import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from biosignal_toolbox.utils import getAbsolutePath
from biosignal_toolbox.motion_lib import MotionData

#? create a quali motion data object
qualisys_data = MotionData(data_path= "data/quali/BU62D/", filename='24_07_2025_BU62D_0g_grasp_1.tsv')
#? calculate the torques from quali .tsv
qualisys_data.calculateTorque(body_weight_kg=80, 
                              obj_weight_g=0, 
                              subject_biological_sex="male",
                              subject_hand_length_mm=113,
                              method='com')
#? save the torques into individual .npy files
qualisys_data.saveTorques_npy(save_torques=True, save_dir="results/BU62D", joint_to_save=['all'])

#? plot front shoulder torque profile
numpy_file_path = getAbsolutePath("results/BU62D/quali_torque_elbow_24_07_2025_BU62D_0g_grasp_1.npy")
data = np.load(numpy_file_path)
plt.plot(qualisys_data.time_axis, data)
plt.grid(linestyle=':', linewidth=1.5)
plt.title('Elbow Torque')
plt.xlabel('time (sec.)')
plt.ylabel('Joint Torque (N-m)')
plt.minorticks_on()
plt.show()
