import numpy as np
import matplotlib.pyplot as plt
from biosignal_toolbox.motion_lib import MotionData

demo_quali = MotionData(data_path='/home/dfki.uni-bremen.de/kschari/kc_ws/repos/biosignal_toolbox/data/', filename='complex_0g_set6.tsv')

# demo_quali.loadQualisysData()
demo_quali.calculateTorque(save_torques=True, save_path='../../data/test/elbow_torque.npy')

data = np.load('../../data/test/elbow_torque.npy')
print(data)
plt.plot(data)
plt.show()