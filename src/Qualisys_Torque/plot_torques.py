import numpy as np
import matplotlib.pyplot as plt
from biosignal_toolbox.utils import getAbsolutePath
from biosignal_toolbox.motion_lib import MotionData
from os import listdir
from os.path import isfile, join

data_path = "data/jte/quali/BU62D/tsv"
joint_names = np.array(["elbow", "shoulder_front", "shoulder_side"])
qualisys_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]

for file in qualisys_files:
    qualisys_data = MotionData(data_path= data_path, filename=file)

    _, _, _, subject, weight, movement, set_no = qualisys_data.parseFilename()

    for joint in joint_names:
        numpy_file_path = getAbsolutePath(f"results/{subject}/{weight}/quali_torque_{joint}_24_07_2025_{subject}_{weight}_{movement}_{set_no}.npy")
        data = np.load(numpy_file_path)
        save_dir = getAbsolutePath(f"plots/{subject}/{weight}")
        fullpath = save_dir / f"quali_torque_{joint}_24_07_2025_{subject}_{weight}_{movement}_{set_no}.png"
        fullpath.parent.mkdir(parents=True, exist_ok=True)
        plt.plot(qualisys_data.time_axis, data)
        plt.grid(linestyle=':', linewidth=1.5)
        plt.title(subject+" "+weight+" "+movement.title()+" Movement "+joint.title()+' Torque (Set '+set_no+")")
        plt.xlabel('time (sec.)')
        plt.ylabel('Joint Torque (N-m)')
        plt.minorticks_on()
        plt.show()
        plt.close()
