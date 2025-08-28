from biosignal_toolbox.motion_lib import MotionData
from os import listdir
from os.path import isfile, join


data_path = "data/qualisys/WW06D/"

qualisys_files = [f for f in listdir(data_path) if (isfile(join(data_path, f)))]

for file in qualisys_files:
    qualisys_data = MotionData(data_path= data_path, filename=file)

    _, _, _, subject, weight, movement, set_no = qualisys_data.parseFilename()

    qualisys_data.calculateTorque()
    qualisys_data.saveTorques_npy(save_torques=True, save_dir=f"results/{subject}/{weight}", joint_to_save=['all'])

