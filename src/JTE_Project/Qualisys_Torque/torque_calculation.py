from biosignal_toolbox.motion_lib import MotionData
from os import listdir
from os.path import isfile, join


data_path = "data/quali/WW06D/"

qualisys_files = [f for f in listdir(data_path) if (isfile(join(data_path, f)))]

for file in qualisys_files:
    qualisys_data = MotionData(data_path= data_path, filename=file)

    _, _, _, subject, obj_weight, movement, set_no = qualisys_data.parseFilename()

    qualisys_data.calculateTorque(body_weight_kg=config_param['data_param']['subject_weight'], obj_weight_g=int(obj_weight.strip('g')),
                                  subject_biological_sex=config_param['data_param']['subject_bio_sex'], subject_hand_length_mm=config_param['data_param']['subject_hand_len'],
                                  method='com')
    qualisys_data.saveTorques_npy(save_torques=True, save_dir=f"results/{subject}/{obj_weight}", joints_to_save=['all'])

