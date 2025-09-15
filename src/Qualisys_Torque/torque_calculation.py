from biosignal_toolbox.motion_lib import MotionData
from os import listdir
from os.path import isfile, join
from biosignal_toolbox.utils import loadConfig

# ! load config file
from types import SimpleNamespace

config_filename = 'new_jte.yaml'

def namespace_to_dict(ns):
    if isinstance(ns, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
    elif isinstance(ns, dict):
        return {k: namespace_to_dict(v) for k, v in ns.items()}
    else:
        return ns

config_param = namespace_to_dict(loadConfig(filename=config_filename))

data_path = "data/quali/BU62D/"

qualisys_files = [f for f in listdir(data_path) if (isfile(join(data_path, f)))]

for file in qualisys_files:
    qualisys_data = MotionData(data_path= data_path, filename=file)

    _, _, _, subject, obj_weight, movement, set_no = qualisys_data.parseFilename()

    qualisys_data.calculateTorque(body_weight_kg=config_param['data_param']['subject_weight'], obj_weight_g=int(obj_weight.strip('g')))
    qualisys_data.saveTorques_npy(save_torques=True, save_dir=f"results/{subject}/{obj_weight}", joint_to_save=['all'])
