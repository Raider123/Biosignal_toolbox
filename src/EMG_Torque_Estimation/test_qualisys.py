import numpy as np

from biosignal_toolbox.motion_lib import MotionData

demo_quali = MotionData(data_path='/home/dfki.uni-bremen.de/kschari/kc_ws/repos/biosignal_toolbox/data/', filename='complex_0g_set6.tsv')

# demo_quali.loadQualisysData()
print([demo_quali.channel_names.index(ch) for ch in ['w_l_x', 'w_l_y']])