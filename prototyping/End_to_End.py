import json
from pprint import pprint
import numpy as np
Settings = json.load(open('settings.txt'))
pprint(Settings)
import sys
sys.path.append('../')
from cabbage.regression.Regression import ReadOnlyRegression, get_W_mot16_02_dmax100
from cabbage.MultiplePeopleTracking import GraphGenerator
from cabbage.features.deepmatching import ReadOnlyDeepMatching
from cabbage.features.ReId import StoredReId, StackNet64x64
from experiments import MOT16_Experiments

root = Settings['data_root']

mot16 = MOT16_Experiments(root)
video_name = 'MOT16-11'
video = mot16.mot16_11_X
dmax = 100

W = get_W_mot16_02_dmax100(root)
print(W.shape)


# set the correct DM's first!
dm = ReadOnlyDeepMatching(root, dmax)

#reid = StoredReId(root, dmax)
#reid.set_mot16_11_dmax100_true_predictions3349()
reid = StackNet64x64(root)

#Dt = mot16.mot16_11_true_detections_no_pid
Dt = mot16.mot16_11_detections

gg = GraphGenerator(root, video, Dt, dmax, W, video_name=video_name,
                    DM_object=dm, reid_object=reid,
                    is_memorized_reid=False)
