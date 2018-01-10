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
from cabbage.features.ReId import StoredReId
from experiments import MOT16_Experiments

root = Settings['data_root']

mot16 = MOT16_Experiments(root)
video_name = 'MOT16-02'
video = mot16.mot16_02_X
dmax = 100

#regression = ReadOnlyRegression(root, 'MOT16-11', dmax)
#W = regression.get_weights()
W = get_W_mot16_02_dmax100()
print(W.shape)


# set the correct DM's first!
dm = ReadOnlyDeepMatching(root, dmax)

reid = StoredReId(root, dmax)
reid.set_mot16_02_dmax100_true_predictions3105()


Dt = mot16.mot16_02_detections
Hy = mot16.mot16_11_true_detections

gg = GraphGenerator(root, video, Dt, dmax, W, video_name=video_name,
                    DM_object=dm, reid_object=reid,
                    is_memorized_reid=True)
