import json
from pprint import pprint
import numpy as np
Settings = json.load(open('../settings.txt'))
pprint(Settings)
import sys
sys.path.insert(0,'../../')
sys.path.append('../')
# ---
from cabbage.regression.Regression import Regression
from experiments import MOT16_Experiments
from cabbage.regression.Regression import ReadOnlyRegression
from cabbage.MultiplePeopleTracking import GraphGenerator
from cabbage.features.deepmatching import ReadOnlyDeepMatching
from cabbage.features.ReId import StoredReId

root = Settings['data_root']


mot16 = MOT16_Experiments(root)

video_name = 'MOT16-02'
video = mot16.mot16_02_X
dmax = 100

# --------------
dm = ReadOnlyDeepMatching(root, dmax)
reid = StoredReId(root, dmax)
reid.set_mot16_02_dmax100_true_predictions3105()
# --------------

Hy = mot16.mot16_02_true_detections

regression = Regression(Hy, root,  video_name, video, dmax,
                        DM_object=dm, reid_object=reid,
                        is_memorized_reid=True)

#regression.restore_features()

regression.run()

W = regression.get_weights()

print(W.shape)
