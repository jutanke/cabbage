import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import json
from pprint import pprint
import numpy as np
Settings = json.load(open('settings.txt'))
from cabbage.features.ReId import StoredReId
pprint(Settings)
import numpy as np
import sys
sys.path.append('../')
from cabbage.regression.Regression import ReadOnlyRegression
from cabbage.MultiplePeopleTracking import GraphGenerator
from experiments import MOT16_Experiments
root = Settings['data_root']

mot16 = MOT16_Experiments(root)
video_name = 'MOT16-02'
video = mot16.mot16_02_X
dmax = 5

#Dt = mot16.mot16_02_detections
Dt = mot16.mot16_02_true_detections_no_pid

reid = StoredReId(root, 5)

reid.memorize(Dt, video, video_name)
