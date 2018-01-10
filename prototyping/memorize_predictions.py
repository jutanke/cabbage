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
video_name = 'MOT16-11'
video = mot16.mot16_11_X
dmax = 100

#Dt = mot16.mot16_02_detections
Dt = mot16.mot16_11_true_detections_no_pid
#Dt = mot16.mot16_02_true_detections_no_pid

reid = StoredReId(root, dmax)

reid.memorize(Dt, video, video_name + '_dmax100')
