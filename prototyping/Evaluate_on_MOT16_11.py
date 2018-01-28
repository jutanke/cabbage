import json
from pprint import pprint
import numpy as np
Settings = json.load(open('settings.txt'))
pprint(Settings)
import numpy as np
import sys
sys.path.append('../')
from cabbage.regression.Regression import get_W_mot16_02_dmax100
from cabbage.MultiplePeopleTracking import GraphGenerator, BatchGraphGenerator, AABBLookup
from cabbage.features.deepmatching import ReadOnlyDeepMatching
from cabbage.features.ReId import StoredReId, StackNet64x64, get_element
from experiments import MOT16_Experiments
from cabbage.data.video import VideoData
from time import time
root = Settings['data_root']
print("\n")

reid = StackNet64x64(root)
dm = ReadOnlyDeepMatching(root, 100)  # deep matches are set-up for 100 frames

dmax = 30


mot16 = MOT16_Experiments(root)
video_name = 'MOT16-11'
X = mot16.mot16_11_X

#Dt = mot16.mot16_11_detections
Dt = mot16.mot16_11_true_detections_no_pid
vd = VideoData(Dt)
#Dt = vd.get_n_first_frames(50)

W = get_W_mot16_02_dmax100(root)

print("\nDt", Dt.shape)
print("\n")


generator = BatchGraphGenerator(root, reid=reid, dm=dm, dmax=dmax, video_name=video_name)
generator.build(Dt, X, W, batch_size=1700)
