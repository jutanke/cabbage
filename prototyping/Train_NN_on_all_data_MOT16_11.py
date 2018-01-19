import matplotlib.pyplot as plt
import json
from pprint import pprint
import numpy as np
Settings = json.load(open('settings.txt'))
pprint(Settings)
import numpy as np
import sys
sys.path.append('../')
from cabbage.regression.Regression import get_W_mot16_02_dmax100
from cabbage.MultiplePeopleTracking import GraphGenerator
from cabbage.features.deepmatching import ReadOnlyDeepMatching
from cabbage.features.ReId import StoredReId, StackNet64x64, get_element
from experiments import MOT16_Experiments
from cabbage.data.video import VideoData
from time import time
from keras.applications.vgg16 import preprocess_input

print("\n")

dmax = 100

root = Settings['data_root']

mot16 = MOT16_Experiments(root)
video_name = 'MOT16-11'
X = mot16.mot16_11_X

Dt = mot16.mot16_11_detections
vd = VideoData(Dt)
Dt = vd.get_n_first_frames(100)

n, _ = Dt.shape
W, H = 64, 64

print("Dt", Dt.shape)
print("\n")

Im = np.zeros((n, H, W, 3), 'uint8')

IDS_IN_FRAME = [None] * (X.shape[0] + 1)
LAST_FRAME = X.shape[0]

for i, (frame, x, y, w, h, _) in enumerate(Dt):
    frame_i = int(frame)
    im = get_element(X[frame_i-1], (x,y,w,h), (W, H), True)
    Im[i] = im
    
    if IDS_IN_FRAME[frame_i] is None:
        IDS_IN_FRAME[frame_i] = []
    IDS_IN_FRAME[frame_i].append(i)


ALL_PAIRS = []
for frame_i, ids_in_frame in enumerate(IDS_IN_FRAME):
    if ids_in_frame is None:
        continue
    
    for i in ids_in_frame:
        for j in ids_in_frame:
            if i != j:
                ALL_PAIRS.append((i,j))
        
        for frame_j in range(frame_i + 1, min(frame_i + dmax + 1, LAST_FRAME)):
            if IDS_IN_FRAME[frame_j] is None:
                continue
            for j in IDS_IN_FRAME[frame_j]:
                ALL_PAIRS.append((i, j))
    
    print('handle frame ' + str(frame_i) + " from " + str(LAST_FRAME))
                
print(len(ALL_PAIRS))

reid = StoredReId(root, 100)

end = len(ALL_PAIRS)
for i in range(0, end, 500):
    PAIRS = np.array(ALL_PAIRS[i:i+500])
    a = PAIRS[:,0]
    b = PAIRS[:,1]
    A = preprocess_input(Im[a].astype('float64'))
    B = preprocess_input(Im[b].astype('float64'))
    X = np.concatenate([A, B], 3)    
    reid.batch_memorize(a, b, X)
    print('run batch from ' + str(i) + ' to ' + str(i+500))


reid.save('batch_mot16-11_dmax100')