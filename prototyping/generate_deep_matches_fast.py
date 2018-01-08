import matplotlib.pyplot as plt
import numpy as np
from os import makedirs, listdir
from os.path import join, isfile, isdir, exists, splitext
from pprint import pprint
import json
Settings = json.load(open('settings.txt'))
pprint(Settings)
print("")
from pak.datasets.MOT import MOT16
from pak import utils
import sys
sys.path.append('../')

from cabbage.features.deepmatching import DeepMatching
import cabbage.features.spatio as st
from cabbage.features.ReId import StackNet64x64, get_element

root = Settings['data_root']
mot16 = MOT16(root)

delta_max = 30
dm = DeepMatching(Settings['deepmatch'], join(root, 'deep_matching'),
                 delta_max=delta_max)

VIDEO = "MOT16-11"
img_loc = mot16.get_test_imgfolder(VIDEO)

X, Y_det, Y_gt = mot16.get_train(VIDEO, memmapped=True)

dm.generate_matches(img_loc, VIDEO)