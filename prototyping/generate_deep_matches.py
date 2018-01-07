import json
from pprint import pprint
import numpy as np
Settings = json.load(open('settings.txt'))
root = Settings['data_root']
from math import ceil, floor
pprint(Settings)
print("")
from pak.datasets.MOT import MOT16
import subprocess
from os import makedirs, listdir
from os.path import join, isfile, isdir, exists, splitext

deepmatch_loc = Settings['deepmatch']
assert(isfile(deepmatch_loc))

root = Settings['data_root']
mot16 = MOT16(root)

# ---------------------

VIDEO = "MOT16-11"
delta_max = 100

# --- start process ---

img_loc = mot16.get_test_imgfolder(VIDEO)

frames = sorted([join(img_loc, f) for f in listdir(img_loc) \
                  if f.endswith('.jpg')])

def deepmatch(img1, img2):
    args = (deepmatch_loc, img1, img2, '-downscale', '3', '-nt', '16')
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    B = np.fromstring(popen.stdout.read(), sep=' ')
    n = B.shape[0]
    assert(floor(n) == ceil(n))
    assert(floor(n/6) == ceil(n/6))
    B = B.reshape((int(n/6), 6))
    return B


# check if folder exists
folder_name = join(root, 'DM_' + VIDEO)
if isdir(folder_name):
    # recover from previous state!
    already_calc = [f for f in listdir(folder_name) if f.endswith('.npy')]
    start_i = len(already_calc)
else:
    makedirs(folder_name)
    start_i = 0

    
#TOTAL = []
for i in range(start_i, len(frames)):
    curr_frame = []
    for j in range(i, min(i+delta_max+1, len(frames))):
        print("solve " + str(i) + " -> " + str(j))
        M = deepmatch(frames[i], frames[j])
        curr_frame.append(M)
    
    fname = "f" + "%06d" % (i+1,) + '.npy'
    np.save(join(folder_name, fname), np.array(curr_frame))
