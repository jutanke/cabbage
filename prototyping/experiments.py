from pppr import aabb
import numpy as np
from pak.datasets.MOT import MOT16
from pak import utils
from pppr import aabb
from time import time

# ===========================================
# Helper functions
# ===========================================

def get_visible_pedestrains(Y_gt, frame):
    Y_gt_frame1 = utils.extract_eq(Y_gt, col=0, value=frame)
    Y_gt_frame1 = utils.extract_eq(Y_gt_frame1, col=7, value=1)
    Y_gt_frame1 = utils.extract_eq(Y_gt_frame1, col=8, value=1)
    return Y_gt_frame1

def get_visible_pedestrains_det(Y_det, frame):
    Y_det_frame1 = utils.extract_eq(Y_det, col=0, value=frame)
    return Y_det_frame1

# ===========================================
# Experiments implementation
# ===========================================
verbose = True

class MOT16_Experiments:
    def __init__(self, folder):
        """ For the experiments we need MOT16-02 and 
            MOT16-11 for the analysis
            
            The detections will have the following structure:
                0: frame_nbr
                1: person id
                2: detection top-left x position
                3: detection top-left y position
                4: detection bb width
                5: detection bb height 
                6: detection output score
        """
        global verbose
        mot16 = MOT16(folder, verbose=True)
        
        mot16_02 = mot16.get_train("MOT16-02", memmapped=True)
        mot16_11 = mot16.get_train("MOT16-11", memmapped=True)
        
        true_detections_per_video = []
        for X, Y_det, Y_gt in [mot16_02, mot16_11]:
            # -- run for each video --
            # this is not the most efficient way but not important atm..
            true_detections = []
            true_detections_per_video.append(true_detections)
            frames = X.shape[0]
            TIMING_start = time()
            for frame in range(frames):
                y_gt = get_visible_pedestrains(Y_gt, frame)
                y_det = get_visible_pedestrains_det(Y_det, frame)
            
                for ped in y_det:
                    i, _,l, t, w, h, score, _, _,_ = ped
                    for ped_ in y_gt:
                        j, pid, l_gt, t_gt, w_gt, h_gt, _, _, _ = ped_
                        assert(i == j)
                        if aabb.IoU((l,t,w,h), (l_gt,t_gt,w_gt,h_gt)) > 0.5:
                            true_detections.append(
                                np.array([i, pid, l, t, w, h, score]))
            TIMING_end = time()
            if verbose:
                print("Handling " + str(frames) + " frames in " + \
                      str(TIMING_end - TIMING_start) + " seconds")
        
        self.mot16_02_true_detections = np.array(true_detections_per_video[0])
        self.mot16_11_true_detections = np.array(true_detections_per_video[1])