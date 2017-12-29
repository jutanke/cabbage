from pppr import aabb
import numpy as np
from pak.datasets.MOT import MOT16
from pak import utils
from pppr import aabb
from time import time
from cselect import color as cs

# ===========================================
# Helper functions
# ===========================================

def get_visible_pedestrains(Y_gt, frame):
    Y_gt_frame1 = utils.extract_eq(Y_gt, col=0, value=frame)
    #Y_gt_frame1 = utils.extract_eq(Y_gt_frame1, col=7, value=1)
    #Y_gt_frame1 = utils.extract_eq(Y_gt_frame1, col=8, value=1)
    return Y_gt_frame1

def get_visible_pedestrains_det(Y_det, frame):
    Y_det_frame1 = utils.extract_eq(Y_det, col=0, value=frame)
    return Y_det_frame1

def get_center(d):
    """ full detection has 7 parameters:
        full_detection: (frame, pid, x, y, w, h, score)
    """
    x, y, w, h = d[2], d[3], d[4], d[5]
    return x+w/2, y+h/2
    

# ===========================================
# Experiments implementation
# ===========================================
verbose = False

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
        mot16 = MOT16(folder, verbose=verbose)
        
        mot16_02 = mot16.get_train("MOT16-02", memmapped=True)
        mot16_11 = mot16.get_train("MOT16-11", memmapped=True)
        
        self.mot16_02_X = mot16_02[0]
        self.mot16_11_X = mot16_11[0]
        
        gt_per_video = []
        true_detections_per_video = []
        color_lookups_per_video = []
        for X, Y_det, Y_gt in [mot16_02, mot16_11]:
            # --- run for each video ---
            # this is not the most efficient way but not important atm..
            Y_gt = MOT16.simplify_gt(Y_gt)
            gt_bbs = []
            true_detections = []
            true_detections_per_video.append(true_detections)
            gt_per_video.append(gt_bbs)
            frames = X.shape[0]
            TIMING_start = time()
            for frame in range(1, frames+1):
                y_gt = get_visible_pedestrains(Y_gt, frame)
                y_det = get_visible_pedestrains_det(Y_det, frame)
            
                for ped_ in y_gt:
                    j, pid, l_gt, t_gt, w_gt, h_gt = ped_
                    gt_bbs.append((j, pid, l_gt, t_gt, w_gt, h_gt))
            
                for ped in y_det:
                    i, _,l, t, w, h, score, _, _,_ = ped
                    for ped_ in y_gt:
                        j, pid, l_gt, t_gt, w_gt, h_gt = ped_
                        assert(i == j)
                        if aabb.IoU((l,t,w,h), (l_gt,t_gt,w_gt,h_gt)) > 0.5:
                            true_detections.append(
                                np.array([i, pid, l, t, w, h, score]))
            TIMING_end = time()
            if verbose:
                print("Handling " + str(frames) + " frames in " + \
                      str(TIMING_end - TIMING_start) + " seconds")
        
            # --- figure out coloring ---
            Y = np.array(true_detections)
            U = np.unique(Y[:,1])
            Color_lookup = {}
            Colors = cs.lincolor(len(U), random_sat=True, random_val=True)
            #Colors = cs.poisson_disc_sampling_Lab(len(U))
            Colors = np.array(Colors, 'float32') / 255
            for u,c in zip(U, Colors):
                Color_lookup[u] = c
            color_lookups_per_video.append(Color_lookup)
        
        self.mot16_02_gt_bbs = np.array(gt_per_video[0])
        self.mot16_11_gt_bbs = np.array(gt_per_video[1])
        
        self.mot16_02_true_detections = np.array(true_detections_per_video[0])
        self.mot16_11_true_detections = np.array(true_detections_per_video[1])
        
        self.mot16_02_color_lookup = color_lookups_per_video[0]
        self.mot16_11_color_lookup = color_lookups_per_video[1]
    
    
    def get_MOT16_02_gt_trajectories(self):
        return self.get_detections_as_trajectories(
            self.mot16_02_gt_bbs)
    
    def get_MOT16_02_trajectories(self):
        return self.get_detections_as_trajectories(
            self.mot16_02_true_detections)
    
    
    def get_detections_as_trajectories(self, true_detections):
        trajectories = []
        for d in true_detections:
            x,y = get_center(d)
            frame = d[0]
            pid = d[1]
            trajectories.append((frame, pid, x, y))
        return np.array(trajectories)
    
    
    def plot_frame_MOT16_02(self, ax, frame, with_gt=False):
        self.plot_frame(ax, 
                        self.mot16_02_X, 
                        self.mot16_02_true_detections,
                        self.mot16_02_color_lookup,
                        frame, with_gt, self.mot16_02_gt_bbs)
        
    
    def plot_frame(self, ax, X, true_detections, id_colors, frame, 
                   with_gt, gt_bbs):
        """ plots the frame with its true detections
        """
        Y = utils.extract_eq(true_detections, col=0, value=frame)
        X = X[frame]
        
        ax.imshow(X)
        
        for _, pid, x, y, w, h, score in Y:
            ax.text(x, y, str(int(pid)), color='white', fontsize=17,
                   bbox={'facecolor': 'red', 'alpha': 0.5})
            
            bbX, bbY = utils.bb_to_plt_plot(x, y, w, h)
            ax.plot(bbX, bbY, linewidth=2, color=id_colors[pid])
            
        if with_gt:
            Y = utils.extract_eq(gt_bbs, col=0, value=frame)
            for _, pid, x, y, w, h in Y:
                bbX, bbY = utils.bb_to_plt_plot(x, y, w, h)
                ax.plot(bbX, bbY, 'g--', linewidth=4)
            
# -------------