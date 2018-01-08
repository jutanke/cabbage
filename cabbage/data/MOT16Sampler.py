import numpy as np
from os import makedirs, listdir
from os.path import join, isfile, isdir, exists, splitext
from pak.datasets.MOT import MOT16
from pak import utils

def get_visible_pedestrains(Y_gt):
    """ return people without distractors
    """
    #Y_gt = utils.extract_eq(Y_gt, col=0, value=frame)
    Y_gt = utils.extract_eq(Y_gt, col=7, value=1)
    #Y_gt = utils.extract_eq(Y_gt, col=8, value=1)
    return Y_gt


class MOT16Sampler:
    """ Sample ids from MOT16
    """

    def __init__(self, root):
        mot16 = MOT16(root)

        for f in mot16.get_train_folders():
            X, Y_det, Y_gt = mot16.get_train(f, memmapped=True)
            Y_gt = get_visible_pedestrains(Y_gt)
            print(f + " visible: ", Y_gt.shape)

            del X
            del Y_det
            del Y_gt
