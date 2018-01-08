import numpy as np
from math import sqrt

def calculate(bb1, bb2):
    """ calculate the spatio-temporal distance between two
        Axis-Aligned bounding boxes

        bb1: {aabb} (x,y,w,h)
        bb2: {aabb}
    """
    x1, y1, bb_w1, bb_h1 = bb1
    x2, y2, bb_w2, bb_h2 = bb2

    h = (bb_h1 + bb_h2) / 2.0
    f_st =  sqrt((x1 - x2)**2 + (y1 - y2)**2) / h

    if f_st <= 0:
        f_st = 1e-10  # to prevent numerical issues

    return f_st
