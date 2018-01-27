# Transform data to fit the MOTA evaluation strategy from the pak library
import numpy as np

def general_transform(X, transform_fun):
    """ general transform function
    """
    n,m = X.shape
    assert m == 6

    result = []
    for frame, pid, x, y, w, h in X:
        result.append([frame, pid, *transform_fun(x,y,w,h)])

    return np.array(result)


def aabb_to_center_point(X):
    """ Transforms the data X into data with a point centered at the aabb

        X: {np.array} [ (frame, pid, x, y, w, h)]
    """
    return general_transform(X, lambda x,y,w,h: (x + w/2, y + h/2))


def aabb_to_floor_point(X):
    """ Transforms the data X into data with a point centered at the aabb

        X: {np.array} [ (frame, pid, x, y, w, h)]
    """
    return general_transform(X, lambda x,y,w,h: (x + w/2, y + h))
