import numpy as np
from os import makedirs, listdir
from os.path import join, isfile, isdir, exists, splitext
from pak.datasets.MOT import MOT16
from pak import utils
from time import time
from skimage.transform import resize

def get_element(X, bb, shape):
    """ returns the bounding box area from the image X
    """
    x,y,w,h = bb
    x,y,w,h = int(x),int(y),int(w),int(h)
    return resize(X[y:y+h,x:x+w], shape, mode='constant')


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

    def __init__(self, root, shape):
        mot16 = MOT16(root, verbose=False)

        data_loc = join(root, 'mot16_data_sampler')
        if not isdir(data_loc):
            makedirs(data_loc)

        self.lookup = {}

        start = time()
        for F in mot16.get_train_folders():

            fX = join(data_loc, 'X_' + F + '.npy')
            fY = join(data_loc, 'Y_' + F + '.npy')

            X_is_loaded, Y_is_loaded = False, False
            if isfile(fX):
                _X = np.load(fX)
                X_is_loaded = True

            if isfile(fY):
                _Y = np.load(fY)
                Y_is_loaded = True

            if Y_is_loaded and X_is_loaded:
                self.lookup[F] = (_X, _Y)
            else:
                X, Y_det, Y_gt = mot16.get_train(F, memmapped=True)
                Y_gt = get_visible_pedestrains(Y_gt)

                _X = []
                _Y = []
                for i, (f, pid, x, y, w, h, _, _, _) in enumerate(Y_gt):
                    pid = int(pid)
                    I = X[int(f)-1]  # f starts at 1, not at 0
                    try:
                        person = get_element(I, (x,y,w,h), shape)
                        _X.append(person)
                        _Y.append(pid)
                    except:
                        print("skip in " + F, (x,y,w,h))
                assert len(_X) == len(_Y)
                assert len(_X) > 0

                _X = np.array(_X, 'uint8')
                _Y = np.array(_Y, 'int32')
                if not X_is_loaded:
                    np.save(fX, _X)

                if not Y_is_loaded:
                    np.save(fY, _Y)

                self.lookup[F] = (_X, _Y)

                del X  # free memory
                del Y_det
                del Y_gt

            end = time()
            print(F + " .. elapsed", (end-start))
