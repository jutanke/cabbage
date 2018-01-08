import numpy as np
from os import makedirs, listdir
from os.path import join, isfile, isdir, exists, splitext
from pak.datasets.MOT import MOT16
from pak import utils
from time import time
from skimage.transform import resize
from cabbage.data.ReId import get_positive_pairs_by_index

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
    Y_gt = utils.extract_eq(Y_gt, col=8, value=1)
    return Y_gt


class MOT16Sampler:
    """ Sample ids from MOT16
    """

    def get_all_batch(self, num_pos, num_neg):
        """
        """
        X_ = []
        Y_ = []
        for key, _ in self.pos_pairs.items():
            X, Y = self.get_named_batch(key, num_pos, num_neg)
            X_.append(X)
            Y_.append(Y)

        X = np.vstack(X_)
        Y = np.vstack(Y_)

        n = len(self.pos_pairs.items()) * (num_pos + num_neg)
        order = np.random.choice(n, size=n, replace=False)

        return X[order], Y[order]


    def get_named_batch(self, key, num_pos, num_neg):
        """ get a batch based on the video (e.g. MOT16-02)
        """
        assert key in self.lookup
        assert key in self.pos_pairs
        assert num_pos > 0
        assert num_neg > 0
        X, Y = self.lookup[key]
        pos_pairs = self.pos_pairs[key]

        pos_indx = np.random.choice(len(pos_pairs), size=num_pos, replace=False)
        sampled_pos_pairs = pos_pairs[pos_indx]
        sampled_neg_pairs = []
        assert len(X) == len(Y)
        n = len(X)
        while len(sampled_neg_pairs) < num_neg:
            a, b = np.random.choice(n, size=2, replace=False)
            if Y[a] != Y[b]:
                sampled_neg_pairs.append((a,b))
        sampled_neg_pairs = np.array(sampled_neg_pairs)

        Ap = sampled_pos_pairs[:,0]
        Bp = sampled_pos_pairs[:,1]
        An = sampled_neg_pairs[:,0]
        Bn = sampled_neg_pairs[:,1]
        X_a_pos = X[Ap]
        X_b_pos = X[Bp]
        X_a_neg = X[An]
        X_b_neg = X[Bn]

        X_a = np.concatenate([X_a_pos, X_a_neg])
        X_b = np.concatenate([X_b_pos, X_b_neg])

        X = np.concatenate((X_a, X_b), axis=3)
        Y = np.array([(1, 0)] * num_pos + [(0, 1)] * num_neg)

        return X, Y


    def __init__(self, root, shape):
        mot16 = MOT16(root, verbose=False)

        data_loc = join(root, 'mot16_data_sampler')
        if not isdir(data_loc):
            makedirs(data_loc)

        self.lookup = {}
        self.pos_pairs = {}

        for F in mot16.get_train_folders():
        #for F in ['MOT16-02']:
            start = time()
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

                for f, pid, x, y, w, h, _, _, _ in Y_gt:
                    pid = int(pid)
                    I = X[int(f)-1]  # f starts at 1, not at 0
                    try:
                        person = get_element(I, (x,y,w,h), shape)
                        _X.append(person * 255)
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

            print("finished generating X and Y for " + F)

            fPos_pairs = join(data_loc, "pos_pairs_" + F + ".npy")
            if isfile(fPos_pairs):
                print('load positive pairs from disk')
                self.pos_pairs[F] = np.load(fPos_pairs)
            else:
                print('generate positive pairs')
                pos_pairs = get_positive_pairs_by_index(_Y)
                np.save(fPos_pairs, pos_pairs)
                self.pos_pairs[F] = pos_pairs

            print("pos pairs:", self.pos_pairs[F].shape)


            end = time()
            print(F + " .. elapsed", (end-start))
