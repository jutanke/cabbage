import numpy as np
from os import makedirs, listdir
from os.path import join, isfile, isdir, exists, splitext
from skimage.transform import resize
import urllib.request
import shutil
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input


def get_element(X, bb, shape, force_uint=False):
    """ returns the bounding box area from the image X
    """
    x,y,w,h = bb
    x,y,w,h = int(x),int(y),int(w),int(h)
    if force_uint:
        I = resize(X[y:y+h,x:x+w], shape, mode='constant').copy()
        if I.dtype != np.uint8:
            if np.max(I) < 1.01:
                I *= 255
                I = I.astype('uint8')
        return I
    else:
        return resize(X[y:y+h,x:x+w], shape, mode='constant')


class ReId:
    """ Generic ReId class
    """

    def __init__(self, root, verbose):
        self.root = join(root, 'reid_models')
        if not isdir(self.root):
            makedirs(self.root)
        self.verbose = verbose
        self.model = None


    def predict(self, X):
        """ uses the model to predict the data is the same
        """
        Y = self.model.predict(X)
        return Y[0][0]


    def load_model(self, model_name, model_url):
        """ loads a model if it is not found
        """

        fname = join(self.root, model_name)
        if not isfile(fname):
            if self.verbose:
                print("Could not find " + fname + ".. attempt download")
            with urllib.request.urlopen(model_url) as res, open(fname, 'wb') as f:
                shutil.copyfileobj(res, f)
            if self.verbose:
                print("Download complete.. model: " + fname)
        elif self.verbose:
            print("Found model " + fname + "! :)")


        model = load_model(fname)
        self.model = model

class StoredReId(ReId):
    """ Stores the exact prediction
    """

    def __init__(self, root, dmax):
        model_name = 'stacknet64x64_84_BOTH.h5'
        model_url = 'http://188.138.127.15:81/models/stacknet64x64_84_BOTH.h5'
        ReId.__init__(self, root, True)
        self.dmax = dmax
        self.load_model(model_name, model_url)
        self.Broken_pair = None
        self.Prediction = None


    def create_key(self, i, j):
        return str(i) + ':' + str(j)


    def get_predictions_file(self, name):
        file_name = 'predict_' + name + '.npy'
        return join(self.root, file_name)


    def get_broken_file(self, name):
        file_name = 'broken_' + name + '.npy'
        return join(self.root, file_name)


    def load_memory(self, pred_file, broken_file):
        self.Broken_pair = np.load(broken_file)
        self.Prediction = np.load(pred_file).item()


    def predict(self, i, j):
        assert self.Prediction is not None
        assert self.Broken_pair is not None
        key = self.create_key(i, j)
        if key in self.Broken_pair:
            return 0
        elif key in self.Prediction:
            return self.Prediction[key]
        else:
            raise Exception('Could not find pair ' + key)


    def memorize(self, Dt, X, name):
        """
            Dt: {np.array} [(frame, x, y, w, h, score), ..]
            X: {np.array} (n, w, h, 3) video
        """
        n, _ = Dt.shape
        dmax = self.dmax


        Broken_pair = []
        Prediction = {}

        for i in range(n):
            frame1, x,y,w,h, _ = Dt[i]
            bb1 = (x,y,w,h)
            I1 = X[int(frame1-1)]

            for j in range(i+1, n):
                frame2, x,y,w,h,_ = Dt[j]
                delta = int(abs(frame2-frame1) )
                if delta < dmax:
                    bb2 = (x,y,w,h)
                    I2 = X[int(frame2-1)]
                    keyA, keyB = self.create_key(i, j), self.create_key(j, i)
                    assert keyA not in Prediction and keyB not in Prediction
                    try:
                        a = get_element(I1, bb1, (64,64), force_uint=True)
                        b = get_element(I2, bb2, (64,64), force_uint=True)
                        pred = StackNet64x64.predict(self, a, b)
                        Prediction[keyA] = pred
                        Prediction[keyB] = pred
                    except:
                        Broken_pair.append(keyA)
                        Broken_pair.append(keyB)

            print('handled ' + str(i) + " out of " + str(n))

        fname = self.get_predictions_file(name)
        fname_broken = self.get_broken_file(name)
        self.Broken_pair = set(Broken_pair)
        self.Prediction = Prediction

        np.save(fname, Prediction)
        np.save(fname_broken, self.Broken_pair)



class StackNet64x64(ReId):
    """ StackNet for 64x64x3 images using VGG16
    """

    def __init__(self, root, verbose=True):
        ReId.__init__(self, root, verbose)
        model_name = 'stacknet64x64_84acc.h5'
        #model_name = 'stacknet64x64_cheat.h5'
        #url = 'http://188.138.127.15:81/models/stacknet64x64_cheating.h5'
        #url = 'http://188.138.127.15:81/models/stacknet64x64_77acc.h5'
        url = 'http://188.138.127.15:81/models/stacknet64x64_84acc.h5'
        self.load_model(model_name, url)


    def predict(self, A, B):
        """ uses the model to predict if A and B are the same
        """
        w1,h1,c1 = A.shape
        w2,h2,c2 = B.shape
        assert w1 == w2 and h1 == h2 and c1 == c2
        assert w1 == 64 and h1 == 64 and c1 == 3
        assert np.max(A) > 1 and np.max(B) > 1

        X = np.concatenate([A, B], axis=2)
        X = np.expand_dims(X, axis=0)
        X = preprocess_input(X.astype('float64'))

        return ReId.predict(self, X)
