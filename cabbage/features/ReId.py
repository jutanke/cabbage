import numpy as np
from os import makedirs, listdir
from os.path import join, isfile, isdir, exists, splitext
from skimage.transform import resize
import urllib.request
import shutil
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input


def get_element(X, bb, shape):
    """ returns the bounding box area from the image X
    """
    x,y,w,h = bb
    x,y,w,h = int(x),int(y),int(w),int(h)
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
        return Y
        #return Y[0][0]


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


class StackNet64x64(ReId):
    """ StackNet for 64x64x3 images using VGG16
    """

    def __init__(self, root, verbose=True):
        ReId.__init__(self, root, verbose)
        model_name = 'stacknet64x64_84acc.h5'
        url = 'http://188.138.127.15:81/models/stacknet64x64_77acc.h5'
        #url = 'http://188.138.127.15:81/models/stacknet64x64_84acc.h5'
        self.load_model(model_name, url)


    def predict(self, A, B):
        """ uses the model to predict if A and B are the same
        """
        w1,h1,c1 = A.shape
        w2,h2,c2 = B.shape
        assert w1 == w2 and h1 == h2 and c1 == c2
        assert w1 == 64 and h1 == 64 and c1 == 3

        X = np.concatenate([A, B], axis=2)
        X = np.expand_dims(X, axis=0)
        X = preprocess_input(X.astype('float64'))

        return ReId.predict(self, X)


def predict_stacknet64x64(A1, A2):
    """ predicts the probability that bb1 and bb2 are the
        same image in X

        A1: {image}
        A2: {image}
    """


    x,y,w,h = bb1
