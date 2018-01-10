import numpy as np
from os import makedirs, listdir, remove
from os.path import join, isfile, isdir, exists, splitext

from cabbage.features.GenerateFeatureVector import pairwise_features
from cabbage.features.ReId import StackNet64x64, get_element
from cabbage.features.deepmatching import DeepMatching
import cabbage.features.spatio as st

import cabbage.regression.LogisticRegression as LR

class Regression:

    def __init__(self, Hy, root, video_name, video, dmax, d=None,
        deep_matching_binary=None, dm_data_loc=None, bbshape=(64,64)):
        """ Regression
            root: {string} data path
            video_name: {string} name of the video
            video: {np.array} (n, w, h, 3)
            dmax: {int} max range for regression
            d: {int} range from where to switch to lifted cuts
            deep_matching_binary: {string} folder to deepmatching binary
        """
        if d is None:
            d = int(dmax/2)
        self.video_name = video_name
        self.root = root
        self.Hy = Hy
        self.data_root = join(root, 'regression_' + video_name + '_dmax_' + str(dmax))
        if not isdir(self.data_root):
            makedirs(self.data_root)
        self.dmax = dmax
        self.d = d
        self.X = video
        self.bbshape = bbshape

        self.deep_matching_binary = deep_matching_binary
        self.dm_data_loc=dm_data_loc


        #dm_only_eval = deep_matching_binary is None

        #if dm_data_loc is None:
        #    dm_data_loc = join(root, 'deep_matching')

        #self.dm = DeepMatching(deep_matching_binary,
        #    dm_data_loc, dmax, only_eval=dm_only_eval)

        #self.reid = StackNet64x64(root)

    def get_filename_thetas(self):
        """
        """
        file_name = 'theta.npy'
        return join(self.data_root, file_name)


    def get_filename_for_features(self, t):
        """ generate the filename for features
        """
        file_name = "f" + "%08d" % (t,) + '.npy'
        return join(self.data_root, file_name)


    def get_weights(self):
        """ gets the weights
        """
        fname = self.get_filename_thetas()
        if not isfile(fname):
            self.run()
        assert isfile(fname)
        W = np.load(fname)
        return W


    def run(self):
        """ run the regression
        """
        Hy = self.Hy
        print ("run")
        video_name = self.video_name

        n, _ = Hy.shape
        delta_max=self.dmax


        gen = pairwise_features(self.root,delta_max,self.deep_matching_binary,self.dm_data_loc)

        #pairwise_vectors = [[] for _ in range(delta_max)]
        #labels = [[] for _ in range(delta_max)]
        i_start, pairwise_vectors, labels = self.restore_features()


        for i in range(i_start, n):
            frame1, id1, x1, y1, w1, h1,conf1 = Hy[i]
            I1 = self.X[int(frame1-1)]
            for j in range(i+1, n):
                frame2, id2, x2, y2, w2, h2,conf2 = Hy[j]

                I2 = self.X[int(frame2-1)]
                delta = int(abs(frame2-frame1) )
                if delta >= delta_max :
                    continue

                try:
                    pair_vec = gen.get_pairwise_vector(
                        video_name ,
                        I1, I2,
                        frame1,frame2,
                        (x1, y1, w1, h1),
                        (x2, y2, w2, h2),
                        conf1,
                        conf2)

                    assert delta < delta_max, "delta too big:" + str(delta)
                    pairwise_vectors[delta].append(pair_vec)
                    labels[delta].append(1 if id1==id2 else 0)
                except:
                    print("ignore frame " + str(frame1) + " -> " + str(frame2))
                #print ("same" if id1==id2 else "different")
            print("detection: ",i," out of ",n)

            self.store_features_per_delta(i, pairwise_vectors, labels)

            if i > 0:
                self.delete_features_per_delta(i-1)


        #TODO implement Regression
        weights = []

        for i in range(delta_max):
            #X_ = np.array( pairwise_vectors[i])
            #n, count_features = X_.shape
            #X = np.ones((n, count_features+1))
            #X[:,1:] = X_
            X = np.array( pairwise_vectors[i])
            
            Y = np.array(labels[i])
            w = LR.get_params(X,Y)
            weights.append(w)

        fname = self.get_filename_thetas()
        W = np.array(weights)
        np.save(fname, W)


    def get_filenames_for_feature(self, i, delta):
        """ generate the filename for feature per delta
        """
        file_name = "i" + "%08d" % (i,) + "d%05d" % (delta,) + '.npy'
        file_name_label = 'label_' + file_name
        return join(self.data_root, file_name), join(self.data_root, file_name_label)


    def restore_features(self):
        """ attempts to restore the features
        """

        files = sorted([f for f in listdir(self.data_root) if \
            f.endswith('.npy') and f.startswith('i')])

        pairwise_vectors = [[] for _ in range(self.dmax)]
        labels = [[] for _ in range(self.dmax)]
        i = 0

        if len(files) > 0:
            f = files[-1]
            i = int(f[1:9])
            i += 1  # so we start from the NEXT hypothesis

            for f in files:
                f_label = 'label_' + f
                delta = int(f[10:15])
                assert delta <= self.dmax

                v = np.load(join(self.data_root, f))
                pairwise_vectors[delta] = v.tolist()

                l = np.load(join(self.data_root, f_label))
                labels[delta] = l.tolist()

        return i, pairwise_vectors, labels



    def delete_features_per_delta(self, i):
        """
        """
        for delta in range(self.dmax):
            fname, fname_label = self.get_filenames_for_feature(i, delta)
            if isfile(fname):
                remove(fname)
            if isfile(fname_label):
                remove(fname_label)


    def store_features_per_delta(self, i, pairwise_vectors, labels):
        """
        """
        assert len(pairwise_vectors) == self.dmax
        assert len(labels) == self.dmax

        for delta, (v,l) in enumerate(zip(pairwise_vectors, labels)):
            assert len(v) == len(l), 'Features and Labels must be same length'
            fname, fname_label = self.get_filenames_for_feature(i, delta)
            if len(v) > 0:
                v, l = np.array(v), np.array(l)
                np.save(fname, v)
                np.save(fname_label, l)



# --
class ReadOnlyRegression(Regression):
    """ allows only data-access
    """

    def __init__(self, root, video_name, dmax):
        """
        """
        self.video_name = video_name
        self.root = root
        self.data_root = join(root, 'regression_' + video_name + '_dmax_' + str(dmax))
        if not isdir(self.data_root):
            raise Exception(self.data_root + " MUST exist (dir)")

    def get_weights(self):
        """ gets the weights
        """
        fname = self.get_filename_thetas()
        if not isfile(fname):
            raise Exception(fname + ' MUST exist (file)')
        W = np.load(fname)
        return W
