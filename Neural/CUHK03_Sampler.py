import numpy as np
from os import makedirs, listdir
from os.path import join, isfile, isdir, exists, splitext
from pak import utils
import json
from pprint import pprint
from pak.datasets.CUHK03 import cuhk03

class CUHK03_Sampler:

    def __init__(self, target_w=112, target_h=112, T=100, settings_url='../prototyping/settings.txt'):
        """ctor
        """
        Settings = json.load(open(settings_url))
        pprint(Settings)
        root = Settings['data_root']
        self.root = root
        cuhk = cuhk03(root, target_w=target_w, target_h=target_h)
        X, Y = cuhk.get_labeled()

        self.X = X
        self.Y = Y

        index_test, index_train = [], []
        for i, y in enumerate(Y):
            if y <= T:
                index_test.append(i)
            else:
                index_train.append(i)

        self.index_test = np.array(index_test)
        self.index_train = np.array(index_train)

        # test pairs
        fpairs_test = self.get_pos_pairs_file_name('test')
        if isfile(fpairs_test):
            self.test_pos_pair = np.load(fpairs_test)
        else:
            self.test_pos_pair = []
            for i in index_test:
                for j in index_test:
                    if Y[i] == Y[j]:
                        self.test_pos_pair.append((i, j))
            self.test_pos_pair = np.array(self.test_pos_pair)
            np.save(fpairs_test, self.test_pos_pair)
        print("positive test pairs:", len(self.test_pos_pair))

        # train pairs
        fpairs_train = self.get_pos_pairs_file_name('train')
        if isfile(fpairs_train):
            self.train_pos_pair = np.load(fpairs_train)
        else:
            self.train_pos_pair = []
            for i in index_train:
                for j in index_train:
                    if Y[i] == Y[j]:
                        self.train_pos_pair.append((i, j))
            self.train_pos_pair = np.array(self.train_pos_pair)
            np.save(fpairs_train, self.train_pos_pair)
        print("positive train pairs:", len(self.train_pos_pair))


    def get_pos_pairs_file_name(self, folder):
        root = self.root
        file_name = 'cuhk03_sampler_' + folder + '.npy'
        return join(root, file_name)


    def get_test_batch(self, num_pos, num_neg):
        return self.get_batch(num_pos, num_neg,
                              self.test_pos_pair, self.index_test)

    def get_train_batch(self, num_pos, num_neg):
        return self.get_batch(num_pos, num_neg,
                              self.train_pos_pair, self.index_train)


    def get_batch(self, num_pos, num_neg, pos_pairs, valid_indexes):
        """ generic batch function
        """
        pos_indx = np.random.choice(len(pos_pairs), size=num_pos, replace=False)
        sampled_pos_pairs = pos_pairs[pos_indx]
        sampled_neg_pairs = []
        Y = self.Y
        X = self.X

        n_all_indexes = len(valid_indexes)
        while len(sampled_neg_pairs) < num_neg:
            a, b = np.random.choice(
                n_all_indexes, size=2, replace=False)
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

        n = num_pos + num_neg
        order = np.random.choice(n, size=n, replace=False)

        return X[order], Y[order]


# ~~~~~~~~~~~~~~~~~~~~~~~~~
