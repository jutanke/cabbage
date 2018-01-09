import numpy as np
from os import makedirs, listdir
from os.path import join, isfile, isdir, exists, splitext


class Regression:

    def __init__(self, root, dmax):
        """ Regression
        """
        self.root = join(root, 'regression' + str(dmax))
        if not isdir(self.root):
            makedirs(self.root)
        self.dmax = dmax


    def get_filename_for_theta(self, t):
        """ generate the filename for theta_t
        """
        file_name = "w_" + str(t) + '.npy'
        return join(self.root, file_name)


    def run(self):
        """ run the regression
        """
        #TODO implement
        pass
