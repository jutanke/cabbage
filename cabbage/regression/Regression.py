import numpy as np
from os import makedirs, listdir
from os.path import join, isfile, isdir, exists, splitext

from cabbage.features.ReId import StackNet64x64, get_element
from cabbage.features.deepmatching import DeepMatching
import cabbage.features.spatio as st


class Regression:

    def __init__(self, root, video_name, video, dmax, d=None,
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
        self.root = join(root, 'regression_' + video_name + '_dmax_' + str(dmax))
        if not isdir(self.root):
            makedirs(self.root)
        self.dmax = dmax
        self.d = d
        self.X = video
        self.bbshape = bbshape

        dm_only_eval = deep_matching_binary is None

        if dm_data_loc is None:
            dm_data_loc = join(root, 'deep_matching')

        self.dm = DeepMatching(deep_matching_binary,
            dm_data_loc, dmax, only_eval=dm_only_eval)

        self.reid = StackNet64x64(root)


    def get_bb_image(self, frame, bb):
        """ gets the transformed image for the bb from the given image

            frame: {int} frame-number starting with 1!
            bb: {aabb} (x,y,w,h)
        """
        assert frame > 0, 'frame must be positive .. ' + str(frame)
        n, _, _, _ = self.X.shape
        assert frame <= n, 'frame must be in the video boundaries ' + str((frame, n))
        i = frame - 1
        x = self.X[i]
        return get_element(x, bb, self.bbshape, force_uint=True)


    def get_filename_for_theta(self, t):
        """ generate the filename for theta_t
        """
        file_name = "w_" + str(t) + '.npy'
        return join(self.root, file_name)


    def run(self, Hy):
        """ run the regression
        """
        video_name = self.video_name
        dm = self.dm
        reid = self.reid

        #f_st = st.calculate(bb1, bb2)
        #f_dm = dm.calculate_cost(video_name, frame1, bb1, frame2, bb2)

        # get bounding box image:
        # I1 = get_bb_image(frame1, bb1)  # no need to do anything else!

        # f_reid =reid.predict(I1, I2)

        #TODO implement
