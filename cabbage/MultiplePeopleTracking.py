import numpy as np
from time import time
from os import makedirs, listdir, remove
from os.path import join, isfile, isdir, exists, splitext

class GraphGenerator:
    """
    """


    def __init__(self, root, video, detections, dmax, W, d=None, video_name=None):
        """
            root: {string} path to data root
            detections: {np.array} ([f, x, y, w, h, score], ...)
            dmax: {int}
            W: {np.array} calculated theta scores

        """
        if d is None:
            d = int(dmax/2)
        assert dmax > d
        assert W.shape[0] == dmax

        if video_name is None:
            video_name = "video" + str(time)

        self.root = root
        self.X = video
        self.dmax = dmax
        self.d = d
        self.detections = detections
        self.video_name = video_name

        data_loc = self.get_data_folder()
        if not isdir(data_loc):
            makedirs(data_loc)


    def get_data_folder(self):
        """ gets the data directory
        """
        return join(join(self.root, 'graph_generator'), self.video_name)
