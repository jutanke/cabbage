import numpy as np
from time import time
from os import makedirs, listdir, remove
from os.path import join, isfile, isdir, exists, splitext
from cabbage.features.GenerateFeatureVector import pairwise_features

from time import time

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

        n, _ = self.detections.shape
        n= 100
        edges = []
        lifted_edges = []

        ALL_EDGES = []

        gen = pairwise_features(self.root,dmax)

        #for i, entry in enumerate(self.detections):
        for i in range(n):
            frame1, x1, y1, w1, h1,conf1 = detections[i]
            I1 = self.X[int(frame1-1)]
            for j in range(i+1, n):
                frame2, x2, y2, w2, h2,conf2 = detections[j]
                delta = int(abs(frame2-frame1) )
                if delta >= dmax :
                    continue

                I2 = self.X[int(frame2-1)]

                try:
                    vec = gen.get_pairwise_vector(
                            video_name ,
                            I1, I2,
                            frame1,frame2,
                            (x1, y1, w1, h1),
                            (x2, y2, w2, h2),
                            conf1,
                            conf2)

                    cost = -1*(W[delta]@np.array(vec))

                    if delta > d:
                        if (cost > 0):
                            # lifted edge
                            lifted_edges.append((i,j,cost))
                    else:
                        # normal edge
                        edges.append((i,j,cost))

                except:
                    print("ignore frame " + str(frame1) + " -> " + str(frame2))
                #print('cost:', cost)
                #cost = 10 if pid == pid_o else -1
            print("edges for detection: ",i," out of ",n)


        edges = np.array(edges)
        lifted_edges = np.array(lifted_edges)

        print('Edges', edges.shape)
        print('Lifted Edges', lifted_edges.shape)

        fmt = '%d %d %f'

        np.savetxt('edges.txt', edges, delimiter=';', fmt=fmt)
        np.savetxt('lifted_edges.txt', lifted_edges, delimiter=';', fmt=fmt)


    def get_data_folder(self):
        """ gets the data directory
        """
        return join(join(self.root, 'graph_generator'), self.video_name)
