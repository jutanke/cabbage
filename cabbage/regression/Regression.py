import numpy as np
from os import makedirs, listdir
from os.path import join, isfile, isdir, exists, splitext

from cabbage.features.GenerateFeatureVector import pairwise_features
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
        self.root = root
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


    def get_filename_for_theta(self, t):
        """ generate the filename for theta_t
        """
        file_name = "w_" + str(t) + '.npy'
        return join(self.data_root, file_name)
    
    def get_weights(self, t):
        assert t < self.dmax


    def run(self, Hy):
        """ run the regression
        """
        print ("run")
        video_name = self.video_name
        
        n, _ = Hy.shape
        delta_max=self.dmax
        
        
        gen = pairwise_features(self.root,delta_max,self.deep_matching_binary,self.dm_data_loc)
        
        pairwise_vectors = [[] for _ in range(delta_max)]
        labels = [[] for _ in range(delta_max)]
        
        for i in range(n):
            frame1, id1, x1, y1, w1, h1,conf1 = Hy[i]
            I1 = self.X[int(frame1-1)]
            for j in range(i+1, n):
                frame2, id2, x2, y2, w2, h2,conf2 = Hy[j]

                I2 = self.X[int(frame2-1)]
                delta = int(abs(frame2-frame1) )
                if delta >= delta_max :
                    continue 

                pair_vec = gen.get_pairwise_vector(video_name ,I1,I2,frame1,frame2,(x1, y1, w1, h1),(x2, y2, w2, h2),conf1,conf2)
                
                assert delta < delta_max, "delta too big:" + str(delta)
                pairwise_vectors[delta].append(pair_vec)
                labels[delta].append(1 if id1==id2 else 0)
                #print ("same" if id1==id2 else "different")
            print("detection: ",i," out of ",n)
           
            
        #TODO implement Regression 
