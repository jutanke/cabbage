import sys
from os.path import join

from cabbage.features.deepmatching import DeepMatching
import cabbage.features.spatio as st
from cabbage.features.ReId import StackNet64x64, get_element

from time import time

import json


class pairwise_features:
    """ generate pairwise features
    """
    def __init__(self,root,delta_max,deep_matching_binary=None, dm_data_loc=None,
            DM_object=None, reid_object=None):
        """ ctor
        """

        self.root = root

        dm_only_eval = deep_matching_binary is None
        if dm_data_loc is None:
            dm_data_loc = join(root, 'deep_matching')

        if DM_object is None:
            self.dm = DeepMatching(deep_matching_binary,dm_data_loc, delta_max, only_eval=dm_only_eval)
        else:
            self.dm = DM_object

        if reid_object is None:
            self.stacknet = StackNet64x64(self.root)
        else:
            self.stacknet = reid_object



    def get_pairwise_vector(self,video_name,I1, I2, frame1,frame2,bb1,bb2,conf1,conf2,
        i1=None, i2=None):
        if i1 is None:
            i1 = get_element(I1, bb1, (64,64), force_uint=True)
        if i2 is None:
            i2 = get_element(I2, bb2, (64,64), force_uint=True)

        st_cost = st.calculate(bb1, bb2)
        dm_cost = self.dm.calculate_cost(video_name, frame1, bb1, frame2, bb2)
        reid_cost = self.stacknet.predict(i1, i2)
        min_conf = min(conf1,conf2)

        return (1,st_cost , dm_cost , reid_cost , min_conf , \
                st_cost**2,st_cost * dm_cost,st_cost * reid_cost, st_cost * min_conf, \
                dm_cost**2,dm_cost * reid_cost ,dm_cost * min_conf , \
                reid_cost**2,reid_cost * min_conf , \
                min_conf**2)
