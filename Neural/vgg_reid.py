import os
import urllib.request
import shutil
from os import makedirs, listdir
from os.path import join, isfile, isdir, exists, splitext

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.misc 
import scipy.io


def debug(txt):
    if __name__ == '__main__' or False:
        print(txt)

class vgg_reid:
    """ vgg reid network
    """
    
    def __init__(self):
        """ ctor
        """
        cwd = os.getcwd()
        self.VGG_PATH = cwd + "/weights/imagenet-vgg-verydeep-19.mat"
        
        if not isfile(self.VGG_PATH):
            if not exists(cwd + "/weights"):
                makedirs(cwd + "/weights")
            url = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'
            with urllib.request.urlopen(url) as res, open(self.VGG_PATH, 'wb') as f:
                print ("Download . . . ")
                shutil.copyfileobj(res, f)
        
        print ("Packages loaded.")
    

    def debug(self):
        print('debugging')

    
    def get_stacked_reid(self,image_size):
        """ stuff
        """
        layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
        )
        data_path = self.VGG_PATH
        data = scipy.io.loadmat(data_path)
        mean = data['normalization'][0][0][0]
        mean_pixel = np.mean(mean, axis=(0, 1))
        weights = data['layers'][0]
    
        net = {}
        
        img_width , img_hight = image_size
        img_placeholder = tf.placeholder(tf.float32, shape=(None, img_width, img_hight, 6))
        current = img_placeholder
        for i, name in enumerate(layers):
            kind = name[:4]
            debug(name)
            if name == 'conv1_1':
                kernels, bias = weights[i][0][0][0][0]
                kernels = np.concatenate((kernels,kernels),axis=2)
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                bias = bias.reshape(-1)
                
                current = self._conv_layer(current, kernels, bias)
            elif kind == 'conv':
                kernels, bias = weights[i][0][0][0][0]
                # matconvnet: weights are [width, height, in_channels, out_channels]
                # tensorflow: weights are [height, width, in_channels, out_channels]
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                bias = bias.reshape(-1)
                current = self._conv_layer(current, kernels, bias)
            elif kind == 'relu':
                current = tf.nn.relu(current)
            elif kind == 'pool':
                current = self._pool_layer(current)
            net[name] = current
            
        assert len(net) == len(layers)
        
        _ ,h,w,z = current.shape
        current = tf.reshape(current, [-1,int(h*w*z)])
        net["flat"] = current
        current = tf.layers.dense(current,1024)
        net["fc6"] = current
        current = tf.nn.relu(current)
        net["relu6"]=current
        
        current = tf.layers.dense(current,2)
        net["fc7"] = current
        cuurent = tf.nn.softmax(current)
        net["out"] = current
        return net,img_placeholder, mean_pixel
   
    
    # -- private functions --
    
    def _private_func(self):
        pass
    
    
    def _conv_layer(self,input, weights, bias):
        debug (weights.shape)
        debug (bias.shape)
        conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
                padding='SAME')
        return tf.nn.bias_add(conv, bias)
    def _pool_layer(self,input):
        return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                padding='SAME')
 
# --------------------------    
if __name__ == '__main__':
    print('interactive mode')
    
    nn = vgg_reid()
    nn.debug()
    nn.get_stacked_reid()
    
    