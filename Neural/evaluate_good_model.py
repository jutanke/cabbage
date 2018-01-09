import json
from pprint import pprint
Settings = json.load(open('../prototyping/settings.txt'))
pprint(Settings)
from CUHK03_Sampler import CUHK03_Sampler
from keras.callbacks import ModelCheckpoint, TerminateOnNaN
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from stacknet import get_model
from os.path import join, isfile, isdir, exists, splitext
from pak.evaluation import one_hot_classification as ohc
import numpy as np
import sys

sys.path.insert(0,"../")

from cabbage.data import ReId
from cabbage.data.MOT16Sampler import MOT16Sampler


root = Settings['data_root']
model_root = join(root, 'good_models')
filepath = join(model_root, 'stacknet64x64_84acc.h5')
#filepath = join(root, 'stacknet_64x64_model.h5')
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint, TerminateOnNaN()]

if isfile(filepath):
    model = load_model(filepath)
else:
    raise Exception("Could not find model!")

model.summary()

#sampler = ReId.DataSampler(root,64,64)
sampler = MOT16Sampler(root, (64, 64))

X, Y = sampler.get_named_batch('MOT16-02', 5, 10)
X = preprocess_input(X.astype('float64'))

Y_ = model.predict(X)

print("Y_", Y_)
print("Y", Y)

#Y_clipped = (Y_[:,0] > 0.5) * 1
#Yclipped =  (Y[:,0] > 0.5) * 1

#accuracy = np.sum( (Y_clipped == Yclipped) * 1) / len (Yclipped)

accuracy = ohc.accuracy(Y, Y_)

print("accuracy :   \t", accuracy)
