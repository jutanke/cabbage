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
import numpy as np
import sys

sys.path.insert(0,"../")

from cabbage.data import ReId


root = Settings['data_root']
filepath = join(root, 'stacknet_model.h5')
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint, TerminateOnNaN()]

if isfile(filepath):
    model = load_model(filepath)
else:
    raise Exception("Could not find model!")

model.summary()

sampler = ReId.DataSampler(root,112,112)
X, Y = sampler.get_test_batch(1000, 1000)
X = preprocess_input(X.astype('float64'))

Y_ = model.predict(X)

Y_clipped = (Y_[:,0] > 0.5) * 1
Yclipped =  (Y[:,0] > 0.5) * 1

accuracy = np.sum( (Y_clipped == Yclipped) * 1) / len (Yclipped)


print("accuracy :   \t", accuracy)

