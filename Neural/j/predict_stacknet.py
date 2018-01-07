import json
from pprint import pprint
Settings = json.load(open('../../prototyping/settings.txt'))
pprint(Settings)
import sys
sys.path.append('../../')
from keras.callbacks import ModelCheckpoint, TerminateOnNaN
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from stacknet import get_model
from os.path import join, isfile, isdir, exists, splitext
import numpy as np

from cabbage.data import ReId

root = Settings['data_root']
filepath = join(root, 'stacknet_64x64_model.h5')
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint, TerminateOnNaN()]

if isfile(filepath):
    model = load_model(filepath)
else:
    raise Exception("Could not find model!")

model.summary()

sampler = ReId.DataSampler(root, 64, 64)

X, Y = sampler.get_train_batch(3, 3)
X = preprocess_input(X.astype('float64'))

Y_ = model.predict(X)

print("Y\t", np.squeeze(Y))
print("Y_hat\t", np.squeeze(Y_))
