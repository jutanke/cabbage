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

from cabbage.data import ReId
from cabbage.data.MOT16Sampler import MOT16Sampler

import numpy as np

root = Settings['data_root']
filepath = join(root, 'stacknet_64x64_model.h5')
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint, TerminateOnNaN()]

if isfile(filepath):
    model = load_model(filepath)
else:
    model = get_model(lr=0.0001, w=64, h=64, train_upper_layers=True)

model.summary()

sampler = ReId.DataSampler(root, 64, 64)
sampler_train = MOT16Sampler(root, (64, 64))

def generate_training_data():
    global sampler
    while True:
        X, Y = sampler_train.get_all_batch(20, 20)
        X_, Y_ = sampler.get_train_batch(100, 100)
        X = np.concatenate([X, X_])
        Y = np.concatenate([Y, Y_])
        X = preprocess_input(X.astype('float64'))
        yield X, Y

def generate_validation_data():
    global sampler
    while True:
        X, Y = sampler_train.get_all_batch(20, 20)
        X_, Y_ = sampler.get_test_batch(100, 100)

        X = np.concatenate([X, X_])
        Y = np.concatenate([Y, Y_])

        X = preprocess_input(X.astype('float64'))
        yield X, Y


model.fit_generator(generate_training_data(),
                    validation_data=generate_validation_data(),
                    validation_steps=5,
                    steps_per_epoch=100,
                    epochs=1000,
                    callbacks=callbacks_list)

# load_model to load model
