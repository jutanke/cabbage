import json
from pprint import pprint
Settings = json.load(open('../prototyping/settings.txt'))
pprint(Settings)
from CUHK03_Sampler import CUHK03_Sampler
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from stacknet import get_model
from os.path import join, isfile, isdir, exists, splitext


root = Settings['data_root']
filepath = join(root, 'stacknet_model.h5')
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

if isfile(filepath):
    model = load_model(filepath)
else:
    model = get_model()

model.summary()

sampler = CUHK03_Sampler()

def generate_training_data():
    global sampler
    while True:
        X, Y = sampler.get_train_batch(100, 100)
        yield X, Y

def generate_validation_data():
    global sampler
    while True:
        X, Y = sampler.get_test_batch(100, 100)
        yield X, Y


model.fit_generator(generate_training_data(),
                    validation_data=generate_validation_data(),
                    validation_steps=5,
                    steps_per_epoch=100,
                    epochs=1000,
                    callbacks=callbacks_list)

# load_model to load model
