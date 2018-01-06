import json
from pprint import pprint
Settings = json.load(open('../prototyping/settings.txt'))
pprint(Settings)
from CUHK03_Sampler import CUHK03_Sampler

from stacknet import get_model

model = get_model()
model.summary()

def generate_training_data():
    sampler = CUHK03_Sampler()
    while True:
        X, Y = sampler.get_train_batch(100, 100)
        yield X, Y

model.fit_generator(generate_training_data(), samples_per_epoch=10, 
                    nb_epoch=10)