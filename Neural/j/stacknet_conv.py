from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dropout, Flatten, Dense, Activation, Conv2D, MaxPooling2D
from keras.models import Sequential
import numpy as np
from keras.optimizers import SGD

def get_model(lr=0.01, train_upper_layers=True):
    vgg_model = VGG16(input_shape=(64, 64, 3), weights='imagenet',
                      include_top=False)

    model = Sequential()
    block1_conv1 = Conv2D(64, (3, 3), input_shape=(64,64,6),
                          padding='same', name='block1_conv1',
                          trainable=train_upper_layers)
    model.add(block1_conv1)
    model.add(Conv2D(64, (3,3), padding='same', name='block1_conv2',
                     trainable=train_upper_layers))
    model.add(MaxPooling2D(pool_size=(2,2), name='block1_pool'))

    model.add(Conv2D(128, (3,3), padding='same', name='block2_conv1',
                     trainable=train_upper_layers))
    model.add(Conv2D(128, (3,3), padding='same', name='block2_conv2',
                     trainable=train_upper_layers))
    model.add(MaxPooling2D(pool_size=(2,2), name='block2_pool'))

    model.add(Conv2D(256, (3,3), padding='same', name='block3_conv1',
                     trainable=train_upper_layers))
    model.add(Conv2D(256, (3,3), padding='same', name='block3_conv2',
                     trainable=train_upper_layers))
    model.add(Conv2D(256, (3,3), padding='same', name='block3_conv3',
                     trainable=train_upper_layers))
    model.add(MaxPooling2D(pool_size=(2,2), name='block3_pool'))

    model.add(Conv2D(512, (3,3), padding='same', name='block4_conv1',
                     trainable=train_upper_layers))
    model.add(Conv2D(512, (3,3), padding='same', name='block4_conv2',
                     trainable=train_upper_layers))
    model.add(Conv2D(512, (3,3), padding='same', name='block4_conv3',
                     trainable=train_upper_layers))
    model.add(MaxPooling2D(pool_size=(2,2), name='block4_pool'))

    model.add(Conv2D(512, (3,3), padding='same', name='block5_conv1',
                     trainable=train_upper_layers))
    model.add(Conv2D(512, (3,3), padding='same', name='block5_conv2',
                     trainable=train_upper_layers))
    model.add(Conv2D(512, (3,3), padding='same', name='block5_conv3',
                     trainable=train_upper_layers))
    model.add(MaxPooling2D(pool_size=(2,2), name='block5_pool'))
    
    #model.add(Conv2D(1024, kernel_size=3, padding='valid', name='consolidate',
    #                 trainable=train_upper_layers))
    
    #model.add(Conv2D(64, kernel_size=1, padding='valid', name='consolidate2',
    #                 trainable=train_upper_layers))
    

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    # --- set the fixed weights ---
    W, bias = vgg_model.layers[1].get_weights()
    W = np.concatenate((W,W),axis=2)
    block1_conv1.set_weights([W, bias])
    # -----------------------------

    for i in range(1, len(vgg_model.layers)-1):
        model.layers[i].set_weights(vgg_model.layers[i+1].get_weights())

    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model


