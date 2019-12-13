from keras.engine import  Model
from keras.layers import Flatten, Dense, Input, Dropout, Activation
#from keras_vggface.vggface import VGGFace
from keras.applications import VGG19
from keras import models
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import os 
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping

def transfer_vgg19(input_shape, num_classes, hidden_dim):
    conv_base = VGG19(weights='imagenet',
                  include_top=False,
                  input_shape=input_shape)
    conv_base.trainable = False
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(hidden_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='sigmoid'))
    print(model.summary())
    return model