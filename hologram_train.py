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
from architectures import transfer_vgg19


hologram_model = transfer_vgg19(input_shape=(48, 52, 3), num_classes=2, hidden_dim=256)
#hologram_model = transfer_vgg16_block5(input_shape=(48, 52, 3), num_classes=2, hidden_dim=256)


work_dir = './hologram_data_for_binaryclassification'
batch_s=20
target_s=(48, 52)

train_dir = os.path.join(work_dir, 'train')
validation_dir =  os.path.join(work_dir, 'validation')
#test_dir = os.path.join(work_dir, 'test')

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_s,
        batch_size=batch_s,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=target_s,
        batch_size=batch_s,
        class_mode='categorical')

hologram_model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5), 
              metrics=['acc'])

early_stopping =EarlyStopping(monitor='val_loss', patience=2)
model_name = 'hologram_model_vgg19.hdf5'
model_checkpoint = ModelCheckpoint(model_name, 'val_loss', verbose=1,
                                                    save_best_only=True)

history_vgg16 = hologram_model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=200,
      validation_data=validation_generator,
      callbacks=[model_checkpoint],
	  verbose=1,
      validation_steps=50)
accuracy_rslt=history_vgg16.history["acc"]
val_accuracy_rslt=history_vgg16.history["val_acc"]
loss_rslt=history_vgg16.history["loss"]
val_loss_rslt=history_vgg16.history["val_loss"]

with open("./vgg19_model_sonuc.txt","a") as myfile:
    myfile.write("accuracy={0}\n,val_accuracy={1}\n,loss={2}\n,val_loss={3}\n".format(accuracy_rslt,
                                                                                      val_accuracy_rslt, 
                                                                                      loss_rslt,
                                                                                      val_loss_rslt))
model_name_w='hologram_model_vgg19.h5'
hologram_model.save_weights(model_name_w)


