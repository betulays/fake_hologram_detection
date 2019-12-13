from keras.models import load_model
from keras.preprocessing import image
#import matplotlib.pyplot as plt
import numpy as np
import os
from os import listdir
from os.path import isfile, join



hologram_model=load_model('./hologram_model_vgg19.hdf5')


img_path='./test1/o1.jpg'

label_list={0:'fake',1:'real'}

img = image.load_img(img_path, target_size=(48, 52))
img_tensor = image.img_to_array(img)                    
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

#plt.imshow(img_tensor[0])                           
#plt.axis('off')

hologram_probability=hologram_model.predict(img_tensor)
predicted_label = np.argmax(hologram_probability)
hologram_text = label_list[predicted_label]
print("hologram: {} ({:.2f}%)".format(hologram_text, hologram_probability[0][predicted_label] * 100))
