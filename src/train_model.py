import tensorflow as tf
from tensorflow  import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random
import pickle      



# ImageDataGenerator loads images from my folder automatically and feeds them to the model in small batches

Datadirectory = r"C:\Users\HP\Downloads\eye dataset\data\train"
img_size = 224
batch_size = 32


# Normalise the input data

train_datagen = ImageDataGenerator(rescale = 1./255, validation_split=0.2)    # because the input data i.e. image contains pixel value 0(black) and 255(white) so normalise it from 0 to 1

# loads 32 images from Datadirectory folder

train_data = train_datagen.flow_from_directory(Datadirectory, target_size=(img_size, img_size), batch_size=batch_size, class_mode ='binary', subset="training", shuffle =True)


val_data = train_datagen.flow_from_directory(Datadirectory, target_size=(img_size, img_size), batch_size=batch_size, class_mode="binary", subset="validation")



print("train_data.class_indices:", train_data.class_indices)

base_model = tf.keras.applications.MobileNet(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

#inlcude_top = False   ,,removes final layer from the MobileNet model (which was giving output in 1000 classes )  instead we will add our own layers
#weights = 'iamgenet'    load pre-trained knowledge



# Freeze base model  (don't change MobileNet's knowledge as it is alrready familier with edges, corners, shapes, textures, patterns)
for layer in base_model.layers:
    layer.trainable = False


x = base_model.output                    #take features extracted by model (edges, shapes , patterns)
x = layers.GlobalAveragePooling2D()(x)   #compressing feature maps into smaller form
x = layers.Dense(64, activation='relu')(x)  #add learning layer with 64 neurons
x = layers.Dense(1, activation='sigmoid')(x) # final decision layer with one layer for binary classification




#creating final model  with input same as prev and output as x 
model = models.Model(inputs=base_model.input, outputs=x)

#compile model 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# fit model 
model.fit(train_data, validation_data=val_data, epochs=5)  #with single epoch and steps_per_epoch as 200




#  SAVE MODEL
model.save(r"C:\Users\HP\OneDrive\Documents\Etc\OpenCV\Drowsiness Detection\model.h5")

print("✅ Model saved successfully!")









