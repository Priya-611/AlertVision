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

# To read a image

# # img_array = cv2.imread(r"C:\Users\HP\Downloads\eye dataset\data\train\awake\s0001_01856_0_0_1_0_0_01.png", cv2.IMREAD_GRAYSCALE)
# # plt.imshow(img_array, cmap="gray")
# # plt.show()
# # print(img_array.shape)



# To read all the images 
# # Datadirectory = r"C:\Users\HP\Downloads\eye dataset\data\train"
# # classes = ["awake", "sleepy"]
# # for c in classes:
# #     path = os.path.join(Datadirectory, c)   #join the datadirectory and classes (one by one)
# #     for img in os.listdir(path):     
# #         img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)   #joining and reading each image inside the 'path' in grayscale format
# #         backToRGB = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)   #conversion to rgb channel for cnn training
# #         # plt.imshow(img_array, cmap="gray")
# #         # plt.show()
# #         break
# #     break


# Resizing each image
# # img_size = 224 

# # new_array = cv2.resize(backToRGB, (img_size, img_size))   # resizing the coloured image
# # plt.imshow(new_array , cmap="gray")
# # plt.show()






# Reading all image and converting them into an array for data and labels


# Datadirectory = r"C:\Users\HP\Downloads\eye dataset\data\train"
# classes = ["awake", "sleepy"]

# img_size = 224

# training_data = []

# def create_training_Data():
#     for c in classes:
#         path = os.path.join(Datadirectory, c)
#         class_num = classes.index(c)  #gives numeric label for each class ( 0 - 'awake', 1 - 'sleepy')
#         for img in os.listdir(path):
#             try:
#                 img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
#                 backToRGB = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
#                 new_array = cv2.resize(backToRGB, (img_size, img_size))

#                 training_data.append([new_array, class_num])     #img_array , 0/1
#             except Exception as e:
#                 print("Error: ", e)


# create_training_Data()   #50937

# # print(len(training_data))


 
# random.shuffle(training_data)   #shuffle training_data list so model don't learn smae pattern


# # separting x and y

# x = []
# y = []   

# for feature, label in training_data:
#     x.append(feature)    #img_array  as input
#     y.append(label)      #0/1 as output

# x = np.array(x)  # converting x and yfrom list to numpyarray
# y = np.array(y)


# print(x.shape)     # (total_samples , 224, 224 , 3)     each image of 224 x 224 size and each has 3 channel RGB

# # Normalise input data
# x = x/255.0     # because the input data i.e. image contains pixel value 0(black) and 255(white) so normalise it from 0 to 1

# # save x
# with open("x.pickle", "wb") as f:
#     pickle.dump(x, f)

# # save y 
# with open("y.pickle", "wb") as f:
#     pickle.dump(y, f)


# # load x
# with open("x.pickle", "rb") as f:
#     pickle.load(f)

# # load y
# with open("y.pickle", "rb") as f:
#    pickle.load(f)



# model = tf.keras.applications.mobilenet.MobileNet()
 
# # print(model.summary())

# base_input = model.layers[0].input
# base_output = model.layers[-4].output 

# Flat_layer = layers.Flatten()(base_output)
# final_output = layers.Dense(1)(Flat_layer)
# final_output = layers.Activation('sigmoid')(final_output)




# new_model = keras.Model(inputs=base_input, outputs =final_output)
# print(new_model.summary())

# # new_model.compile(loss="binary_crossentropy", optimizer="adam", metrics =["accuracy"])
# # new_model.fit(x,y, epochs=1 , validation_split=0.1)




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



# # img =cv2.imread(r"C:\Users\HP\Downloads\awake_woman.jpg")
# img =cv2.imread(r"C:\Users\HP\Downloads\eye dataset.zip\data\train\sleepy\s0001_00025_0_0_0_0_0_01.png")
# # img = cv2.resize(img, (224,224))


# #show image
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()


# # x_input = np.array(img, dtype="float32")/ 255.0
# # x_input = x_input.reshape(1,224,224,3)

# # Prediction
# # prediction = model.predict(x_input)
# # print(prediction)

# # if prediction[0][0] > 0.5:
# #     result ="Sleepy"
# # else:
# #     result ="Awake"

# # print("Final Result ", result)


# faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

# eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")


# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# faces = faceCascade.detectMultiScale(gray, 1.1, 5, minSize=(80,80))




# for (fx,fy,fw,fh) in faces:
#     cv2.rectangle(img, (fx, fy), (fx+fw, fy+fh), (255,0,0), 2)

#     roi_gray = gray[fy:fy+fh , fx:fx+fw]
#     roi_color = img[fy:fy+fh , fx:fx+fw]

#     # Slightly relaxed parameters to improve detection when eyes are partially/fully closed.
#     eyes = eyeCascade.detectMultiScale(roi_gray, 1.05, minNeighbors=1, minSize=(15,15))


#     if len(eyes)>0:
#         for i, (x,y,w,h) in enumerate(eyes):
#             if i >= 2:
#                 break
#             eye_img = roi_color[y:y+h, x:x+w]

#             # eye_rgb = cv2.cvtColor(eye_img, cv2.COLOR_GRAY2BGR)
#             eye_resized = cv2.resize(eye_img, (224,224))
#             eye_resized = cv2.cvtColor(eye_resized, cv2.COLOR_BGR2RGB)


#             x_input = np.array(eye_resized, dtype="float32")/255.0
#             x_input = x_input.reshape(1,224,224,3)

#             prediction = model.predict(x_input)
            
#             if prediction[0][0] > 0.4:
#                 result ="Sleepy"
#             else:
#                 result ="Awake"



#             cv2.rectangle(roi_color, (x, y), (x+w, y+h), (0,255,0), 2)
#             cv2.putText(roi_color,result, (x,y-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,255,0), 2)

#     else: 
#         #  print("No eye detected")
#         # face_rgb = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
#         # face_resized = cv2.resize(face_rgb, (224,224))

#         eye_region = roi_color[int(fh * 0.15): int(fh*0.6), :]
#         eye_resized = cv2.resize(eye_region, (224,224))
#         eye_resized = cv2.cvtColor(eye_resized, cv2.COLOR_BGR2RGB)
        
#         x_input = np.array(eye_resized, dtype="float32")/255.0
#         x_input = x_input.reshape(1,224,224,3)

#         prediction = model.predict(x_input)
 
#         if prediction[0][0] > 0.4:
#             result ="Sleepy"
#         else:
#             result ="Awake"




#         # cv2.rectangle(img, (fx, fy), (fx+fw, fy+fh), (0,255,0), 2)
#         cv2.putText(img,result, (fx,fy-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,255,0), 2)


        


# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()





# ✅ SAVE MODEL
model.save(r"C:\Users\HP\OneDrive\Documents\Etc\OpenCV\Drowsiness Detection\model.h5")

print("✅ Model saved successfully!")









