#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 09:52:52 2023

@author: aysegulsezen
"""

import skimage.io as io 
import scipy.io
import numpy as np

import tensorflow as tf
import os
from PIL import Image
import glob
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy


#this method below will be 2. step of the project. Now,It is not used.
def ExtactImagesFromVideo():
    cam = cv2.VideoCapture("IMG_1677.MOV")
    
    currentframe = 0
  
    while(True):
        ret,frame = cam.read()
        
        if ret:
            # if video is still left continue creating images
            name = './data/person/image' + str(currentframe) + '.png'
            print ('Creating...' + name)
  
            # writing the extracted images
            cv2.imwrite(name, cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE))
  
            currentframe += 1
        else:
            break
    cam.release()
    cv2.destroyAllWindows()

def FindFaceSimilarity():
    
    # Get images from directory and label them
    data=tf.keras.utils.image_dataset_from_directory('data',image_size=(128,128))
       
    
    # draw first 11 images and show their labels
    data_iterator = data.as_numpy_iterator()
    batch= data_iterator.next()
   
    fig,ax=plt.subplots(ncols=11, figsize=(20,20))
    for idx,img in enumerate(batch[0][:11]):
        ax[idx].imshow(img.astype(int))
        ax[idx].title.set_text(batch[1][idx])
    
    
    data = data.map(lambda x,y: (x/255,y))
    scaled_iterator = data.as_numpy_iterator()
    
    batch = scaled_iterator.next()
    
    fig,ax=plt.subplots(ncols=11, figsize=(20,20))
    for idx,img in enumerate(batch[0][:11]):
        ax[idx].imshow(img)
        ax[idx].title.set_text(batch[1][idx])
    
    # Determine training , validation and test images count 
    train_size= int(len(data)* .7)
    val_size= int(len(data)*.2)
    test_size = int(len(data)*.1)
    
    print('train_size:',train_size)
    print('val_size:',val_size)
    print('test_size:',test_size)
    
    train = data.take(train_size)
    val=data.skip(train_size).take(val_size)
    test=data.skip(train_size+val_size).take(test_size)
    
    # Start ML
    model = Sequential()
    
    model.add(Conv2D(16,(3,3),1,activation='relu',input_shape=(128,128,3)))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(32,(3,3),1,activation='relu'))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(16,(3,3),1,activation='relu'))
    model.add(MaxPooling2D())
    
    model.add(Flatten())
    
    model.add(Dense(128,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    
    model.compile('adam',loss=tf.losses.BinaryCrossentropy(),metrics=['accuracy'])
    model.summary()
    
    logdir='logs'
    tensorboard_callback=tf.keras.callbacks.TensorBoard(log_dir=logdir)
    
    hist=model.fit(train,epochs=20,validation_data=val,callbacks=[tensorboard_callback])
    
    ####### PREDICT images  ######################################
        
    img= Image.open('targetPersonImage1.png').convert(mode='RGB')    
    resize= tf.image.resize(img,(128,128))
    np.expand_dims(resize,0)
    yhat=model.predict(np.expand_dims(resize/255,0))    
    print('Predict target person image1:',yhat)

    img= Image.open('targetPersonImage2.png').convert(mode='RGB')
    resize= tf.image.resize(img,(128,128))
    np.expand_dims(resize,0)
    yhat=model.predict(np.expand_dims(resize/255,0))    
    print('Predict target person image2:',yhat)
    
    img= Image.open('targetPersonImage3.png').convert(mode='RGB')
    resize= tf.image.resize(img,(128,128))
    np.expand_dims(resize,0)
    yhat=model.predict(np.expand_dims(resize/255,0))    
    print('Predict target person image3:',yhat)
    
    img2= Image.open('01290.png').convert(mode='RGB')
    resize2= tf.image.resize(img2,(128,128))
    np.expand_dims(resize2,0)
    yhat2=model.predict(np.expand_dims(resize2/255,0))    
    print('Predict other person1 image:',yhat2)
       
    img2= Image.open('01715.png').convert(mode='RGB')
    resize2= tf.image.resize(img2,(128,128))
    np.expand_dims(resize2,0)
    yhat2=model.predict(np.expand_dims(resize2/255,0))
    print('Predict other person2 image:',yhat2)
       
    img2= Image.open('01716.png').convert(mode='RGB')
    resize2= tf.image.resize(img2,(128,128))
    np.expand_dims(resize2,0)
    yhat2=model.predict(np.expand_dims(resize2/255,0))
    print('Predict other person3 image:',yhat2)
    
    
    # Find accuracy
    pre=Precision()
    re=Recall()
    acc=BinaryAccuracy()

    for batch in test.as_numpy_iterator():
        X,y = batch
        yhat = model.predict(X)
        pre.update_state(y,yhat)
        re.update_state(y,yhat)
        acc.update_state(y,yhat)
        
    
    print('Pre:',pre.result().numpy(),' Re:',re.result().numpy(), ' acc:',acc.result().numpy())
    
    
    

#ExtactImagesFromVideo()  #this method will be next step of this project. Finding face similarity between video and image. Is it same person on video and this image.
#
FindFaceSimilarity()