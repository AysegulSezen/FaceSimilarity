#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 12:25:29 2023

@author: aysegulsezen
"""


import face_recognition
import cv2
import os
import glob


def faceRecognition(firstImg,secondImg):
    picture_of_me = face_recognition.load_image_file(firstImg)
    my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

# my_face_encoding now contains a universal 'encoding' of my facial features that can be compared to any other picture of a face!

    unknown_picture = face_recognition.load_image_file(secondImg)
    unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]

# Now we can see the two face encodings are of the same person with `compare_faces`!

    results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)

    if results[0] == True:
        print("It's target picture!")
    else:
        print("It's not target picture!")
       
#Extracting images from given video.
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

#Comparing all images in video to target images
def CheckAllVideoImages():
    filePath='data/'
    folderList=os.listdir(filePath)
    myIdentityImg="IMG_1680.png"
    
    for fldr in folderList:
        if not fldr.startswith('.'):
            filePathF = filePath + fldr +'/*.*'
            for filename in glob.glob(filePathF):
                if fldr=='person':
                    faceRecognition(filename,myIdentityImg)



ExtactImagesFromVideo()
CheckAllVideoImages()