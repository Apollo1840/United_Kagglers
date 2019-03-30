# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 15:33:48 2019

@author: zouco
"""
from keras.models import model_from_json
import numpy as np
import matlibplot.pyplot as plt
from keras.applications.xception import preprocess_input
import cv2

from tools.keras_model_handler import save_model, load_model
triplet_model = load_model(MODEL_NAME)

# now we just use "triplet_model" to finetune the weights and "metric" to get the similarity between two inputs
# Input:    2 different samples of the same class, 1 sample from a different class
#           The difference between input 1 and the same-class sample should be smaller than the difference between
#           input 1 and the different-class sample

# utility function for creating images with simple shapes
def create_im(imtype = 'circle'):
    im = np.zeros((229,229,3),dtype=np.float32)
    if imtype == 'circle':
        center = (np.random.randint(10,209,1)[0],np.random.randint(10,209,1)[0])
        radius = np.random.randint(10,60,1)[0]
        cv2.circle(im,center,radius,(0,255,0),-1)
        center = (np.random.randint(10,209,1)[0],np.random.randint(10,209,1)[0])
        radius = np.random.randint(10,60,1)[0]
        cv2.circle(im,center,radius,(0,255,0),-1)
        x,y = np.random.randint(10,209,1)[0],np.random.randint(10,209,1)[0]
        w,h = np.random.randint(10,60,1)[0],np.random.randint(10,60,1)[0]
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),-1)
    elif imtype == 'rectangle':
        x,y = np.random.randint(10,209,1)[0],np.random.randint(10,209,1)[0]
        w,h = np.random.randint(10,60,1)[0],np.random.randint(10,60,1)[0]
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),-1)
        x,y = np.random.randint(10,209,1)[0],np.random.randint(10,209,1)[0]
        w,h = np.random.randint(10,60,1)[0],np.random.randint(10,60,1)[0]
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),-1)
        center = (np.random.randint(10,209,1)[0],np.random.randint(10,209,1)[0])
        radius = np.random.randint(10,60,1)[0]
        cv2.circle(im,center,radius,(0,255,0),-1)
    im = im.reshape(1,229,229,3)
    return preprocess_input(im)

# utility function for scaling input to image range (for ploting)
def scale_img(x):
    x = np.float32(x)
    x-=np.nanmin(x)
    if np.nanmax(x)!=0:
        x/=np.nanmax(x)
    return np.uint8(255*x)

# Test on some dummy images:
for k in range(4):
    im1 = create_im()
    im2 = create_im()
    im3 = create_im(imtype = 'rectangle')
    
    distances = triplet_model.predict([im1,im2,im3])[0]

    plt.subplot(4,3,(k*3)+1)
    plt.imshow(scale_img(im1[0]))
    plt.subplot(4,3,(k*3)+2)
    if distances[0]<=distances[1]:
        plt.imshow(scale_img(im2[0]))
        plt.subplot(4,3,(k*3)+3)
        plt.imshow(scale_img(im3[0]))
    else:
        plt.imshow(scale_img(im3[0]))
        plt.subplot(4,3,(k*3)+3)
        plt.imshow(scale_img(im2[0]))
        
    if k == 0:
        plt.subplot(4,3,(k*3)+1)
        plt.title('input 1')
        plt.subplot(4,3,(k*3)+2)
        plt.title('more similar to input 1')
        plt.subplot(4,3,(k*3)+3)
        plt.title('less similar to input 1')
plt.savefig('example.png')