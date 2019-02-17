# -*- coding: utf-8 -*-
import requests as req
from time import time

PROJECT_NAME = '005_landmarks_retrieval'
DATA_PATH = 'datasets\\{}\\'.format(PROJECT_NAME)

from tools.data_loader import load_data
data_train, data_test = load_data(DATA_PATH)

print(data_train.columns)
print(data_train.shape)


# read the image
from skimage import io

t0 = time()
images = []
num = 5
for i in range(num):
    print("loading {}. image ...".format(i))
    images.append(io.imread(data_train.loc[i,"url"]))
    
print(images[0].shape)
print("load {} images take {} min".format(num,(time()-t0)/60))


# show the image
import cv2

for i in range(num):
    cv2.imshow('image {}'.format(i),images[i])

cv2.waitKey(0)
cv2.destroyAllWindows()


'''
    findings:
        1. images are not in same size

'''