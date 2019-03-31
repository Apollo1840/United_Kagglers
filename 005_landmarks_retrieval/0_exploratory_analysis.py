# -*- coding: utf-8 -*-
import requests as req
from time import time

PROJECT_NAME = '005_landmarks_retrieval'
DATA_PATH = 'datasets\\{}\\'.format(PROJECT_NAME)

from tools.data_loader import load_data
train_data, test_data = load_data(DATA_PATH)

print(train_data.columns)
print(train_data.shape)


# -----------------------------------------------
# primary analysis

# missing value
from tools.pandas_extend import NA_refiner
nar = NA_refiner(train_data)
nar.show()
# comments: there is no missing value

# how much images for one landmark
print(train_data.landmark_id.value_counts().describe())
# comments: very less


# ------------------------------------------------------
# read the image
from skimage import io

t0 = time()
images = []
num = 5
for i in range(num):
    print("loading {}. image ...".format(i))
    images.append(io.imread(train_data.loc[i,"url"]))
    
print(images[0].shape)
print("load {} images take {} min".format(num,(time()-t0)/60))


# show the image
import cv2

for i in range(num):
    cv2.imshow('image {}'.format(i),images[i])

cv2.waitKey(0)
cv2.destroyAllWindows()


'''
# or show the image by display the HTML
# works in Jupyter notebook?
from IPython.display import Image
from IPython.core.display import HTML 

def display_category(urls, category_name):
    img_style = "width: 180px; margin: 0px; float: left; border: 1px solid black;"
    images_list = ''.join([f"<img style='{img_style}' src='{u}' />" for _, u in urls.head(20).iteritems()])

    display(HTML(images_list))

urls = train_data['url']
display_category(urls, "")
'''

# read the image locally
# mark: run the image_downloader first (see README to understand how to run image_downloader)
from image_loader import imageLoader

il = imageLoader(DATA_PATH + "images") 

num = 5

images = []
for image_id in train_data.loc[:num, "id"]:
    images.append(il.load(image_id))

print(images[0])    




'''
    findings:
        1. images are not in same size

'''