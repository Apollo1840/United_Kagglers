# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 19:06:25 2019

@author: zouco
"""

"""
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
"""

from image_loader import imageLoader

il = imageLoader("..//datasets//005_landmarks_retrieval//imgs")
img = il.load("00cfd9bbf55a241e")

print(img.shape)

imgs = [img]

from image_transformer import imageTransformer

it = imageTransformer(125)
imgs.append(it.transformer(img))
imgs.append(it.transformer(img))
imgs.append(it.transformer(img))


from tools.plot_image import plot_imgs

plot_imgs(imgs)


