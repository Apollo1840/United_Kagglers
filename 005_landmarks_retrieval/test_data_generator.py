# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 08:30:38 2019

@author: zouco

"""


from data_generator import triplet_generation
from data_generator import DataGenerator


tg = triplet_generation()

ID = "00cfd9bbf55a241e"
img3 = tg.get_one_input_tensor(ID)
print(len(img3))
print(img3[0].shape)

from tools.plot_image import plot_imgs

plot_imgs(img3)
