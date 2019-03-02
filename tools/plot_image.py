# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 19:14:39 2019

@author: zouco
"""

import matplotlib.pyplot as plt


def plot_imgs(list_img):
    for i, img in enumerate(list_img):
        ax = plt.subplot(1, len(list_img), i + 1)
        plt.tight_layout()
        ax.set_title(str(i))
        plt.imshow(img)
    plt.show()