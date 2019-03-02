# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 19:38:40 2019

@author: zouco
"""

import os
import pandas as pd
import matplotlib

from image_loader import imageLoader


class imgBank():
    
    def __init__(self, imgs_folder = "..//datasets//005_landmarks_retrieval//imgs"):
        i = 0

        il = imageLoader(imgs_folder)        
        df = pd.DataFrame({"img_id":[],"size_x":[],"size_y":[]})
        
        for (dirpath, dirnames, filenames) in os.walk(imgs_folder):
            for filename in filenames:
                img_id = filename.split(".")[0]
                img = il.load(img_id)
                
                if img is not None:
                    df = df.append({"img_id":img_id, "size_x": img.shape[0], "size_y":img.shape[1]}, ignore_index=True)
                if i%1000 == 0:
                    print(i)
                i+=1
        self.data = df
        self.preprocess()
        
    def preprocess(self):
        self.data["img_shape"] = ["({},{})".format(x,y) for x,y in zip(self.data["size_x"], self.data["size_y"])]
        self.data["xy_raito"] = [x/y for x,y in zip(self.data["size_x"], self.data["size_y"])]
    
    @property
    def num_img(self):
        return self.data.shape[0]
    
if __name__ == "__main__":
    iB = imgBank()
    
    print(iB.data.img_shape.value_counts())
    
    """
    (192.0,256.0)    9083
    (256.0,192.0)    2555
    (170.0,256.0)    2217
    (171.0,256.0)     619
    (256.0,171.0)     546
    """
    
    print(iB.data.xy_ratio.value_counts())
    print(iB.data.xy_ratio.describe())
    
    """
    mean         0.877384
    std          0.278738
    min          0.558594
    25%          0.750000
    50%          0.750000
    75%          0.753906
    max          1.777778
    """