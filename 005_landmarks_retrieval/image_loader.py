# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 11:57:58 2019

@author: zouco
"""

import cv2
import os
import pandas as pd
import sys
import numpy as np


class imageLoader():
    
    def __init__(self, images_folder):
        self.path = images_folder
    
    def load(self, image_id):
        try:
            imgdir = os.getcwd() + "\\{}\\{}.jpg".format(self.path,image_id)
            img = cv2.imread(imgdir)
        except:
            print("failed to get img from {}".format(imgdir))
        return img


def update_data(data_file, imgdir):
    
    df = pd.read_csv(data_file)
    
    il = imageLoader(imgdir) 

    if "image" not in df.columns:
        df["image"] = [None for _ in range(df.shape[0])]
        
    for i in range(df.shape[0]):
        
        if(i%1000 == 0):
            print("data updated, processed: " + str(i))
            df.to_csv(data_file, index=False)

        if df.loc[i, "image"] is None or np.isnan(df.loc[i, "image"]):
            print(il.load(df.loc[i,"id"]).shape)
            #df.loc[i, "image"] = il.load(df.loc[i,"id"])
    
    print("data updated (finished)")
    df.to_csv(data_file, index=False)
           

def main():
    if len(sys.argv) != 3:
        print('Syntax: {} <data_file.csv> <images dir>'.format(sys.argv[0]))
        sys.exit(0)
        
    (data_file, imgdir) = sys.argv[1:]
    
    update_data(data_file, imgdir)
    
    
           

# arg1 : data_file.csv
# arg2 : images_dir
if __name__ == '__main__':
    main()