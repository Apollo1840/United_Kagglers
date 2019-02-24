# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 11:42:18 2019

@author: zouco
"""

PROJECT_NAME = '005_landmarks_retrieval'
DATA_PATH = 'datasets\\{}\\'.format(PROJECT_NAME)

from tools.data_loader import load_data
train_data, test_data = load_data(DATA_PATH)

train_data.iloc[:100,:].to_csv(PROJECT_NAME + "\\testData_imageDownloader.csv", index=False)
