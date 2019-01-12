# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 16:13:27 2019

@author: zouco
"""

from tools.data_loader import load_data

PROJECT_NAME = '002_house_price'
DATA_PATH = 'datasets\\{}\\'.format(PROJECT_NAME)

data_train, data_test = load_data(DATA_PATH)

print(data_train.shape)