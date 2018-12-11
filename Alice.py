# -*- coding: utf-8 -*-
'''
    There are 3 steps of data analysis
        1, load data
        2, check column
        3, check column correlation

'''

# some predefined value for illustration
PROJECT_FOLDER = 'United_Kagglers'
PROJECT_NAME = '001_titanic'
DATA_PATH = 'datasets\\{}\\'.format(PROJECT_NAME)



###########################################################################
# 1 load data
import os

def change_dir_to_UKa():
    path = os.getcwd()
    while(os.path.basename(path) != PROJECT_FOLDER):
        path = os.path.dirname(path)
    os.chdir(path)


import pandas as pd 

def load_data():
    change_dir_to_UKa()
    
    # data_train = pd.read_csv(os.path.dirname(__file__)+'\\datasets\\k000_titanic\\train.csv')
    data_train = pd.read_csv(DATA_PATH + 'train.csv')
    data_test = pd.read_csv(DATA_PATH + 'test.csv')
    
    return data_train, data_test


data_train, data_test = load_data()


'''
    how to load data?
    1) download the data to datasets, put it into the project folder
    2) copy the whole thing into your py file
    
'''

###########################################################################
# 2 check column







###########################################################################
# 3 check column relationship
import numpy as np


os.chdir(os.getcwd()+'\\tools')

from ploters import plot_distribution, plot_categories, plot_correlation_map
from ploters import StackedPloter










