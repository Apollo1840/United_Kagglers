# -*- coding: utf-8 -*-

import pandas as pd 
import os

# some predefined value for illustration
PROJECT_FOLDER = 'United_Kagglers'

def change_dir_to_UKa():
    path = os.getcwd()
    while(os.path.basename(path) != PROJECT_FOLDER):
        path = os.path.dirname(path)
    os.chdir(path)



def load_data(DATA_PATH):
    change_dir_to_UKa()
    
    # data_train = pd.read_csv(os.path.dirname(__file__)+'\\datasets\\k000_titanic\\train.csv')
    data_train = pd.read_csv(DATA_PATH + 'train.csv')
    data_test = pd.read_csv(DATA_PATH + 'test.csv')
    
    return data_train, data_test


'''
    how to load data?
    1) download the data to datasets, put it into the project folder
    2) in py file, define the PROJECT_FOLDER
    3) from tools.data_loader import load_data
    4) data_train, data_test = load_data(DATA_PATH)
    
'''