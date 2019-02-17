# -*- coding: utf-8 -*-

PROJECT_NAME = '005_landmarks_retrieval'
DATA_PATH = 'datasets\\{}\\'.format(PROJECT_NAME)

from tools.data_loader import load_data
data_train, data_test = load_data(DATA_PATH)

print(data_train.columns)
print(data_train.shape)


