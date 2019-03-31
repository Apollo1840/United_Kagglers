# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 16:21:39 2019

@author: zouco
"""

import os
import numpy as np
import keras
from tools.data_loader import load_data
from image_loader import imageLoader
from image_transformer import imageTransformer

# import uuid
# ID = uuid.uuid4().hex


PROJECT_NAME = '005_landmarks_retrieval'
DATA_PATH = 'datasets\\{}\\'.format(PROJECT_NAME)


class CaseApi():
    
    def __init__(self):
        self.il = imageLoader("datasets//005_landmarks_retrieval//imgs")
        self.it = imageTransformer(125)
        
    def load_img(self, ID):    
        return self.il.load(ID)
    
    @staticmethod
    def get_valid_imgIDs():
        v = []
        for r, d, f in os.walk(DATA_PATH):
            for file in f:
                if '.jpg' in file:
                    v.append(file.split(".")[0])
        return v
    
    @staticmethod    
    def get_training_part_ids():
        return CaseApi.get_valid_imgIDs()
    
    @staticmethod
    def get_validation_part_ids():
        return np.random.choice(CaseApi.get_valid_imgIDs(), 10000)


class original_db():
    train_data, test_data = load_data(DATA_PATH)


class triplet_generation(original_db):
    
    def __init__(self):
        self.train_data = self.train_data.loc[self.train_data.id.isin(CaseApi.get_valid_imgIDs()), :]
        self.api = CaseApi()
    
    def get_one_input_tensor(self, imgID=None):
        a_triple = []
        for imgID in self.get_a_triplet_tuple(imgID):
            a_triple.append(self.get_normalized_img(imgID))
        return a_triple
        
    def get_a_triplet_tuple(self, imgID):
        if imgID is None:
            start_id = np.random.choice(self.train_data.id.values,1)[0]
        else:
            start_id = imgID
        sim_id = self.get_a_similar_img(start_id)
        diff_id = self.get_a_diff_img(start_id)
        return (start_id, sim_id, diff_id)
        
    def get_a_similar_img(self, imgID, db = "train"):
        """
        return a similar (in the sense of content) imgID
        """
        if db=="train":
            the_landmark_id = self.get_landmark_id(imgID)
            subset = self.train_data.loc[self.train_data.landmark_id == the_landmark_id, "id"].values       
            return self.choose_an_imgID(subset, imgID)
            
        return None
    
    def get_a_diff_img(self, imgID, db = "train"):
        '''
        return a different (in the sense of content) imgID
        '''
        if db=="train":
            the_landmark_id = self.get_landmark_id(imgID)            
            subset = self.train_data.loc[self.train_data.landmark_id != the_landmark_id, "id"].values       
            return self.choose_an_imgID(subset, imgID)
            
        return None
    
    def get_url(self, imgID):
        return self.train_data.loc[self.train_data.id == imgID, "url"].values[0]
    
    @staticmethod
    def choose_an_imgID(subset, imgID):
        "from a list of imgID, randomly choose a different one."
        
        s = set(subset)
        s = s - set((None, imgID, np.nan))
        if len(s)>0:
            return np.random.choice(tuple(s),1)[0]
        return imgID 
    
    def get_landmark_id(self, imgID):
        return self.train_data.loc[self.train_data.id == imgID, "landmark_id"].values[0]
    
    def get_normalized_img(self, imgID):
        return self.api.it.transformer(self.api.il.load(imgID))
    

def test_tools():
    tg = triplet_generation()
    imgID2 = tg.get_a_similar_img("00cfd9bbf55a241e")
    print(tg.get_url("00cfd9bbf55a241e"))
    print(tg.get_url(imgID2))
    
    tg.get_a_triplet_tuple()
    
    data = tg.get_a_line_of_data()
    print(data)


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, 
                 dim=(32,32,32), n_channels=1,
                 n_classes=2, shuffle=True):
        
        'Initialization'
        self.tg = triplet_generation()
        
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        
        if list_IDs == "train":
            self.list_IDs = CaseApi.get_training_part_ids()
        elif list_IDs == "validation":
            self.list_IDs = CaseApi.get_validation_part_ids()
        else:
            self.list_IDs = list_IDs
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self.tg.get_one_input_tensor(ID)

            # Store class
            y[i] = [1, 0]
        
        y = keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, y