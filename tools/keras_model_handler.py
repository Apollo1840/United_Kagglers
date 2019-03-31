# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 16:07:56 2019

@author: zouco
"""

from keras.models import model_from_json


def save_model(model, path):
    # serialize model to JSON
    model_json = model.to_json()
    with open(path + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model .save_weights(path + ".h5")
    print("Saved model to disk")
    

def load_model(path):
    with open(path + '.json', 'r') as json_file:
        loaded_model_json = json_file.read()

    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path + '.h5')
    print("Loaded model from disk")
    
    return loaded_model