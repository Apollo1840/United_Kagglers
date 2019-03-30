# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 15:33:19 2019

@author: zouco
"""

MODEL_NAME = "triplet_model"

from tools.keras_model_handler import save_model, load_model
triplet_model = load_model(MODEL_NAME)


"""
here we train the model:
    triplet_model.fit(....)


"""

save_model(triplet_model, "trained_" + MODEL_NAME)


