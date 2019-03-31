# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 15:33:19 2019

@author: zouco
"""

MODEL_NAME = "triplet_model"

from tools.keras_model_handler import save_model, load_model
triplet_model = load_model(MODEL_NAME)

from data_generator import myDataGenerator

# params = {"dim": (32,32,32)} 

# Datasets
training_generator = myDataGenerator('train',  **params)
validation_generator = myDataGenerator('validation',  **params)

# training 
triplet_model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6)

save_model(triplet_model, "trained_" + MODEL_NAME)


