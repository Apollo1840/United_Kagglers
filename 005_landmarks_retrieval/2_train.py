# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 15:33:19 2019

@author: zouco
"""

MODEL_NAME = "triplet_model"

from tools.keras_model_handler import save_model, load_model
triplet_model = load_model(MODEL_NAME)

from data_generator import DataGenerator

# params = {"dim": (32,32,32)} 

# Datasets
training_generator = DataGenerator('train', labels=None)
validation_generator = DataGenerator('validation', labels=None) #because labels are always [1,0]

# compiling
triplet_model.compile(optimizer='sgd', loss='categorical_crossentropy')

# training 
print("################# fittinng.... ####################")
triplet_model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=4)

print("################# saving   ########################")
save_model(triplet_model, "trained_" + MODEL_NAME)


