# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 15:32:16 2019

@author: zouco
"""

from keras.layers import Input,Lambda,subtract,GlobalMaxPooling2D,Dense,GlobalAveragePooling2D,concatenate,Activation
from keras.applications.xception import Xception as Net
from keras.models import Model

def create_model(d1,d2):
    # The triplet network takes 3 input images: 2 of the same class and 1 out-of-class sample
    input_tensor1 = Input(shape=(d1, d2, 3))
    input_tensor2 = Input(shape=(d1, d2, 3))
    input_tensor3 = Input(shape=(d1, d2, 3))
    # load a pretrained model (try, except block because the kernel would not let me download the weights for the network)
    try:
        base_model = Net(input_shape=(d1,d2,3),weights='imagenet',include_top=False)
    except:
        print('Could not download weights. Using random initialization...')
        base_model = Net(input_shape=(d1,d2,3),weights=None,include_top=False)
    # predefine a summation layer for calculating the distances:
    # the weights of this layer will be set to ones and fixed  (since they
    # are shared we could also leave them trainable to get a weighted sum)
    summation = Dense(1,activation='linear',kernel_initializer='ones',bias_initializer='zeros',name='summation')
    # feed all 3 inputs into the pretrained keras model
    x1 = base_model(input_tensor1)
    x2 = base_model(input_tensor2)
    x3 = base_model(input_tensor3)
    
    # flatten/summarize the models output:
    # (here we could also use GlobalAveragePooling or simply Flatten everything)
    x1 = GlobalMaxPooling2D()(x1)
    x2 = GlobalMaxPooling2D()(x2)
    x3 = GlobalMaxPooling2D()(x3)
    # calculate something proportional to the euclidean distance
    #   a-b
    d1 = subtract([x1,x2])
    d2 = subtract([x1,x3])
    #   (a-b)**2
    d1 = Lambda(lambda val: val**2)(d1)
    d2 = Lambda(lambda val: val**2)(d2)
    # sum((a-b)**2)
    d1 = summation(d1)
    d2 = summation(d2)
    #  concatenate both distances and apply softmax so we get values from 0-1
    d = concatenate([d1,d2])
    d = Activation('softmax')(d)
    
    # build the model and show a summary
    model = Model(inputs=[input_tensor1,input_tensor2,input_tensor3], outputs=d)
    
    # a second model that can be used as metric between input 1 and input 2
    metric = Model(inputs=[input_tensor1,input_tensor2], outputs=d1)
    model.summary()
    # draw the network (it looks quite nice)
    try:
        from keras.utils.vis_utils import plot_model as plot
        plot(model, to_file = 'Triplet_Dense121.png')
    except ImportError:
        print('It seems like the dependencies for drawing the model (pydot, graphviz) are not installed')
    # fix the weights of the summation layer (since the weight of this layer
    # are shared we could also leave them trainable to get a weighted sum)
    for l in model.layers:
        if l.name == 'summation':
            print('fixing weights of summation layer')
            l.trainable=False
    # compile model
    model.compile(optimizer='sgd', loss='categorical_crossentropy')
        
    return model,metric
    
triplet_model, metric = create_model(229,229)

from tools.keras_model_handler import save_model
save_model(triplet_model, "triplet_model")

