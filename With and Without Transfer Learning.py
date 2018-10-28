from keras.applications.inception_v3 import InceptionV3
import gc
import os
from keras.preprocessing import image
from keras import backend as K
import os
import tensorflow as tf
from keras.utils import plot_model
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Lambda, Embedding,constraints,GlobalAveragePooling2D,\
Dropout, Activation,GRU,Bidirectional,Subtract, Permute, TimeDistributed, Reshape
from keras.layers import Conv1D,Conv2D,MaxPooling2D,GlobalAveragePooling1D,GlobalMaxPooling1D, MaxPooling1D, Flatten
from keras.layers import CuDNNGRU, CuDNNLSTM, SpatialDropout1D,Layer, initializers, regularizers
from keras.layers.merge import concatenate, Concatenate, Average, Dot, Maximum, Multiply, Subtract
from keras.models import Model
from keras.optimizers import RMSprop,Adam,SGD
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from keras.activations import softmax
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet121



#----------------------------------------get data----------------------------------#

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
        
train_generator = train_datagen.flow_from_directory(                  
        '/home/dataset/ultraspnic/ult_data_train/',  
        target_size=(150, 150),                                       
        batch_size=32,
        class_mode='categorical')                                          


test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow_from_directory(
        '/home/dataset/ultraspnic/ult_data_test/',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')
		
		
def initialize_model(pretrain_model, num_unfreezed_layers, show = True):
    
    if pretrain_model == 'inceptionV3':
        base_model = InceptionV3(weights='imagenet', include_top=False)
        
    elif pretrain_model == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False)
        
    elif pretrain_model == 'resnet':
        base_model = ResNet50(weights='imagenet', include_top=False)
        
    elif pretrain_model == 'densenet':
        base_model = DenseNet121(weights='imagenet', include_top=False)
        
    else:
        raise ValueError ('unexpected pretrain model %s'%pretrain_model)
        
    #add GlobelAvg2D layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    #add 2 FC layers 
    x = Dense(512, activation='relu')(x)

    predictions = Dense(4, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    #unfreeze layers
    if num_unfreezed_layers != 'all':
        for layer in base_model.layers[:-num_unfreezed_layers]:
            layer.trainable = False
    if show:
        print('*'*50)
        print('%s pretrain model is employed' %pretrain_model)
        print('unfreezed layer is %s'%num_unfreezed_layers)
        print('*'*50)
    #compile model
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
	
def main_function(pretrain_model):
    
    if pretrain_model == 'inceptionV3':
        layer_list = [0, 60, 120, 180, 240, 'all']
        
    elif pertrain_model == 'vgg16':
        layer_list = [0, 3, 6, 9, 12, 'all']
        
    elif pretrain_model == 'resnet':
        layer_list = [0, 35, 70, 105, 140, 'all']
        
    elif pretrain_model == 'densenet':
        layer_list = [0, 85, 170, 255, 340, 'all']
        
    else:
        raise ValueError ('unexpected pretrain model %s'%pretrain_model)
        
    
	

#------------------------------pre-train model with random weights--------------------------------------#
base_model = DenseNet121(weights='None', include_top=False)
x = base_model.output
x = BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
predictions = Dense(4, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
	
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) 
	
early_stopping =EarlyStopping(monitor='val_acc', patience=5)
callbacks = [early_stopping,]

model.fit_generator(
        train_generator,
        steps_per_epoch=1024,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=256, callbacks=callbacks)

		
#-----------------------------pre-train model with different freeze ratio---------------------------------#
#take DenseNet as an example


#all layers are trainable

model = initialize_model('densenet', 0, show = False)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) 
	
early_stopping =EarlyStopping(monitor='val_acc', patience=5)
callbacks = [early_stopping,]

model.fit_generator(
        train_generator,
        steps_per_epoch=1024,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=256, callbacks=callbacks)
		
#first 85 layers are untrainable

model = initialize_model('densenet', 85, show = False)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) 
	
early_stopping =EarlyStopping(monitor='val_acc', patience=5)
callbacks = [early_stopping,]

model.fit_generator(
        train_generator,
        steps_per_epoch=1024,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=256, callbacks=callbacks)

#first 170 layers are untrainable

model = initialize_model('densenet', 170, show = False)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) 
	
early_stopping =EarlyStopping(monitor='val_acc', patience=5)
callbacks = [early_stopping,]

model.fit_generator(
        train_generator,
        steps_per_epoch=1024,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=256, callbacks=callbacks)
		
#first 255 layers are untrainable

model = initialize_model('densenet', 255, show = False)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) 
	
early_stopping =EarlyStopping(monitor='val_acc', patience=5)
callbacks = [early_stopping,]

model.fit_generator(
        train_generator,
        steps_per_epoch=1024,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=256, callbacks=callbacks)
		
#first 340 layers are untrainable

model = initialize_model('densenet', 340, show = False)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) 
	
early_stopping =EarlyStopping(monitor='val_acc', patience=5)
callbacks = [early_stopping,]

model.fit_generator(
        train_generator,
        steps_per_epoch=1024,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=256, callbacks=callbacks)
		
#all layers are untrainable

model = initialize_model('densenet', 'all', show = False)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) 
	
early_stopping =EarlyStopping(monitor='val_acc', patience=5)
callbacks = [early_stopping,]

model.fit_generator(
        train_generator,
        steps_per_epoch=1024,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=256, callbacks=callbacks)
		
#----------------------------------------get data----------------------------------#

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
        
train_generator = train_datagen.flow_from_directory(                  
        '/home/dataset/ultraspnic/ult_data_train/',  
        target_size=(150, 150),                                       
        batch_size=32,
        class_mode='categorical')                                          

train_datagen2 = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)		

train_generator2 = train_datagen2.flow_from_directory(                  
        '/home/dataset/ultraspnic/ult_data_train2/',  
        target_size=(150, 150),                                       
        batch_size=32,
        class_mode='categorical')
		
test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow_from_directory(
        '/home/dataset/ultraspnic/ult_data_test/',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')
		
	

#------------------------------build pre-train model--------------------------------------#
base_model = DenseNet121(weights='None', include_top=False)
x = base_model.output
x = BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
predictions = Dense(4, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
	
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) 
	
early_stopping =EarlyStopping(monitor='val_acc', patience=5)
callbacks = [early_stopping,]

model.fit_generator(
        train_generator,
        steps_per_epoch=1024,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=256, callbacks=callbacks)

		
#-----------------------------Transfer Leanring---------------------------------#
#take DenseNet as an example


#all layers are trainable

	
early_stopping =EarlyStopping(monitor='val_acc', patience=5)
callbacks = [early_stopping,]

model.fit_generator(
        train_generator2,
        steps_per_epoch=1024,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=256, callbacks=callbacks)
		
