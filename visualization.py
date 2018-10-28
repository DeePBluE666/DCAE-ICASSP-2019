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
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


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
		
	
#-------------------Build model-------------------#
base_model = DenseNet121(weights='None', include_top=True)
x = base_model.output
x = BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu', name = 'outputs')(x)
x = BatchNormalization()(x)
predictions = Dense(4, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
	
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) 

get_output = Model(inputs=model.input,outputs=model.get_layer('outputs').output)
	
early_stopping =EarlyStopping(monitor='val_acc', patience=5)
callbacks = [early_stopping,]

model.fit_generator(
        train_generator,
        steps_per_epoch=1024,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=256, callbacks=callbacks)


#------------------------------visualization--------------------------------#
features = get_output.predict_generator(validation_generator)
label = validation_generator.classes

min_max_scaler = MinMaxScaler()
ss = StandardScaler()

features = min_max_scaler.fit_transform(features)


#---------------------T-SNE-----------------------#
X_tsne = TSNE().fit_transform(features)

data_tsne = pd.DataFrame(X_tsne)

data_tsne['label'] = label

plt_data_tsne = data_tsne

scatter =Scatter("", title_pos ='center')

scatter.add("k", plt_data_tsne[plt_data_tsne['label'] == 0][1], \
            plt_data_tsne[plt_data_tsne['label'] == 0][0], legend_top='bottom')
scatter.add("l", plt_data_tsne[plt_data_tsne['label'] == 1][1], \
            plt_data_tsne[plt_data_tsne['label'] == 1][0], legend_top='bottom')
scatter.add("p", plt_data_tsne[plt_data_tsne['label'] == 2][1], \
            plt_data_tsne[plt_data_tsne['label'] == 2][0], legend_top='bottom')
scatter.add("t", plt_data_tsne[plt_data_tsne['label'] == 3][1], \
            plt_data_tsne[plt_data_tsne['label'] == 3][0], legend_top='bottom')


scatter.render()
scatter


#-----------------PCA-------------------------#

X_pca = PCA(n_components=2, random_state=10).fit_transform(features)

X_pca = pd.DataFrame(X_pca)

X_pca['pred'] = pred
X_pca['label'] = label

plt_data_pca = X_pca

scatter =Scatter("\n The Scatter of last layer's outputs of DenseNet using PCA", title_pos ='center')

scatter.add("k", plt_data_pca[plt_data_pca['label'] == 0][1], \
            plt_data_pca[plt_data_pca['label'] == 0][0], legend_top='bottom')
scatter.add("l", plt_data_pca[plt_data_pca['label'] == 1][1], \
            plt_data_pca[plt_data_pca['label'] == 1][0], legend_top='bottom')
scatter.add("p", plt_data_pca[plt_data_pca['label'] == 2][1], \
            plt_data_pca[plt_data_pca['label'] == 2][0], legend_top='bottom')
scatter.add("t", plt_data_pca[plt_data_pca['label'] == 3][1], \
            plt_data_pca[plt_data_pca['label'] == 3][0], legend_top='bottom')


scatter.render()
scatter