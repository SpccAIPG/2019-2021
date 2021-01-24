from keras.initializers import RandomNormal
from keras.models import Model, Input, Sequential
from keras.layers import Conv3D, Conv2D, Conv3DTranspose, Conv2DTranspose, Flatten, Dense
from keras.layers import LeakyReLU, Activation, Cropping2D
from keras.layers import Dropout, Lambda, Reshape, Concatenate, BatchNormalization
from keras.layers import MaxPool2D , UpSampling2D, GlobalAveragePooling2D
from keras import initializers, regularizers, constraints, optimizers
from keras import backend as K
#from keras.utils import print_summary
from keras.applications import Xception

import cv2
import numpy as np

from utils import InstanceNormalization, gram_matrix
from parameterizing_trick import SampleLayer


def resnet_block(n_filters, input_layer, encode=True, activation='leakyrelu'):
    # weight initialization
    init = RandomNormal(stddev=0.02)

    if encode:
        g = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=init)(input_layer)
        g = InstanceNormalization(axis=-1)(g)
        g = LeakyReLU()(g)
        g = Conv2D(n_filters, (2, 2), strides=(2, 2), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization(axis=-1)(g)
        if activation == 'leakyrelu':
            g = LeakyReLU()(g)
        else:
            g = Activation(activation)
        skip_connect = MaxPool2D((2, 2), strides=(2, 2), padding='same')(input_layer)
        g = Concatenate()([g, skip_connect])
    else:
        g = Conv2DTranspose(n_filters, (3, 3), padding='same', kernel_initializer=init)(input_layer)
        g = InstanceNormalization(axis=-1)(g)
        g = LeakyReLU()(g)
        g = Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization(axis=-1)(g)
        g = LeakyReLU()(g)
        skip_connect = UpSampling2D((2, 2))(input_layer)
        g = Concatenate()([g, skip_connect])

    return g

class ContentExtractor(): 
    def __init__(self, image_shape, style_vector_shape, training=True):    
        self.image_shape = image_shape
        self.style_vector_shape = style_vector_shape
        self.training = training
    
    def construct(self):
        in_image = Input(shape=self.image_shape)

        c = Concatenate(axis = -1)([in_image, in_image, in_image])

        extractor = Xception(weights = "imagenet", include_top = False)

        c = extractor(c)

        c = Flatten()(c)

        content_z = Dense(200)(c)
        content_z = Activation('softmax')(content_z)
        
        model = Model(in_image, content_z)
        model.name = "content_extractor"
        print_summary(model)

        return model


class StyleExtractor(): 
    def __init__(self, image_shape, content_vector_shape, training=True):    
        self.image_shape = image_shape
        self.content_vector_shape = content_vector_shape
        self.training = training
    
    def construct(self):
        in_image = Input(shape = self.image_shape)
        style_vector = Input(shape = self.content_vector_shape)
        style = Dense(self.image_shape[0]**2, activation='tanh')(style_vector)
        style = Reshape((self.image_shape[0], self.image_shape[1], 1))(style)

        s = Concatenate(axis=-1)([in_image, style])

        s = resnet_block(16, s, encode=True)
        s = resnet_block(32, s, encode=True)
        s = resnet_block(64, s, encode=True)
        s = resnet_block(128, s, encode=True)
        s = resnet_block(256, s, encode=True)

        s = Flatten()(s)

        style_z = Dense(200)(s)
        style_z = Activation('softmax')(style_z)

        #mean = Dense(200)(s)
        #mean = Activation('softmax')(mean)
        #logvar = Dense(200)(s)
        #logvar = Activation('softmax')(logvar)

        #style_z = SampleLayer()([mean, logvar], training=self.training)
        
        model = Model([in_image, style_vector], style_z)
        model.name = "style_extractor"
        print_summary(model)

        return model


class Decoder():
    def __init__(self, content_vector_shape, style_vector_shape):
        self.content_vector_shape = content_vector_shape
        self.style_vector_shape = style_vector_shape
        self.init = RandomNormal(stddev=0.02)

    def construct(self):
        content_vector = Input(shape = self.content_vector_shape)
        style_vector = Input(shape = self.style_vector_shape)
        d = Concatenate(axis=1)([content_vector, style_vector])
        
        d = Dense(64)(d)
        d = LeakyReLU()(d)
        d = Reshape((8, 8, 1))(d)

        d = resnet_block(256, d, encode=False)
        d = resnet_block(128, d, encode=False)
        d = resnet_block(64, d, encode=False)
        out_image = Conv2D(1, (1, 1), padding='valid', activation='sigmoid')(d)

        # define model
        model = Model([content_vector, style_vector], out_image)
        model.name = "decoder"
        print_summary(model)
        return model


class ParallelModel():
    def __init__(self, style_vector_shape,
                    content_vector_shape,
                    image_shape,
                    content_extractor,
                    style_extractor,
                    decoder):

        self.style_vector_shape = style_vector_shape
        self.content_vector_shape = content_vector_shape
        self.image_shape = image_shape
        self.c_extractor = content_extractor
        self.s_extractor = style_extractor
        self.decoder = decoder

    def construct(self):
        main_image1 = Input(shape=self.image_shape)
        main_image2 = Input(shape=self.image_shape)
        main_content = Input(shape=self.content_vector_shape)
        compare_image1 = Input(shape=self.image_shape)
        compare_image2 = Input(shape=self.image_shape)
        compare_content = Input(shape=self.content_vector_shape)

        #main loop
        main_input_content_vector = self.c_extractor(main_image1)
        main_input_style_vector = self.s_extractor([main_image2, main_content])
        main_output_from_main_input = self.decoder([main_input_content_vector, main_input_style_vector])
        
        #compare procedure loop
        compare_input_content_vector = self.c_extractor(compare_image1)
        compare_input_style_vector = self.s_extractor([compare_image2, compare_content])
        compare_output_from_main_input = self.decoder([compare_input_content_vector, compare_input_style_vector])

        model = Model([main_image1, main_image2, main_content, compare_image1, compare_image2, compare_content],
                        [main_output_from_main_input,
                        main_input_content_vector,
                        main_input_style_vector,
                        compare_output_from_main_input,
                        compare_input_content_vector,
                        compare_input_style_vector,])

        model.name = "parallel_model"
        print_summary(model)

        return model

