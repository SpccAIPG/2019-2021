from keras.initializers import RandomNormal
from keras.models import Model, Input, Sequential
from keras.layers import Conv3D, Conv2D, Conv3DTranspose, Conv2DTranspose, Flatten, Dense
from keras.layers import LeakyReLU, Activation, Cropping2D, Cropping1D
from keras.layers import Dropout, Lambda, Reshape, Concatenate, BatchNormalization
from keras.layers import MaxPooling2D , UpSampling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers import Layer
from keras import initializers, regularizers, constraints, optimizers
from keras.models import load_model
from keras.losses import KLDivergence
from keras import backend as K
from keras.utils import print_summary
from keras.applications import VGG16, VGG19, MobileNetV2
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import preprocess_input

import cv2
import numpy as np

from utils import InstanceNormalization, gram_matrix
from parameterizing_trick import SampleLayer


def resnet_block(n_filters, input_layer, encode=True, skip=True, activation='leakyrelu'):
    # weight initialization
    init = RandomNormal(stddev=0.02)

    if encode:
        g = Conv2D(n_filters, (3, 3), padding='same')(input_layer)
        g = BatchNormalization(axis=-1)(g)
        g = LeakyReLU()(g)
        g = Conv2D(n_filters, (2, 2), strides=(2, 2), padding='same')(g)
        g = BatchNormalization(axis=-1)(g)

        if activation == 'leakyrelu':
            g = LeakyReLU()(g)
        else:
            g = Activation(activation)

        if skip:
            skip_connect = AveragePooling2D((2, 2), strides=(2, 2), padding='same')(input_layer)
            g = Concatenate()([g, skip_connect])
    else:
        g = Conv2DTranspose(n_filters, (3, 3), padding='same')(input_layer)
        g = BatchNormalization(axis=-1)(g)
        g = LeakyReLU()(g)
        g = Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(g)
        g = BatchNormalization(axis=-1)(g)
        g = LeakyReLU()(g)

        if skip:
            skip_connect = UpSampling2D((2, 2))(input_layer)
            g = Concatenate()([g, skip_connect])

    return g

class Encoder():
    def __init__(self, image_shape):
        self.image_shape = image_shape

    def construct(self):
        #img = Input(shape=self.image_shape)
        img = Input(shape=self.image_shape)

        s = resnet_block(16, img, skip=True, encode=True)
        s = resnet_block(32, s, skip=True, encode=True)
        s = resnet_block(64, s, skip=True, encode=True)
        s = resnet_block(128, s, skip=True, encode=True)
        s = resnet_block(256, s, skip=True, encode=True)
        s = Flatten()(s)
        s = Dense(768)(s)
        
        model = Model(img, s)
        model.name = 'encoder'
        return model


class Decoder0():
    def __init__(self):
        pass

    def construct(self):
        #z1 = Input(shape=(512, ))
        #z2 = Input(shape=(256, ))
        z_combined = Input(shape=(768, )) #Concatenate(axis=1)([z1, z2])

        s = Dense(256*4*4)(z_combined)
        s = Activation("tanh")(s)
        s = Reshape((4, 4, 256))(s)
        s = resnet_block(128, s, skip=False, encode=False)
        s = resnet_block(64, s, skip=False, encode=False)
        s = resnet_block(32, s, skip=False, encode=False)
        s = resnet_block(16, s, skip=False, encode=False)
        s = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(s)
        
        model = Model(z_combined, s) # [z1, z2]
        model.name = "decoder"

        return model


class AutoEncoder():
    def __init__(self, image_shape, encoder, decoder):
        self.image_shape = image_shape
        self.e = encoder
        self.d = decoder
        self.content_vector_shape = 512
        self.style_vector_shape = 256

    def construct(self):
        main_img = Input(shape=self.image_shape)
        c_img = Input(shape=self.image_shape)
        s_img = Input(shape=self.image_shape)

        main_vector = self.e(main_img)
        main_vector = Reshape((self.content_vector_shape + self.style_vector_shape, 1))(main_vector)
        main_c = Cropping1D(cropping=(0, self.style_vector_shape))(main_vector)
        main_c = Reshape((self.content_vector_shape, ))(main_c)
        main_s = Cropping1D(cropping=(self.content_vector_shape, 0))(main_vector)
        main_s = Reshape((self.style_vector_shape, ))(main_s)

        main_fake = self.d(Reshape((self.content_vector_shape + self.style_vector_shape, ))(main_vector))
        
        compare_vector_c = self.e(c_img)
        compare_vector_c = Reshape((self.content_vector_shape + self.style_vector_shape, 1))(compare_vector_c)
        compare_c = Cropping1D(cropping=(0, self.style_vector_shape))(compare_vector_c)
        compare_c = Reshape((self.content_vector_shape, ))(compare_c)

        compare_vector_s = self.e(s_img)
        compare_vector_s = Reshape((self.content_vector_shape + self.style_vector_shape, 1))(compare_vector_s)
        compare_s = Cropping1D(cropping=(self.content_vector_shape, 0))(compare_vector_s)
        compare_s = Reshape((self.style_vector_shape, ))(compare_s)

        compare_fake = self.d(Concatenate(axis = 1)([compare_c, compare_s]))

        def kl(x):
            #if K.ndim(x) == 1:
            kld = KLDivergence()
            return kld(x[0], x[1])
            #else: 
            #    raise ValueError('The input tensor should be either a [y_true, y_pred] list')

        model = Model([main_img, c_img, s_img],
                        [main_fake, compare_fake, main_c, main_s, compare_c, compare_s])
        print_summary(model)
        model.name = 'autoencoder'
        return model