import tensorflow
import numpy as np
import cv2
import statistics
import os

from keras.initializers import RandomNormal
from keras.models import Model, Input, Sequential
from keras.layers import Conv3D, Conv2D, Conv3DTranspose, Conv2DTranspose, Flatten, Dense, Cropping2D
from keras.layers import LeakyReLU, Activation
from keras.layers import Dropout, Lambda, Reshape, Concatenate, BatchNormalization
from keras.models import load_model
from keras import initializers, regularizers, constraints, optimizers
from tensorflow.keras import backend as K
from keras.utils import print_summary
from keras.utils import CustomObjectScope
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam as Adam
from keras.applications.vgg19 import VGG19
from keras.losses import kullback_leibler_divergence
from keras.losses import KLD
from keras.backend import variable
from keras.backend import is_keras_tensor

from numpy import asarray
from os import listdir
from matplotlib import image

from utils import InstanceNormalization, create_main_img, create_comparison_img
from models import ContentExtractor, Decoder, ParallelModel, FeatureExtractor

#ITERATIONS = 30 # not using anymore as we can iterate all the words in single fit
#magnify = 6 # not using anymore as we only feed in single word
font_no = 300
N_WORDS = 30
image_shape = (75 , 75, 1) # or 3, same reason as above
style_vector_shape = (75 , ) # just a random number right now
content_vector_shape = (75 , ) # just a random number right now
#pred_list = list() # what is the use of this?
loaded_images = np.load('D:/AI Project Group/FONT/DATA/loaded_data.npy', allow_pickle=True)

# Creating models
ContentExtractor = ContentExtractor(image_shape, style_vector_shape)
content_extractor = ContentExtractor.construct()
Decoder = Decoder(content_vector_shape, style_vector_shape)
decoder = Decoder.construct()
ParallelModel = ParallelModel(style_vector_shape, content_vector_shape, image_shape, content_extractor, decoder)
Model = ParallelModel.construct()

FeatureExtractor = FeatureExtractor(image_shape)
feature_extractor = FeatureExtractor.construct()

Model.compile(optimizer="adam", loss = ["binary_crossentropy", "binary_crossentropy", KLD, KLD])

# Generating input data
combined_images = np.load('D:/AI Project Group/FONT/DATA/whole_data.npy', allow_pickle = True)

combined_images_list = []
for font in range(300):
    combined_images_list.append([])
    for i in range(N_WORDS * 75):
        combined_images_list[font].append(combined_images[font][i])
combined_images = np.array(combined_images_list)/255
'''
inter_combined = []

for i in range(font_no):
    intermed = []
    for width in range(magnify):
        intermed.append(combined_images[i][int(N_WORDS * 75 / magnify * width) : int(N_WORDS * 75 / magnify * (width + 1 ))])
    new_width = np.concatenate(intermed[0:magnify], axis = 1)
    inter_combined.append(new_width)
combined_images = np.array(inter_combined)

print(combined_images.shape)
'''
#for iteration in range(ITERATIONS):
main_img = create_main_img(combined_images, iteration, N_WORDS, font_no)
comparison_img = create_comparison_img(iteration, main_img, N_WORDS, font_no, combined_images)
concat_main_img = np.concatenate((main_img[iteration], main_img[iteration], main_img[iteration]), axis = 3)
concat_compare_img = np.concatenate((comparison_img[iteration], comparison_img[iteration], comparison_img[iteration]), axis = 3)
for font in font_no:
    loss = Model.fit([main_img[font],
                        feature_extractor.predict(comparison_img[font]),
                        comparison_img[font],
                        feature_extractor.predict(concat_compare_img[font])],
                        [main_img[font],
                        comparison_img[font],
                        content_extractor.predict(main_img[font]),
                        content_extractor.predict(comparison_img[font])],
                        epochs = 30)

    print('Iteration %s\tLoss: %s' % (font + 1, loss))
