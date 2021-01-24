import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 
import tensorflow as tf
import numpy as np
import cv2
import pickle
from PIL import Image
import keras.backend as K

from keras import initializers, regularizers, constraints, optimizers
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam, SGD
from keras.losses import KLDivergence, BinaryCrossentropy, MeanAbsoluteError
from keras.applications import VGG16
from keras.models import load_model
#from keras.utils import print_summary
#rom keras.models import Model

from numpy import asarray
from os import listdir
from matplotlib import image
from matplotlib import pyplot as plt
from math import floor
from tqdm import tqdm
from statistics import mean

from utils import InstanceNormalization, get_layer_output_grad #gray2bgr
from models import ContentExtractor, StyleExtractor, Decoder, ParallelModel
from new_models import Encoder1, Encoder2, Decoder0, AutoEncoder# AutoEncoder2
#from sep_models import Encoder, Decoder0, AutoEncoder
from data_loader import data_loader#

from varname import nameof

EPOCH = 100
BATCH_SIZE = 1
TRAIN_RATIO = 1 # normal should be 0.8

n_fonts = 50
n_words = 50
vgg_image_shape = (64, 64, 3, )
image_shape = (64, 64, 1, ) # or 3
style_vector_shape = (512, )
content_vector_shape = (512, )

n_batches = n_fonts * n_words // BATCH_SIZE
n_t_batches = n_batches * TRAIN_RATIO 
n_v_batches = n_batches - n_t_batches

PATH = 'C:/Users/Lennon no microsoft/Desktop/AI materials September/English'
pretrain_path = 'D:/AI Project Group/FONT/content_extractor_parallel/models/Model_v2'
#DIRECTORY = "C:/Users/Lennon no microsoft/Desktop/AI materials September/21. try overfit one word/"

# Creating models
KLDLoss = KLDivergence(reduction='sum')


E1 = Encoder1(image_shape)
e1 = E1.construct()
E2 = Encoder2(image_shape)
e2 = E2.construct()
D = Decoder0()
d = D.construct()
DAE = AutoEncoder(image_shape, e1, e2, d)
Model = DAE.construct()

'''
E = Encoder(image_shape)
e = E.construct()
D = Decoder0()
d = D.construct()
DAE = AutoEncoder(image_shape, e, d)
Model = DAE.construct()
'''

Model.compile(optimizer=Adam(learning_rate=1e-4),
                loss=['mae', 'mae', 'mae', 'mae', 'mae', 'mae'], 
                loss_weights=[1.0, 1.0, 0.001, 0.001, 0.001, 0.001])


#Model = load_model('models/Model_v2.h5')
print(Model.summary)

t_loss_hist = []
v_loss_hist = []
for epoch in range(1, EPOCH+1):
    epoch_loss = []
    n_val = 1
    batch_train_loss = []
    batch_val_loss = []

    pbar = tqdm(total=n_batches, postfix={'loss': 0}) # just to test
    for i, (imgs, c_img, s_img) in enumerate(data_loader(PATH, n_fonts, batch_size=BATCH_SIZE)):
        #print("i = " + str(i)) #just to check for the index of the image, see if it is 0, 1, 2, 3 or 1, 2, 3, 4 for max = 4 
        c_img = np.reshape(c_img, (c_img.shape[0], c_img.shape[1], c_img.shape[2], 1))
        s_img = np.reshape(s_img, (s_img.shape[0], s_img.shape[1], s_img.shape[2], 1))

        main_c = Model.predict([imgs, c_img, s_img])[2]
        main_s = Model.predict([imgs, c_img, s_img])[3]
        compare_c = Model.predict([imgs, c_img, s_img])[4]
        compare_s = Model.predict([imgs, c_img, s_img])[5]

        # training set   
        if i < n_t_batches:
            #print(s_img)
            #cv2.imshow("images", s_img[0]) # peek input
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            loss = Model.train_on_batch([imgs, c_img, s_img],
                                        [imgs, imgs, compare_c, compare_s, main_c, main_s])

            main_mae = loss[0]
            compare_mae = loss[1]
            kl = mean([loss[2], loss[3], loss[4], loss[5]])

            all_losses = {"main_mae": main_mae, "compare_mae": compare_mae, "kl": kl}

            batch_train_loss.append(loss)

        # validation set
        elif i < n_batches:
            pred = Model.predict([imgs, c_img, s_img])
            main_pred = pred[0]
            compare_pred = pred[1]
            main_c_vector = pred[2]
            main_s_vector = pred[3]
            compare_c_vector = pred[4]
            compare_s_vector = pred[5]
            #cv2.imshow("images", main_pred[0]) #peek products

            main_mae = MeanAbsoluteError()(imgs, main_pred).numpy()
            compare_mae = MeanAbsoluteError()(imgs, compare_pred).numpy()
            main_c_kl = KLDLoss(compare_c, main_c_vector).numpy()
            main_s_kl = KLDLoss(compare_s, main_s_vector).numpy()
            compare_c_kl = KLDLoss(main_c, compare_c_vector).numpy()
            compare_s_kl = KLDLoss(main_s, compare_s_vector).numpy()
            kl = mean([main_c_kl, main_s_kl, compare_c_kl, compare_s_kl])
            
            all_losses = {"main_mae": main_mae, "compare_mae": compare_mae, "kl": kl}

            batch_val_loss.append([main_mae, compare_mae, main_c_kl, main_s_kl, compare_c_kl, compare_s_kl])

        else:
            break
    

        pbar.update(1) #just to test
        pbar.set_postfix({'main_mae': main_mae, 'compare_mae': compare_mae, 'kl': kl}) # just to test
        if i % BATCH_SIZE == 0:
                pred = Model.predict([imgs, c_img, s_img])
                main_pred = pred[0]
                #print("main_pred" + str(main_pred)) # just to check
                compare_pred = pred[1]

                all_imgs = {
                    "show_img": np.reshape(imgs, (imgs.shape[0]*imgs.shape[1], imgs.shape[2], imgs.shape[3])),
                    "main_pred": np.reshape(main_pred, (main_pred.shape[0]*main_pred.shape[1], main_pred.shape[2], main_pred.shape[3])),
                    "compare_pred": np.reshape(compare_pred, (compare_pred.shape[0]*compare_pred.shape[1], compare_pred.shape[2], compare_pred.shape[3])),
                    "c_show": np.reshape(c_img, (c_img.shape[0]*c_img.shape[1], c_img.shape[2], c_img.shape[3])),
                    "s_show": np.reshape(s_img, (s_img.shape[0]*s_img.shape[1], s_img.shape[2], s_img.shape[3]))
                }

                '''
                show_img = np.reshape(imgs, (imgs.shape[0]*imgs.shape[1], imgs.shape[2], imgs.shape[3]))
                main_pred = np.reshape(main_pred, (main_pred.shape[0]*main_pred.shape[1], main_pred.shape[2], main_pred.shape[3]))
                compare_pred = np.reshape(compare_pred, (compare_pred.shape[0]*compare_pred.shape[1], compare_pred.shape[2], compare_pred.shape[3]))
                c_show = np.reshape(c_img, (c_img.shape[0]*c_img.shape[1], c_img.shape[2], c_img.shape[3]))
                s_show = np.reshape(s_img, (s_img.shape[0]*s_img.shape[1], s_img.shape[2], s_img.shape[3]))
                '''

                j = 0
                for im in all_imgs:
                    if j == 0:
                        img = all_imgs[im]
                    else:
                        img = np.concatenate((img, all_imgs[im]), axis=1)
                    j += 1
                
                # to print the whole numpy array not dot dot dot
                import numpy
                import sys
                numpy.set_printoptions(threshold=sys.maxsize)
                #print(img)
                #print(img.shape) # just to check
                #cv2.imshow("image", img)
                
                
                if not os.path.exists('validation/%s' % epoch):
                        os.makedirs('validation/%s' % epoch)
                        os.makedirs('validation/%s/train' % epoch)
                        os.makedirs('validation/%s/val' % epoch)

                if i < n_t_batches:
                    #im = Image.fromarray(img.astype("float32"))
                    #im.save('validation/%s/train/train_%s.jpg' % (epoch, i), "L")
                    cv2.imwrite('validation/%s/train/train_%s.jpg' % (epoch, i), img*255)
                    #print("writen") # just to check
                    
                    os.makedirs('validation/%s/train/data' % epoch)
                    os.makedirs('validation/%s/train/data/pickle' % epoch)
                    pickle.dump(all_losses, open('validation/%s/train/data/pickle/%s.h5' % (epoch, i), 'wb'))
                    
                    os.makedirs('validation/%s/train/data/txt' % epoch)
                    f = open('validation/%s/train/data/txt/%s.txt' % (epoch, i), "a")
                    f.write(str(all_losses))
                    f.close
                else:
                    #im = Image.fromarray(img)
                    #im.save('validation/%s/val/validation_%s.jpg' % (epoch, i), "L")
                    #print(img)
                    cv2.imwrite('validation/%s/val/validation_%s.jpg' % (epoch, i), img*255)
                    #print("writen") # just to check

                    os.makedirs('validation/%s/val/data' % epoch)
                    os.makedirs('validation/%s/val/data/pickle' % epoch)
                    pickle.dump(all_losses, open('validation/%s/val/data/pickle/%s.h5' % (epoch, i), 'wb'))
                    
                    os.makedirs('validation/%s/val/data/txt' % epoch)
                    f = open('validation/%s/val/data/txt/%s.txt' % (epoch, i), "a")
                    f.write(all_losses)
                    f.close
                
                
    
    # start of model saving

    # create directories for models to be saved
    if not os.path.exists('models/%s' % epoch):
            os.makedirs('models/%s' % epoch)
    
    # save the models in the model folder/epoch number/(Model or encoder or decoder)
    Model.save('models/%s/Model_v2.h5' % epoch)
    e1.save('models/%s/encoder_1_v2.h5' % epoch)
    e2.save('models/%s/encoder_2_v2.h5' % epoch)
    d.save('models/%s/decoder_v2.h5' % epoch)

    # end of model saving
    

    # start to store loss information in history lists
    t_loss_hist.append(batch_train_loss)
    t_loss_hist_plt = []
    for i in t_loss_hist:
        for j in i:
            t_loss_hist_plt.append(j)

    v_loss_hist.append(batch_val_loss)

    # print loss
    print('Epoch %s\tLoss: %s' % (epoch, v_loss_hist[-1]))
    
    def save_loss(loss_name):
        f = open("losses/" + nameof(loss_name) + "_epoch_" + str(epoch) + ".txt", "a")
        pickle.dump({'train_loss': t_loss_hist, 'train_loss_plt': t_loss_hist_plt, 'val_loss': v_loss_hist}, open("losses/" + nameof(loss_name) + "_epoch_" + str(epoch) + ".txt", 'wb'))
    
    for i in [t_loss_hist, t_loss_hist_plt, v_loss_hist]:
        save_loss(i)
    
    # end of storing loss information in history lists

# start of plotting graphs
    
plt.plot(t_loss_hist_plt)
plt.ylabel('loss')
plt.xlabel("epoch")
plt.show()

pickle.dump({'train_loss': t_loss_hist, 'val_loss': v_loss_hist}, open('models/Model_v2.h5', 'wb'))
