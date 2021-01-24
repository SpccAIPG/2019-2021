import os
from PIL import Image
import numpy as np
import pygame
import io
import cv2 as cv

'''
def gray(im):
    im = 255 * (im / im.max())
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    return ret

pygame.init()
display = pygame.display.set_mode((350, 350))
x = np.arange(0, 300)
y = np.arange(0, 300)
X, Y = np.meshgrid(x, y)
Z = X + Y
Z = 255 * Z / Z.max()
Z = gray(Z)
surf = pygame.surfarray.make_surface(Z)

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    display.blit(surf, (0, 0))
    pygame.display.update()
pygame.quit()
'''
def get_rand_idx():
    randint1 = np.random.randint(65, 90)
    randint2 = np.random.randint(97, 122)
    randint0 = np.random.randint(0, 1)
    if randint0:
        return str(randint1)
    else:
        return str(randint2)
#Path.iterdir()
import zipfile

# unicode for english --> 65-90 97-122
# font index --> 1 - 3000+
def data_loader(path, n_font, batch_size=36):
    batch = 0
    main_img = []
    c_img = []
    s_img = []
    dis_img = []
    
    with zipfile.ZipFile('C:/Users/Lennon no microsoft/Desktop/AI materials July/imagesfont200-20200724T024222Z-001.zip', 'r') as f:
        names1 = f.namelist()
    with zipfile.ZipFile('C:/Users/Lennon no microsoft/Desktop/AI materials July/imagesfont200-20200724T024222Z-002.zip', 'r') as f:
        names2 = f.namelist()
    
    #with zipfile.ZipFile('C:/Users/Lennon no microsoft/Desktop/AI materials July/English.zip', 'r') as f:
    #    names = f.namelist()
    
    names = names1 + names2
    f = zipfile.ZipFile('C:/Users/Lennon no microsoft/Desktop/AI materials July/imagesfont200-20200724T024222Z-001.zip', 'r')
    g = zipfile.ZipFile('C:/Users/Lennon no microsoft/Desktop/AI materials July/imagesfont200-20200724T024222Z-002.zip', 'r')
    
    #f = zipfile.ZipFile('C:/Users/Lennon no microsoft/Desktop/AI materials July/English.zip', 'r')
    

    for i, img in enumerate(names, 0):
        font = img.split('_')[0] # img = '1_64' --> 1 is the font index; 64 is the unicode for the word
        word = img.split('_')[1] # img = '1_64' --> 1 is the font index; 64 is the unicode for the word
        
        if img in names1:
            img_data = f.read(img)
        elif img in names2:
            img_data = g.read(img)
        
        #img_data = f.read(img)        
        imgfile = io.BytesIO(img_data)
        main_image = Image.open(imgfile)
        main_data = np.asarray(main_image)
        #print(main_data.shape)
        main_img.append(main_data)
        
        ok = False
        while not ok:
            try:
                font_index = str(np.random.randint(1, 1125)) #n_font
                c_name = font_index + '_' + word
                if c_name in names1:
                    img_data = f.read(c_name)
                elif c_name in names2:
                    img_data = g.read(c_name)
                #img_data = f.read(c_name)   
                imgfile = io.BytesIO(img_data)
                c_image = Image.open(imgfile)
                c_data = np.asarray(c_image)
                c_data = np.reshape(c_data, (main_data.shape[0], main_data.shape[1], 1))
                c_img.append(c_data) 
                ok = True
            except FileNotFoundError:
                ok = False

        ok = False
        while not ok:
            try:
                word_index = str(np.random.randint(1, 52)) #n_word
                s_name = font + '_' + word_index + '.png'
                if s_name in names1:
                    img_data = f.read(s_name)
                elif s_name in names2:
                    img_data = g.read(s_name)
                #img_data = f.read(s_name)
                imgfile = io.BytesIO(img_data)
                s_image = Image.open(imgfile)
                s_data = np.asarray(s_image)
                s_data = np.reshape(s_data, (main_data.shape[0], main_data.shape[1], 1))
                s_img.append(s_data)
                ok = True
            except FileNotFoundError:
                ok = False
        if i >= 1:
            break
        if i % int(batch_size/ 9) == 0:
            main = np.array(main_img)
            content = np.array(c_img)
            style = np.array(s_img)
            #disentangle = np.array(dis_img)

            main_img = []
            c_img = []
            s_img = []
            dis_img = []
            #main_east, content_east, style_east, main_north, content_north, style_north, main_cut, content_cut, style_cut, main_rotate, content_rotate, style_rotate = [], [], [], [], [], [], [], [], [], [], [], []


            yield main, content, style
        