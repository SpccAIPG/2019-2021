import os
from PIL import Image
import numpy as np


def get_rand_idx():
    randint1 = np.random.randint(65, 90)
    randint2 = np.random.randint(97, 122)
    randint0 = np.random.randint(0, 1)
    if randint0 == 0:
        return str(randint1)
    else:
        return str(randint2)


# unicode for english --> 65-90 97-122
# font index --> 1 - 3000+
def data_loader(path, n_font, batch_size):
    main_img = []
    c_img = []
    s_img = []
    dis_img = []
    for i, img in enumerate(os.listdir(path), 1):
        #print(img) # check the path n
        font = img.split('_')[0] # img = '1_64' --> 1 is the font index; 64 is the unicode for the word
        word = img.split('_')[1] # img = '1_64' --> 1 is the font index; 64 is the unicode for the word

        main_image = Image.open(os.path.join(path, img))
        main_data = np.asarray(main_image)
        #print(main_data)
        main_img.append(main_data)
        
        ok = False
        while not ok:
            try:
                font_index = str(np.random.randint(1, n_font))
                c_name = font_index + '_' + word
                c_image = Image.open(os.path.join(path, c_name))
                c_data = np.asarray(c_image)
                c_img.append(c_data) 
                ok = True
            except FileNotFoundError:
                ok = False

        ok = False
        while not ok:
            try:
                word_index = get_rand_idx()
                s_name = font + '_' + word_index + '.png'
                s_image = Image.open(os.path.join(path, s_name))
                s_data = np.asarray(s_image)
                s_img.append(s_data)
                ok = True
            except FileNotFoundError:
                ok = False

        if i % batch_size == 0:
            main = np.array(main_img) / 255
            main = np.reshape(main, (main.shape[0], main.shape[1], main.shape[2], 1))
            content = np.array(c_img) / 255
            content = np.reshape(content, (content.shape[0], content.shape[1], content.shape[2], 1))
            style = np.array(s_img) / 255
            style = np.reshape(style, (style.shape[0], style.shape[1], style.shape[2], 1))

            main_img = []
            c_img = []
            s_img = []
            

            yield main, content, style