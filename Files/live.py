import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model, Sequential
import cv2


def predictor(sdir, csv_path, crop_image=False):
    # read in the csv file
    class_df = pd.read_csv(csv_path, encoding='cp1252')
    # img_height=int(class_df['height'].iloc[0])
    img_height = int(class_df['width'].iloc[0])
    img_width = int(class_df['width'].iloc[0])
    img_size = (img_width, img_height)
    scale = 1
    try:
        s = int(scale)
        s2 = 1
        s1 = 0
    except:
        split = scale.split('-')
        s1 = float(split[1])
        s2 = float(split[0].split('*')[1])
        print(s1, s2)
    path_list = []
    paths = sdir
    print('path', paths)
    # for f in paths:
    path_list.append(paths)
    image_count = 1
    index_list = []
    prob_list = []
    cropped_image_list = []
    good_image_count = 0
    for i in range(image_count):

        img = cv2.imread(path_list[i])
        # print('i',img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if crop_image == True:
            status, img = crop(img)
        else:
            status = True
        if status == True:
            good_image_count += 1
            img = cv2.resize(img, img_size)
            cropped_image_list.append(img)
            img = img * s2 - s1
            img = np.expand_dims(img, axis=0)
            p = np.squeeze(model.predict(img))
            index = np.argmax(p)
            prob = p[index]
            index_list.append(index)
            prob_list.append(prob)
    if good_image_count == 1:
        print(class_df.columns.tolist())
        class_name = class_df['class'].iloc[index_list[0]]
        symtom = class_df['symtoms '].iloc[index_list[0]]
        # symtom='1'
        medicine = class_df['medicine'].iloc[index_list[0]]
        wht = class_df['what is'].iloc[index_list[0]]
        probability = prob_list[0]
        img = cropped_image_list[0]
        # plt.title(class_name, color='blue', fontsize=16)
        # plt.axis('off')
        # plt.imshow(img)
        print(class_name)
        # print(symtom,medicine,wht)
        return class_name, probability, symtom, medicine, wht
    elif good_image_count == 0:
        return None, None, None, None, None
    most = 0
    for i in range(len(index_list) - 1):
        key = index_list[i]
        keycount = 0
        for j in range(i + 1, len(index_list)):
            nkey = index_list[j]
            if nkey == key:
                keycount += 1
        if keycount > most:
            most = keycount
            isave = i             
    best_index=index_list[isave]    
    psum=0
    bestsum=0
    for i in range (len(index_list)):
        psum += prob_list[i]
        if index_list[i]==best_index:
            bestsum += prob_list[i]  
    img= cropped_image_list[isave]/255    
    class_name=class_df['class'].iloc[best_index]
    symtom=class_df['symtoms '].iloc[best_index]
    medicine=class_df['medicine'].iloc[best_index]
    wht=class_df['what is'].iloc[best_index]
    # print(symtom,medicine,wht)
    # plt.title(class_name, color='blue', fontsize=16)
    # plt.axis('off')
    # plt.imshow(img)
    return class_name, bestsum/image_count,symtom,medicine,wht
img_size=(300, 300)
import cv2

cam = cv2.VideoCapture(0)

while True:
    check, frame = cam.read()
    cv2.imwrite("C:/Users/PYTHONFABHOST/Desktop/new project work/skin disease using efficientnet/app/out.jpg",frame)
    path="C:/Users/PYTHONFABHOST/Desktop/new project work/skin disease using efficientnet/app/out.jpg"
        # store_path="store_path.jpg"
        # cv2.imwrite(path, img)
    csv_path="C:/Users/PYTHONFABHOST/Desktop/new project work/skin disease using efficientnet/app/class.csv"
    model_path="C:/Users/PYTHONFABHOST/Desktop/new project work/skin disease using efficientnet/EfficientNetB3-skindisease-83.00.h5"
    class_name, probability,symtom,medicine,wht=predictor(path, csv_path,  crop_image = False)
        # print(symtom,medicine,wht)
    # print(symtom)
    cv2.imshow('video', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
        
