import tensorflow as tf
import os
import random
import numpy as np
from tkinter import *
from tkinter.filedialog import askopenfilename

from skimage.data import camera
from skimage.filters import frangi, hessian
from os import listdir
from tqdm import tqdm 
from skimage.transform import resize

import cv2
from sklearn.model_selection import KFold
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

#DEFINE CONSTANTS
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

def get_model():
    #Build the model
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    #Contraction path
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    #Expansive path 
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='mean_squared_logarithmic_error', metrics=['accuracy'])
    return model
	
def get_loaded_models():
    model_files = [file for file in listdir(os.curdir+"/model")]
    models = []
    for model in model_files:
        m = get_model()
        m.load_weights(os.curdir+"/model/"+model)
        models.append(m)
    return models
	
def slice_image(image, height=128, width=128):
    sliced_image = []
    img_width, img_height, _ = image.shape
    for i in range(0, img_height, height):
        for j in range(0, img_width, width):
            sliced_image.append(image[i:i+height,j:j+width])
    return sliced_image
	
def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

def conc(arr):
    ret = []
    for x in range(10):
        ret.append([])
        for y in range(10):
            ret[x].append(arr[x*10+y])    
    return concat_tile(ret)
	
def predict_image(image):
    result = np.zeros((len(image),IMG_HEIGHT,IMG_WIDTH,1))
    models = get_loaded_models()
    for model in models:
        result += model.predict(np.array(image), verbose = 1)
    result/=len(models)
    result = (result > 0.99).astype(np.uint8)
    return result    
	
def predict(image):
    image_shape = (image.shape[0], image.shape[1])
    image = cv2.resize(image,(1280,1280))
    image = slice_image(image)
    image = predict_image(image)
    image = conc(image)
    image = cv2.resize(image,(image_shape[1],image_shape[0]))
    return image


def mask(image):
    image = image[:,:,2]
    _,mask = cv2.threshold(image,10,255,cv2.THRESH_BINARY)
    return mask

def simple(image):
    kernel_ero = np.ones((3,3),np.uint8)
    kernel_open = np.ones((11,11),np.uint8)
    kernel_dil = np.ones((6,6),np.uint8)
    
    maskO = mask(image)
    
    image = image[:,:,1]
    
    image = cv2.GaussianBlur(image,(3,3),0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
    image = clahe.apply(image)
    image = cv2.adaptiveThreshold(image,255,cv2.THRESH_BINARY,\
                cv2.THRESH_BINARY,121,3)
    image = cv2.dilate(image,kernel_dil,iterations = 1)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_open)
    image = cv2.erode(image,kernel_ero,iterations = 1)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_open)

    image = frangi(image)

    image = image * maskO
    image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)
    image[image>=0.9] = 1
    image[image<0.1] = 0
    return image

def use_mask(image, mask):
    image = image*1
    for idx,x in enumerate(mask):
        for idy,y in enumerate(x):
            if(y > 0):
                image[idx][idy][0] = 255
                image[idx][idy][1] = 255
                image[idx][idy][2] = 255
                            
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def choose_file(window_title):
    root = Tk()
    filename = askopenfilename(initialdir = "./data/",title = window_title,filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    root.destroy()
    return filename
    
def statistics(reference_mask, mask, algorithm_name):
    TN = 0
    TP = 0
    FN = 0
    FP = 0
    for idx,x in enumerate(reference_mask):
        for idy,y in enumerate(x):
            if (reference_mask[idx][idy]==0 and mask[idx][idy]==0):
                TN+=1
            if (reference_mask[idx][idy]>0 and mask[idx][idy]>0):
                TP+=1
            if (reference_mask[idx][idy]==0 and mask[idx][idy]>0):
                FP+=1
            if (reference_mask[idx][idy]>0 and mask[idx][idy]==0):
                FN+=1
    acc = (TP+TN)/(TN+TP+FN+FP)
    spe = TN/(FP+TN)
    sen = TP/(FN+TP)
    return algorithm_name+"\nTrafność: "+str(acc)+"%\nCzułość: "+str(sen)+"%\nSwoistość: "+str(spe)+"%"        

def start():
    filename = choose_file("Wybierz zdjęcie dna oka")
    image = cv2.imread(filename)
    mask = cv2.imread(os.curdir+"/mask/"+filename.split("/")[-1])[:,:,2]
    print("Obliczanie maski za pomocą prostego algorytmu...")
    image_3 = simple(image)
    print("Obliczanie maski za pomocą sieci neuronowych...")
    image_5 = predict(image)
    print("Nakładanie maski eksperckiej na obraz...")
    masked_image = use_mask(image,mask)
    print("Nakładanie maski prostego algorytmu na obraz...")
    masked_image_simple = use_mask(image,image_3)
    print("Nakładanie maski sieci neuronowej na obraz...")
    masked_image_nn = use_mask(image,image_5)
    fig = plt.figure(figsize=(20,20))
    plt.subplot(2,2,1)
    plt.title('Obraz dna oka')
    plt.axis('off')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.subplot(2,2,2)
    plt.title('Maska ekspercka')
    plt.axis('off')
    plt.imshow(masked_image,cmap='gray')
    plt.subplot(2,2,3)
    plt.title("Proste filtrowania")
    plt.axis('off')
    plt.imshow(masked_image_simple,cmap='gray')
    plt.subplot(2,2,4)
    plt.title("Sieć neuronowa")
    plt.axis('off')
    plt.imshow(masked_image_nn,cmap='gray')
    plt.show()
    print("Obliczanie statystyk prostego algorytmu...")
    print(statistics(mask,image_3,"Prosty algorytm"))
    print("Obliczanie statystyk sieci neuronowej...")
    print(statistics(mask,image_5,"Sieć neuronowa"))
    
    