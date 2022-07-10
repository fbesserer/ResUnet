import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.layers import MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import concatenate, add
import tensorflow.keras.backend as K

import numpy as np
# import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os


def label2OneHot(labelImage, noclasses):
    # erstellen eines labelImage.shape(0) x labelImage.shape(1) x noclasses arrays
    labelsOneHot = np.eye(N=noclasses)[labelImage]

    return labelsOneHot


def LoadSplitData(pathImages, pathLabels, split):
    images = os.listdir(pathImages)
    labels = os.listdir(pathLabels)
    len_images = len(images)
    # Ausgabe Arrays initialisieren
    images_4d = np.zeros((len_images, 128, 128, 3))
    labels_4d = np.zeros((len_images, 128, 128, 4))

    assert len(images) == len(labels), "Anzahl der Bilder und Labels ist unterschiedlich"

    for i in range(len_images):
        # Bilder der Reihe nach einlesen, Labels umwandeln und im Ausgabearray speichern
        image = images.pop()
        label = labels.pop()
        assert image[:5] == label[:5], f"{image} und {label} stimmen nicht überein"
        # print(image, " ", label)

        png_image = mpimg.imread(pathImages + "\\" + image)
        png_label = mpimg.imread(pathLabels + "\\" + label)
        png_label = (png_label[:, :, 0] * 255).astype('int')
        png_label = label2OneHot(png_label, 4)

        images_4d[i, ::] = png_image[:, :, 0:3]
        labels_4d[i, ::] = png_label

    # images und labels mischen
    shuffle = np.random.permutation(len_images)
    print(shuffle)
    images_4d = images_4d[shuffle]
    labels_4d = labels_4d[shuffle]

    split = np.ceil(len_images * split).astype('int')
    xtrain = images_4d[split:, ::]
    xval = images_4d[:split, ::]
    ytrain = labels_4d[split:, ::]
    yval = labels_4d[:split, ::]

    return xtrain, ytrain, xval, yval


def Conv2DBN(x, fs, kernel_size=(3, 3), strides=(1, 1), activation='relu', bn=True):
    # fs = feature maps
    # https://www.tensorflow.org/tutorials/customization/custom_layers
    y = Conv2D(fs, kernel_size=kernel_size, strides=strides, activation=activation, padding='same', use_bias=False)(x)
    if bn:
        y = BatchNormalization(axis=3, scale=False)(y)

    return y


def TransConv2D(x, fs, kernel_size=(2, 2), strides=(2, 2), bn=True):
    # Upsampling bzw. Up Convolution
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose
    y = Conv2DTranspose(filters=fs, kernel_size=kernel_size, strides=strides, use_bias=False, padding='same')(x)
    if bn:
        y = BatchNormalization(axis=3, scale=False)(y)

    return y


def ResBlock1(x, fs):
    f33 = fs // 6
    f55 = fs // 3
    f77 = fs // 2
    f11 = f33 + f55 + f77

    conv2f33 = Conv2DBN(x, f33)
    conv2f55 = Conv2DBN(conv2f33, f55, kernel_size=(5, 5))
    conv2f77 = Conv2DBN(conv2f55, f77, kernel_size=(7, 7))
    conv2f11 = Conv2DBN(x, f11, kernel_size=(1, 1), strides=(1, 1), activation=None)

    concat = concatenate([conv2f77, conv2f55, conv2f33], axis=3)
    y = BatchNormalization(axis=3)(concat)  # vlt scale=False?
    y = add([y, conv2f11])
    y = Activation(activation='relu')(y)
    y = BatchNormalization(axis=3)(y)

    return y


def ResBlock2(x, fs):
    conv2_1 = Conv2DBN(x, fs, kernel_size=(3, 3))
    conv2_2 = Conv2DBN(x, fs, kernel_size=(1, 1), activation=None)
    y = add([conv2_1, conv2_2])
    y = BatchNormalization(axis=3)(y)

    return y


def connectH(x, fs, layer, n_layers=4):
    k = n_layers - layer
    if layer == n_layers:
        # kein ResBlock2
        return x

    fs = fs * np.power(2, layer - 1)  # kann hier auch 1/2 durch layer 0 vorkommen bzw ei nicht integer wert?
    y = ResBlock2(x, fs=fs)
    for _ in range(k - 1):  # k-1, da der erste ResBlock for loop bereits erstellt wird
        y = ResBlock2(y, fs)

    return y


def encoder(x, fs, layer):
    fs = fs * np.power(2, layer - 1)  # kann hier auch 1/2 durch layer 0 vorkommen bzw ei nicht integer wert?
    y2 = ResBlock1(x, fs)
    y1 = MaxPooling2D(pool_size=(2, 2))(y2)

    return y1, y2


def decoder(x1, x2, fs, layer, n_layers=4):
    fs = fs * np.power(2, layer - 1)  # kann hier auch 1/2 durch layer 0 vorkommen bzw ei nicht integer wert?
    transConv = TransConv2D(x1, fs, kernel_size=(2, 2), strides=(2, 2))

    # if x2:  # tiefste Schicht hat keinen Input x2
    transConv = concatenate([transConv, x2], axis=3)
    y = ResBlock1(transConv, fs)

    return y


def ResUnet(input_size=(128, 128, 3), n_classes=4, fs1=32, numb_layers=4, use_connect_block=True):
    inputs = x2 = Input(input_size)
    encoders = list()

    for layer in range(1, numb_layers + 1):  # Layers sind 1 indexiert
        # erstellt encoder für alle Schichten
        x1, x2 = encoder(x2, fs1, layer)
        encoders.append(x1)

    # x2 = None
    for idx in range(len(encoders) - 1, -1, -1):
        x1 = encoders[idx]
        if use_connect_block:
            x1 = connectH(x1, fs1, layer=idx + 1, n_layers=numb_layers)
        x2 = decoder(x1, x2, fs1, layer=idx + 1, n_layers=numb_layers)

    # 1x1 Convolution
    y = Conv2DBN(x2, fs=n_classes, kernel_size=(1, 1), strides=(1, 1), activation='softmax')

    m = Model(input=inputs, output=y)

    return m


if __name__ == "__main__":
    pathImages = os.getcwd() + '\\Daten\\images'
    pathLabels = os.getcwd() + '\\Daten\\labels'

    split = 0.2
    for i in range(3):
        xtrain, ytrain, xval, yval = LoadSplitData(pathImages, pathLabels, split)

    input_size = (xtrain.shape[1], xtrain.shape[2], xtrain.shape[3])
    n_classes = 4
    fs1 = 16
    n_layers = 3
    use_connect = False

    model_1 = ResUnet(input_size=input_size, n_classes=n_classes, fs1=fs1, numb_layers=n_layers,
                      use_connect_block=use_connect)
