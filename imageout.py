import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import imageio
from imutils import paths
import cv2
import os

ia.seed(1)

seq = iaa.Sequential(
    [
        iaa.Fliplr(0.5),             
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.3))),        
        iaa.ContrastNormalization((0.75, 1.25)), #mayor 1 mas contraste al original         
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),    
    ],
    random_order=True)  # apply augmenters in random order



ruta = "dataset/limon_sutil_segmentado/"
imagenRutas = list(paths.list_images(ruta))
verbose = 10
total = len(imagenRutas)
numero = 1
for (i, imagenRutas) in enumerate(imagenRutas):
    img = cv2.imread(imagenRutas)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    nombre = ""
    if numero <= 350:
        nombre = "maduro"+"_"+str(numero)
    elif numero > 350 and numero <= 701:
        nombre = "pinton"+"_"+str(numero)
    else:
        nombre = "verde"+"_"+str(numero)

    print(nombre)
    numero += 1

    #img = imageio.imread("limon1.jpg") #read you image
    images = np.array([img for _ in range(10)], dtype=np.uint8)  #crear 10 enhanced images using following methods.
    images_aug = seq.augment_images(images)
    for i in range(10):
        imageio.imwrite(nombre+str(i)+".jpg", images_aug[i])  #guardar las imagenes generadas en el disco

