import pdb

import numpy as np
from skimage.transform import rescale
import scipy
import scipy.ndimage
import cv2
import os

FILENAME = 'teste_img.jpg'
PATH = os.path.join('assets', FILENAME)
img = cv2.imread(PATH)

blue = rescale(img[:, :, 0], 0.5)
green = rescale(img[:, :, 1], 0.5)
red = rescale(img[:, :, 2], 0.5)
img = np.stack([blue, green, red], axis=2)
cv2.imshow('Imagem original', img)

# Box Blur
box_blur = (1/9) * np.array([[1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1]])
# box_factor= 1 / box_blur.sum() * box_blur -> resultado é 1/9

# Box Blur 10x10
box_blur_10 = (1/100) * np.ones((10, 10))

# Gaussian blur 3x3
gaussian_3 = (1/16) * np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]])

# Gaussian Blur 3x3
gaussian_5 = (1/256) * np.array([[1, 4, 6, 4, 1],
                                 [4, 16, 24, 16, 4],
                                 [6, 24, 36, 24, 6],
                                 [4, 16, 24, 16, 4],
                                 [1, 4, 6, 4, 1]])

kernels_dict = {
                    'Box Blur': box_blur, 
                    '10x10 Box Blur': box_blur_10, 
                    '3x3 Gaussian Blur': gaussian_3, 
                    '5x5 Gaussian Blur': gaussian_5,
                }

for name, kernel in kernels_dict.items():
    conv_im1 = scipy.ndimage.convolve(img, np.atleast_3d(kernel), mode='nearest')
    cv2.imshow(name, conv_im1)

cv2.waitKey(delay=0)