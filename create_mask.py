# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 18:34:03 2018

@author: Conor
"""

import math
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from skimage import util as skutil
from sklearn.model_selection import train_test_split
from scipy.ndimage.morphology import binary_fill_holes
from PIL import Image


img = plt.imread('Images\\ir_frame3_contour.png')
contour_red = img[:,:,0] == 1
contour_green = img[:,:,1] == 0
mask = binary_fill_holes(contour_red * contour_green)
plt.imshow(mask)

mask_img = Image.fromarray(mask.astype('uint8')*255)
mask_img.save('Images\\ir_frame3_mask.png')