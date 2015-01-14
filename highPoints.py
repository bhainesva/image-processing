#!bin/python2
from PIL import Image
import numpy as np
import matplotlib
import scipy
from scipy import ndimage
from scipy import misc
from matplotlib import pyplot as plt
import pylab
import sys
import argparse

neighborhood_size = 5
threshold = 125

data = misc.imread(sys.argv[1])
data_max = ndimage.maximum_filter(data, neighborhood_size)
maxima = (data == data_max)
data_min = ndimage.minimum_filter(data, neighborhood_size)
diff = ((data_max - data_min) > threshold)

def isGray(img):
    return img.mode != 'RGB'

def toGray(tup):
    return 0.299*tup[0] + 0.587*tup[1] + 0.114*tup[2] 


img = Image.open(sys.argv[1])
if not isGray(img):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            dripLength = -1
            while (dripLength < 1):
                dripLength = int(np.random.normal(50, 10))
            if diff[i,j].any():
                toSort = data[i:i+dripLength,j]
                didSort = sorted(toSort, key=toGray)
                data[i:i+dripLength,j] = didSort

else:
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            dripLength = int(np.random.normal(50, 10))
            if diff[i,j]:
                toSort = data[i:i+dripLength,j]
                didSort = sorted(toSort)
                data[i:i+dripLength,j] = didSort

plt.gray()
plt.imshow(data)
plt.show()
misc.imsave('out.png', data)
