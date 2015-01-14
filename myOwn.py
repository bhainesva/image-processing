#!bin/python2
from __future__ import division
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

brightnessValue = 70;

img = misc.imread(sys.argv[1])
height, width = img.shape[0], img.shape[1]


def lumosity(pixel):
    return 0.21*pixel[0] + 0.72*pixel[1] + 0.07*pixel[2]

def sort(arr, mode=0):
    x=getFirstBright(arr)
    xend = 0
    first = True
    while (xend < len(arr)-1):
        if not first:
            x = getNextBright(arr, x)
        if x == -1:
            break
        xend = getNextDark(arr, x)
        if x == xend:
            break
        arr[x:xend] = sorted(arr[x:xend], key=lumosity)
        x = xend+1
        first = False

def getFirstBright(arr):
    for i in range(len(arr)):
        if (lumosity(arr[i]) > brightnessValue):
            return i
    return -1

def getNextBright(arr, x):
    if (lumosity(arr[x]) > brightnessValue and x == 0):
        return 0
    for i in range(x+1, len(arr)):
        if (lumosity(arr[i]) > brightnessValue):
            return i
    return len(arr)-1

def getFirstDark(arr):
    for i in range(len(arr)):
        if (lumosity(arr[i]) < brightnessValue):
            return i
    return -1

def getNextDark(arr, x):
    for i in range(x+1, len(arr)):
        if (lumosity(arr[i]) < brightnessValue):
            return i
    return len(arr)-1


for i in range(height):
    sort(img[:,i])

plt.imshow(img)
plt.show()

