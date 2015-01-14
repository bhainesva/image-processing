#!bin/python2
from PIL import Image
import numpy as np
import matplotlib
import scipy
from scipy import misc
import pylab
import sys
import argparse

def toGray(tup):
    return 0.299*tup[0] + 0.587*tup[1] + 0.114*tup[2] 

def grayShellSort(array, grain = 50):
     if grain < 1:
         grain = 50
     gap = len(array) // 2
     # loop over the gaps
     while gap > grain:
         # do the insertion sort
         for i in range(gap, len(array)):
             val = array[i]
             j = i
             while j >= gap and array[j - gap] > val:
                 array[j] = array[j - gap]
                 j -= gap
             array[j] = val
         gap //= 2

def shellSort(array, grain = 50):
     if grain < 1:
         grain = 50
     gap = len(array) // 2
     # loop over the gaps
     while gap > grain:
         # do the insertion sort
         for i in range(gap, len(array)):
             val = array[i]
             j = i
             while j >= gap and toGray(array[j - gap]) > toGray(val):
                 array[j] = array[j - gap]
                 j -= gap
             array[j] = val
         gap //= 2

def isGray(img):
    return img.mode != 'RGB'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="the image to have its pixels sorted")
    parser.add_argument("-r", "--rows", help="sorts rows",
                        action="store_true")
    parser.add_argument("-c", "--columns", help="sorts columns",
                        action="store_true")
    parser.add_argument("-g", "--grayscale", help="converts image to grayscale",
                        action="store_true")
    parser.add_argument("-R", "--rowgrain", type=int,
                        help="gapsize for rows")
    parser.add_argument("-C", "--colgrain", type=int,
                        help="gapsize for columns")
    args = parser.parse_args()

    img = Image.open(args.image)
    if (not isGray(img) and args.grayscale):
        img = img.convert('L')
        img.save('greyscale.png')
        lena = misc.imread('greyscale.png')
        pylab.gray()

        if (args.columns):
            for i in range(lena.shape[1]):
                col = lena[:,i]
                grayShellSort(col, args.colgrain)

        if (args.rows):
            for i in range(lena.shape[0]-1):
                grayShellSort(lena[i], args.rowgrain)
        
    else:
        lena = misc.imread(args.image)

        if (args.columns):
            for i in range(lena.shape[1]):
                col = lena[:,i]
                shellSort(col, args.colgrain)

        if (args.rows):
            for i in range(lena.shape[0]-1):
                shellSort(lena[i], args.rowgrain)

    pylab.imshow(lena)
    pylab.show()
