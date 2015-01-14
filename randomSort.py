#!bin/python2
from PIL import Image
import numpy as np
import matplotlib
import scipy
from scipy import misc
import pylab
import sys
import argparse
from random import randint
from random import gauss

def toGray(tup):
    return 0.299*tup[0] + 0.587*tup[1] + 0.114*tup[2] 

def isGray(img):
    return img.mode != 'RGB'

def chunks(l, n):
    if n < 1:
        n = 1
    return [l[i:i + n] for i in range(0, len(l), n)]

def slice_list(arr, size):
    input_size = len(arr)
    slice_size = input_size / size
    remain = input_size % size
    result = []
    iterator = iter(arr)
    for i in range(size):
        result.append([])
        for j in range(slice_size):
            result[i].append(iterator.next())
        if remain:
            result[i].append(iterator.next())
            remain -= 1
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="the image to have its pixels sorted")
    parser.add_argument("-g", "--grayscale", help="converts image to grayscale",
                        action="store_true")
    args = parser.parse_args()

    args = parser.parse_args()

    img = Image.open(args.image)
    if (not isGray(img) and args.grayscale):
        img = img.convert('L')
        img.save('greyscale.png')
        lena = misc.imread('greyscale.png')
        pylab.gray()

        for i in range(lena.shape[1]):
            col = lena[:,i]
            offset = randint(0, (lena.shape[0]-1)/25)
            col = np.concatenate((col[offset:], col[:offset]))
            chunked = [sorted(x) for x in chunks(col, randint(lena.shape[1]/100, lena.shape[1]/70))]
            lena[:,i] = [item for sublist in chunked for item in sublist]

    else:
        lena = misc.imread(args.image)

        for i in range(lena.shape[1]):
            col = lena[:,i]
            if (randint(0, 10) <= 8):
                #offset = randint(0, (lena.shape[0]-1)/40)
                #col = np.concatenate((col[offset:], col[:offset]))
                chunked = [sorted(x, key=toGray) for x in slice_list(col, randint(2, 3))]
            else:
                chunked = [x for x in slice_list(col, 1)]
            lena[:,i] = [item for sublist in chunked for item in sublist]



    pylab.imshow(lena)
    pylab.show()
