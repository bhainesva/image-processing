#!../bin/python2
from __future__ import division
import math
import sys
import numpy as np
import random
import scipy
from ImgFunctions2 import *
from scipy import misc
from scipy import ndimage
import skimage
from matplotlib import pyplot as plt
from collections import defaultdict

with open('face_coords.txt', 'r') as f:
    lines = [line.rstrip('\n') for line in f]

location = './patches_face/'

count = 0
for line in lines:
    if count > 99:
        location = './patches_face_unused/'
    line = line.split()
    filename = line.pop(0).split('.')[0] + '.png'
    leyex, leyey, reyex, reyey, _, _, lmouthx, lmouthy, cmouthx, cmouthy, rmouthx, rmouthy = [int(float(x)) for x in line]

    img = misc.imread("./tmp/" + filename)

    xstart = min(leyex, lmouthx)
    ystart = min(leyey, reyey)
    xlen = max(reyex - xstart, rmouthx - xstart)
    ylen = max(lmouthy-ystart, cmouthy-ystart, rmouthy-ystart)
    tmplen = max(xlen, ylen)
    xlen = ylen = tmplen * 1.3
    xstart = xstart - int((xlen-tmplen)/2)
    ystart = ystart - int((ylen-tmplen)/2)

    face = img[ystart:ystart+ylen, xstart:xstart+xlen]
    print filename

    newface = misc.imresize(face, (12, 12))
    misc.imsave(location + filename.split('.')[0] + '_' + str(count) + '.png', newface)
    count += 1

pointDict = defaultdict(list)
count = 0
for line in lines:
    line = line.split()
    filename = line.pop(0)
    leyex, leyey, reyex, reyey, nosex, nosey, lmouthx, lmouthy, cmouthx, cmouthy, rmouthx, rmouthy = [int(float(x)) for x in line]
    pointDict[filename].append((leyex, leyey))
    pointDict[filename].append((reyex, reyey))
    pointDict[filename].append((lmouthx, lmouthy))
    pointDict[filename].append((cmouthx, cmouthy))
    pointDict[filename].append((rmouthx, rmouthy))
    pointDict[filename].append((nosex, nosey))

location = './patches_noface/'
while True:
    for name in pointDict.keys():
        img = misc.imread("./tmp/" + name.split('.')[0] + '.png')
        for point in pointDict[name]:
            valid = False
            if count > 99:
                location = './patches_noface_unused/'
            if count > 169:
                sys.exit()
            while not valid:
                x, y = random.randint(0, img.shape[1]-12), random.randint(0, img.shape[0]-12)
                valid = True
                for point in pointDict[name]:
                    if (x < point[0] < x+12) and (y < point[1] < y+12):
                        valid = False
        patch = img[y:y+12, x:x+12]
        misc.imsave(location + filename.split('.')[0] + "_" + str(count) + '.png', patch)
        count += 1 

            

