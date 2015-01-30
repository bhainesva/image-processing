from __future__ import division
import math
import numpy as np
import scipy
from ImgFunctions import *
from scipy import misc
from scipy import ndimage
import skimage
from matplotlib import pyplot as plt
from skimage.filter import hsobel, vsobel, gaussian_filter

def find_nearest(a, a0):
    index = np.abs(a - a0).argmin()
    return a[index]

def evalGaussian(x, y, sigma):
    return (1/((math.pi * 2) * sigma**2)) * math.e ** (-(x**2 + y**2)/(2*sigma**2))

def evalGaussianXDeriv(x, y, sigma):
    return -((x * math.e**(-(x**2 + y**2)/(2*sigma**2)))/((2 * math.pi) * sigma**3))

def evalGaussianYDeriv(x, y, sigma):
    return -((y * math.e**(-(x**2 + y**2)/(2*sigma**2)))/((2 * math.pi) * sigma**3))

def newGradient(size, sigma, img):
    r = (size - 1)/2
    xKernel = np.zeros((size, size))
    yKernel = np.zeros((size, size))
    J_y = np.zeros(img.shape)
    J_x = np.zeros(img.shape)

    it = np.nditer(xKernel, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        y, x = it.multi_index
        newx = x-r
        newy = y-r
        it[0] = evalGaussianXDeriv(newx, newy, sigma)
        yKernel[y, x] = evalGaussianYDeriv(newx, newy, sigma)
        it.iternext()
    if np.amax(yKernel) != 0:
        yKernel /= np.amax(yKernel)
        print yKernel
    if np.amax(xKernel) != 0:
        xKernel /= np.amax(xKernel)
    print xKernel
    print yKernel
    J_y = scipy.signal.convolve2d(img, yKernel, boundary='symm')
    J_x = scipy.signal.convolve2d(img, xKernel, boundary='symm')
    return (J_x, J_y)

def newGaussian(size, sigma):
    r = (size - 1)/2
    kernel = np.zeros((size, size))

    it = np.nditer(kernel, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        y, x = it.multi_index
        x = x-r
        y = y-r
        it[0] = evalGaussian(x, y, sigma)
        it.iternext()
    kernel /= np.amax(kernel)
    return kernel

    

def degreediff(x, y):
    while x < 0:
        x += 2*math.pi
    while y < 0:
        y += 2*math.pi
    return min(abs(x-y), math.pi*2-abs(x-y))

def getDir(near):
    directions = {0 : (1, 0), math.pi : (1, 0), math.pi/4 : (1, 1), math.pi*5 / 4 : (1, 1), math.pi/2 : (0, 1), math.pi*3/2 : (0, 1), 3 * math.pi/4 : (-1, 1), math.pi*7/4 : (-1, 1)}
    smallest_key = min(directions.keys(), key=lambda x: degreediff(x, near))
    return directions[smallest_key]
    

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

#def makeGaussian(size, sigma = 3):
#    x = np.arange(0, size, 1, float)
#    y = x[:,np.newaxis]
#
#    x0 = y0 = size // 2
#
#    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / sigma**2)

#def gradient(img):
#    J_y = np.zeros(img.shape)
#    J_x = np.zeros(img.shape)
#
#    #it = np.nditer(img, flags=['multi_index'], op_flags=['readwrite'])
#    #while not it.finished:
#    verticalSobel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
#    horizontalSobel = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
#    J_y = scipy.signal.convolve2d(img, verticalSobel, boundary='symm')
#    J_x = scipy.signal.convolve2d(img, horizontalSobel, boundary='symm')
#    return J_x, J_y


