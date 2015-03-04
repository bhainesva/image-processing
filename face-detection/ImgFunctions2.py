from __future__ import division
import math
import numpy as np
import scipy
from scipy import misc
from scipy import ndimage
import skimage
from matplotlib import pyplot as plt

# Evaluate 2D gaussian function at point (x, y) with given sigma
def evalGaussian(x, y, sigma):
    return (1/((math.pi * 2) * sigma**2)) * math.e ** (-(x**2 + y**2)/(2*sigma))

# Evaluate partial derivative of 2D gaussian by x
def evalGaussianXDeriv(x, y, sigma):
    return -((x * math.e**(-(x**2 + y**2)/(2*sigma**2)))/((2 * math.pi) * sigma**3))

# Evaluate partial derivative of 2D gaussian by y
def evalGaussianYDeriv(x, y, sigma):
    return -((y * math.e**(-(x**2 + y**2)/(2*sigma**2)))/((2 * math.pi) * sigma**3))

# Create kernels and convolve with original image to produce horizontal and vertical gradients
def newGradient(r, sigma, img):
    size = r*2 + 1
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
    if np.amax(xKernel) != 0:
        xKernel /= np.amax(xKernel)
    J_y = scipy.signal.convolve2d(img, yKernel, mode='same', boundary='symm')
    J_x = scipy.signal.convolve2d(img, xKernel, mode='same', boundary='symm')
    return (J_x, J_y)

# Construct a kernel for applying a gaussian filter
def newGaussian(r, sigma):
    size = r*2 + 1
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

# Compute the covariance matrix at a given point
def computeC(x, y, length, jx, jy):
    Ex2 = np.sum(getSubMatrix(x, y, length, jx)**2)
    Ey2 = np.sum(getSubMatrix(x, y, length, jy)**2)
    Exy = np.sum(getSubMatrix(x, y, length, jx) * getSubMatrix(x, y, length, jy))
    return np.asarray([[Ex2, Exy],[Exy, Ey2]])

# Return a submatrix centered at a certain point with given size
def getSubMatrix(x, y, length, img):
    (ymax, xmax) = img.shape
    xstart = max(x-length, 0)
    xend = min(x+length+1, xmax)
    ystart = max(y-length, 0)
    yend = min(y+length+1, ymax)
    return img[ystart:yend, xstart:xend]

# Sets all values in a submatrix to 0 except for the middle one
def clearSubMatrix(x, y, length, img):
    tmp = img[y, x]
    (ymax, xmax) = img.shape
    xstart = max(x-length, 0)
    xend = min(x+length+1, xmax)
    ystart = max(y-length, 0)
    yend = min(y+length+1, ymax)
    img[ystart:yend, xstart:xend] = 0
    img[y, x] = tmp
    
# Calculate difference between two angles
def degreediff(x, y):
    while x < 0:
        x += 2*math.pi
    while y < 0:
        y += 2*math.pi
    return min(abs(x-y), math.pi*2-abs(x-y))

# Approximate actual angle by one of 8 angles
def getDir(near):
    directions = {0 : (1, 0), math.pi : (1, 0), math.pi/4 : (1, 1), math.pi*5 / 4 : (1, 1), math.pi/2 : (0, 1), math.pi*3/2 : (0, 1), 3 * math.pi/4 : (-1, 1), math.pi*7/4 : (-1, 1)}
    smallest_key = min(directions.keys(), key=lambda x: degreediff(x, near))
    return directions[smallest_key]
    
# Just to help me do the writeup
def kerToSig(ker):
    return 0.3 * (ker/2 - 1) + 0.8

# Function for drawing boxes around points. Used for displaying the output of the corner detection.
def outline(x, y, r, w, img):
    img[max(0,y-r):min(y-r+w, img.shape[0]),max(x-r, 0):min(x+r+1, img.shape[1])] = 1
    img[max(0,y+r-w+1):min(y+r+1, img.shape[0]),max(x-r, 0):min(x+r+1, img.shape[1])] = 1
    img[max(0,y-r):min(y+r, img.shape[0]),max(0, x-r):min(img.shape[1], x-r+w)] = 1
    img[max(0, y-r+w):min(img.shape[0], y+r-w+1), max(x+r-w+1, 0):min(img.shape[1], x+r+1)]=1

# Convert RGB image to grayscale
def grayscale(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

