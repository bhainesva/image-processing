#!bin/python2
from __future__ import division
import math
import numpy as np
import scipy
from scipy import misc
from scipy import ndimage
import skimage
from matplotlib import pyplot as plt
from skimage.filter import hsobel, vsobel, gaussian_filter

filename = 'circ.png'
sigma = 3
kernel_size = 5

def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return a[idx]

def degreediff(x, y):
    #while x < 0:
    #    x += 2*math.pi
    #while y < 0:
    #    y += 2*math.pi
    return min(abs(x-y), math.pi*2-abs(x-y))

def getDir(near):
    directions = {0 : (1, 0), math.pi/4 : (1, 1), math.pi/2 : (0, 1), 3 * math.pi/4 : (-1, 1)}
    smallest_key = min(directions.keys(), key=lambda x: degreediff(x, near))
    return directions[smallest_key]
    #return directions[find_nearest(directions.keys(), near)]
    

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

img = misc.imread(filename)
#img = rgb2gray(img)

# PART 1 - CANNY_ENHANCER
# Apply gaussian filter
kernel = makeGaussian(kernel_size, sigma)
img = scipy.signal.convolve2d(img, kernel, boundary='symm')

 #Compute gradient components
J_y, J_x = np.gradient(img)

# Strength image
strImg = (J_x**2 + J_y**2)**0.5

# Orientation Image
J_y = np.asfarray(J_y, dtype='float')
J_x = np.asfarray(J_x, dtype='float')

ornImg = np.arctan2(J_y, J_x)
#ornImg[np.isnan(ornImg)]=math.pi/2


# PART 2 - NONMAX_SUPPRESSION
I_n = np.zeros(strImg.shape)

it = np.nditer(ornImg, flags=['multi_index'], op_flags=['readwrite'])
dirImg = {}
while not it.finished:
    x, y = it.multi_index[1], it.multi_index[0]
    x_max, y_max = ornImg.shape[1], ornImg.shape[0]
    actual_dir = it[0]
    approx_dir = getDir(actual_dir)
    dirImg[(x, y)] = approx_dir

    x_off, y_off = approx_dir[0], approx_dir[1]
    if x + x_off < x_max and y + y_off < y_max and x-x_off > -1 and y-y_off > -1 and x + x_off > 0 and y + y_off > 0 and x-x_off < x_max and y-y_off < y_max:
        if strImg[y, x] < strImg[y+y_off, x+x_off] or strImg[y,x] < strImg[y+y_off, x+x_off]:
            I_n[y, x] = 0
        else:
            I_n[y, x] = strImg[y, x]

    it.iternext()
    

# PART 3 - HYSTERESIS THRESH
    # PART i. - BUILDING CHAINS

# Set up thresholds
t_l = 0
t_h = .2
chains = []
visited = np.zeros(strImg.shape)
it = np.nditer(I_n, flags=['multi_index'], op_flags=['readwrite'])

def visit(I_n, dirImg, visited, x, y, t_h):
    chain = []
    if visited[y, x] == 1:
        return chain
    visited[y, x] = 1
    if (I_n[y, x] < t_h):
        return chain
    chain.append((x, y))
    off = dirImg[(x, y)]
    if (I_n[y + off[1], x+off[0]] > t_h):
        chain += visit(I_n, dirImg, visited, x, y+1, t_h)
    if (I_n[y - off[1], x-off[0]] > t_h):
        chain += visit(I_n, dirImg, visited, x+1, y, t_h)
    return chain


for x in range(strImg.shape[1]):
    for y in range(strImg.shape[0]):
        if (visited[y, x] == 0):
            chain = visit(I_n, dirImg, visited, x, y, t_h)
            chains.append(chain)

final = np.zeros(strImg.shape)
list2 = [x for x in chains if x != []]

somelist = [x for x in list2 if max(x, key=lambda y : I_n[y[1], y[0]]) > t_l]
print len(list2)
print len(somelist)
for x in somelist:
    for pix in x:
        final[pix[1], pix[0]] = I_n[pix[1], pix[0]]

plt.imshow(final)
plt.gray()
plt.show()

