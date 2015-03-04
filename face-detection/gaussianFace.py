#!../bin/python2
from __future__ import division
import math
import sys
import copy
import os
import numpy as np
import random
import scipy
from ImgFunctions2 import *
from scipy import misc
from numpy import linalg as LA
from scipy import ndimage
import skimage
from matplotlib import pyplot as plt
from collections import defaultdict

# Parameters
tau = 0.5
filename = 'smallmom.jpg'

# Define functions ####
# Evaluates 144 dimensional gaussian given a patch, mean, sigma, and determinate.
# Note that sigma in this case is not actually the covariance matrix but 
# Ek computed below
def gaussian(x, mu, sigmaInv, sigmaDet):
    x = np.asarray(x).ravel()
    return (1/((2*math.pi)**72 * math.sqrt(abs(sigmaDet)))) * math.exp(-0.5 * (x - mu).T.dot(sigmaInv).dot(x-mu))

# Given a patch evaluates the face and nonface gaussian and returns which class is more likely
def classify(patch, fmu, fsigmaInv, fsigmaDet, nmu, nsigmaInv, nsigmaDet):
    f =  gaussian(patch, fmu, fsigmaInv, fsigmaDet) 
    n =  gaussian(patch, nmu, nsigmaInv, nsigmaDet)
    return (f > n, f)

# Get list of filenames for face and nonface patches
faceFilenames = os.listdir("./patches_face")
nofaceFilenames = os.listdir("./patches_noface")

# Convert each image into a length 144 vector and add it to a list
faces = [np.asarray(misc.imread("./patches_face/" + x)).ravel() for x in faceFilenames]
nofaces = [np.asarray(misc.imread("./patches_noface/" + x)).ravel() for x in nofaceFilenames]

# Take the average face and nonface patches to find the sample means
sampleFaceMean = [sum(x)/len(faces) for x in zip(*faces)]
sampleNofaceMean = [sum(x)/len(nofaces) for x in zip(*nofaces)]

# Resize and save the sample mean images
faceMean = np.asarray(sampleFaceMean).reshape(12, 12)
nofaceMean = np.asarray(sampleNofaceMean).reshape(12, 12)
faceMean = misc.imresize(faceMean, (48, 48))
nofaceMean = misc.imresize(nofaceMean, (48, 48))
misc.imsave('faceMean.jpg', faceMean)
misc.imsave('nofaceMean.jpg', nofaceMean)

# Subtract class sample mean from each element of each class
faces = [x-sampleFaceMean for x in faces]
nofaces = [x-sampleNofaceMean for x in nofaces]

# Construct the 100x144 matrix A that will be used to compute the covariance matrix
faceA = np.vstack(faces)
nofaceA = np.vstack(nofaces)

# Sigma is the product of A transpose and A
faceSigma = faceA.T.dot(faceA) / 100
nofaceSigma = nofaceA.T.dot(nofaceA) / 100

# Use singular value decomposition to get U, S, Uprime
faceU, faceS, faceUp = LA.svd(faceSigma)
nofaceU, nofaceS, nofaceUp = LA.svd(nofaceSigma)

# Plot values of S
#plt.plot(nofaceS[10:-10])
#plt.ylabel('Specific Values')
#plt.savefig('foo.png', bbox_inches='tight')

# Select elements of faceS that are above the determined threshold
faceSthresh = []
for i in faceS:
    if i > tau:
        faceSthresh.append(i)
    else:
        break

# Select the first k columns of U
k = len(faceSthresh)
faceUk = faceU[:,:k]

# Find upper left submatrix of S containing only values above the threshold
faceSk = np.diag(faceSthresh)

# Construct Ek, Ek^-1, and |Ek|
faceEk = faceUk.dot(faceSk).dot(faceUk.T)
faceEkinv = faceUk.dot(LA.inv(faceSk)).dot(faceUk.T)
facedetEk = np.prod(faceSthresh)

# Repeat the above process for the nonface images
nofaceSthresh = []
for i in nofaceS:
    if i > tau:
        nofaceSthresh.append(i)
    else:
        break

k = len(nofaceSthresh)
nofaceUk = nofaceU[:,:k]
nofaceSk = np.diag(nofaceSthresh)

nofaceEk = nofaceUk.dot(nofaceSk).dot(nofaceUk.T)
nofaceEkinv = nofaceUk.dot(LA.inv(nofaceSk)).dot(nofaceUk.T)
nofacedetEk = np.prod(nofaceSthresh)


# Read in image
img = misc.imread(filename)

# Convert image to grayscale if it isn't already
if len(img.shape) > 2:
    img = grayscale(img)

# Setup array of 0s to hold output
out = np.zeros(img.shape)
rect = copy.deepcopy(img)
points = []
pointMatrix = np.zeros(img.shape)

# For every pixel in the image, classify the 12x12 square
# for which that pixel is the upper left corner. 
# If the patch is determined to be a face, make appropriate changes
for x in range(img.shape[1] - 12):
    for y in range(img.shape[0] - 12):
        patch = img[y:y+12, x:x+12]
        truth, value = classify(patch, sampleFaceMean, faceEkinv, facedetEk, sampleNofaceMean, nofaceEkinv, nofacedetEk)
        if truth:
            out[y:y+12, x:x+12] = 1
            points.append((x, y, value))
            pointMatrix[y, x] = value
            

# Nonmaximum suppression 
points = sorted(points, key=lambda x: -x[2])
for point in points:
    if (pointMatrix[point[1], point[0]] != 0):
            clearSubMatrix(point[0], point[1], 15, pointMatrix)

# Draw face boxes
it = np.nditer(pointMatrix, flags=['multi_index'], op_flags=['readwrite'])
while not it.finished:
    (y, x) = it.multi_index
    if pointMatrix[y,x] != 0:
        outline(x+6, y+6, 6, 1, rect)
    it.iternext()

# Save the output images.
misc.imsave("binary.png", out)
misc.imsave("outline.png", rect)

# Display result
plt.imshow(rect)
plt.gray()
plt.show()
            
###### TESTING ########
# Testing accuracy on training images
testFaces = os.listdir("./patches_face/")
testnoFaces = os.listdir("./patches_noface/")


testpoints = [(np.asarray(misc.imread("./patches_face/" + x)), 1) for x in testFaces]
testpoints += [(np.asarray(misc.imread("./patches_noface/" + x)), 0) for x in testnoFaces]


truepos = 0
trueneg = 0
false = 0
for x in testpoints:
    if classify(x[0], sampleFaceMean, faceEkinv, facedetEk, sampleNofaceMean, nofaceEkinv, nofacedetEk)[0] and x[1] == 1:
        truepos += 1
    elif not (classify(x[0], sampleFaceMean, faceEkinv, facedetEk, sampleNofaceMean, nofaceEkinv, nofacedetEk)[0]) and x[1] == 0:
        trueneg += 1
    else:
        false += 1

print truepos, trueneg, false
print (truepos + trueneg)/len(testpoints)

# Test accuracy on patches not used for training
testFaces = os.listdir("./patches_face_unused/")
testnoFaces = os.listdir("./patches_noface_unused/")


testpoints = [(np.asarray(misc.imread("./patches_face_unused/" + x)), 1) for x in testFaces]
testpoints += [(np.asarray(misc.imread("./patches_noface_unused/" + x)), 0) for x in testnoFaces]


truepos = 0
trueneg = 0
false = 0
count = 0
for x in testpoints:
    if classify(x[0], sampleFaceMean, faceEkinv, facedetEk, sampleNofaceMean, nofaceEkinv, nofacedetEk)[0] and x[1] == 1:
        truepos += 1
        print count,
    elif not (classify(x[0], sampleFaceMean, faceEkinv, facedetEk, sampleNofaceMean, nofaceEkinv, nofacedetEk)[0]) and x[1] == 0:
        trueneg += 1
        print count,
    else:
        false += 1
    count+= 1

print 
print truepos, trueneg, false
print (truepos + trueneg)/len(testpoints)
