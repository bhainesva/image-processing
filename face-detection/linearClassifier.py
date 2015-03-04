#!../bin/python2
from __future__ import division
import math
import os
import numpy as np
import scipy
from ImgFunctions2 import *
from scipy import misc
from numpy import linalg as LA
from scipy import ndimage
from matplotlib import pyplot as plt
import copy

# Paramters
learningRate = 0.5
iterations = 7500

# Initialize weight vector to 0
w = np.zeros(145,)

# Get list of filenames
faceFilenames = os.listdir("./patches_face")
nofaceFilenames = os.listdir("./patches_noface")


# Convert 12x12 face patches into length 144 vectors, insert into tuple with 1, insert into list of points
points = [(np.asarray(misc.imread("./patches_face/" + x)).ravel(), 1) for x in faceFilenames]

# Repeat with nonfaces. Tuples consist now of the image vector and 0.
points += [(np.asarray(misc.imread("./patches_noface/" + x)).ravel(), 0) for x in nofaceFilenames]

# Scale every patch to between 0 and 1
points = [(x[0], x[1]) for x in points]

# Append 1 to the end of each image vector resulting in vectors of length 145
points = [(np.append(x[0], 1), x[1]) for x in points]

# Define classifier function. 
# Params: xi - image vector with length 145
#         w - weight vector
# Notes: Because math.exp() cannot handle values larger than ~700, the function is
#        clamped and simply returns 0 in these cases.
def g(xi, w):
    exp = -(w.dot(xi))
    if exp > 700:
        return 0
    return 1/(1+math.exp(exp))

def g2(xi, w):
    return -(w.dot(xi))

# Train classifier
# Notes: Potentially try online version rather than batch (see linked notes)
#        Maybe huge iterations?- but I go until it stops moving at all..
#        Scaling?
#        Maybe to compensate make the learning rate really tiny? - doesn't seem to have an effect
#        I think it's implemented correctly, it's pretty simple...
#        Where are negative values coming from?

#testFaces = os.listdir("./patches_face/")
#testnoFaces = os.listdir("./patches_noface/")
#testpoints = [(np.asarray(misc.imread("./patches_face/" + x)).ravel(), 1) for x in testFaces]
#testpoints += [(np.asarray(misc.imread("./patches_noface/" + x)).ravel(), 0) for x in testnoFaces]
#testpoints = [(np.append(x[0], 1), x[1]) for x in testpoints]
#accuracy = []

for i in xrange(iterations):
    total = 0
    for point in points:
        total += (point[1] - g(point[0], w)) * point[0]
    total *= learningRate
    w += total
    correct = 0
#    for point in testpoints:
#        if ((g(point[0], w) > 0.5) and point[1] == 1) or ((g(point[0], w) <= 0.5) and point[1] == 0):
#            correct += 1
#    accuracy.append(correct/len(testpoints))

# Plot accuracy
#plt.axis([0.0,7500.0, 0.0,1.0])
#ax = plt.gca()
#ax.set_autoscale_on(False)
#plt.plot(accuracy)
#plt.ylabel('Accuracy')
#plt.xlabel('Iterations')
#plt.savefig('foo.png', bbox_inches='tight')
    


# Read in image to test
img = misc.imread('judybatstest.jpg')
# Convert image to grayscale if it isn't already
if len(img.shape) > 2:
    img = grayscale(img)
# Setup output image
out = np.zeros(img.shape)

#img = np.asarray(img).ravel()
#img = img/np.amax(img)
#img = np.append(img, 1)
#if g(img, w) > 0.5:
#    print "TRUE"
#else:
#    print "FALSE"

# Iterate over every 12x12 patch in the image and check if it contains a face. 
# Store scores for each patch and make somethreshold?
# MxM neighborhood thresholding
# Some cutoff score combined with an above technique
# Weight results close to center/eyelevel
total = 0
rect = copy.deepcopy(img)
coords = []
pointMatrix = np.zeros(img.shape)

for x in range(img.shape[1] - 12):
    for y in range(img.shape[0] - 12):
        patch = img[y:y+12, x:x+12]
        patch = np.append(np.asarray(patch).ravel(), 1)
        a = g(patch, w)
        #print a
        if a > .5:
            exp = g2(patch, w)
            coords.append((x, y, exp))
            total+= 1
            out[y:y+12, x:x+12] = 1
            pointMatrix[y, x] = exp
            #outline(x+6, y+6, 6, 1, rect) 

# Nonmaximum suppression 
coords = sorted(coords, key=lambda x: x[2])
for coord in coords:
    if (pointMatrix[coord[1], coord[0]] != 0):
            clearSubMatrix(coord[0], coord[1], 15, pointMatrix)

# Draw face boxes
it = np.nditer(pointMatrix, flags=['multi_index'], op_flags=['readwrite'])
while not it.finished:
    (y, x) = it.multi_index
    if pointMatrix[y,x] != 0:
        outline(x+6, y+6, 6, 1, rect)
    it.iternext()

print "PERCENT FACES", total/(img.shape[0] * img.shape[1])
plt.imshow(rect)
plt.gray()
plt.imsave('rect.png', rect)
plt.show()

correct = 0
for point in points:
    if ((g(point[0], w) > 0.5) and point[1] == 1) or ((g(point[0], w) <= 0.5) and point[1] == 0):
        correct += 1

print "TRAINING", correct/len(testpoints)
        
testFaces = os.listdir("./patches_face_unused/")
testnoFaces = os.listdir("./patches_noface_unused/")


testpoints = [(np.asarray(misc.imread("./patches_face_unused/" + x)).ravel(), 1) for x in testFaces]
testpoints += [(np.asarray(misc.imread("./patches_noface_unused/" + x)).ravel(), 0) for x in testnoFaces]
testpoints = [(np.append(x[0], 1), x[1]) for x in testpoints]

correct2 = 0
for point in testpoints:
    if ((g(point[0], w) > 0.5) and point[1] == 1) or ((g(point[0], w) <= 0.5) and point[1] == 0):
        correct2 += 1

print "UNUSED", correct2/len(testpoints)

            

