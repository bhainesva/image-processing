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
import os
import skimage
from matplotlib import pyplot as plt
from collections import defaultdict

faceFilenames = os.listdir("./patches_face")
nofaceFilenames = os.listdir("./patches_noface")

faces = [misc.imresize(np.asarray(misc.imread("./patches_face/" + x)), (24, 24)) for x in faceFilenames]
nofaces = [misc.imresize(np.asarray(misc.imread("./patches_noface/" + x)), (24, 24)) for x in nofaceFilenames]

faces = np.asarray(faces)
faces = faces.reshape(10, 10, 24, 24)

rows = []
for i in faces:
    rows.append(np.hstack(i))

faces = np.vstack(rows)

nofaces = np.asarray(nofaces)
print len(faces)
print faces.shape
nofaces = nofaces.reshape(10, 10, 24, 24)

rows = []
for i in nofaces:
    rows.append(np.hstack(i))

nofaces = np.vstack(rows)
final = np.hstack((faces, nofaces))
plt.imshow(final)
misc.imsave('collage.png', final)
plt.gray()
plt.show()
