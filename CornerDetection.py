#!bin/python2
from ImgFunctions import *
from numpy import linalg as LA

# Input / Parameters

threshold = 20000
kernelSize = 3
sigma = 0.3 * (kernelSize/2 - 1) + 0.8
length = 2
img = misc.imread('checker.jpg')

img = rgb2gray(img)
#J_y, J_x = np.gradient(img)
J_x, J_y = newGradient(kernelSize, sigma, img)

#def tupleDistance(tup1, tup2):
#    return math.sqrt((tup1[0] - tup2[0])**2 + (tup1[1] - tup2[1])**2)
#
def computeC(x, y, length, jx, jy):
    Ex2 = np.sum(getSubMatrix(x, y, length, jx)**2)
    Ey2 = np.sum(getSubMatrix(x, y, length, jy)**2)
    Exy = np.sum(getSubMatrix(x, y, length, jx) * getSubMatrix(x, y, length, jy))
    return np.asarray([[Ex2, Exy],[Exy, Ey2]])

def getSubMatrix(x, y, length, img):
    (ymax, xmax) = img.shape
    xstart = max(x-length, 0)
    xend = min(x+length+1, xmax)
    ystart = max(y-length, 0)
    yend = min(y+length+1, ymax)
    return img[ystart:yend, xstart:xend]

def clearSubMatrix(x, y, length, img):
    tmp = img[y, x]
    (ymax, xmax) = img.shape
    xstart = max(x-length, 0)
    xend = min(x+length+1, xmax)
    ystart = max(y-length, 0)
    yend = min(y+length+1, ymax)
    img[ystart:yend, xstart:xend] = 0
    img[y, x] = tmp

interestPoints = []

smallesteigs = np.zeros(img.shape)
eigs = np.zeros(img.shape)
final = np.zeros(img.shape)

it = np.nditer(img, flags=['multi_index'])
while not it.finished:
    (y, x) = it.multi_index
    C = computeC(x, y, length, J_x, J_y)
    #C = np.asarray([[Ex2[y, x], Exy[y, x]], [Exy[y, x], Ey2[y, x]]])
    smallestEig = min(LA.eig(C)[0])
    #smallesteigs[y, x] = smallestEig
    if smallestEig > threshold:
        eigs[y, x] = smallestEig
    it.iternext()

print J_x
plt.gray()
plt.imshow(eigs)
plt.imsave('eigs', eigs)
plt.show()
#plt.imshow(smallesteigs)
#plt.show()

it = np.nditer(eigs, flags=['multi_index'])
pointList = []
while not it.finished:
    if it[0] == 0:
        it.iternext()
        continue
    (y, x) = it.multi_index
    if (it[0] == np.amax(getSubMatrix(x, y, 2*length, eigs))):
        final[y, x] = it[0]
        clearSubMatrix(x, y, length, eigs)
        pointList.append((x, y))
    it.iternext()


plt.gray()
for x in pointList:
    img[x[1], x[0]] = 0

plt.imshow(final)
plt.show()
plt.imshow(img)
plt.show()
misc.imsave('corners2.jpg', final)





