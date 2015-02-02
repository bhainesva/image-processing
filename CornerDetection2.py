#!bin/python2
from ImgFunctions2 import *
from numpy import linalg as LA
import copy

# Input / Parameters
######################################
threshold = 900000
N = 3
gradientKernelSize = 5
gradientSigma = 0
supressionRadius = 8
filename = "building.jpg"
######################################

fileout = filename.split('.')[0] + "_" + str(gaussianKernelSize) + "_" + str(gradientKernelSize) + "_" + str(gaussianSigma).replace('.', '-') + "_" + str(gradientSigma).replace('.', '-') + "CORN." + filename.split('.')[1]
print fileout
# Set sigmas if not already set
if gradientSigma == 0:
    gradientSigma = 0.3 * (gradientKernelSize/2 - 1) + 0.8

# Read in image and convert it to grayscale if it isn't already
img = misc.imread(filename)
if len(img.shape) > 2:
    img = grayscale(img)

# Compute horizontal and vertical gradient components
J_x, J_y = newGradient(gradientKernelSize, gradientSigma, img)

# Setup images to store lowest eigenvalues and final corners
eigs = np.zeros(img.shape)
final = np.zeros(img.shape)

# Array to store points of interest
pointList = []

# Iterate through the image and compute covariance matrix at each point
it = np.nditer(img, flags=['multi_index'])
while not it.finished:
    (y, x) = it.multi_index
    C = computeC(x, y, N, J_x, J_y)
    smallestEig = min(LA.eig(C)[0])

    # If the smallest eigenvalue is larger than our threshold, add it to the list of potential corners
    if smallestEig > threshold:
        eigs[y, x] = smallestEig
        pointList.append((x, y))
    it.iternext()

# Sort list by decreasing value of lowest eigenvalue at that point
plt.imshow(eigs)
plt.show()
pointList = sorted(pointList, key=lambda x: -eigs[x[1], x[0]]) 
misc.imsave(fileout.split('.')[0] + "STR." + fileout.split('.')[1], eigs)

# Nonmaximum suppression 
for point in pointList:
    if (eigs[point[1], point[0]] != 0):
            clearSubMatrix(point[0], point[1], supressionRadius, eigs)
   # for x in range(max(0, point[0]-N), min(eigs.shape[1], point[0]+N+1)):
   #     for y in range(max(0, point[1]-N), min(eigs.shape[0], point[1]+N+1)):
   #         if x==point[0] and y==point[1]:
   #             continue
   #         if (x, y) in pointList: 
   #             pointList.remove((x, y))

for x in pointList:
    final[x[1], x[0]] = 255

holder = copy.deepcopy(eigs)

#misc.imsave("outline.jpg", eigs)

#misc.imsave(fileout, final)
#misc.imsave("out.jpg", img)
misc.imsave(fileout.split('.')[0] + "MAXSUP." + fileout.split('.')[1], eigs)

pos = eigs > 0

it = np.nditer(eigs, flags=['multi_index'], op_flags=['readwrite'])
while not it.finished:
    (y, x) = it.multi_index
    if pos[y, x]:
        print x, y
        it[0] = 255
        outline(x, y, 4, 2, eigs)
    it.iternext()


it = np.nditer(eigs, flags=['multi_index'], op_flags=['readwrite'])
while not it.finished:
    (y, x) = it.multi_index
    if it[0] > 0:
        img[y, x] = 100
    it.iternext()

misc.imsave(fileout, img)
misc.imsave(fileout.split('.')[0] + "JX." + fileout.split('.')[1], J_x)
misc.imsave(fileout.split('.')[0] + "JY." + fileout.split('.')[1], J_y)
#strImg = (J_x**2 + J_y**2)**0.5


