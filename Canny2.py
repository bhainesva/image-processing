#!bin/python2 
from ImgFunctions2 import *

# Parameters
######################################################
filename = 'bikewarrior.jpg'

# This is the radius of the kernel, so 3 would produce a kernel of size 7x7
kernel_size = 8
kernel_sigma = 0
gradient_size = 8
gradient_sigma =0

# Thresholds for hysteresis thresholding
t_l = 20
t_h = 25

fileout = filename.split('.')[0] + str(kernel_size) + "_" + str(kernel_sigma).replace(".", "-") + "_" + str(gradient_size) + "_" + str(gradient_sigma).replace(".","-") + "_" + str(t_l).replace(".","-") + "_" + str(t_h) + ".png" 
######################################################

# Calculate sigma if not set
if kernel_sigma == 0:
    kernel_sigma = 0.3 * (kernel_size/2 - 1) + 0.8
if gradient_sigma == 0:
    gradient_sigma = 0.3 * (gradient_size/2 - 1) + 0.8

# Read in image
origImg = misc.imread(filename)

# Convert image to grayscale if it isn't already
if len(origImg.shape) > 2:
    origImg = rgb2gray(origImg)

# PART 1 - CANNY_ENHANCER
# Create gaussian kernel to convolve with image.
kernel = newGaussian(kernel_size, kernel_sigma)

# Convolve kernel with image to get blurred image.
img = scipy.signal.convolve2d(origImg, kernel, boundary='symm')

#Compute gradient components
J_x, J_y = newGradient(gradient_size, gradient_sigma, origImg)

# Compute strength image
strImg = (J_x**2 + J_y**2)**0.5

# Calculate orientation image
ornImg = np.arctan2(J_y, J_x)

# PART 2 - NONMAX_SUPPRESSION
# Initialize ouput image to contain all 0s
I_n = np.zeros(strImg.shape)

# Dictionary to store direction approximations for each pixel
dirImg = {}

# Iterate through all pixels in image, set to 0 if not maximum along edge normal
it = np.nditer(ornImg, flags=['multi_index'], op_flags=['readwrite'])
while not it.finished:
    x, y = it.multi_index[1], it.multi_index[0]
    x_max, y_max = ornImg.shape[1], ornImg.shape[0]
    actual_dir = it[0]
    approx_dir = getDir(actual_dir)
    dirImg[(x, y)] = approx_dir

    x_off, y_off = approx_dir[0], approx_dir[1]
    if -1 < x + x_off < x_max and -1 < y + y_off < y_max and -1 < x-x_off < x_max and -1 < y-y_off < y_max:
        if strImg[y, x] < strImg[y+y_off, x+x_off] or strImg[y,x] < strImg[y-y_off, x-x_off]:
            I_n[y, x] = 0
        else:
            I_n[y, x] = strImg[y, x]

    it.iternext()
    

# PART 3 - HYSTERESIS THRESH
    # PART i. - BUILDING CHAINS

# List to hold chains of pixels
chains = []
# Boolean array to remember what pixels have been visited
visited = np.zeros(strImg.shape)

# Function to recursively visit pixels and add them to chains
def visit(x, y, dirImg, img, visited):
    chain = [(x, y)]
    visited[y, x] = True
    (x_off, y_off) = dirImg[(x, y)]
    (ymax, xmax) = img.shape
    if (-1 < y + y_off < ymax and -1 < x + x_off < xmax and not visited[y+y_off, x+x_off]):
        if (img[y + y_off, x + x_off] > t_l):
            visited[y+y_off, x+x_off] = True
            chain += visit(x+x_off, y+y_off, dirImg, img, visited)
    if (-1 < y - y_off < ymax and -1 < x - x_off < xmax and not visited[y-y_off, x-x_off]):
        if (img[y - y_off, x - x_off] > t_l):
            visited[y-y_off, x-x_off] = True
            chain += visit(x-x_off, y-y_off, dirImg, img, visited)
    return chain



# Normalize I_n to between 0 and 100
if np.amax(I_n) != 0:
    I_n = (I_n/np.amax(I_n)) * 100

# Iterate through I_n building chains
it = np.nditer(I_n, flags=['multi_index'], op_flags=['readwrite'])
while not it.finished:
    (y, x) = it.multi_index
    if it[0] < t_h:
        it.iternext()
        continue
    chains.append(visit(x, y, dirImg, I_n, visited))
    it.iternext()

final = np.zeros(strImg.shape)
list2 = [x for x in chains if x != []]

# Construct final image
for x in list2:
    for pix in x:
        final[pix[1], pix[0]] = I_n[pix[1], pix[0]]

misc.imsave(fileout.split(".")[0] + "JX." + fileout.split(".")[1], J_x)
misc.imsave(fileout.split(".")[0] + "JY." + fileout.split(".")[1], J_y)
misc.imsave(fileout.split(".")[0] + "STR." + fileout.split(".")[1], strImg)
misc.imsave(fileout.split(".")[0] + "MAXSUP." + fileout.split(".")[1], I_n)
misc.imsave(fileout, final)

print "DONE"
