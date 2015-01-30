#!bin/python2
from ImgFunctions import *

# Parameters
filename = 'building.jpg'
fileout = 'mybuildingout.png'
sigma = 8
kernel_size = 5

origImg = misc.imread(filename)
origImg = rgb2gray(origImg)

# PART 1 - CANNY_ENHANCER
# Apply gaussian filter
# Create gaussian kernel to convolve with image.
kernel = newGaussian(kernel_size, sigma)

# Convolve kernel with image
img = scipy.signal.convolve2d(origImg, kernel, boundary='symm')

#Compute gradient components
#J_y, J_x = np.gradient(img)
J_x, J_y = newGradient(3, 1, origImg)

# Strength image
strImg = (J_x**2 + J_y**2)**0.5

# Calculate gradients and orientation image
#J_y = np.asfarray(J_y, dtype='float')
#J_x = np.asfarray(J_x, dtype='float')

ornImg = np.arctan2(J_y, J_x)
#ornImg[np.isnan(ornImg)]=math.pi/2


# PART 2 - NONMAX_SUPPRESSION
# Initialize ouput image to contain all 0s
I_n = np.zeros(strImg.shape)

# Dictionary to store direction approximations for each pixel
dirImg = {}

# Iterate through all pixels in image
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

# Set up thresholds
t_l = 180
t_h = 200
chains = []
visited = np.zeros(strImg.shape)
it = np.nditer(I_n, flags=['multi_index'], op_flags=['readwrite'])

def visit(I_n, dirImg, visited, x, y, t_l):
    (ymax, xmax) = I_n.shape
    chain = []
    if visited[y, x] == 1:
        return chain
    visited[y, x] = 1
    if (I_n[y, x] < t_l):
        return chain
    chain.append((x, y))
    off = dirImg[(x, y)]
    if (0 < y + off[1] < ymax and 0 < x + off[1] < xmax):
        if (I_n[y + off[1], x+off[0]] > t_l):
            chain += visit(I_n, dirImg, visited, x+off[0], y+off[1], t_l)
    if (0 < y - off[1] < ymax and 0 < x - off[1] < xmax):
        if (I_n[y - off[1], x-off[0]] > t_l):
            chain += visit(I_n, dirImg, visited, x+off[0], y+off[1], t_l)
    return chain


for x in range(strImg.shape[1]):
    for y in range(strImg.shape[0]):
        if (visited[y, x] == 0):
            chain = visit(I_n, dirImg, visited, x, y, t_l)
            chains.append(chain)

final = np.zeros(strImg.shape)
list2 = [x for x in chains if x != []]

somelist = [x for x in list2 if max(x, key=lambda y : I_n[y[1], y[0]]) > t_h]
print len(list2)
print len(somelist)
for x in somelist:
    for pix in x:
        final[pix[1], pix[0]] = I_n[pix[1], pix[0]]

plt.imshow(final)
plt.imsave('I_n', I_n)
plt.gray()
plt.show()
misc.imsave(fileout, final)
