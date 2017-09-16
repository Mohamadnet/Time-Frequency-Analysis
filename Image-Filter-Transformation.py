import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.fftpack as fp
import scipy.ndimage as ndimage
from numpy import pi, exp, sqrt

# Session 10


img = ndimage.imread('Cersei.jpg')
plt.imshow(img, interpolation='nearest')
plt.title('Original')
plt.show()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


gray = rgb2gray(img)
plt.imshow(gray, cmap = plt.get_cmap('gray'))
plt.title('Gray')
plt.show()

A1 = fp.fft2(gray)
A2 = fp.fftshift(A1)
psd2D = np.abs(A2)**2
sd2D = np.abs(A2)
plt.imshow(np.log10(sd2D))
plt.title('Gray FFT')
plt.show()



gray = rgb2gray(img)
rows, columns = gray.shape
arr_noise = gray + (50*np.random.rand(rows, columns)-(150/2))
A1 = fp.fft2(arr_noise)
A2 = fp.fftshift(A1)
psd2D = np.abs(A2)**2
sd2D = np.abs(A2)
print(sd2D.shape)
plt.imshow(np.log10(sd2D))
plt.title('Gray + random noise    FFT')
plt.show()

# Note the 0 sigma for the last axis, we don't wan't to blurr the color planes together!
# Thus step is generally to add Gaussian noise to the original signal
gray = ndimage.gaussian_filter(arr_noise, sigma=(5, 5), order=0)
plt.imshow(gray, interpolation='nearest')
plt.title('Gray + Gaussian filter 1')
plt.show()



# There is a nice tutorial on fft of image in Python
# http://www.astrobetter.com/blog/2010/03/03/fourier-transforms-of-images-in-python/

# Take the fourier transform of the image.
A1 = fp.fft2(gray)
# Now shift the quadrants around so that low spatial frequencies are in
# the center of the 2D fourier transformed image.
A2 = fp.fftshift(A1)
# Calculate a 2D power spectrum
psd2D = np.abs(A2)**2
sd2D_gaussian = np.abs(A2)
plt.imshow(np.log10(sd2D_gaussian))
plt.title('Gray + Gaussian filter     FFT  1')
plt.show()
'''
# Gaussian filter 2
def _gaussianMatrix(self, dim, sigma):
    """
    Create and return a 2D matrix filled with a gaussian distribution. The
    returned matrix will be of shape (dim, dim). The mean of the gaussian
    will be in the center of the matrix and have a value of 1.0.
    """

    gaussian = lambda x, sigma: exp(-(x**2) / (2*(sigma**2)))

    # Allocate the matrix
    m = np.empty((600, 800), dtype=np.double)

    # Find the center
    center = (dim - 1) / 2.0

    # TODO: Simplify using numpy.meshgrid
    # Fill it in
    for y in range(600):
      for x in range(800):
        dist = np.sqrt((x-401)**2 + (y-301)**2)
        m[y,x] = gaussian(dist, sigma)

    return m

mean_x=201
mean_y=301
#x, y = np.mgrid[-size:size+1, -size_y:size_y+1]
# make these smaller to increase the resolution
dx, dy = 1, 1

# generate 2 2d grids for the x & y bounds
x, y = np.mgrid[slice(1, 600 + dy, dy),
                slice(1, 800 + dx, dx)]
sigma = 100
gaussian_filter = _gaussianMatrix(np.ones(shape=x.shape),dim=10,sigma=sigma)
plt.pcolor(y, x, gaussian_filter)
plt.title('Guassian Filter 2')
# set the limits of the plot to the limits of the data
#plt.axis([x.min(), x.max(), y.min(), y.max()])
plt.colorbar()
plt.show()
'''


img = cv2.imread('Cersei.jpg',0)
img_noise = img + (50*np.random.rand(rows, columns)-(150/2))
dft = cv2.dft(np.float32(img_noise),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

plt.subplot(121),plt.imshow(img_noise, cmap = 'gray')
plt.title(' OpenCV Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('OpenCV Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()


rows, cols = img.shape
crow,ccol = np.int(rows/2) , np.int(cols/2)

# create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows,cols,2),np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

# apply mask and inverse DFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

