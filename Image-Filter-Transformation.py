import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
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
arr_noise = gray + (150*np.random.rand(rows, columns)-(150/2))
A1 = fp.fft2(arr_noise)
A2 = fp.fftshift(A1)
psd2D = np.abs(A2)**2
sd2D = np.abs(A2)
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

# Gaussian filter 2
def gaussian_kernel(size, size_y=None, mean_x=301, mean_y=401):
    size = int(size)
    if not size_y:
        size_y = size
    else:
        size_y = int(size_y)
    #x, y = np.mgrid[-size:size+1, -size_y:size_y+1]
    x, y = np.mgrid[1:(2*mean_x) - 1, 1:(2*mean_y) - 1]
    g = np.exp(-((x-mean_x)**2/float(size)+(y-mean_y)**2/float(size_y)))
    return g / g.sum()

gaussian_filter = gaussian_kernel(150)

# make these smaller to increase the resolution
dx, dy = 1, 1

# generate 2 2d grids for the x & y bounds
y, x = np.mgrid[slice(1, 600 + dy, dy),
                slice(1, 800 + dx, dx)]
plt.pcolor(x, y, gaussian_filter)
plt.title('Guassian Filter 2')
# set the limits of the plot to the limits of the data
plt.axis([x.min(), x.max(), y.min(), y.max()])
plt.colorbar()
plt.show()

freq_filter = sd2D * gaussian_filter
image_filter = np.fft.irfft2(np.fft.fftshift(freq_filter))
plt.imshow(image_filter, interpolation='nearest')
plt.title('Image after Gaussian filter')
plt.show()

'''
gray = rgb2gray(img)
plt.imshow(gray, cmap = plt.get_cmap('gray'))
plt.show()

A1 = fp.fft2(gray)
A2 = fp.fftshift(A1)
psd2D = np.abs(A2)**2
sd2D = np.abs(A2)
plt.imshow(np.log10(sd2D))
plt.show()
'''


'''
img = mpimg.imread('Cersei.jpg')
plt.imshow(img)
plt.show()
print(img.shape)
'''

'''
fname = 'Cersei.jpg'
image = Image.open(fname).convert("L")
arr = np.asarray(image)
print(arr.shape)
rows, columns = arr.shape
plt.imshow(arr, cmap='gray')
plt.show()

arr_noise = arr + (150*np.random.rand(rows, columns)-(150/2))

plt.imshow(arr_noise, cmap='gray')
plt.show()
'''


