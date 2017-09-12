import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.fftpack as fp

# Session 10
'''
img = mpimg.imread('Cersei.jpg')
plt.imshow(img)
plt.show()
print(img.shape)
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

# There is a nice tutorial on fft of image in Python
# http://www.astrobetter.com/blog/2010/03/03/fourier-transforms-of-images-in-python/

# Take the fourier transform of the image.
F1 = fp.fft2(arr)
# Now shift the quadrants around so that low spatial frequencies are in
# the center of the 2D fourier transformed image.
F2 = fp.fftshift(F1)
# Calculate a 2D power spectrum
psd2D = np.abs(F2)**2

plt.imshow(np.log10(psd2D))
plt.show()
