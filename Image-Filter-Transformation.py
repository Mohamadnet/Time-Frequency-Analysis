import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

img = mpimg.imread('Cersei.jpg')
imgplot = plt.imshow(img)
plt.show()
print(img.size)

img2 = Image.open('Cersei.jpg')
imgplot = plt.imshow(img2)
print(img2.size)

# Session 10
