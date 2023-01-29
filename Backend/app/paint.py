# import required libraries
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt



# read the image 
image = cv2.imread('Backend\\assets\\phone.png')

# resize the image
image_resize = cv2.resize(image, None, fx = 0.5, fy = 0.5)

# clear the impurities
img_clear = cv2.medianBlur(image_resize, 3)
img_clear = cv2.medianBlur(img_clear, 3)
img_clear = cv2.medianBlur(img_clear, 3)


img_clear = cv2.edgePreservingFilter(img_clear, sigma_s = 5)

# gaussian filter would also blur the corner of the image
# so bilateral filter was used to preserve the corners in the image from the blur effect
img_filter = cv2.bilateralFilter(img_clear, 3, 10, 5)

for bi in range(2):
  image_filter = cv2.bilateralFilter(img_filter, 3, 20, 10)

for bi in range(3):
  image_filter = cv2.bilateralFilter(img_filter, 5, 30, 10)


# remove the blur effect from the bilateralFilter we used guassian filter and mask
guassian_masks = cv2.GaussianBlur(img_filter, (7, 7), 2)


img_sharp = cv2.addWeighted(img_filter, 1.5, guassian_masks, -0.5, 0)
img_sharp = cv2.addWeighted(img_sharp, 1.4, guassian_masks, -0.6, 10)

plt.imshow(img_sharp[:,:,::-1])