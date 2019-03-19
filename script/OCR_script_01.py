import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt

############### preprocessing stage ###############

# read page to be processed as grayscale
img = cv2.imread('../raw_images/largepreview.png', 1)
out = img.copy()

# we don't need three channels, just grayscale
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# we work with the inverse (more intuitive)
invGray = cv2.bitwise_not(grayImg)

# threshold image to reduce undesired artefacts
retVal, binaryThresh = cv2.threshold(invGray,10,255,cv2.THRESH_BINARY) # the lower this threshold is, the bigger the area of the letter. start around 50

############### matching stage ###############

threshold = 0.8 # final determinant for match. lower values mean more matches, but also increases false positives
# preprocess just one template letter
templateLetter = cv2.imread('../template_letters_lowerscore/largepreview_0w.png', 0)
templateLetter = cv2.bitwise_not(templateLetter)
retVal, templateLetter = cv2.threshold(templateLetter,60,255,cv2.THRESH_BINARY) # we make template slightly thinner than img letters so it fits more
w, h = templateLetter.shape[::-1]

# Apply template Matching for one template
res_one = cv2.matchTemplate(binaryThresh,templateLetter,cv2.TM_CCOEFF_NORMED)
loc = np.where( res_one >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(out, pt, (pt[0] + w, pt[1] + h), (255,0,0), 1)
    out[pt[1]:pt[1]+h, pt[0]:pt[0]+w] = cv2.cvtColor(templateLetter,cv2.COLOR_GRAY2RGB)

############### presentation stage ###############

# show intermediate stages
plt.subplot(131)
plt.imshow(img)
plt.title('original image')

plt.subplot(132)
plt.imshow(binaryThresh, 'gray')
plt.title('binary thresholding')

plt.subplot(133)
plt.imshow(out, 'gray')
plt.title('Original with overlayed matches')

plt.show()
