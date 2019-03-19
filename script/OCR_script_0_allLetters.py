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

threshold = 0.65 # final determinant for match. lower values mean more matches, but also increases false positives

#load template images
tmp_letters = [cv2.imread(file, 0) for file in glob.glob("../template_letters_lowerscore/*.png")]
res = []

# for reach template, make the matching 
for i, s in enumerate(tmp_letters):
    # preprocess one template letter
    tmp_letters[i] = cv2.bitwise_not(tmp_letters[i])
    retVal, tmp_letters[i] = cv2.threshold(tmp_letters[i],60,255,cv2.THRESH_BINARY) #this threshold works well at around 60
    w, h = tmp_letters[i].shape[::-1]
    res.append(cv2.matchTemplate(binaryThresh,tmp_letters[i],cv2.TM_CCOEFF_NORMED))
    loc = np.where( res[i] >= threshold)
    # draw rectangles at the matched regions for one template
    for pt in zip(*loc[::-1]):
        cv2.rectangle(out, pt, (pt[0] + w, pt[1] + h), (255,0,0), 1)        
        out[pt[1]:pt[1]+h, pt[0]:pt[0]+w] = cv2.cvtColor(tmp_letters[i],cv2.COLOR_GRAY2RGB)

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
