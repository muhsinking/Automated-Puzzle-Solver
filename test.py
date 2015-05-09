#import numpy as np
import cv2

# Perform thresholding
im = cv2.imread('contourtest.png')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
chainwarden = imgray[0][0]-1
ret, imthresh = cv2.threshold(imgray, chainwarden, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours = cv2.findContours(imthresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(im, contours[1], -1, (0, 255, 0), 0)

print(contours[1][1])

cv2.imshow("Window", im)
cv2.waitKey()