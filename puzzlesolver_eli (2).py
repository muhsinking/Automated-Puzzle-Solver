import numpy as np
import cv2
import sys
from puzzlepiece import *

np.set_printoptions(threshold = 10000, linewidth = 1000)

def getInternalPixels(img, contour, nullval=-1):
	maxx, maxy = 0, 0
	minx, miny = sys.maxsize, sys.maxsize

	#print(img)

	for val in contour:
		if val[0][0] > maxx :
			maxx = val[0][0]
		if val[0][0] < minx :
			minx = val[0][0]
		if val[0][1] > maxy : 
			maxy = val[0][1] 
		if val[0][1] < miny : 
			miny = val[0][1]

	#retPx = np.zeros((maxx-minx+3, maxy-miny+3, 3))
	retPx = np.zeros((maxy-miny+3, maxx-minx+3, 3))
	print("Maxx: "+str(maxx)+" Minx: "+str(minx)+" Maxy: "+str(maxy)+" Miny "+str(miny))

	ri, ci = 0, 0

	for row in range(miny-1, maxy+2):
		ci = 0
		for col in range(minx-1, maxx+2):
			if cv2.pointPolygonTest(contour, (col, row), False) > 0:
				retPx[ri][ci] = img[row][col]
			else:
				retPx[ri][ci] = nullval
			ci += 1
		ri += 1

	return retPx


pieces = []

# Perform thresholding
#im = cv2.imread('puzzle_scrambled_rotated.png')
im = cv2.imread('contourtest_color.png')
print(im)
cv2.imshow("Window", im)
cv2.waitKey()

imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
chainwarden = imgray[0][0]-1
ret, imthresh = cv2.threshold(imgray, chainwarden, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours = cv2.findContours(imthresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#cv2.drawContours(im, contours[1], -1, (0, 255, 0), 0)

#cv2.imshow("Window", im)
#cv2.waitKey()

for c in contours[1]:
	newpiece = PuzzlePiece(c, getInternalPixels(im, c, (255, 255, 255)))
	pieces.append(newpiece)

print(pieces[0].getPixels())

cv2.imshow("Window", pieces[0].getPixels())
cv2.waitKey()
