import numpy as np
import cv2
import sys
import math 
from puzzlepiece import *

np.set_printoptions(threshold = 10000, linewidth = 1000)

def getInternalPixels(img, contour, nullval=-1):
	maxx, maxy = 0, 0
	minx, miny = sys.maxsize, sys.maxsize

	for val in contour:
		if val[0][0] > maxx :
			maxx = val[0][0]
		if val[0][0] < minx :
			minx = val[0][0]
		if val[0][1] > maxy : 
			maxy = val[0][1] 
		if val[0][1] < miny : 
			miny = val[0][1]

	retPx = np.zeros((maxy-miny+3, maxx-minx+3, 3), np.uint8)

	ri, ci = 0, 0

	for row in range(miny-1, maxy+2):
		ci = 0
		for col in range(minx-1, maxx+2):
			if cv2.pointPolygonTest(contour, (col, row), False) >= 0:
				retPx[ri][ci] = img[row][col]
			else:
				retPx[ri][ci] = nullval
			ci += 1
		ri += 1

	return retPx

def getAngle3Pts(a, b, c):
	leg1 = math.hypot(a[0]-b[0], a[1]-b[1])
	leg2 = math.hypot(b[0]-c[0], b[1]-c[1])
	hyp = math.hypot(a[0]-c[0], a[1]-c[1])

	return math.degrees(math.acos((leg1 * leg1 + leg2 * leg2 - hyp * hyp)/(2.0 * leg2 * hyp)))

pieces = []

# Perform thresholding
#im = cv2.imread('puzzle_scrambled_rotated.png')
im = cv2.imread('puzzlepiece_whitebg.png')

cv2.imshow("Window", im)
cv2.waitKey()

imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
chainwarden = imgray[0][0]-1
ret, imthresh = cv2.threshold(imgray, chainwarden, 255, cv2.THRESH_BINARY_INV)
imForContours = np.copy(imthresh)

# Find contours
contours = cv2.findContours(imForContours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#cv2.drawContours(imthresh, contours[1], -1, (0, 127, 0), 3)

for c in contours[1]:
	rgbpx = getInternalPixels(im, c, (255, 255, 255))
	graypx = getInternalPixels(imthresh, c, (0, 0, 0))
	graypx = cv2.cvtColor(graypx, cv2.COLOR_BGR2GRAY)

	localContour = cv2.findContours(graypx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1][0]

	newpiece = PuzzlePiece(localContour, rgbpx)
	pieces.append(newpiece)

#print(pieces[0].getPixels())
#cv2.cvtColor(pieces[0].getPixels(), cv2.COLOR_BGR2GRAY)
for piece in pieces:
	approx = cv2.approxPolyDP(piece.getContour(), 1, True)
	length = approx.shape[0]
	angles = np.zeros(length)

	for i in range(1, length):
		prev = approx[i-1][0]
		current = approx[i][0]
		next = approx[(i+1)%length][0]

		
		angles[i%length] = getAngle3Pts(prev, current, next)
		print(prev, current, next, angles[i%length])

		temp = np.copy(piece.getPixels())
		cv2.circle(temp, (prev[0], prev[1]), 3, (255, 0, 0), -1)
		cv2.circle(temp, (current[0], current[1]), 3, (0, 0, 255), -1)
		cv2.circle(temp, (next[0], next[1]), 3, (0, 255, 0), -1)
		cv2.imshow("Window", temp)
		cv2.waitKey()

	print(angles)








	cv2.polylines(piece.getPixels(), approx, True, (0, 255, 0), 3, 2)
	cv2.imshow("Window", piece.getPixels())
	cv2.waitKey()


