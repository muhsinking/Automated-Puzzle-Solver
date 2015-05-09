import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
from puzzlepiece import *
from math import *

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


pieces = []

# Perform thresholding
im = cv2.imread('puzzle_scrambled_rotated.png')
#im = cv2.imread('2pieces_matching.png')

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
    contour = cv2.approxPolyDP(piece.getContour(), 1, True)
    cv2.polylines(piece.getPixels(), contour, True, (0, 255, 0), 3, 2)
    cv2.imshow("Window", piece.getPixels())
    angles = np.zeros(contour.shape[0])
    for i in range(0,contour.shape[0]-1):
        #find angle between two points
        p1 = contour[i][0]
        p2 = contour[i+1][0]
        xDiff = p2[0] - p1[0]
        yDiff = p2[1] - p1[1]
        angle = atan2(yDiff, xDiff) * 180 / pi
        angles[i] = angle   
        print (i)
        print(contour[i][0])
        #print(xDiff)
        #print(yDiff)
        print(angles[i])
        print(' ')
        #print(np.diff(angles)[i])
    #plt.plot(np.arange(angles.shape[0]), angles, 'b', np.arange(np.diff(angles).shape[0]), np.diff(angles), 'r')
    #plt.plot(np.arange(angles.shape[0]), angles, 'b')
    plt.plot(np.arange(np.diff(angles).shape[0]), np.diff(angles), 'r')
    plt.show()
    cv2.waitKey()

