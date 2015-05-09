import numpy as np
import cv2
import sys
#import matplotlib.pyplot as plt
from puzzlepiece import *
from math import *

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

def getAngle3pts(p1,p2,p3):
    # find the angle between three points (p1 is the vertex)

    p12 = sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    p13 = sqrt((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2)
    p23 = sqrt((p2[0] - p3[0])**2 + (p2[1] - p3[1])**2)

    return acos((p12**2 + p13**2 - p23**2) / (2 * p12 * p13)) * 180 / pi


def getAngle2pts(p1,p2):
    # find the angle between two points

    xDiff = p2[0] - p1[0]
    yDiff = p2[1] - p1[1]

    angle = atan2(yDiff, xDiff) * 180 / pi
    
    if angle < 0:
        angle = 360 - angle

np.set_printoptions(threshold = 10000, linewidth = 1000)

pieces = []

# Perform thresholding
im = cv2.imread('puzzle_big_2.png')
#im = cv2.imread('2pieces_matching.png')

#cv2.imshow("Window", im)
#cv2.waitKey()

imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
chainwarden = imgray[0][0]-1
ret, imthresh = cv2.threshold(imgray, chainwarden, 255, cv2.THRESH_BINARY_INV)
imForContours = np.copy(imthresh)

# Find contours
contours = cv2.findContours(imForContours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print(type(contours[1][0][0][0][0]))
cv2.drawContours(im, contours[1], -1, (0, 127, 0), 3)

# cv2.imshow("Window", im)
# cv2.waitKey()

#print (len(contours[1]))

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


    length = contour.shape[0]

    # offset

    flag = True

    while flag:
        flag = False
        deletions = 0
        k = 0
        j = 0
        for k in range(length):
            j = k+deletions

            if (j >= contour.shape[0]):
                break
            if (contour[j].any != -1):

                p1 = contour[j][0]
                p3 = contour[(j+1)%contour.shape[0]][0]

                xDiff = p3[0] - p1[0]
                yDiff = p3[1] - p1[1]

                distance = sqrt( xDiff**2 + yDiff**2 )

                if (distance < 8):
                    #print (j)
                    #print (contour[j])
                    #print (contour[j][0])
                    contour[j][0] = ( (p3[0]+p1[0])/2 , (p3[1]+p1[1])/2 )
                    deletions += 1
                    contour = np.delete(contour,(j+1)%contour.shape[0],0)
                    flag = True

    # contour_copy = np.zeros((contour.shape[0]-deletions),(1,2))
    # print (contour_copy)
    # deletions = 0

    # for i in range(1,length):
    #     if (contour[i].any != -1):
    #         contour_copy[i-deletions] = contour[i]
    #     else:
    #         deletions += 1

    length = contour.shape[0]
    angles = np.zeros(contour.shape[0])
    xAxis = np.zeros(angles.shape[0])

    p2 = contour[length-1][0]
    p1 = contour[0][0]
    p3 = contour[1][0]

    angle = getAngle3pts(p1,p2,p3)

    angles[0] = angle

    for i in range(1,length):

        #print (i)

        p2 = contour[i-1][0]
        p1 = contour[i][0]
        p3 = contour[(i+1)%length][0]

        # if (i == contour.shape[0]-1):
        #     p1 = contour[i][0]
        #     p2 = contour[0][0]
        # else:
        #     p1 = contour[i][0]
        #     p2 = contour[i+1][0]


        # finds the sliding window of three points (p1 is the vertex)


        xDiff = p3[0] - p1[0]
        yDiff = p3[1] - p1[1]

        distance = sqrt( xDiff**2 + yDiff**2 )


        #print (p2,p1,p3)

        
        angle = getAngle3pts(p1,p2,p3)
        #print (angle)
        angles[i%length] = angle


        # find the euclidian distance between points
        # to scale the x-axis for the approximated contours

        
        if (i > 0):
            distance += xAxis[i-1]
        
        xAxis[i%length] = distance

        # temp = np.copy(piece.getPixels())
        # cv2.circle(temp, (p2[0], p2[1]), 3, (255, 0, 0), -1)
        # cv2.circle(temp, (p1[0], p1[1]), 3, (0, 0, 255), -1)
        # cv2.circle(temp, (p3[0], p3[1]), 3, (0, 255, 0), -1)
        # cv2.imshow("Window", temp)
        # cv2.waitKey()

    piece.setContour(contour)

    #minus_max_array = np.copy(np.diff(angles))
    minus_max_array = np.copy(angles)

    cv2.polylines(piece.getPixels(), contour, True, (0, 255, 0), 3, 2)  

    # cv2.imshow("Window", piece.getPixels())
    # cv2.waitKey()

    # for i in range(4):
    #     maxdTh = np.argmax(minus_max_array)
    #     minus_max_array[maxdTh] = 0
    #     cv2.circle(piece.getPixels(), (contour[maxdTh][0][0], contour[maxdTh][0][1]), 4, (0, 0,255),-1)
    #     print (maxdTh)
    
    minus_min_array = np.copy(np.diff(angles))  
    minus_min_array = np.copy(angles)  

    cornerIndices = np.zeros(4)

    for i in range(4):
        mindTh = np.argmin(minus_min_array)
        minus_min_array[mindTh] = sys.maxsize
        cv2.circle(piece.getPixels(), (contour[mindTh][0][0], contour[mindTh][0][1]), 4, (255,0,0),-1)

        cornerIndices[i] = mindTh

    start = np.amin(cornerIndices)

    edgelength = 0
    firstcorner = start
    secondcorner = 0

    edgeCount = 0
    edges = []
    c1 = start
    c2 = 0
    i = start
    while edgeCount < 4:
        edge = []
        edge.append(contour[c1%length][0].tolist())
        i += 1
        
        while not(i%length in cornerIndices):
            edge.append(contour[i%length][0].tolist())
            i += 1
        edge.append(contour[i%length][0].tolist())
        edges.append(edge)
        edgeCount += 1
        c1 = i  

    piece.setEdges(edges)


    
    cv2.imshow("Window", piece.getPixels())

    imEdges = np.zeros((800, 800))

    # edgeArray = np.zeros((len(piece.getEdges()[0]), 1, 2), np.uint8)

    # for i in range(len(piece.getEdges()[0])):
    #     edgeArray[i][0][0] = piece.getEdges()[0][i][0]
    #     edgeArray[i][0][1] = piece.getEdges()[0][i][1]  

    edgeContours = []
    for edge in piece.getEdges():

        edgeArray = np.zeros((len(edge), 1, 2), np.int32)
        for i in range(len(edge)):
            edgeArray[i][0][0] = edge[i][0]
            edgeArray[i][0][1] = edge[i][1]  

        #print(piece.getContour())    
        #print(edgeArray)
        dim = piece.getDimensions()
        imEdges = np.zeros((dim[0], dim[1]), np.uint8)
        #cv2.polylines(imEdges, edgeArray, False, 255, 3)

        edgeContours = [] 
        for i in range(edgeArray.shape[0]-1):
            cv2.line(imEdges, (edgeArray[i][0][0], edgeArray[i][0][1]), (edgeArray[i+1][0][0], edgeArray[i+1][0][1]), 355, 1)

        edgeContour = cv2.findContours(np.copy(imEdges), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imEdges, edgeContour[1], -1, 127, 1)

        edgeContours.append(edgeContour)

        cv2.imshow("Edge", imEdges)
        # print(piece.getEdgeContours()[2])
        # print(cv2.matchShapes(piece.getEdgeContours()[0][0], piece.getEdgeContours()[1][0], 1, 0))
        cv2.waitKey()

    piece.setEdgeContours(edgeContours)
    #print(cv2.matchShapes(piece.getEdgeContours()[0][0], piece.getEdgeContours()[1][0], 1, 0))



    #plt.plot(np.arange(angles.shape[0]), angles, 'b', np.arange(np.diff(angles).shape[0]), np.diff(angles), 'r')
    #plt.plot(np.arange(angles.shape[0]), angles, 'b')
    #plt.plot(np.arange(np.diff(angles).shape[0]), np.diff(angles), 'r')
    
    # delete the last element 
    #xAxis = np.delete(xAxis,xAxis.shape[0]-1)
    #plt.plot(xAxis, np.diff(angles), 'r')
    
    # plt.plot(xAxis, angles, 'b')

    # plt.show()
    cv2.waitKey()
    

