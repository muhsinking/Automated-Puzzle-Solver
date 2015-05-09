import numpy as np
import cv2
import sys
from puzzlepiece import *
from math import *

# finds the internal pixels of a contour
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

# finds the angle between three points (p1 is the vertex)
# using the law of cosines
def getAngle3pts(p1,p2,p3):
    p12 = sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    p13 = sqrt((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2)
    p23 = sqrt((p2[0] - p3[0])**2 + (p2[1] - p3[1])**2)

    return acos((p12**2 + p13**2 - p23**2) / (2 * p12 * p13)) * 180 / pi

# finds the angle between two points
# this is legacy code which we discuss in our report but do not use in our
# final implementation
def getAngle2pts(p1,p2):
    xDiff = p2[0] - p1[0]
    yDiff = p2[1] - p1[1]

    angle = atan2(yDiff, xDiff) * 180 / pi
    
    # makes negative angles loop back around
    if angle < 0:
        angle = 360 - angle

np.set_printoptions(threshold = 10000, linewidth = 1000)

pieces = []

# Load image and perform thresholding
im = cv2.imread('puzzlepiece.png')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
chainwarden = imgray[0][0]-1
ret, imthresh = cv2.threshold(imgray, chainwarden, 255, cv2.THRESH_BINARY_INV)
imForContours = np.copy(imthresh)

# Find contours
contours = cv2.findContours(imForContours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(im, contours[1], -1, (0, 127, 0), 3)

# Get internal pixels of each peace, and create contours
numpieces = 0

for c in contours[1]:
    rgbpx = getInternalPixels(im, c, (255, 255, 255))
    graypx = getInternalPixels(imthresh, c, (0, 0, 0))
    graypx = cv2.cvtColor(graypx, cv2.COLOR_BGR2GRAY)

    localContour = cv2.findContours(graypx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1][0]

    newpiece = PuzzlePiece(numpieces, localContour, rgbpx, )
    pieces.append(newpiece)
    numpieces += 1

# Main contour loop
for piece in pieces:
    contour = cv2.approxPolyDP(piece.getContour(), 1, True)


    length = contour.shape[0]

    # Cleans the contour by averaging points that are too close together
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
                    contour[j][0] = ( (p3[0]+p1[0])/2 , (p3[1]+p1[1])/2 )
                    deletions += 1
                    contour = np.delete(contour,(j+1)%contour.shape[0],0)
                    flag = True

    length = contour.shape[0]
    angles = np.zeros(contour.shape[0])
    xAxis = np.zeros(angles.shape[0])

    # Compute the first angle outside of the loop
    # (probably a better way to do this)
    p2 = contour[length-1][0]
    p1 = contour[0][0]
    p3 = contour[1][0]

    angle = getAngle3pts(p1,p2,p3)

    angles[0] = angle

    for i in range(1,length):

        # Find the three-point sliding window (p1 is the vertex)
        p2 = contour[i-1][0]
        p1 = contour[i][0]
        p3 = contour[(i+1)%length][0]

        # Compute the distance between the vertex and the next point
        xDiff = p3[0] - p1[0]
        yDiff = p3[1] - p1[1]

        distance = sqrt( xDiff**2 + yDiff**2 )
        
        # Get the angle between the three points with p1 as the vertex
        angle = getAngle3pts(p1,p2,p3)
        angles[i%length] = angle


        # Find the euclidian distance between points
        # to scale the x-axis for the approximated contours
        # (this is necessary because the approximation removes
        # all contours between the corners ofa flat side)        
        if (i > 0):
            distance += xAxis[i-1]
        
        xAxis[i%length] = distance

    piece.setContour(contour)
    cv2.polylines(piece.getPixels(), contour, True, (0, 255, 0), 3, 2)  
    
    # Find corners by computing the minimum angles
    minus_min_array = np.copy(np.diff(angles))  
    minus_min_array = np.copy(angles)  
    cornerIndices = np.zeros(4)
    for i in range(4):
        minTheta = np.argmin(minus_min_array)
        minus_min_array[minTheta] = sys.maxsize
        cv2.circle(piece.getPixels(), (contour[minTheta][0][0], contour[minTheta][0][1]), 4, (255,0,0),-1)
        cornerIndices[i] = minTheta

    start = np.amin(cornerIndices)

    # Build the edges of the piece from its corners
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
    
    #cv2.imshow("Window", piece.getPixels())


    # Show the piece's edges, one at a time
    edgeContours = np.array([[],[],[],[]])
    edgeCount = 0
    print (edgeContours)
    for edge in piece.getEdges():

        edgeArray = np.zeros((len(edge), 1, 2), np.int32)
        for i in range(len(edge)):
            edgeArray[i][0][0] = edge[i][0]
            edgeArray[i][0][1] = edge[i][1]  
        dim = piece.getDimensions()
        imEdges = np.zeros((dim[0], dim[1]), np.uint8)

        for i in range(edgeArray.shape[0]-1):
            cv2.line(imEdges, (edgeArray[i][0][0], edgeArray[i][0][1]), (edgeArray[i+1][0][0], edgeArray[i+1][0][1]), 355, 1)

        edgeContour = cv2.findContours(np.copy(imEdges), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        imPieceEdges = np.copy(piece.getPixels())
        cv2.drawContours(imPieceEdges, edgeContour[1], -1, (0, 0, 255), 4)

        edgeContours[3]
        edgeContours[0] = edgeContour[1]
        print(edgeContours)
        cv2.imshow("Edge", imPieceEdges)

        # print(edgeContour[1])
        
        print(edgeContours.shape)

        edgeCount += 1

        cv2.waitKey()


    piece.setEdgeContours(edgeContours)
    print (piece.getEdgeContours())
    # print (piece.getEdgeContours()[1])
    print(cv2.matchShapes(piece.getEdgeContours()[0], piece.getEdgeContours()[1], 1, 0))    



# for piece1 in pieces
#     for piece2 in pieces
#         if piece2.getID() != piece.getID():
#             for i in range(4):
#                 ret = cv2.matchShapes(piece.getEdgeContours()[0][0], piece.getEdgeContours()[1][0], 1, 0)

