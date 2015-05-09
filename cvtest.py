import numpy as np
import cv2
import sys

im = cv2.imread('tinytest.png')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)