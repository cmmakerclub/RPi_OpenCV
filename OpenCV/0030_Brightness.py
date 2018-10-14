#!/usr/bin/env python

import cv2
import numpy as np

image = cv2.imread( 'image_01.jpg' )

cv2.imshow( 'Original', image )
cv2.waitKey( 0 )

matrix = np.ones(image.shape, dtype = "uint8") * 100

add = cv2.add(image, matrix)

cv2.imshow("Added", add)
cv2.waitKey( 0 )

subtract = cv2.subtract(image, matrix)

cv2.imshow("Subtracted", subtract)
cv2.waitKey( 0 )
