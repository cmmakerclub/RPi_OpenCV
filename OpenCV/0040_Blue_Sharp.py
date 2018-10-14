#!/usr/bin/env python

import cv2
import numpy as np

image = cv2.imread( 'image_01.jpg' )

cv2.imshow( 'Original', image )
cv2.waitKey( 0 )

blur_img = cv2.blur(image,(9,9)) # (9 x 9) filter is used 

cv2.imshow('Blurred', blur_img)
cv2.waitKey(0)

kernel = np.array([[-1,-1,-1], 
                   [-1,9,-1], 
                   [-1,-1,-1]])
sharpened_img = cv2.filter2D(image, -1, kernel)

cv2.imshow('Sharpen', sharpened_img)
cv2.waitKey(0)
