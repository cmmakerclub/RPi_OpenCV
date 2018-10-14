#!/usr/bin/env python

import cv2
import numpy as np

image = cv2.imread( 'image_01.jpg' )

cv2.imshow( 'Original', image )
cv2.waitKey( 0 )

canny_img = cv2.Canny(image, 50, 200)

cv2.imshow( 'Canny Edge Detection', canny_img )
cv2.waitKey( 0 )

