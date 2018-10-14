#!/usr/bin/env python

import cv2

image = cv2.imread( 'image_01.jpg' )

cv2.imshow( 'Original', image )
cv2.waitKey( 0 )

cv2.imwrite( 'saved_image_01.jpg', image )

