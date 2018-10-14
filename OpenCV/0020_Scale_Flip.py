#!/usr/bin/env python

import cv2

image = cv2.imread( 'image_01.jpg' )

cv2.imshow( 'Original', image )
cv2.waitKey( 0 )

cubic_img = cv2.resize(image, None, fx=0.3, fy=0.3, interpolation = cv2.INTER_CUBIC)

cv2.imshow( 'Cubic Scaled', cubic_img )
cv2.waitKey( 0 )

linear_img = cv2.resize(image, None, fx=0.3, fy=0.3, interpolation = cv2.INTER_LINEAR)

cv2.imshow( 'Linear Scaled', linear_img )
cv2.waitKey( 0 )

scale_img = cv2.resize(image, (400, 100), interpolation = cv2.INTER_AREA)

cv2.imshow( 'Area Interpolate', scale_img )
cv2.waitKey( 0 )

hflip_img = cv2.flip(image, 1)

cv2.imshow( 'Horizontal Flip', hflip_img )
cv2.waitKey( 0 )

vflip_img = cv2.flip(image, 0)

cv2.imshow( 'Vertical Flip', vflip_img )
cv2.waitKey( 0 )

bothflip_img = cv2.flip(image, 0)

cv2.imshow( 'Horiaontal & Vertical Flip', bothflip_img )
cv2.waitKey( 0 )




