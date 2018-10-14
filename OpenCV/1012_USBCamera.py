#!/usr/bin/env python

import cv2

cam = cv2.VideoCapture(0)
ret, image = cam.read()


cv2.imshow("Image", image)
cv2.waitKey(0)

cam.release()
