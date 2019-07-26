#!/usr/bin/env python

import cv2
import numpy as np

image = cv2.imread('image_01.jpg')

cv2.imshow('Original', image)
cv2.waitKey(0)

gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Gray', gray_img)
cv2.waitKey(0)

ret, threshold_img = cv2.threshold(gray_img, 27, 255, cv2.THRESH_BINARY)

cv2.imshow('Binary Thresholding', threshold_img)
cv2.waitKey(0)
