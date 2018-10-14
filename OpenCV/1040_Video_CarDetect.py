#!/usr/bin/env python

import time
import cv2
import numpy as np

cars_detect = cv2.CascadeClassifier('haarcascade_car.xml')

capture = cv2.VideoCapture('cars.avi')

while True:
	time.sleep(.05)
	
	ret, frame = capture.read()
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)	
	
	cars_detected = cars_detect.detectMultiScale(gray, 1.25, 2)
	for (x,y,w,h) in cars_detected:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)	
        
	cv2.imshow('Detecting Cars', frame)
	
	key = cv2.waitKey(1) & 0xFF
 

	if key == 27:
		break
        	
capture.release()
