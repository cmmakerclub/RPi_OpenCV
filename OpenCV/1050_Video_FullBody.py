#!/usr/bin/env python

import time
import cv2
import numpy as np

fullbody_detect = cv2.CascadeClassifier('haarcascade_fullbody.xml')

capture = cv2.VideoCapture('people.avi')

while True:
	time.sleep(.05)
	
	ret, frame = capture.read()
	
	frame = cv2.resize(frame, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)	
	
	body_detected = fullbody_detect.detectMultiScale(gray, 1.15, 2)
	for (x,y,w,h) in body_detected:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)	
        
	cv2.imshow('Detecting body', frame)
	
	key = cv2.waitKey(1) & 0xFF
 

	if key == 27:
		break
        	
capture.release()
