#!/usr/bin/env python

import cv2
import numpy as np

face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if face_detect.empty():
	raise IOError('Unable to haarcascade_frontalface_alt.xml file')

eyes_detect = cv2.CascadeClassifier('haarcascade_eye.xml')
if eyes_detect.empty():
	raise IOError('Unable to load haarcascade_eye.xml file')

noise_detect = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
if noise_detect.empty():
	raise IOError('Unable to load haarcascade_mcs_nose.xml file')

capture = cv2.VideoCapture(0)

while True:
	# Start capturing frames
	ret, image = capture.read()
	
	resize_frame = cv2.resize(image, None, fx=0.4, fy=0.4, 
            interpolation=cv2.INTER_LINEAR)	
	
	gray = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2GRAY)	
	
	face_detection = face_detect.detectMultiScale(gray, 1.2, 5)
	for (x,y,w,h) in face_detection:
		cv2.rectangle(resize_frame, (x,y), (x+w,y+h), (0,0,255), 4)  	

		gray_roi = gray[y:y+h, x:x+w]
		color_roi = resize_frame[y:y+h, x:x+w]
		   
		eye_detector = eyes_detect.detectMultiScale(gray_roi)
			
		for (eye_x, eye_y, eye_w, eye_h) in eye_detector:
			cv2.rectangle(color_roi,(eye_x,eye_y),(eye_x + eye_w, eye_y + eye_h),(255,0,0),2)
						   
		nose_detector = noise_detect.detectMultiScale(gray_roi, 1.3, 5)

		for (nose_x, nose_y, nose_w, nose_h) in nose_detector:
			cv2.rectangle(color_roi, (nose_x, nose_y), (nose_x + nose_w, nose_y + nose_h), (0,255,0), 2) 

	cv2.imshow("Frame", resize_frame)
	key = cv2.waitKey(1) & 0xFF
 

	if key == 27:
		break
        	
capture.release()
