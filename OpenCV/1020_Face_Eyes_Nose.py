#!/usr/bin/env python

from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

time.sleep(0.1)

face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if face_detect.empty():
	raise IOError('Unable to haarcascade_frontalface_alt.xml file')

eyes_detect = cv2.CascadeClassifier('haarcascade_eye.xml')
if eyes_detect.empty():
	raise IOError('Unable to load haarcascade_eye.xml file')

noise_detect = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
if noise_detect.empty():
	raise IOError('Unable to load haarcascade_mcs_nose.xml file')


for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	image = frame.array
	
	resize_frame = cv2.resize(image, None, fx=0.3, fy=0.3, 
            interpolation=cv2.INTER_LINEAR)	
	
	gray = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2GRAY)	
	
	face_detection = face_detect.detectMultiScale(gray, 1.05, 5)
	for (x,y,w,h) in face_detection:
		print "Frame", x, y, w, h
		cv2.rectangle(resize_frame, (x,y), (x+w,y+h), (0,0,255), 5)  	
 
	# show the frame
	cv2.imshow("Frame", resize_frame)
	key = cv2.waitKey(1) & 0xFF
 
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)

	if key == 27:
		break
        	
  
