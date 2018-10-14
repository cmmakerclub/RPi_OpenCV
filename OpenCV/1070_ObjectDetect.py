#!/usr/bin/env python

import cv2
import numpy as np

def ORB(input_image, stored_image):
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    orb_detector = cv2.ORB(1600, 1.3)

    (keypoints_1, descriptor_1) = orb_detector.detectAndCompute(gray, None)

    (keypoints_2, descriptor_2) = orb_detector.detectAndCompute(stored_image, None)

    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches_found =  brute_force.match(descriptor_1,descriptor_2)

    matches_found = sorted(matches_found, key=lambda val: val.distance)

    return len(matches_found)

capture = cv2.VideoCapture(0)

stored_image = cv2.imread('raspberry_pi.jpg', 0) 

while True:
	ret, capturing = capture.read()
	
	capturing = cv2.resize(capturing, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    
	frame_height, frame_width = capturing.shape[:2]

	x1_top_left = frame_width / 3
	y1_top_left = (frame_height / 2) + (frame_height / 4)
	x2_bottom_right = (frame_width / 3) * 2
	y2_bottom_right = (frame_height / 2) - (frame_height / 4)
    
	cv2.rectangle(capturing, (x1_top_left,y1_top_left), (x2_bottom_right,y2_bottom_right), (0,0,255), 4)
    
	cropped_box = capturing[y2_bottom_right:y1_top_left , x1_top_left:x2_bottom_right]

	capturing = cv2.flip(capturing,1)
    
	matches_found = ORB(cropped_box, stored_image)
    
	string = "Matches = " + str(matches_found)
	cv2.putText(capturing, string, (30,220), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
    
	set_threshold = 400
    
	if matches_found > set_threshold:
		cv2.rectangle(capturing, (x1_top_left,y1_top_left), (x2_bottom_right,y2_bottom_right), (0,255,0), 4)
		
		cv2.putText(capturing,'Object Detected',(200,50), cv2.FONT_HERSHEY_COMPLEX, 1 ,(0,255,0), 2)
    
	cv2.imshow('Real-time Object Detection', capturing)
    
	c = cv2.waitKey(1)
	if c == 27:
		break

capture.release()

cv2.destroyAllWindows()
