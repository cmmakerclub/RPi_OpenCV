import numpy as np

import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

capture = cv2.VideoCapture(0)
face_count = 0

while True:
    ret, capturing = capture.read()
    capturing = cv2.resize(capturing, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Frame", capturing)
    cv2.waitKey(1)

capture.release()
cv2.destroyAllWindows()
