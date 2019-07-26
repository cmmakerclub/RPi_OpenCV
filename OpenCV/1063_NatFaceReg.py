import numpy as np

import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

face_count = 0


def draw_red_rect(image, x, y, w, h):
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 4)


def crop_image(image, x, y, w, h):
    cropped = image[y:y + h, x:x + w]
    cropped = cv2.resize(cropped, (250, 250))
    return cropped


def face_detector(image, size=0.5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    is_face_detected = face_cascade.detectMultiScale(gray, 1.2, 5)

    if is_face_detected is ():
        return None, image, None, gray

    for (x, y, w, h) in is_face_detected:
        draw_red_rect(image, x, y, w, h)
        cropped = crop_image(image, x, y, w, h)
        cv2.putText(image, str(face_count), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)

    return True, image, cropped, gray


capture = cv2.VideoCapture(0)

while True:
    ret, capturing = capture.read()
    capturing = cv2.resize(capturing, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    face_detected, image, cropped, gray = face_detector(capturing)
    cv2.putText(image, str(face_count), (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    if face_detected != None:
        face_count = face_count + 1
        pass

    cv2.imshow("Frame", gray)
    cv2.imshow("Capture", image)
    # if cropped != False:
    #     cv2.imshow("face", cropped)
    cv2.waitKey(1)

capture.release()
cv2.destroyAllWindows()
