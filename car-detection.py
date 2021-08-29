import cv2
import time
import numpy as np

car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')
cap=cv2.VideoCapture('video.mp4')

while True:
    ret, frames = cap.read()
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    for(x,y,w,h) in cars:
        cv2.rectangle(frames, (x,y), (x+w,y+h), (0,0,225), 2)

    cv2.imshow('video2', frames)

    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()
