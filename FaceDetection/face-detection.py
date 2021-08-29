import cv2 
import time
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
originalImage = cv2.imread('couple.JPG')

grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    
faces = face_cascade.detectMultiScale(grayImage, 1.3, 5) 

for (x,y,w,h) in faces:
    cv2.rectangle(originalImage,(x,y),(x+w,y+h),(255,255,0),2) 
    roi_gray = grayImage[y:y+h, x:x+w] 
    roi_color = originalImage[y:y+h, x:x+w] 

cv2.imshow('Face Detection using Haar Cascade',originalImage) 
cv2.waitKey(50)

time.sleep(10) 
cv2.destroyAllWindows() 
