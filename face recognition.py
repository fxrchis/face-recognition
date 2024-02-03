import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img
    
    for (x,y,w,h)  in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),2)
    return img

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = detect_face(frame)

    cv2.imshow("Face", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()