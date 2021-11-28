import os
import cv2

ctr = 0
# import face detection cascade
face_cascade = cv2.CascadeClassifier(
    '/Users/aishaandatt/Downloads/IIT_Temp/haarcascade_frontalface_default.xml')
# create capture object
cap = cv2.VideoCapture(0)


while True:
    # capture frame-by-frame
    ret, img = cap.read()
    # convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    # for each face draw a rectangle around and copy the face

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_color = img[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]
        # SAVE THE DETECTED FACES LOCALLY
        roi_gray = cv2.resize(roi_color, (250, 250))
        cv2.imwrite('faces'+'/'+str(ctr)+'.jpg', roi_gray)
        break
    # display the resulting frame
    cv2.imshow('frame', img)

    key = cv2.waitKey(1)
    if key == 32:  # if space key is pressed
        break
