import numpy
import sys
import os
from tkinter import font
import uuid
import cv2
import numpy as np
from PIL import Image
import face_recognition
from pymongo import MongoClient
import tkinter as tk
client = MongoClient('localhost', 27017)
db = client['faces-test2']
collection = db['faces3']
window = tk.Tk()
window.title('Face Recognition')
window.geometry('1400x720')
window.configure(background='#4c4c4c')
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)
message = tk.Label(text='Face Recognition',
                   font=('roboto', 30, 'bold'))
message.place(x=600, y=20)
lbl1 = tk.Label(window, text="Enter Name", width=20, fg="black",
                bg="#DEBA9D", height=1, font=('roboto', 15, ' bold '))
lbl1.place(x=400, y=100)
txt1 = tk.Entry(window, width=20,
                font=('roboto', 15, ' bold '))
txt1.place(x=700, y=100)
lbl2 = tk.Label(window, text="Enter ID (Not if Reg)", width=20, fg="black",
                bg="#DEBA9D", height=1, font=('roboto', 15, ' bold '))
lbl2.place(x=400, y=200)
txt2 = tk.Entry(window, width=20,
                font=('roboto', 15, ' bold '))
txt2.place(x=700, y=200)
lbl3 = tk.Label(window, text="Enter Path (Webcam)", width=20, fg="black",
                bg="#DEBA9D", height=1, font=('roboto', 15, ' bold '))
lbl3.place(x=400, y=300)
txt3 = tk.Entry(window, width=20,
                font=('roboto', 15, ' bold '))
txt3.place(x=700, y=300)
lbl3 = tk.Label(window, text="Notification : ", width=20,
                height=1, font=('roboto', 15, ' bold underline '))
lbl3.place(x=400, y=400)
message = tk.Label(window, text="", width=20,
                   height=1, activebackground="#DEBA9D", font=('roboto', 15, ' bold '))
message.place(x=700, y=400)


def clear():
    txt2.delete(0, 'end')
    res = ""
    message.configure(text=res)


def clear2():
    txt3.delete(0, 'end')
    res = ""
    message.configure(text=res)


ctr = 0
# import face detection cascade
face_cascade = cv2.CascadeClassifier(
    '/Users/aishaandatt/Downloads/IIT_Temp/haarcascade_frontalface_default.xml')


def Register():
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    cv2.imwrite('faces'+'/'+str(ctr)+'.jpg', img)
    path_reg = '/Users/aishaandatt/Downloads/IIT_Temp/faces/0.jpg'
    cv2.imshow('frame', img)
    cap.release()
    cv2.destroyAllWindows()
    name = (str)(txt1.get())

    image = face_recognition.load_image_file('{}'.format(path_reg))
    if(collection.find().count() != 0):
        k = verification2()
        if(k == True):
            message.configure(text='Already Registered')
        else:
            pil_image = Image.fromarray(image)

            path_store = '/Users/aishaandatt/Downloads/IIT_Temp/faces-full/{}.jpg'.format(
                name)
            pil_image.save(path_store)
            ID = uuid.uuid4()
            dict1 = [{'name': name, 'ID': (str)(ID), 'enc': (face_recognition.face_encodings(image)[0]).tolist(),
                      'image': path_store}]
            message.configure(text='{} is Registered'.format(name))
            collection.insert(dict1)


def verification2():
    ctr = 0
    cap = cv2.VideoCapture(0)
    # capture frame-by-frame
    ret, img = cap.read()
    cv2.imwrite('faces'+'/'+str(ctr)+'.jpg', img)
    path_img = '/Users/aishaandatt/Downloads/IIT_Temp/faces/0.jpg'
    # display the resulting frame
    cv2.imshow('frame', img)
    cap.release()
    cv2.destroyAllWindows()
    namepath = 'face.jpg'
    from PIL import Image
    webcam = face_recognition.load_image_file('{}'.format(path_img))
    my_face_encoding = face_recognition.face_encodings(webcam)[0]
    # db load and encode
    ctr = 0
    for r in db['faces3'].find({}):
        print(r['image'])
        ctr = ctr+1
        results = face_recognition.compare_faces(
            [my_face_encoding], np.array([r['enc']])[0])
        if(results[0] == True):
            message.configure(text=r['name'])
            print(r['name'])
            break
        else:
            message.configure(text='Not Registered')
    print(ctr)

    return results[0]


# verification2('/Users/aishaandatt/Downloads/IIT_Temp/Penélope-Cruz.jpeg')
clearButton2 = tk.Button(window, text="Clear", command=clear, fg="black", bg="#DEBA9D",
                         width=10, height=1, activebackground="#DEBA9D", font=('roboto', 15, ' bold '))
clearButton2.place(x=950, y=200)
clearButton3 = tk.Button(window, text="Clear", command=clear2, fg="black", bg="#DEBA9D",
                         width=10, height=1, activebackground="#DEBA9D", font=('roboto', 15, ' bold '))
clearButton3.place(x=950, y=300)
takeImg = tk.Button(window, text="Verify", command=verification2, fg="black", bg="#DEBA9D",
                    width=10, height=1, activebackground="#DEBA9D", font=('roboto', 15, ' bold '))
takeImg.place(x=200, y=600)
takeImg = tk.Button(window, text="Register", command=Register, fg="black", bg="#DEBA9D",
                    width=10, height=1, activebackground="#DEBA9D", font=('roboto', 15, ' bold '))
takeImg.place(x=500, y=600)
# trainImg = tk.Button(window, text="Train", command=TrainImages, fg="black",
#                      bg="#DEBA9D", width=20, height=3, activebackground="black", font=('roboto', 15, ' bold '))
# trainImg.place(x=500, y=500)
# trackImg = tk.Button(window, text="Authenticate", command=TrackImages, fg="black",
#                      bg="#DEBA9D", width=20, height=3, activebackground="black", font=('roboto', 15, ' bold '))
# trackImg.place(x=800, y=500)
quitWindow = tk.Button(window, text="Quit", command=window.destroy, fg="black", bg="#DEBA9D",
                       width=10, height=1, activebackground="#DEBA9D", font=('roboto', 15, ' bold '))
quitWindow.place(x=1100, y=600)
window.mainloop()
