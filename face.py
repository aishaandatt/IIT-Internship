import os
from tkinter import font
import uuid
import cv2
from PIL import Image
import face_recognition
from pymongo import MongoClient
import tkinter as tk
client = MongoClient('localhost', 27017)
db = client['faces']
collection = db['faces1']
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


# def verification():
#     IDs = (str)(txt2.get())
#     path_img = (str)(txt3.get())
#     namepath = 'face.jpg'
#     # webcam_path='/Users/aishaandatt/Downloads/IIT_Temp/abhay3.jpeg'
#     from PIL import Image
#     webcam = face_recognition.load_image_file('{}'.format(path_img))
#     face_locations = face_recognition.face_locations(
#         webcam, number_of_times_to_upsample=0, model="cnn")
#     for face_location in face_locations:
#         top, right, bottom, left = face_location
#         face_image = webcam[top:bottom, left:right]
#         pil_image = Image.fromarray(face_image)
#         k = pil_image.resize((250, 250))

#         k.save('/Users/aishaandatt/Downloads/IIT_Temp/webcam-imgs/{}'.format(namepath))
#     # webcam load and encode
#     webcam_load = face_recognition.load_image_file(
#         '/Users/aishaandatt/Downloads/IIT_Temp/webcam-imgs/{}'.format(namepath))
#     my_face_encoding = face_recognition.face_encodings(webcam_load)[0]
#     x = (collection.find_one({'ID': IDs}))
#     # db load and encode
#     db_img = face_recognition.load_image_file(x['image'])
#     db_enc = face_recognition.face_encodings(db_img)[0]
#     results = face_recognition.compare_faces([my_face_encoding], db_enc)
#     message.configure(text=results[0])
#     return results[0]


#print(verification('/Users/aishaandatt/Downloads/IIT_Temp/abhay3.jpeg', 2))


def Register():
    # Load the jpg file into a numpy array
    path_reg = (str)(txt3.get())
    name = (str)(txt1.get())
    image = face_recognition.load_image_file('{}'.format(path_reg))
    k = verification2()
    if(k == True):
        message.configure(text='Already Registered')
    else:
        face_locations = face_recognition.face_locations(
            image, number_of_times_to_upsample=0, model="cnn")
        print("Found {} face(s) in this photograph.".format(len(face_locations)))
        for face_location in face_locations:
            # Print the location of each face in this image
            top, right, bottom, left = face_location
            print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(
                top, left, bottom, right))

            # You can access the actual face itself like this:
            face_image = image[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)
            # pil_image.show()
            k = pil_image.resize((250, 250))
            Name = name
            path_store = '/Users/aishaandatt/Downloads/IIT_Temp/prototype_faces/{}.jpg'.format(
                Name)
            k.save(path_store)
            ID = uuid.uuid4()
            dict1 = [{'name': Name, 'ID': (str)(ID),
                      'image': path_store}]
            message.configure(text='{} is Registered'.format(Name))
            collection.insert(dict1)


def verification2():
    namepath = 'face.jpg'
    # webcam_path='/Users/aishaandatt/Downloads/IIT_Temp/abhay3.jpeg'
    from PIL import Image
    path_img = (str)(txt3.get())
    webcam = face_recognition.load_image_file('{}'.format(path_img))
    face_locations = face_recognition.face_locations(
        webcam, number_of_times_to_upsample=0, model="cnn")
    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_image = webcam[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        k = pil_image.resize((250, 250))

        k.save('/Users/aishaandatt/Downloads/IIT_Temp/webcam-imgs/{}'.format(namepath))
    # webcam load and encode
    webcam_load = face_recognition.load_image_file(
        '/Users/aishaandatt/Downloads/IIT_Temp/webcam-imgs/{}'.format(namepath))
    my_face_encoding = face_recognition.face_encodings(webcam_load)[0]
    # db load and encode
    ctr = 0
    for r in db['faces1'].find({}):
        print(r['image'])
        ctr = ctr+1
        db_img = face_recognition.load_image_file('{}'.format(r['image']))
        db_enc = face_recognition.face_encodings(db_img)[0]
        # print(ctr, mydoc)
        results = face_recognition.compare_faces(
            [my_face_encoding], db_enc)
        if(results[0] == True):
            message.configure(text=r['name'])
            break
        else:
            message.configure(text='Not Registered')
    # if(results[0] == False):
    #     Register()
    print(ctr)

    return results[0]


# verification2('/Users/aishaandatt/Downloads/IIT_Temp/Penélope-Cruz.jpeg')
clearButton2 = tk.Button(window, text="Clear", command=clear, fg="black", bg="#DEBA9D",
                         width=10, height=1, activebackground="#DEBA9D", font=('roboto', 15, ' bold '))
clearButton2.place(x=950, y=200)
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
