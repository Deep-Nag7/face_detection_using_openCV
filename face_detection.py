import numpy as np
import cv2
import face_recognition
import os
from datetime import datetime
from datetime import date

def find_encoding(images):
    encodelist = []
    for i in images:
        img = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist


def webcam():
    path = './FaceDetection'
    images = []
    classNames = []
    mylist = os.listdir(path)
    print(mylist)
    for i in mylist:
        curImg = cv2.imread(f'{path}/{i}')
        images.append(curImg)
        classNames.append(os.path.splitext(i)[0])
    print(classNames)

    encodeListKnown = find_encoding(images)
    print("The Encoding is done")

    cap = cv2.VideoCapture(0)  # for vdo capture
    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # This is for making the image size small
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        # in the webcam we might have multiple faces for that we have to find the location of the faces
        faceCurFrane = face_recognition.face_locations(imgS)  # find the all the location in our small image
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrane)

        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrane):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name)
                y1, x2, y2, x1 = faceLoc
                # 0, 3, 1, 2  ->> Locations
                x1 = x1 * 4
                x2 = x2 * 4
                y1 = y1 * 4
                y2 = y2 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # rectangle for image
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)  # rectangle for text
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imshow('webcam', img)
        cv2.waitKey(1)

    # Allte cooding part is done........The design part is remaining.
    # Allte cooding part is done........The design part is remaining.

def withimage():
    path = './FaceDetection'
    images = []
    classNames = []
    mylist = os.listdir(path)
    print(mylist)
    for i in mylist:
        curImg = cv2.imread(f'{path}/{i}')
        images.append(curImg)
        classNames.append(os.path.splitext(i)[0])
    print(classNames)
    encodeListKnown = find_encoding(images)
    print("The Encoding is done")
    Ritu = face_recognition.load_image_file('./test.jfif') #loading the image
    Ritu = cv2.cvtColor(Ritu,cv2.COLOR_BGR2RGB) #converting the image colored to rgb
    faceLoc = face_recognition.face_locations(Ritu)[0]
    encodeRitu = face_recognition.face_encodings(Ritu)[0]

    matches = face_recognition.compare_faces(encodeListKnown, encodeRitu)
    faceDis = face_recognition.face_distance(encodeListKnown, encodeRitu)
    matchIndex = np.argmin(faceDis)
    if matches[matchIndex]:
        name = classNames[matchIndex].upper()
        print(name)
        y1, x2, y2, x1 = faceLoc
        cv2.rectangle(Ritu, (x1, y1), (x2, y2), (0, 255, 0), 2) 
        cv2.rectangle(Ritu, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)  # rectangle for text
        cv2.putText(Ritu, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Rituparna Singh',Ritu)
        cv2.waitKey(0)





if __name__ == '__main__':
    print(" 1 for image base face detection")
    print(" 2 for webcam base face detection")
    ch=int(input("Enter your choice : "))
    if ch==1:
        withimage()
    else:
        webcam()