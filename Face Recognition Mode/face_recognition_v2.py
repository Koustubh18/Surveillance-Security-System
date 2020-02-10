import cv2
import numpy as np
import pickle
import os


def position_list_():
        data_dir = "C:/Users/Pravin/Desktop/k/programming/rpi/fr_mode1/DataSet"
        return os.listdir(data_dir)


def draw_box(Image, x, y, w, h, c):
        cv2.line(Image, (x, y), (int(x + (w / 5)), y), c, 2)
        cv2.line(Image, (int(x + ((w / 5) * 4)), y), (x + w, y), c, 2)
        cv2.line(Image, (x, y), (x, int(y + (h / 5))), c, 2)
        cv2.line(Image, (x + w, y), (x + w, int(y + (h / 5))), c, 2)
        cv2.line(Image, (x, int(y + (h / 5 * 4))), (x, y + h), c, 2)
        cv2.line(Image, (x, (y + h)), (int(x + (w / 5)), y + h), c, 2)
        cv2.line(Image, (int(x + ((w / 5) * 4)), y + h), (x + w, y + h), c, 2)
        cv2.line(Image, (x + w, int(y + (h / 5 * 4))), (x + w, y + h), c, 2)


def select_color():
        i = 0
        pos_col = {}
        pos_list = position_list_()
        col_list = [[0, 255, 255], [255, 255, 255], [255, 0, 0], [0, 255, 0], [255, 0, 255]]
        for k in pos_list:
                pos_col[k] = col_list[i]
                i += 1
        return pos_col


def read_pickle():
        if os.listdir("dataframe/"):
                labels = {}
                with open("dataframe/labeldata_format.pickle", "rb") as f:
                        labels = pickle.load(f)
                return labels


def face_recog():
        cap = cv2.VideoCapture(0)

        face_cas = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("trainner/train.yml")

        label_data = read_pickle()
        select_col = select_color()
        print(select_col)
        while True:
                ret, fr = cap.read()
                gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                faces = face_cas.detectMultiScale(gray, 1.3, 4)
                for (x, y, w, h) in faces:
                        roi = gray[y:y + h, x:x + w]
                        id, conf = recognizer.predict(roi)
                        if id:
                                print(label_data[id][0], label_data[id][1])
                                draw_box(fr, x, y, w, h, select_col[label_data[id][1]])

                        else:
                                print("not recognized")
                                draw_box(fr, x, y, w, h, [0, 0, 255])

                cv2.imshow('org', fr)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        cap.release()
        cv2.destroyAllWindows()


def face_detect():
        cap = cv2.VideoCapture(0)
        face_cas = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')
        glass_cas = cv2.CascadeClassifier('haar/haarcascade_eye.xml')

        while True:
                ret, fr = cap.read()
                gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                faces = face_cas.detectMultiScale(gray, 1.3, 4)
                for (x, y, w, h) in faces:
                        draw_box(fr, x, y, w, h, 'R')

                cv2.imshow('org', fr)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        cap.release()
        cv2.destroyAllWindows()


def final_decision():
        if os.listdir("trainner/"):
                face_recog()

        else:
                print("You Didnt Created Any Face ID!")
                face_detect()


final_decision()
