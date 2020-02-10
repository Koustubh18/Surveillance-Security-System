import os
import numpy as np
from PIL import Image
import cv2
import pickle


def position_list_():
    data_dir = "C:/Users/Pravin/Desktop/k/programming/rpi/fr_mode1/DataSet"
    return os.listdir(data_dir)


def create_pickle(file, name):
    if file:
        with open("dataframe/" + str(name) + ".pickle", "wb") as f:
            pickle.dump(file, f)


def dict_format(dt):
    dt = {v[0]: [k, v[1]] for k, v in dt.items()}
    return create_pickle(dt, 'labeldata_format')


def train_data():
    face_cas = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    data_dir = "C:/Users/Pravin/Desktop/k/programming/rpi/fr_mode1/DataSet"

    label_number = 1

    label_list = []
    train_list = []
    position_list = position_list_()

    label_dict = {}

    for i in position_list:
        for root, dirs, files in os.walk(data_dir + "/" + str(i)):
            for file in files:
                if file.endswith("png") or file.endswith("jpg"):
                    path = os.path.join(root, file)
                    label = os.path.basename(root)
                    if not label in label_dict:
                        label_dict[label] = [label_number, i]
                        label_number += 1

                    label_value = label_dict[label][0]
                    pil_image = Image.open(path).convert("L")
                    image_array = np.array(pil_image, "uint8")
                    train_list.append(image_array)
                    label_list.append(label_value)

    print(label_dict)
    if label_list:
        create_pickle(label_dict, 'labeldata')
        dict_format(label_dict)
        recognizer.train(train_list, np.array(label_list))
        recognizer.save("trainner/train.yml")


train_data()
