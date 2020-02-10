import cv2
import os
import shutil


def create_folder(user_id, status):
    status_list = [i for i in os.listdir("C:/Users/Pravin/Desktop/k/programming/rpi/fr/DataSet/")]
    while True:
        if status in status_list:
            os.mkdir("C:/Users/Pravin/Desktop/k/programming/rpi/fr/DataSet/" + str(status) + "/" + str(user_id))
            break

        else:
            print("Invalid status! Try again")
            status = input("Access: ")


def remove_id(user_id):
    status_list = [i for i in os.listdir("C:/Users/Pravin/Desktop/k/programming/rpi/fr/DataSet/")]
    for i in status_list:
        for j in os.listdir("C:/Users/Pravin/Desktop/k/programming/rpi/fr/DataSet/" + str(i) + "/"):
            if j == user_id:
                shutil.rmtree("C:/Users/Pravin/Desktop/k/programming/rpi/fr/DataSet/" + str(i) + "/" + str(user_id))
                break


def all_id_list():
    status_list = [i for i in os.listdir("C:/Users/Pravin/Desktop/k/programming/rpi/fr/DataSet/")]
    id_list = []
    for i in status_list:
        for j in os.listdir("C:/Users/Pravin/Desktop/k/programming/rpi/fr/DataSet/" + str(i) + "/"):
            id_list.append(j)
    return id_list


def check_avail_id1(user_id):
    l = all_id_list()
    while user_id in l:
        print("ID is taken,Try Different")
        user_id = input("User_ID:")
    return user_id


def check_avail_id2(user_id):
    l = all_id_list()
    while not user_id in l:
        print("Invalid ID")
        user_id = input("User_ID:")
    return user_id


def draw_box(Image, x, y, w, h):
    WHITE = [255, 255, 255]
    cv2.line(Image, (x, y), (int(x + (w / 5)), y), WHITE, 2)
    cv2.line(Image, (int(x + ((w / 5) * 4)), y), (x + w, y), WHITE, 2)
    cv2.line(Image, (x, y), (x, int(y + (h / 5))), WHITE, 2)
    cv2.line(Image, (x + w, y), (x + w, int(y + (h / 5))), WHITE, 2)
    cv2.line(Image, (x, int(y + (h / 5 * 4))), (x, y + h), WHITE, 2)
    cv2.line(Image, (x, (y + h)), (int(x + (w / 5)), y + h), WHITE, 2)
    cv2.line(Image, (int(x + ((w / 5) * 4)), y + h), (x + w, y + h), WHITE, 2)
    cv2.line(Image, (x + w, int(y + (h / 5 * 4))), (x + w, y + h), WHITE, 2)


def create_face_dateset():
    user_id = input("Enter User_ID:")
    user_id = check_avail_id1(user_id)
    status = input("Access:")
    create_folder(user_id, status)

    cap = cv2.VideoCapture(0)
    face_cas = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')
    c = 1

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cas.detectMultiScale(gray, 1.3, 4)
        for (x, y, w, h) in faces:
            draw_box(frame, x, y, w, h)
            roi = gray[y:y + h, x:x + w]
            cv2.imwrite("DataSet/" + str(status) + "/" + str(user_id) + "/" + str(c) + ".jpg", roi)
            c += 1

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == 27:
            break

        elif c > 100:
            break

    cap.release()
    cv2.destroyAllWindows()


# create_face_dateset()


def update_face_dataset():
    user_id = input("User_ID:")
    user_id = check_avail_id2(user_id)
    remove_id(user_id)

    user_id = input("Enter New User_ID:")
    check_avail_id1(user_id)
    status = input("Access:")
    create_folder(user_id, status)

    cap = cv2.VideoCapture(0)
    face_cas = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')
    c = 1

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cas.detectMultiScale(gray, 1.3, 4)
        for (x, y, w, h) in faces:
            draw_box(frame, x, y, w, h)
            roi = gray[y:y + h, x:x + w]
            cv2.imwrite("DataSet/" + str(status) + "/" + str(user_id) + "/" + str(c) + ".jpg", roi)
            c += 1

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == 27:
            break

        elif c > 100:
            break

    cap.release()
    cv2.destroyAllWindows()


# update_face_dataset()

def fd():
    while True:
        ans = input("task:")
        if ans == 'c':
            create_face_dateset()

        elif ans == 'u':
            update_face_dataset()

        elif ans == 'q':
            break


fd()
