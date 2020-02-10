import cv2
import os
import shutil


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


def create_position_folder(position):
    os.mkdir("C:/Users/Pravin/Desktop/k/programming/rpi/fr_mode1/DataSet/" + str(position))


def remove_position(position):
    if position in os.listdir("C:/Users/Pravin/Desktop/k/programming/rpi/fr_mode1/DataSet/"):
        shutil.rmtree("C:/Users/Pravin/Desktop/k/programming/rpi/fr_mode1/DataSet/" + str(position))


def remove_id(id):
    position_list = [i for i in os.listdir("C:/Users/Pravin/Desktop/k/programming/rpi/fr_mode1/DataSet/")]
    for i in position_list:
        for j in os.listdir("C:/Users/Pravin/Desktop/k/programming/rpi/fr_mode1/DataSet/" + str(i) + "/"):
            if id == j:
                shutil.rmtree("C:/Users/Pravin/Desktop/k/programming/rpi/fr_mode1/DataSet/" + str(i) + "/" + str(id))
                break


def create_id_folder(position, id):
    position_list = os.listdir("C:/Users/Pravin/Desktop/k/programming/rpi/fr_mode1/DataSet/")

    if position in position_list:
        os.mkdir("C:/Users/Pravin/Desktop/k/programming/rpi/fr_mode1/DataSet/" + str(position) + "/" + str(id))

    else:
        create_position_folder(position)
        os.mkdir("C:/Users/Pravin/Desktop/k/programming/rpi/fr_mode1/DataSet/" + str(position) + "/" + str(id))


def all_id_name_list():
    position_list = [i for i in os.listdir("C:/Users/Pravin/Desktop/k/programming/rpi/fr_mode1/DataSet/")]
    id_list = []
    for i in position_list:
        for j in os.listdir("C:/Users/Pravin/Desktop/k/programming/rpi/fr_mode1/DataSet/" + str(i) + "/"):
            id_list.append(j)

    return id_list


def check_avail_id1(id):
    id_list = all_id_name_list()

    while id in id_list:
        print("ID is taken,Try Different")
        id = input("User_ID:")
    return id


def check_avail_id2(id):
    id_list = all_id_name_list()
    while not id in id_list:
        print("Invalid ID")
        id = input("User_ID:")
    return id


def create_face_dataset():
    id = input("Enter User ID:")
    id = check_avail_id1(id)
    position = input("Category:")
    create_id_folder(position, id)

    cap = cv2.VideoCapture(0)
    face_cas = cv2.CascadeClassifier("haar/haarcascade_frontalface_alt2.xml")
    c = 1
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cas.detectMultiScale(gray, 1.3, 4)
        for (x, y, w, h) in faces:
            draw_box(frame, x, y, w, h)
            roi = gray[y:y + h, x:x + w]
            cv2.imwrite("DataSet/" + str(position) + "/" + str(id) + "/" + str(c) + ".jpg", roi)
            c += 1

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == 27:
            break

        elif c > 100:
            break

    cap.release()
    cv2.destroyAllWindows()


# create_face_dataset()


def update_face_dataset():
    id = input("Enter User ID:")
    id = check_avail_id2(id)
    remove_id(id)
    id = input("Enter New User ID:")
    id = check_avail_id1(id)
    position = input("Category:")
    create_id_folder(position, id)

    cap = cv2.VideoCapture(0)
    face_cas = cv2.CascadeClassifier("haar/haarcascade_frontalface_alt2.xml")
    c = 1
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cas.detectMultiScale(gray, 1.3, 4)
        for (x, y, w, h) in faces:
            draw_box(frame, x, y, w, h)
            roi = gray[y:y + h, x:x + w]
            cv2.imwrite("dataset/" + str(position) + "/" + str(id) + "/" + str(c) + ".jpg", roi)
            c += 1

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == 27:
            break

        elif c > 100:
            break

    cap.release()
    cv2.destroyAllWindows()

# update_face_dataset()


def final_decision():
    while True:
        ans = input("task:")
        if ans == 'c':
            create_face_dataset()

        elif ans == 'u':
            update_face_dataset()

        else:
            break


final_decision()
