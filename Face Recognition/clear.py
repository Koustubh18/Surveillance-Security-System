import os
import shutil


def remove():
    data_dir1 = "C:/Users/Pravin/Desktop/k/programming/rpi/fr/DataSet/"
    data_dir2 = "C:/Users/Pravin/Desktop/k/programming/rpi/fr/trainner/"
    data_dir3 = "C:/Users/Pravin/Desktop/k/programming/rpi/fr/dataframe/"

    for f in os.listdir(data_dir1):
        for i in os.listdir("C:/Users/Pravin/Desktop/k/programming/rpi/fr/DataSet/" + str(f)):
            shutil.rmtree("C:/Users/Pravin/Desktop/k/programming/rpi/fr/DataSet/" + str(f) + "/" + str(i))

    os.remove(data_dir2 + "train.yml")
    os.remove(data_dir3 + "labeldata.pickle")
    os.remove(data_dir3 + "labeldata_format.pickle")


remove()
