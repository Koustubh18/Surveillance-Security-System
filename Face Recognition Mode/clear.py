import os
import shutil


def remove():
    data_dir1 = "C:/Users/Pravin/Desktop/k/programming/rpi/fr_mode1/DataSet/"
    data_dir2 = "C:/Users/Pravin/Desktop/k/programming/rpi/fr_mode1/trainner/"
    data_dir3 = "C:/Users/Pravin/Desktop/k/programming/rpi/fr_mode1/dataframe/"

    for f in os.listdir(data_dir1):
        if f:
            shutil.rmtree(data_dir1 + str(f))

    os.remove(data_dir2 + "train.yml")
    os.remove(data_dir3 + "labeldata.pickle")
    os.remove(data_dir3 + "labeldata_format.pickle")


remove()
