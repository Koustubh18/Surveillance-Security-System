import numpy as np
import cv2
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import os


# difference of  time interval in second
def diff_seconds(l1, l2):
    l = []
    for i in range(0, len(l2)):
        if i % 2 == 0:
            mili = round((3600 * l2[i].hour + 60 * l2[i].minute + l2[i].second + l2[i].microsecond / 1000000) - (3600 * l1[i].hour + 60 * l1[i].minute + l1[i].second + l1[i].microsecond / 1000000), 5)
            l.append(mili)
        else:
            l.append(0)
    return l


# area contours
def contours_area(f, cs, x):
    for c in cs:
        area = cv2.contourArea(c)

        if area > x:
            cv2.drawContours(f, cs, -1, (0, 0, 255))


# to extract data
def extract1_data(df, time_list, date_list):
    print(time_list)
    if time_list:
        try:
            if len(time_list) == 2:
                df = df.append({"Date": date_list[0], "Start": time_list[0], "End": time_list[1]}, ignore_index=True)
                df = df.append({"Date": date_list[1], "Start": time_list[1], "End": time_list[1]}, ignore_index=True)

            elif len(time_list) % 2 != 0:
                for i in range(0, len(time_list) - 1):
                    df = df.append({"Date": date_list[i], "Start": time_list[i], "End": time_list[i + 1]}, ignore_index=True)

            elif len(time_list) % 2 == 0 and len(time_list) != 2:
                for i in range(0, len(time_list) - 1):
                    df = df.append({"Date": date_list[i], "Start": time_list[i], "End": time_list[i + 1]}, ignore_index=True)

        except Exception as e:
            print(e)

        else:
            l1 = df.Start.tolist()
            l2 = df.End.tolist()
            l = diff_seconds(l1, l2)
            df2 = pd.DataFrame(np.array(l), columns=['IV'])
            df = df.join(df2)
            df.index += 1
            print(df)
            df.to_pickle('MData_pickle.pickle')
    else:
        print('no motion detected')


def extract2_data(area_time, area_md):
    df = pd.DataFrame({'Time': area_time, 'Area': area_md})
    df.to_pickle('MData2_pickle.pickle')


# data anlayser1
def Data_Analysis1():
    df = pd.read_pickle('MData_pickle.pickle')
    print(df)
    style.use('fivethirtyeight')

    fig = plt.figure()
    ax1 = plt.subplot2grid((1, 1), (0, 0))

    plt.xticks(rotation=45)
    ax1.grid(True)

    ax1.xaxis.label.set_color('k')
    ax1.yaxis.label.set_color('k')

    ax1.tick_params(axis='x', colors='k')
    ax1.tick_params(axis='y', colors='k')

    ax1.plot(df['Start'], df['IV'], '-', color='b', label='IV', alpha=.8, linewidth=2)
    plt.subplots_adjust(left=0.11, bottom=0.24, right=0.90, top=0.90, wspace=0.2, hspace=0)

    plt.xlabel('Time')
    plt.ylabel('Interval')
    plt.title('MD_Interval')
    plt.legend()
    plt.show()


# data analyser2
def Data_Analysis2():
    df = pd.read_pickle('MData2_pickle.pickle')
    style.use('fivethirtyeight')

    fig = plt.figure()
    ax1 = plt.subplot2grid((1, 1), (0, 0))

    plt.xticks(rotation=45)
    ax1.grid(True)

    ax1.xaxis.label.set_color('k')
    ax1.yaxis.label.set_color('k')

    ax1.tick_params(axis='x', colors='k')
    ax1.tick_params(axis='y', colors='k')
    ax1.fill_between(np.array(df['Time']), np.array(df['Area']), 0, facecolor='c', alpha=0.8)
    plt.subplots_adjust(left=0.11, bottom=0.24, right=0.90, top=0.90, wspace=0.2, hspace=0)

    plt.xlabel('Time')
    plt.ylabel('Area')
    plt.title('MD_Area')
    plt.show()


# to track a motion
def TrackMotion():
    i = 0
    path = "C:/Users/Pravin/Desktop/k/programming/rpi/md/t"
    df = pd.DataFrame(columns=['Date', 'Start', 'End'])
    time_list = []
    date_list = []
    area_md = []
    area_time = []
    motion_list = [0]

    cap = cv2.VideoCapture(0)
    ret, f1 = cap.read()
    ret, f2 = cap.read()

    while ret:
        motion = 0

        d = cv2.absdiff(f1, f2)
        gray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(th, np.ones((3, 3), np.uint8), iterations=3)
        eroded = cv2.erode(dilated, np.ones((3, 3), np.uint8), iterations=3)
        a, cs, b = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # take snaps
        for c in cs:
            if cv2.contourArea(c) > 30000:
                cv2.imwrite(os.path.join(path, f'{i}.jpg'), f1)
                i += 1

        s = 0
        for c in cs:
            s = s + cv2.contourArea(c)
            area_time.append(datetime.now().time())
            area_md.append(s)

        if cs:
            cv2.drawContours(f1, cs, -1, (0, 0, 255), 1)
            # contours_area(f1, cs, 500)
            motion = 1

        motion_list.append(motion)
        motion_list = motion_list[-2:]

        if motion_list[-1] == 1 and motion_list[-2] == 0:
            date_list.append(datetime.now().date())
            time_list.append(datetime.now().time())

        if motion_list[-1] == 0 and motion_list[-2] == 1:
            date_list.append(datetime.now().date())
            time_list.append(datetime.now().time())

        cv2.imshow('d', f1)

        if cv2.waitKey(1) == 27:
            if motion == 1:
                time_list.append(datetime.now().time())
                date_list.append(datetime.now().date())

            break

        f1 = f2
        ret, f2 = cap.read()

    cap.release()
    cv2.destroyAllWindows()

    extract1_data(df, time_list, date_list)
    extract2_data(area_time, area_md)
    Data_Analysis1()
    Data_Analysis2()


TrackMotion()

def remove_file():
    l=os.listdir("C:/Users/Pravin/Desktop/k/programming/rpi/md")
    if 'MData_pickle.pickle' in l and 'MData2_pickle.pickle' in l:
        os.remove("MData_pickle.pickle")
        os.remove("MData2_pickle.pickle")


remove_file()
