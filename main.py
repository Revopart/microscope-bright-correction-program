from tkinter import *
import win32gui
import pyautogui
import pyscreenshot as ImageGrab
from sklearn.cluster import KMeans
import cv2
import numpy as np

def callback(hwnd, lparam):
    wind_list.append(hwnd)


def visualize_colors(cluster, centroids):
    # Get the number of different clusters, create histogram, and normalize
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins=labels)
    hist = hist.astype("float")
    hist /= hist.sum()

    # Create frequency rect and iterate through each cluster's color and percentage
    rect = np.zeros((50, 300, 3), dtype=np.uint8)
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
    start = 0
    sum = 0
    step = 0
    for (percent, color) in colors:
        print(color, "{:0.2f}%".format(percent * 100))
        print(step)
        if step < 4:
            sum = sum + percent * 100
            print(sum)
        end = start + (percent * 300)
        step = step + 1
        cv2.rectangle(rect, (int(start), 0), (int(end), 50),
                      color.astype("uint8").tolist(), -1)
        start = end
    if int(sum) <= 10:
        dot = True
    else:
        dot = False
    return rect, dot


def get_cube_root(x):
    if x < 0:
        x = abs(x)
        cube_root = x ** (1 / 3) * (-1)
    else:
        cube_root = x ** (1 / 3)
    return cube_root

def upscale():
    wind_list = []

    def callback(hwnd, lparam):
        wind_list.append(hwnd)

    hwndMain = win32gui.FindWindow(None, "ScopePhoto")
    win32gui.EnumChildWindows(hwndMain, callback, None)
    rect = win32gui.GetWindowRect(wind_list[2])
    x1 = rect[0]
    x2 = rect[2]
    y1 = rect[1]
    y2 = rect[3]
    w = rect[2] - rect[0]
    h = rect[3] - rect[1]
    if w > 50 and h > 50:
        im = ImageGrab.grab(bbox=(x1, y1, x2, y2))  # X1,Y1,X2,Y2
        img = pyautogui.screenshot(region=(x1, y1, w, h))
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        reshape = image.reshape((image.shape[0] * image.shape[1], 3))

        cluster = KMeans(n_clusters=10).fit(reshape)
        visualize, dot = visualize_colors(cluster, cluster.cluster_centers_)
        visualize = cv2.cvtColor(visualize, cv2.COLOR_RGB2BGR)
        cv2.imshow('visualize', visualize)
        cv2.waitKey()

        cv2.imshow("img", img)
        print(dot)

        if dot == True:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            sum = 0
            w = l.shape[0]
            h = l.shape[1]
            for i in range(0, w):
                for j in range(0, h):
                    sum = sum + l[i][j]
            average = int(sum / (w * h))
            print(average)
            delta = 5
            t = average - delta
            part = int(2 * delta ** 3 / 255)
            for i in range(0, w):
                for j in range(0, h):
                    l[i][j] = t + int(get_cube_root(delta ** 3 - (part * l[i][j] + 1)))
            print(t + int(-delta + (part * 0) ** (1. / 3)))

            kernel = np.ones((1, 1), np.uint8)
            temp = cv2.erode(l, kernel, iterations=1)
            l = cv2.dilate(temp, kernel, iterations=1)

            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)

            limg = cv2.merge((cl, a, b))

            final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            cv2.imshow('final', final)
            cv2.waitKey()
        else:

            cv2.imshow("img", img)


            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

            l, a, b = cv2.split(lab)


            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)

            limg = cv2.merge((cl, a, b))

            final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            cv2.imshow('final', final)
            cv2.waitKey()

def exit():
    root.destroy()

root = Tk()

wind_list = []

root.overrideredirect(True)
root.lift()
root.wm_attributes("-topmost", True)
root.wm_attributes("-disabled", False)
root.wm_attributes("-transparentcolor", "white")

hwndMain = win32gui.FindWindow(None, "ScopePhoto")
win32gui.EnumChildWindows(hwndMain, callback, None)
rect = win32gui.GetWindowRect(wind_list[0])
root.geometry("+{}+{}".format(rect[0], rect[1]))

button2=Button(root, text='bright+', bg='lime', command=upscale)
button2.pack(side=LEFT)
button3=Button(root, text='exit', command=exit, bg='red')
button3.pack(side=LEFT)

root.mainloop()