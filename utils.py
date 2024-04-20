import os
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# ========================= SETTINGS
ISIZE = 250
BASE_DIR = r"./data"


# ========================= FUNCTIONS
def read_csv(file_name):
    return pd.read_csv(os.path.join(BASE_DIR, file_name))


def get_column(df, column):
    return df.iloc[:, column].to_numpy()


def convert(img_path):
    global g
    img = np.array(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([36, 0, 0])
    upper_green = np.array([86, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    result = cv2.bitwise_and(img, img, mask=mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        cnt = contours[0]
        cv2.drawContours(result, [cnt], 0, (0, 255, 0), 3)
        mask = np.zeros_like(img)
        cv2.drawContours(mask, [cnt], 0, (255, 255, 255), -1)
        result = cv2.bitwise_and(result, mask)
        moments = cv2.moments(cnt)

        if moments["m00"] != 0:
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
        else:
            cX, cY = 0, 0

        cv2.circle(result, (cX, cY), 10, (255, 0, 0), -1)
        h, w = result.shape[:2]
        start_x = max(0, cX - ISIZE // 2)
        start_y = max(0, cY - ISIZE // 2)

        if start_x + ISIZE > w:
            start_x = w - ISIZE
        if start_y + ISIZE > h:
            start_y = h - ISIZE

        result = result[start_y:start_y + ISIZE, start_x:start_x + ISIZE]
        result[result != 0] = 255
        b, g, r = cv2.split(result)
        g = np.array(g) / 255.0

    return g


def get_data(paths):
    images = defaultdict(list)
    for path in paths:
        imgop = Image.open(os.path.join(BASE_DIR, path))
        category = os.path.basename(os.path.dirname(path))
        images[category].append(convert(imgop))

    return images


def prepare_data(images_dict):
    images_list = []
    labels_list = []
    for category, images in images_dict.items():
        for image in images:
            img = Image.fromarray(image)
            resized_img = img.resize((ISIZE, ISIZE))
            image_np = np.array(resized_img)
            images_list.append(image_np)
            labels_list.append(category)

    images_np = np.array(images_list)
    images_np = images_np.reshape(-1, ISIZE, ISIZE, 1)
    labels_np = np.array(labels_list)

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels_np)
    labels_onehot = to_categorical(labels_encoded)

    return images_np, labels_onehot
