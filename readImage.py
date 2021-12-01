# 定义读取图片函数
import cv2
import numpy as np


def get_img(file_path, img_rows, img_cols):
    img = cv2.imread(file_path)
    img = cv2.resize(img, (img_rows, img_cols))
    if img.shape[2] == 1:
        img = np.dstack([img, img, img])
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)

    return img