import csv
import os
import sys
import cv2
import numpy as np

image_dir = "./train"
list_file = "./bb_landmark/loose_bb_train.csv"
output_dir = "./train_align"


def crop_face(img_dir, img_path, img_dir_out, x0, y0, w, h):
    img_path_full = os.path.join(img_dir, img_path + ".jpg")
    img_path_out_full = os.path.join(img_dir_out, img_path + ".jpg")
    # print(img_path_full)
    if not os.path.exists(img_path_full):
        print("{} does not exist".format(img_path_full))
        return

    img = cv2.imread(img_path_full)
    if len(np.shape(img)) != 3:
        print("Not RGB, Ignore")
        return

    img_h, img_w, img_c = np.shape(img)

    if w > h:
        min_side = h
    else:
        min_side = w

    margin = int(min_side*0.2)
    x0 -= margin
    y0 -= margin
    if x0 < 0:
        x0 = 0
    if y0 < 0:
        y0 = 0

    x1  = x0 + w + margin
    y1  = y0 + h + margin


    if x1 > img_w:
        x1 = img_w
    if y1 > img_h:
        y1 = img_h

    img_crop = img[y0: y1, x0: x1, :]
    if w < h:
        scale = 128.0 / w
        h = int(h*scale)
        w = 128
    else:
        scale = 128.0 / h
        w = int(w*scale)
        h = 128

    print(w, h)
    img_crop_scale = cv2.resize(img_crop, (w, h), 0, 0, cv2.INTER_CUBIC)

    img_path_out_dir = os.path.dirname(img_path_out_full)
    if not os.path.exists(img_path_out_dir):
        os.mkdir(img_path_out_dir)

    cv2.imwrite(img_path_out_full, img_crop_scale)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

if not os.path.exists(image_dir):
    print("{} does not exist".format(image_dir))
    exit()

if not os.path.exists(list_file):
    print("{} does not exist".format(list_file))
    exit()

csv_fd = open(list_file, "rb")
reader = csv.reader(csv_fd)

for idx, item in enumerate(reader):
    if idx == 0:
        continue
    # print(item[1], item[2], item[3], item[4])
    crop_face(image_dir, item[0], output_dir, int(item[1]), int(item[2]), int(item[3]), int(item[4]))

csv_fd.close()
