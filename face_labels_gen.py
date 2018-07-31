#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
import glob
import os
import re
import sys
import cv2
import numpy as np


def generate_labels(img_dir):

    dirs = os.listdir(img_dir)
    label = 0
    _str = ""
    for dir in dirs:
        if os.path.isdir(os.path.join(img_dir, dir)):
            imgs = glob.glob(os.path.join(os.path.join(img_dir, dir), "*.jpg"))
            for img in imgs:
                path = os.path.join(dir, os.path.basename(img))
                label_str = path + " " + str(label) + "\n"
                print(label_str)
                _str += label_str
            label += 1

    return _str


if __name__ == '__main__':
    img_dir = sys.argv[1]
    labels_str = generate_labels(img_dir)

    with open("labels.txt", "w") as labels_file:
        labels_file.writelines(labels_str)

