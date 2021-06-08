#!/usr/bin/env python3
# coding=utf-8

import numpy as np
import argparse
import glob
import cv2
import os

parser = argparse.ArgumentParser(description='Generate the video from the images.')
parser.add_argument('-d', '--dir', type=str, required=True, help='the dir with the images')

args = parser.parse_args()
dir_name = args.dir

filenames = sorted(glob.glob('./img/' + dir_name + '/map_*.png'), key=os.path.getmtime)

frameSize = (640, 480)

out = cv2.VideoWriter('./video/' + dir_name + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), 20, frameSize)

for filename in filenames:
  img = cv2.imread(filename)
  out.write(img)

out.release()
