from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import ultralytics

from model_use import main

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, nargs='+', help="path to YOLO 'deploy'.pt file")
ap.add_argument("-v", "--video", required=True, nargs='+', help="number or url video stream")
args =vars(ap.parse_args())
print(args)
print("[INFO] loading model...")


videos = args['video']; models = args['model']

#start the video
with ThreadPoolExecutor(max_workers=2) as executor:
    for i in range(len(videos)):
        video = videos[i]; model_path = models[i]
        if video == '0':
            future = executor.submit(main, [int(video), model_path])
        else:
            future = executor.submit(main, [video, model_path])
