# importing libraries
import numpy as np
import cv2
import os # dealing with directories
from random import shuffle
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

TRAIN_DIR_WHAT = 'C:/Users/gabri/Documents/Uni/Year 2/Engineering Design Project/Hard drive/Raw Data/DATABASE/What'
TRAIN_DIR_HUNGRY = 'C:/Users/gabri/Documents/Uni/Year 2/Engineering Design Project/Hard drive/Raw Data/DATABASE/Hungry'
TRAIN_DIR_ANGRY = 'C:/Users/gabri/Documents/Uni/Year 2/Engineering Design Project/Hard drive/Raw Data/DATABASE/Angry'
IMG_W_O = 96
IMG_W_F = 54
IMG_H = 54
TIME_FRAMES = 10

def readVideo(videoFile):
    cap = cv2.VideoCapture(videoFile)
    vid = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            img = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(img, (IMG_W_O, IMG_H))
            img = frame[0:54, 20:74]
            vid.append(np.array(img))
        else: 
            break
    cap.release()
    return vid 

def sample(vid):
    sample_width = math.ceil(len(vid) / TIME_FRAMES)
    new_vid = []
    i = 0
    while (i < len(vid)):
        new_vid.append(np.array(vid[i]))
        i = i + sample_width
    while (len(new_vid) > 10):
        new_vid = new_vid[:-1]
    while (len(new_vid) < 10):
        new_vid.append(new_vid[-1])
    new_vid = np.array(new_vid)
    new_vid.reshape(-1, 10, 54, 54, 1)
    return new_vid

def createTrainingData():
    trainingData = []
    for video in tqdm(os.listdir(TRAIN_DIR_WHAT)):
        videoPath = os.path.join(TRAIN_DIR_WHAT, video) # full path of image
        vid = readVideo(videoPath)
        vid_short = sample(vid)
        trainingData.append([np.array(vid_short), [1, 0, 0]])
    for video in tqdm(os.listdir(TRAIN_DIR_HUNGRY)):
        videoPath = os.path.join(TRAIN_DIR_HUNGRY, video) # full path of image
        vid = readVideo(videoPath)
        vid_short = sample(vid)
        trainingData.append([np.array(vid_short), [0, 1, 0]])
    for video in tqdm(os.listdir(TRAIN_DIR_ANGRY)):
        videoPath = os.path.join(TRAIN_DIR_ANGRY, video) # full path of image
        vid = readVideo(videoPath)
        vid_short = sample(vid)
        trainingData.append([np.array(vid_short), [0, 0, 1]])
    np.save('trainingData.npy', trainingData) # save the training data in a numpy file
    return trainingData

train_data = createTrainingData()