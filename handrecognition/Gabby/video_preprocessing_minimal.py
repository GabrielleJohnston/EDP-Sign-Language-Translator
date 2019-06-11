# pre-processing videos
# no modifications (i.e. rotation) to increase data size

# importing libraries
import numpy as np
import cv2
import os # dealing with directories
import math
from tqdm import tqdm

# video directories 
TRAIN_DIR_WHAT = 'C:/Users/gabri/Documents/Uni/Year 2/Engineering Design Project/Hard drive/Raw Data/DATABASE/What'
TRAIN_DIR_HUNGRY = 'C:/Users/gabri/Documents/Uni/Year 2/Engineering Design Project/Hard drive/Raw Data/DATABASE/Hungry'
TRAIN_DIR_ANGRY = 'C:/Users/gabri/Documents/Uni/Year 2/Engineering Design Project/Hard drive/Raw Data/DATABASE/Angry'

# defining parameters
IMG_W_O = 96 # image width before cropping
IMG_W_F = 54 # image width after cropping
IMG_H = 54 # image height
IMG_CROP = 21 # how much to crop by on each side = (IMG_W_O - IMG_W_F)/2
IMG_CROP_1 = IMG_CROP - 1
IMG_CROP_2 = IMG_W_O - IMG_CROP - 1
TIME_FRAMES = 20

# reading in videos
def readVideo(videoFile):
    cap = cv2.VideoCapture(videoFile)
    vid = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            img = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2GRAY) # converting to greyscale
            frame = cv2.resize(img, (IMG_W_O, IMG_H)) # resizing frames to 96 x 54
            img = frame[0:IMG_H, IMG_CROP_1:IMG_CROP_2] # cropping frames to 54 x 54
            vid.append(np.array(img, dtype = np.float32)) # converting to numpy arrays
        else: 
            break
    cap.release()
    return vid 

# only saving a certain number of video frames, equally spaced apart in time
def sample(vid):
    sample_width = math.ceil(len(vid) / TIME_FRAMES) # number of frames between saved frames
    new_vid = []
    i = 0
    while (i < len(vid)):
        new_vid.append(np.array(vid[i]))
        i = i + sample_width
    while (len(new_vid) > TIME_FRAMES):
        new_vid = new_vid[:-1] # remove last frame if too many
    while (len(new_vid) < TIME_FRAMES):
        new_vid.append(new_vid[-1]) # add last frame twice if too few
    new_vid = np.array(new_vid)
    return new_vid

# creating training data 
def createTrainingData():
    trainingData = []
    for video in tqdm(os.listdir(TRAIN_DIR_WHAT)):
        videoPath = os.path.join(TRAIN_DIR_WHAT, video) # full path of video
        vid = readVideo(videoPath)
        vid_short = sample(vid)
        vid_short.reshape(-1, TIME_FRAMES, IMG_H, IMG_W_F, 1) 
        trainingData.append([np.array(vid_short), [1, 0, 0]])
    for video in tqdm(os.listdir(TRAIN_DIR_HUNGRY)):
        videoPath = os.path.join(TRAIN_DIR_HUNGRY, video) # full path of video
        vid = readVideo(videoPath)
        vid_short = sample(vid)
        vid_short.reshape(-1, TIME_FRAMES, IMG_H, IMG_W_F, 1)
        trainingData.append([np.array(vid_short), [0, 1, 0]])
    for video in tqdm(os.listdir(TRAIN_DIR_ANGRY)):
        videoPath = os.path.join(TRAIN_DIR_ANGRY, video) # full path of video
        vid = readVideo(videoPath)
        vid_short = sample(vid)
        vid_short.reshape(-1, TIME_FRAMES, IMG_H, IMG_W_F, 1)
        trainingData.append([np.array(vid_short), [0, 0, 1]])
    np.save('trainingData.npy', trainingData) # save the training data in a numpy file
    return trainingData

train_data = createTrainingData()