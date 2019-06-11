# pre-processing videos with motion detection
# start collecting frame before motion first detected
# stop collecting when motion halts for and does not recommence

# importing libraries
import numpy as np
import cv2
import os # dealing with directories
import math
from tqdm import tqdm
from math import pi
import imutils

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
IMG_AREA = IMG_W_F*IMG_H

# reading in video
def readVideo(videoFile):
    cap = cv2.VideoCapture(videoFile)
    vid = []
    firstFrame = None # first frame of video
    firstMotion = None # first frame with motion
    index = 0 # frame number (starting from 0)
    lastMotion_index = 0 # frame number with last motion
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # converting to greyscale
            frame = cv2.resize(img, (IMG_W_O, IMG_H)) # resizing frames to 96 x 54
            img = frame[0:IMG_H, IMG_CROP_1:IMG_CROP_2] # cropping frames to 54 x 54
            img_blurred = cv2.GaussianBlur(img, (21, 21), 0) 
            if firstFrame is None:
                firstFrame = img_blurred # blurred image so that motion detection is less sensitive
                vid.append(np.array(img, dtype = np.float32))
            elif firstMotion is None:
                frameDelta = cv2.absdiff(firstFrame, img_blurred) # compare blurred frame to blurred first frame
                threshold = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1] 
                threshold = cv2.dilate(threshold, None, iterations=2)
                contours = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = imutils.grab_contours(contours)
                motionArea = 0
                for c in contours:
                    motionArea += cv2.contourArea(c)
                avg = (motionArea*100)/IMG_AREA
                if avg > 1:
                    firstMotion = img # if there is sufficient motion, image is saved
                    vid.append(np.array(img, dtype = np.float32))
                    index += 1
            else:
                vid.append(np.array(img, dtype = np.float32)) # if there already is a first motion, append
                frameDelta = cv2.absdiff(firstFrame, img_blurred) # change this to frame before? 
                threshold = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
                threshold = cv2.dilate(threshold, None, iterations=2)
                contours = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = imutils.grab_contours(contours)
                motionArea = 0
                for c in contours:
                    motionArea += cv2.contourArea(c)
                avg = (motionArea*100)/IMG_AREA
                if avg > 1:
                    lastMotion_index = index # if there is motion, save index as last motion index 
                index += 1
        else: 
            break
    cap.release()
    vid = vid[:lastMotion_index+1] # only save frames up to one after the last frame with motion
    return vid 

# only saving a certain number of video frames, equally spaced apart in time
def sample(vid):
    sample_width = math.ceil(len(vid) / TIME_FRAMES)
    new_vid = []
    i = 0
    while (i < len(vid)):
        new_vid.append(np.array(vid[i]))
        i = i + sample_width
    while (len(new_vid) > TIME_FRAMES):
        new_vid = new_vid[:-1]
    while (len(new_vid) < TIME_FRAMES):
        new_vid.append(new_vid[-1])
    new_vid = np.array(new_vid)
    return new_vid

# creating training data
def createTrainingData():
    trainingData = []
    for video in tqdm(os.listdir(TRAIN_DIR_WHAT)):
        videoPath = os.path.join(TRAIN_DIR_WHAT, video) # full path of image
        vid = readVideo(videoPath)
        vid_short = sample(vid)
        vid_short.reshape(-1, TIME_FRAMES, IMG_H, IMG_W_F, 1)
        trainingData.append([np.array(vid_short), [1, 0, 0]])
    for video in tqdm(os.listdir(TRAIN_DIR_HUNGRY)):
        videoPath = os.path.join(TRAIN_DIR_HUNGRY, video) # full path of image
        vid = readVideo(videoPath)
        vid_short = sample(vid)
        vid_short.reshape(-1, TIME_FRAMES, IMG_H, IMG_W_F, 1)
        trainingData.append([np.array(vid_short), [0, 1, 0]])
    for video in tqdm(os.listdir(TRAIN_DIR_ANGRY)):
        videoPath = os.path.join(TRAIN_DIR_ANGRY, video) # full path of image
        vid = readVideo(videoPath)
        vid_short = sample(vid)
        vid_short.reshape(-1, TIME_FRAMES, IMG_H, IMG_W_F, 1)
        trainingData.append([np.array(vid_short), [0, 0, 1]])
    np.save('trainingDataMotionDetect.npy', trainingData) # save the training data in a numpy file
    return trainingData

train_data = createTrainingData()
