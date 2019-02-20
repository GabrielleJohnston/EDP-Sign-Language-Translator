# USAGE
# python facial_landmarks_video.py --shape-predictor shape_predictor_68_face_landmarks.dat --video video.mp4

from imutils.video import FileVideoStream
from imutils import face_utils
import numpy as np
import datetime
import argparse
import imutils
import time
import dlib
import cv2


# FUNCTION FOR DETECTING HEAD SHAKING
    # return True if head shaken, False if not
def headShaking(visibleFaceWidthOverTime):
    signsOfDiffInFWBetweenFrames = []
    averagedVFWoT = []

    # again, taking the average of each cluster of three elements to try to reduce effect of anomlies
    tempSum = 0
    for i in range(0, len(visibleFaceWidthOverTime)):
        if (i+1)%3 != 0:
            tempSum = tempSum + visibleFaceWidthOverTime[i]
        if(i+1)%3 == 0:
            tempSum = tempSum + visibleFaceWidthOverTime[i]
            averagedVFWoT.append(round(tempSum/3, 2))
            tempSum = 0

    for i in range(0, len(averagedVFWoT) - 1):
        diffInVFWBetweenFrames = averagedVFWoT[i + 1] - averagedVFWoT[i]

        # a buffer value of two was chosen as when the face gets close to the edges of the shake, there are more anomalies
        # where the model struggles to plot all 68 points. This is a value that is large enough to remove these anomalies but
        # not too large that the shake is not properly detected

        # difference between frames is effectively time dependent, therefore a very slow movement to one side from the other
        # may not be registered as a shake
        if diffInVFWBetweenFrames < -2:
            signsOfDiffInFWBetweenFrames.append('-')
            # for negative values

        elif diffInVFWBetweenFrames > 2:
            signsOfDiffInFWBetweenFrames.append('+')
            # for positive values

        else:
            signsOfDiffInFWBetweenFrames.append('o')
            # all values between -2 and 2 are taken as 0

    headshake = False
    for i in range(0, len(signsOfDiffInFWBetweenFrames) - 1):

        # if there is a trend of positive movement then negative, this is registered as shaking the head
        if (signsOfDiffInFWBetweenFrames[i] == '+' and signsOfDiffInFWBetweenFrames[i + 1] == '-'):
            headshake = True
        # also registered as shaking the head for the reverse - negative movement then positive to account for
        # someone preferentially shaking in opposite directions
        elif (signsOfDiffInFWBetweenFrames[i] == '-' and signsOfDiffInFWBetweenFrames[i + 1] == '+'):
            headshake = True

    if headshake == True:
        return True
    else:
        return False

# FUNCTION FOR CHECKING FOR EYEBROW MOVEMENT
    # return 1 if frown
    # return 2 if raising eyebrows
    # return 3 if neither
def eyebrowMovement(normalisedEyetoEyebrowThroughFrames):

    tempSum = 0
    # averageNormalisedEyebrows is an array that will contain an averaged version of normalisedEyetoEyebrowThroughFrames, by
    # taking the average value of each cluster of three elements to try to reduce anomalies from the landmarks jumping between frames
    # tempSum is a temporary value that we use to do this
    averageNormalisedEyebrows = []
    for i in range(0, len(normalisedEyetoEyebrowThroughFrames)):
        if (i+1)%3 != 0:
            tempSum = tempSum + normalisedEyetoEyebrowThroughFrames[i]
        if(i+1)%3 == 0:
            tempSum = tempSum + normalisedEyetoEyebrowThroughFrames[i]
            averageNormalisedEyebrows.append(round(tempSum/3, 2))
            tempSum = 0

    frownCount = 0
    raiseCount = 0

    # select a value of k - will need testing and changing between people. This acts as the 'buffer'
    # value to differentiate between a deliberate raising of the eyebrows and random detections of small movement
    k = 0.1


    # frownCount refers to frowning, raiseCount to eyebrows being raised
    for i in range(0, len(averageNormalisedEyebrows)):
        # x is a boolean value that helps to reset the counters when no eyebrow movement is detected
        x = False

        if averageNormalisedEyebrows[i] <= (normalisedEyetoEyebrowThroughFrames[0] - k/4):
            # buffer value needs testing and may differ between people - more testing required
            frownCount = frownCount + 1
            x = True

        if averageNormalisedEyebrows[i] >= (normalisedEyetoEyebrowThroughFrames[0] + k/2):
            raiseCount = raiseCount + 1
            x = True

        if x == False:
            # if three occaisions in a row of an upward or downward movement are detected, this is classed as a successful result
            if frownCount > 3:
                return 1
            if raiseCount > 3:   # three 'high' values needed in a row to class as success - equivalent to eyebrows raised for approx 1 second
                return 2
    return 0

# FUNCTION THAT DETERMINES IF EYEBROW MOVEMENT IS RAISING
    # returns True if raising, False if not
def eyebrowsRaised(eyebrowMovementResult):
    if eyebrowMovementResult == 2:
        return True
    else:
        return False

# FUNCTION THAT DETERMINES IF EYEBROW MOVEMENT IS FROWNING
    # returns True if frowning, False if not
def eyebrowsFrowned(eyebrowMovementResult):
    if eyebrowMovementResult == 1:
        return True
    else:
        return False


# set arguments required to run program - see USAGE above
# --shape-predictor refers to the trained model referred to to map the facial landmarks
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-v", "--video", required=True,
    help="path to input video file")
args = vars(ap.parse_args())

# start the file video stream
fvs = FileVideoStream(args["video"]).start()

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
# detector is the initialised pre-trained detector from dlib, based on HOG standard

# detector is now the face detector model
detector = dlib.get_frontal_face_detector()

# "shape predictor" is the taught model for landmark detections
# loading this model into 'predictor() function'
predictor = dlib.shape_predictor(args["shape_predictor"])

# array for each required landmark - add as required
# number refers to number on coordinate map - see file on github
three = []
twenty = []
thirty_eight = []
thirty_four = []
twenty_eight = []

# array to hold the differences between points in the x direction for head shaking
visibleFaceWidthOverTime = []

# array to hold the differences in points on the nose in the y direction
# used in normalisation of eyebrow distances
noseLengthThroughFrames = []

# array to store the differences between two coordinates at every frame - the distance between the eye and eyebrow
# to detect eyebrows being raised
eyeToEyebrowOverTime = []

normalisedEyetoEyebrowThroughFrames = []


# loop over the frames from the video stream
while fvs.more():

	# grab the frame from the threaded video file stream, resize it to
	# have a maximum width of 400 pixels, and convert it to
	# grayscale
    frame = fvs.read()
    if frame is not None:
        frame = imutils.resize(frame, width=400)
        grayscaleFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # faces is an array containing four sets of coordinates for the bounding box that encloses the face
        # each element in faces is the bounding box for one face
        faces = detector(grayscaleFrame, 0)

		# loop over the face detections
        # 'for each detected face ...'
        for face in faces:
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
            facialLandmarks = predictor(grayscaleFrame, face)
            facialLandmarks = face_utils.shape_to_np(facialLandmarks)

            # facialLandmarks is a numpy array that contains all the coordinates of the 68 landmarks
            # facialLandmarks is an array of (x, y) coordinates of the landmarks i.e each element in facialLandmarks is an array that
            # contains the coordinates of one of the 68 landmarks

        # landmark counter helps us iterate through facialLandmarks to pull the coordinates of the relevant landmarks
        landmarkCounter = 0
        for j in facialLandmarks:
            if landmarkCounter == 2:
                # the array 'three' now contains the x and y coordinates of the point 3 on the landmakr map
                three = facialLandmarks[landmarkCounter]

            if landmarkCounter == 19:
                twenty = facialLandmarks[landmarkCounter]

            if landmarkCounter == 27:
                twenty_eight = facialLandmarks[landmarkCounter]

            if landmarkCounter == 33:
                thirty_four = facialLandmarks[landmarkCounter]

            if landmarkCounter == 37:
                thirty_eight = facialLandmarks[landmarkCounter]

            landmarkCounter = landmarkCounter + 1

        # coordinates are stored as [x, y] therefore twenty[1] accesses y coordinate of 20

        # eyeToEyebrow is the distance between point 20 and point 38 - distance between eye and eyebrow
        eyeToEyebrow = abs(twenty[1] - thirty_eight[1])

        # append the difference between the two points at every frame to an array
        # each element in eyeToEyebrowOverTime is a value eyeToEyebrow at a frame i.e. first (0) element contains eyeToEyebrow at first frame
        eyeToEyebrowOverTime.append(eyeToEyebrow)

        # noseLength is the distance between point 34 and point 28 - length of the nose
        noseLength = abs(thirty_four[1] - twenty_eight[1])

        # each element in noseLengthThroughFrames contains the value of noseLength at the relevant frame
        noseLengthThroughFrames.append(noseLength)

        # difference in x direction between point 34 and point 3 - visible width of one side of face
        visibleFaceWidth = abs(thirty_four[0] - three[0])

        # each element in visibleFaceWidthOverTime contains the value visibleFaceWidth at the relevant frame
        visibleFaceWidthOverTime.append(visibleFaceWidth)

# create an array, taking the average the differences of every four frames - this aims to reduce the effects of any anomalies/discrepancys in the positions of the coordinates
# the distance in eyeToEyebrow are normalised by dividing by noseLength for each frame, then added to an array of normalisedEyetoEyebrowThroughFrames
for i in range(0, len(eyeToEyebrowOverTime)):
    normalisedEyetoEyebrow = round(eyeToEyebrowOverTime[i]/noseLengthThroughFrames[i], 2)   # elements are rounded to 2dp
    normalisedEyetoEyebrowThroughFrames.append(normalisedEyetoEyebrow)


# BOOLEAN VARIABLES TO PASS INTO FINAL PROGRAM
headshake = headShaking(visibleFaceWidthOverTime)
eyebrowMovementResult = eyebrowMovement(normalisedEyetoEyebrowThroughFrames)
raisedEyebrows = eyebrowsRaised(eyebrowMovementResult)
frowning = eyebrowsFrowned(eyebrowMovementResult)

print("head shaken? : ", headshake)
print("eyebrows raised? : ", raisedEyebrows)
print("frowning? : ", frowning)
