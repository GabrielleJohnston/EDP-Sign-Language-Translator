# USAGE
# python facial_landmarks_video.py --shape-predictor shape_predictor_68_face_landmarks.dat --video video.mp4

# import the necessary packages
from imutils.video import FileVideoStream
from imutils import face_utils
from imutils.video import FPS   # not needed in proper implementation - test feature
import numpy as np
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import xlsxwriter       # not needed in proper implementation - test feature


# construct the argument parse and parse the arguments
# --shape-predictor refers to the trained model referred to to map the facial landmarks
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-v", "--video", required=True,
    help="path to input video file")
args = vars(ap.parse_args())

# start the file video stream thread and allow the buffer to
# start to fill
print("[INFO] starting video file thread...")
fvs = FileVideoStream(args["video"]).start()
time.sleep(1.0)

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")

# detector is the initialised pre-trained detector from dlib, based on HOG standard
# detector is now the face detector model
detector = dlib.get_frontal_face_detector()

# "shape predictor" is the taught model for landmark detections
# loading this model into 'predictor() function'
predictor = dlib.shape_predictor(args["shape_predictor"])

# excel workbook - test functionality
workbook = xlsxwriter.Workbook('landmarks.xlsx')
worksheet = workbook.add_worksheet()
row = 0
col = 0


twenty = []
thirty_eight = []

# array to store the differences between two coordinates at every frame
differences = []

# start frames-per-second counter and timer - only necessary for testing purposes
fps = FPS().start()

# loop over the frames from the video stream
while fvs.more():
	# grab the frame from the threaded video file stream, resize it to
	# have a maximum width of 400 pixels, and convert it to
	# grayscale
    frame = fvs.read()
    if frame is not None:
        frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	    # detect faces in the grayscale frame
        rects = detector(gray, 1)

		# loop over the face detections
        for rect in rects:
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)


        counter = 0
        for j in shape:
            if (counter == 19):
                # create an array containing the x and y coordinates of point 20 on the landmark map
                twenty = shape[counter]
            if (counter == 37):
                # create an array containing the x and y coordinates of point 38 on the landmark map
                thirty_eight = shape[counter]
            counter = counter + 1

        # coordinates are stored as [x, y] - twenty[1] accesses y coordinate of 20
        difference = abs(twenty[1] - thirty_eight[1])

        # append the difference between the two points at every frame to an array
        differences.append(difference)

        fps.update()

	# uncomment the below to show video output with labelled landmarks

	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

	# show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

fps.stop()

# create an array, taking the average the differences of every four frames - this aims to reduce the effects of any anomalies/discrepancys in the positions of the coordinates
sum = 0
average = []
for i in range(0, len(differences)):
    if (i+1)%4 != 0:
        sum = sum + differences[i]
    if(i+1)%4 == 0:
        sum = sum + differences[i]
        average.append(sum/4)
        sum = 0
print(average)

counter1 = 0
for i in range(0, len(average)):
    # write entries in 'average' array to an excel spreadsheet - makes analysis for testing easier, in reality not needed
    worksheet.write(row, col, average[i])
    row = row + 1
    if average[i] > 10:     # buffer value of 10 needs testing and may differ between people - more testing required
        counter1 = counter1 + 1
    else:
        if counter1 >= 3:   # three 'high' values needed in a row to class as success - equivalent to eyebrows raised for approx 1 second
            print("EYEBROWS WERE RAISED!!!")
        counter1 = 0


# prints the time for analysis and fames per second
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
fvs.stop()
workbook.close()
