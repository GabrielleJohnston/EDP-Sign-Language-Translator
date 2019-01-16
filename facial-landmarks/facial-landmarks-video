# USAGE
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --picamera 1

# import the necessary packages
from imutils.video import FileVideoStream
from imutils import face_utils
import numpy as np
import datetime
import argparse
import imutils
import time
import dlib
import cv2

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


coordinates = []
new_coordinates = []

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

		# display the size of the queue on the frame
		# cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
		# 	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

		eyebrows = np.zeros((2), dtype = int)


		# loop over the face detections
		for rect in rects:
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)


			counter = 0
			for j in shape:
				# counter 21 refers to inside of users right eyebrow, 22 for left eyebrow
				if (counter >= 21 and counter <= 22):
					# print(shape[counter])
					# coordinates of right eyebrow first printed
					# second print is coordinates of left eyebrow
					eyebrows = np.vstack((eyebrows, shape[counter]))
				counter = counter + 1

			eyebrows = np.delete(eyebrows, 0)
			difference_x = eyebrows[1] - eyebrows[3]
			difference_y = eyebrows[2] - eyebrows[4]
			difference = [difference_x, difference_y]
			coordinates.append(difference)
			print("x difference: ", abs(difference_x))
			print("y difference: ", abs(difference_y))
			# print("\n")


			# uncomment the below to show video output with labelled landmarks

			# loop over the (x, y)-coordinates for the facial landmarks
			# and draw them on the image
			for (x, y) in shape:
				cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

		# show the frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

# aiming to extract every 5th coordinate to try to avoid the discrepancys and random movement of the coordinates
count = 0
for coordinate in coordinates:
	count = count + 1
	if count % 5 == 0:
		new_coordinates.append(coordinates[coordinate])
print(new_coordinates)

# do a bit of cleanup
cv2.destroyAllWindows()
fvs.stop()
