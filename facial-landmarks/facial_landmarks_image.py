# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/two_people.jpg

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

# construct the argument parser and parse the arguments
# --shape-predictor refers to the trained model referred to to map the facial landmarks
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())



# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor

# detector is the initialised pre-trained detector from dlib, based on HOG standard

# detector is now the face detector model
detector = dlib.get_frontal_face_detector()

# "shape predictor" is the taught model for landmark detections
# loading this model into 'predictor() function'
predictor = dlib.shape_predictor(args["shape_predictor"])

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
# second argument: increasing increases resolution and allows more faces to be detected at the expense of performance (may need drastic increase to be noticeable on an image?)
rects = detector(gray, 1)

# loop over the face detections
# enumerate loops with automatic counter
# rect includes (x, y) coordinates of detection
for (i, rect) in enumerate(rects):

	# i refers to the face: 0 index for first face, 1 for second etc
	# rect has format [(x, y) (x + w, y + h)] - i.e. [(top left bb) (bottom right bb)]

	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array

	# apply function predictor to grayscale image and bounding box
	shape = predictor(gray, rect)

	shape = face_utils.shape_to_np(shape)
	# here shape becomes an array of 'coordinates' - n dimensional numpy array of shape(68,2) where this shape refers to property of numpy array

	# creates an array for the right eyebrow - contains only one zero that is deleted after
	# if dtype int not specified, integer values are followed by a '.'
	right_eyebrow = np.zeros((2), dtype = int)

	# counter is used for the tracking of the number referring to each coordinate point - see image on github for reference although note zero index in python
	counter = 0

	# a numpy array is created that contains the coordinates for the (right eyebrow) landmark.
	# the final array contains 5 coordinates stored in one array, in the form [x1 y1 x2 y2 ... x5 y5]
	for j in shape:
		if (counter >= 17 and counter <= 21):
			print(shape[counter])
			right_eyebrow = np.vstack((right_eyebrow, shape[counter]))
		counter = counter + 1
	right_eyebrow = np.delete(right_eyebrow, 0)
	print(right_eyebrow)
