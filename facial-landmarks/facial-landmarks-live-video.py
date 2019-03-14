# ensure that shape_predictor_68_face_landmarks.dat file is saved in website folder

def faceMoves(a):
	from imutils.video import VideoStream
	from imutils import face_utils
	from imutils.video import FPS   # not needed in proper implementation - test feature
	import numpy as np
	import datetime
	import argparse
	import imutils
	import time
	import dlib
	import cv2
	import keyboard

	# set arguments required to run program - see USAGE above
	# --shape-predictor refers to the trained model referred to to map the facial landmarks
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--shape-predictor", required=True,
		help="path to facial landmark predictor")
	ap.add_argument("-r", "--picamera", type = int, default = -1,
		help = "whether not the Rapsberry Pi camera should be used")
	args = vars(ap.parse_args())

	# start the file video stream
	vs = VideoStream(usePiCamera = args["picamera"] > 0).start()



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
	x_differences = []

	# array to hold the differences in points on the nose in the y direction
	# used in normalisation of eyebrow distances fac
	h_differences = []

	live_ratios = []

	# array to store the differences between two coordinates at every frame
	d_differences = []
	ISTRUE = True

	shape = []

	while ISTRUE:
		try:

			# loop over the frames from the video stream
			# while fvs.more():

			# grab the frame from the threaded video file stream, resize it to
			# have a maximum width of 400 pixels, and convert it to
			# grayscale
			frame = vs.read()

			frame = imutils.resize(frame, width=400)
			gray_vid = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			rects_vid = detector(gray_vid, 1)

			if (len(rects_vid) > 0):
				print("face found")
			elif(len(rects_vid) <=0):
				print("face not found")
				continue


			# loop over the face detections
			for rect in rects_vid:
				# determine the facial landmarks for the face region, then
				# convert the facial landmark (x, y)-coordinates to a NumPy
				# array
				shape = predictor(gray_vid, rect)
				shape = face_utils.shape_to_np(shape)

			counter = 0
			for j in shape:
				if counter == 2:
					three = shape[counter]
				elif counter == 19:
					# create an array containing the x and y coordinates of point 20 on the landmark map
					twenty = shape[counter]
				elif counter == 27:
					twenty_eight = shape[counter]
				elif counter == 33:
					thirty_four = shape[counter]
				elif counter == 37:
					# create an array containing the x and y coordinates of point 38 on the landmark map
					thirty_eight = shape[counter]
				counter = counter + 1

			# coordinates are stored as [x, y] - twenty[1] accesses y coordinate of 20
			d_difference = abs(twenty[1] - thirty_eight[1])

			# append the difference between the two points at every frame to an array
			d_differences.append(d_difference)

			h_difference = abs(thirty_four[1] - twenty_eight[1])

			h_differences.append(h_difference)



			# difference in x direction for head shaking
			x_difference = abs(thirty_four[0] - three[0])

			x_differences.append(x_difference)


			# fps.update()

		# uncomment the below to show video output with labelled landmarks

		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
			for (x, y) in shape:
				cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
		#
		# # show the frame
			cv2.imshow('Frame', frame)
			# if cv2.waitKey(25) & 0xFF == ord('p'):
				#	 print('jbdfubfeubf')
		except KeyboardInterrupt:
			vs.stop()
			cv2.destroyAllWindows()
			break


	vs.stop()
	# fps.stop()
	start = time.time()

	# create an array, taking the average the differences of every four frames - this aims to reduce the effects of any anomalies/discrepancys in the positions of the coordinates

	for i in range(0, len(d_differences)):
		live_ratio = round(d_differences[i]/h_differences[i], 2)
		live_ratios.append(live_ratio)

	sum = 0
	ratios_average = []
	for i in range(0, len(live_ratios)):
		if (i+1)%3 != 0:
			sum = sum + live_ratios[i]
		if(i+1)%3 == 0:
			sum = sum + live_ratios[i]
			ratios_average.append(round(sum/3, 2))
			sum = 0

	counter1 = 0
	counter2 = 0

	# select a value of k - will need testing and changing between people
	k = 0.1
	eyebrowMovement = 'neutral'
	# counter1 refers to frowning, counter2 to eyebrows being raised
	for i in range(0, len(ratios_average)):
		x = False
		if ratios_average[i] <= (live_ratios[0] - k/4):
			# buffer value needs testing and may differ between people - more testing required
			counter1 = counter1 + 1
			x = True
		if ratios_average[i] >= (live_ratios[0] + k/2):
			counter2 = counter2 + 1
			x = True
		if x == False:
			if counter1 > 3:
				print("FROWN")
				eyebrowMovement = 'frown'
			if counter2 > 3:   # three 'high' values needed in a row to class as success - equivalent to eyebrows raised for approx 1 second
				print("EYEBROWS WERE RAISED!!!")
				eyebrowMovement = 'raised'
			counter1 = 0
			counter2 = 0

	delta_array = []
	x_average_differences = []

	sum2 = 0
	for i in range(1, len(x_differences)):
		if (i+1)%3 != 0:
			sum2 = sum2 + x_differences[i]
		if(i+1)%3 == 0:
			sum2 = sum2 + x_differences[i]
			x_average_differences.append(round(sum2/3, 2))
			sum2 = 0


	for i in range(1, len(x_average_differences) - 1):
		delta = x_average_differences[i + 1] - x_average_differences[i]
		if delta < -2:
			delta_array.append('-')
		elif delta > 2:
			delta_array.append('+')
		else:
			delta_array.append('o')

	headshake = False
	for i in range(0, len(delta_array) - 1):
		if (delta_array[i] == '+' and delta_array[i + 1] == '-'):
			headshake = True
		elif (delta_array[i] == '-' and delta_array[i + 1] == '+'):
			headshake = True

	if headshake == True:
		print("head shaken")

	end = time.time()

	print(end - start)


			# do a bit of cleanup
	cv2.destroyAllWindows()

	if headshake == True and eyebrowMovement == 'frown':
		return "shakeFrown")
	elif headshake == True and eyebrowMovement == 'raised':
		return "shakeRaised"
	elif headshake == True and eyebrowMovement == 'neutral':
		return "shakeNeutral"
	elif headshake == False and eyebrowMovement == 'frown':
		return "neutralFrown"
	elif headshake == False and eyebrowMovement == 'raised':
		return "neutralRaised"
	elif headshake == False and eyebrowMovement == 'neutral':
		return "neutralNeutral"
