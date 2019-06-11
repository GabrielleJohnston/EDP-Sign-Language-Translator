# This piece of code normalizes videos in time to frame_number frames.
# It resizes the video to 1/x pixels
# It crops to videos to the desired size
# It turns videos into GREYSCALE colours

# plan argument: 0 for video, 1 for mhi
# path: path of webcam video
# name_out: path of where to save output
# video eg: rf = 30, ss = 36
# mhi eg: rf = 10, ss = 108

def processVideo (path, name_out, resize_factor, side_size, plan):
	# import os, sys
	import numpy as np
	import cv2

	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	frame_number = 20

	#here the video is opened (for the first time) and the number of frames is counted
	cap = cv2.VideoCapture(path)
	ret, frame = cap.read()
	frame_counter = 0

	while (ret == True) :
		frame_counter += 1
		past_frame = frame #this stores the last frame
		ret, frame = cap.read()

	#here frame proportion is computed
	frame_rem = frame_counter % frame_number

	if (frame_rem < frame_number/2):
		frame_par = frame_counter - frame_rem
	if (frame_rem >= frame_number/2):
		frame_par = frame_counter + (frame_number - frame_rem)

	#here the video is opened (for the second time)
	cap = cv2.VideoCapture(path)
	ret, frame = cap.read()
	height = frame.shape[0]
	width = frame.shape[1]


	if (plan == 0):

		name_out = name_out + ".mp4"
		out = cv2.VideoWriter(name_out, fourcc, 5.0, (side_size, side_size) , 0)

		c = 1
		while (ret == True):
			#this lets us grab the frame we are interested in
			if (((c % (frame_par/frame_number)) == 0) or ((frame_rem >= 10) and (frame_rem < 15) and c==1)):

				#this resizes the current frame depending on the resize factor
				roi = cv2.resize(frame, (int(width/resize_factor), int(height/resize_factor)))
				roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

				#here the center of frame is found and the frame is cropped wrt that point
				mid_height = int(roi.shape[0]/2)
				mid_width = int(roi.shape[1]/2)
				roi = roi [int(mid_height-(side_size/2)) : int(mid_height+(side_size/2)), 
					   int(mid_width-(side_size/2)) : int(mid_width+(side_size/2))]
				out.write(roi)

			ret, frame = cap.read()
			c += 1

		#here an additional frame is added when needed
		if (frame_rem >= 10):
			out.write(past_frame)

		out.release()

	if (plan == 1):

		name_out = name_out + ".jpg"
		d = 0 #keeps track of current frame
		frame_sel = 1
		fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(200, 5, 0.7, 3)

		out = np.zeros((side_size, side_size))
		white = 0

		#here we get the mid size of the newly resized frame
		mid_height = int(height/(2*resize_factor))
		mid_width = int(width/(2*resize_factor))

		while (ret == True):

			if ((((d % (frame_par/frame_number)) == 0) or ((frame_rem >= 10) and (frame_rem < 15) and c==1)) 
			    and (d % frame_sel) == 0):
				#here Background substraction algorithm is applied to the frame
				new_image = fgbg.apply(frame)

				#here the current frame is resized to a desirable size
				roi_res = cv2.resize(new_image, (int(width/resize_factor), int(height/resize_factor)))

				#here the frame is cropped to im_height x im_width size
				roi = roi_res [mid_height-int(side_size/2) : mid_height + int(side_size/2), 
					       mid_width - int(side_size/2) : mid_width + int(side_size/2)]

				roi[np.where(roi == [255])] = [white]
				out = out + roi
				white += 10

			ret, frame = cap.read()

			d += 1

		if (frame_rem >= 10):
			new_image = fgbg.apply(past_frame)

			#here the current frame is resized to a desirable size
			roi_res = cv2.resize(new_image, (int(width/resize_factor), int(height/resize_factor)))

			#here the frame is cropped to im_height x im_width size
			roi = roi_res [mid_height-int(side_size/2) : mid_height + int(side_size/2), 
				       mid_width - int(side_size/2) : mid_width + int(side_size/2)]

			roi[np.where(roi == [255])] = [white]
			out = out + roi
			white += 10

		cv2.imwrite(name_out, out)

	cap.release()

# Pre_Processing ("C:\\Users\\lapor\\OneDrive\\Documenti\\IMPERIAL\\Sign Language\\Try_1\\CADOC.mp4" ,"C:\\Users\\lapor\\OneDrive\\Documenti\\IMPERIAL\\Sign Language\\Function_test\\OUTPUT", 10, 100, 0)
# Pre_Processing ("C:\\Users\\lapor\\OneDrive\\Documenti\\IMPERIAL\\Sign Language\\Try_1\\CADOC.mp4" ,"C:\\Users\\lapor\\OneDrive\\Documenti\\IMPERIAL\\Sign Language\\Function_test\\OUTPUT", 10, 108, 1)
