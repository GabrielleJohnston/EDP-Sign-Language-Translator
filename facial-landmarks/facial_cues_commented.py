# input arguments: shapePredictor is a string - filename of shape predictor file
# videoFile is string - video file name
def faceMoves(shapePredictor, videoFile):
    from imutils.video import FileVideoStream
    from imutils import face_utils
    import numpy as np
    import datetime
    import argparse
    import imutils
    import time
    import dlib
    import cv2

################################################################################
#               Pre-processing for facial recognition program                  #
################################################################################

    # The video is opened from the required location on the computer 
    cap = cv2.VideoCapture(videoFile)
    ret, frame = cap.read()
    height = frame.shape[0]
    width = frame.shape[1]

    # the height of the face is determined to be 2/5 of the height of the frame
    # the width of thre face was determined to be 1/4 of the width of the frame
    face_height = int((2/5)*height)
    face_width = int((1/4)*width)

    # the video outputted was set to be of mp4 format, and given the name 
    # output.mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    name_out = videoFile.replace(videoFile, "output.mp4")
    out = cv2.VideoWriter(name_out , fourcc, 25.0, (face_width*2, face_height*2))

    # the output region of interest (roi) is cropped to be the relevent sizes
    while (ret == True):
        roi = frame [0 : face_height, int((width/2)-(face_width/2)) : int((width/2)+(face_width/2))]
        roi = cv2.resize( roi, (face_width*2, face_height*2))
        ret, frame = cap.read()
        out.write(roi)

    out.release()
    cap.release()
    cv2.destroyAllWindows()

################################################################################
#               Main body of facial recognition program                        #
################################################################################



    # start the file video stream
    fvs = FileVideoStream(name_out).start()

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    # detector is the initialised pre-trained detector from dlib, based on HOG 
    # standard

    # detector is now the face detector model
    detector = dlib.get_frontal_face_detector()

    # "shape predictor" is the taught model for landmark detections
    # loading this model into 'predictor() function'
    predictor = dlib.shape_predictor(shapePredictor)

    # array for each required landmark - add as required
    # number (name of the variables) refers to number on coordinate map - see 
    # file on github
    three = []
    twenty = []
    thirty_eight = []
    thirty_four = []
    twenty_eight = []

    # array to hold the differences between points in the x direction for head 
    # shaking
    x_differences = []

    # array to hold the differences in points on the nose in the y direction
    # used in normalisation of eyebrow distances fac
    h_differences = []

    live_ratios = []

    # array to store the differences between two coordinates at every frame
    d_differences = []

    shape = []

    # loop over the frames from the video stream
    while fvs.more():

        # grab the frame from the threaded video file stream, resize it to
        # have a maximum width of 400 pixels, and convert it to
        # grayscale
        frame = fvs.read()
        if frame is not None:
            frame = imutils.resize(frame, width=400)
            gray_vid = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            rects_vid = detector(gray_vid, 1)

            if (len(rects_vid) <= 0):
                continue

            # loop over the face detections
            for rect in rects_vid:
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = predictor(gray_vid, rect)
                shape = face_utils.shape_to_np(shape)

            # coordinates are stored as [x, y] - twenty[1] accesses y coordinate 
            # of 20
            counter = 0
            for j in shape:
                # create an array containing the x and y coordinates of point 3 
                # on the landmark map
                if counter == 2:
                    three = shape[counter]
                # create an array containing the x and y coordinates of point 20
                # on the landmark map
                if counter == 19:
                    twenty = shape[counter]
                # create an array containing the x and y coordinates of point 28
                # on the landmark map
                if counter == 27:
                    twenty_eight = shape[counter]
                # create an array containing the x and y coordinates of point 34
                # on the landmark map
                if counter == 33:
                    thirty_four = shape[counter]
                # create an array containing the x and y coordinates of point 38
                # on the landmark map
                if counter == 37:
                    thirty_eight = shape[counter]

                counter = counter + 1

            # finding the difference between point 20 and 38, i.e. distance
            # between the eye and eyebrow
            d_difference = abs(twenty[1] - thirty_eight[1])
            # append that to an array called d_differences
            d_differences.append(d_difference)

            # finding the difference between point 34 and 28, i.e. distance
            # between the top of the nose and bottom of the nose
            h_difference = abs(thirty_four[1] - twenty_eight[1])
            # append that to an array called h_differences
            h_differences.append(h_difference)

            # finding the difference between point 34 and 3, i.e. distance
            # between the bottom of the nose and the side of the face
            x_difference = abs(thirty_four[0] - three[0])
            # append that to an array called x_differences
            x_differences.append(x_difference)


            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

    # finding the ratio between the eye/eyebrow distance and
    # top-of-the-nose/bottom-of-the-nose distance
    # this results in distance invariance as the eye-eyebrow distance is
    # normalised by top-of-the-nose/bottom-of-the-nose distance
    # top-of-the-nose/bottom-of-the-nose distance would scale up and down
    # depending on the distance between user and the camera
    # this assumes that the user cannot move their nose: a valid assumption
    for i in range(0, len(d_differences)):
        live_ratio = round(d_differences[i]/h_differences[i], 2)
        # append that to an array called live_ratios
        live_ratios.append(live_ratio)

    # create an array of average ratios found earlier called ratios_average
    # this is to reduce errors caused by anomalies
    # this can be due to poor environmental setting and imperfect tracking of
    # the program used
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

    # k is a threshold value - achieved from numbers of testings and may vary
    # between people
    k = 0.05
    eyebrowMovement = 'neutral'
    # counter1 refers to frowning, counter2 to eyebrows being raised
    for i in range(0, len(ratios_average)):
        x = False
        # compare the average ratio with the reference ratio (ratio at the start
        # of the program)
        # if the value is lower than the reference value, frowning is detected
        if ratios_average[i] <= (live_ratios[0] - k/4):
            counter1 = counter1 + 1
            x = True
        # if the value is higher than the reference value, eyebrow raising is
        # detected
        if ratios_average[i] >= (live_ratios[0] + k/2):
            counter2 = counter2 + 1
            x = True
        # 3 consecutive frowning/eyebrow raising detected required to conclude
        # that the user indeed frown/ raising eyebrow
        # this is equivalent to frowning/eyebrow raised for an approximately 1s
        if x == False:
            if counter1 > 3:
                eyebrowMovement = 'frown'
            if counter2 > 3:
                eyebrowMovement = 'raised'
            counter1 = 0
            counter2 = 0


    delta_array = []
    x_average_differences = []

    # similarly, create an array of average x_diferences
    sum2 = 0
    for i in range(1, len(x_differences)):
        if (i+1)%3 != 0:
            sum2 = sum2 + x_differences[i]
        if(i+1)%3 == 0:
            sum2 = sum2 + x_differences[i]
            x_average_differences.append(round(sum2/3, 2))
            sum2 = 0

    # find the change of the distance between the nose and side of face in time
    # the change is considered negative when that change is less negative than -2
    # the change is considered positive when that change is more positive than 2
    
    # the 2 and -2 are threshold values, acquired from testings
    # these are required due to the shape predictor model being unable to
    # accurately plot the landmarks if the head is dramatically turned such that
    # the the face is not entirely visible. 
    
    # values simplified to just '-' for negative change, '+' for positive change
    # and 'o' for no change (also includes range -2 < 0 < 2)
    # store in array called delta_array
    for i in range(1, len(x_average_differences) - 1):
        delta = x_average_differences[i + 1] - x_average_differences[i]
        if delta < -2:
            delta_array.append('-')
        elif delta > 2:
            delta_array.append('+')
        else:
            delta_array.append('o')

    # head is shaken when the face turns left then turn right or vice versa
    # delta is negative when the face is turning left i.e. the distance between
    # the nose and the left side of the face decreases
    # delta is positive when the face is turning right i.e. the distance between
    # the nose and the left side of the face increases
    # hence, when there is a change in sign within a small period of time face
    # is turned
    headshake = False
    for i in range(0, len(delta_array) - 1):
        if (delta_array[i] == '+' and delta_array[i + 1] == '-'):
            headshake = True
        elif (delta_array[i] == '-' and delta_array[i + 1] == '+'):
            headshake = True

    fvs.stop()
    cv2.destroyAllWindows()


    if headshake == True and eyebrowMovement == 'frown':
        return 'shakeFrown'
    elif headshake == True and eyebrowMovement == 'raised':
        return 'shakeRaised'
    elif headshake == True and eyebrowMovement == 'neutral':
        return 'shakeNeutral'
    elif headshake == False and eyebrowMovement == 'frown':
        return 'neutralFrown'
    elif headshake == False and eyebrowMovement == 'raised':
        return 'neutralRaised'
    elif headshake == False and eyebrowMovement == 'neutral':
        return 'neutralNeutral'
