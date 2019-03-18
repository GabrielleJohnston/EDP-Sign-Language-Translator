# ENSURE THAT shape_predictor_68_face_landmarks.dat FILE IS IN THE SAME
# DIRECTORY AS THE CODE, OR THAT PATH IS CHANGED

# ENSURE THAT ALL OTHER SCRIPTS ARE IN SAME DIRECTORY AS CODE, OR THAT PATH IS
# CHANGED

import facial_landmarks_video
import Pre_Processing

shapePredictor = 'shape_predictor_68_face_landmarks.dat'

# name of video file may be different - currently set to test.mp4 in pre-processing program
videoFile = 'test.mp4'

facialCue = facial_landmarks_videos_copy.faceMoves(shapePredictor, videoFile)

print(facialCue)
Pre_Processing.processVideo(videoFile, 'processedvideo', 30, 36, 0)
#facial_cue : headshake, eyebrow_raised, frowning
sign = 'hungry'

if facialCue == 'shakeNeutral':
    print('not', sign)
elif facialCue == 'neutralRaised':
    print(sign+'?')
elif facialCue == 'shakeRaised':
    print('not', sign+'?')
elif sign == 'what' and facialCue == 'neutralFrown':
    print(sign+'?')
else:
    print(sign)
