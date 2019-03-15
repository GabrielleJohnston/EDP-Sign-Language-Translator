import numpy as np
import cv2

cap = cv2.VideoCapture("C:\\Users\\BE-jgs3817\\Desktop\\EDP\\test_video.mp4")

while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)

    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#facial_cue : headshake, eyebrow_raised, frowning
sign = 'hungry'
facial_cue = 'headshake'

if sign == 'hungry' and facial_cue == 'headshake':
    print('not hungry')
elif sign == 'hungry':
    print('hungry')
else:
    print(sign)
