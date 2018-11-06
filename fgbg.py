import cv2
import numpy as np

cap = cv2.VideoCapture('/home/jacob/Python/compression/frame3.jpg')
fgbg = cv2.createBackgroundSubtractorMOG2()

ret, frame = cap.read()
fgmask = fgbg.apply(frame)

cv2.imshow('original', frame)
cv2.imshow('fg', fgmask)

cv2.waitKey(0)
cv2.destroyAllWindows()
