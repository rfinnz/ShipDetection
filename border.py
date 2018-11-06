import numpy as np
import cv2
import sys
x = []
y = []
image = cv2.imread('/home/jacob/Python/compression/maybe.jpg',0)
rbg = cv2.imread('/home/jacob/Python/compression/contours.jpg',cv2.IMREAD_COLOR)
for i in range(len(image)):
    for j in range(len(image[0])):
        if image[i][j] >200 and i > 0 and j > 0:
            x.append(i)
            y.append(j)
right = max(x)
left = min(x)
top = min(y)
bottom = max(y)
print(right,left,top,bottom)
sys.exit()
cv2.rectangle(rbg,(right,top),(left,bottom),(0,255,0),2)
cv2.namedWindow('Display',cv2.WINDOW_NORMAL)
cv2.imshow('Display',rbg)
cv2.waitKey(0)
cv2.destroyAllWindows()
