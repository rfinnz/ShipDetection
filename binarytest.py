import numpy as np
import cv2
import sys
j = 0
image = cv2.imread('/home/jacob/Python/compression/maybe.jpg')
blurred = cv2.pyrMeanShiftFiltering(image,20,51)
gray = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)
ret,threshold = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
_, contours, _ = cv2.findContours(threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
#print(len(contours))
yup = cv2.drawContours(image, contours, -1, (0,0,255), 3)
cv2.imwrite('/home/jacob/Python/compression/contours.jpg',yup)
rbg = cv2.imread('/home/jacob/Python/compression/contours.jpg',cv2.IMREAD_COLOR)
red = [0,0,255]
#print(rbg)
x = []
y = []
#sys.exit()
test = str(rbg)
#sys.exit()
for i in range(len(rbg)):
    for j in range(len(rbg[0])):
        b = rbg[i][j][0]
        g = rbg[i][j][1]
        r = rbg[i][j][2]
        blue = int(b)
        green = int(g)
        red = int(r)
        #print(b,g,r)
        if red > 160 and blue < 100 and green < 100:
            #if i > 0:
            x.append(i)
            #print('yes')
            y.append(j)
#sys.exit()
#print(x,y)
right = max(x)
left = min(x)
top = min(y)
bottom = max(y)
right_index = 0
left_index = 0
top_index = 0
bottom_index = 0
print("right " + str(right) + "top "+ str(top) + "left " + str(left)+ "bottom " +str(bottom))
#sys.exit()
cv2.rectangle(rbg,(right,top),(left,bottom),(0,255,0),2)
cv2.namedWindow('Display',cv2.WINDOW_NORMAL)
cv2.imshow('Display',rbg)
cv2.waitKey(0)
cv2.destroyAllWindows()

sys.exit()
