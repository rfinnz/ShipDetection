import cv2
import sys
j = 0
path = '/home/jacob/Python/compression/frame3.jpg'
#img = cv2.imread(path,0)
img = cv2.imread(path,0)
#cv2.imshow('window')
#cv2.WAITKEY(0)
#cv2.DESTROYALLWINDOWS()
#sys.exit()
for i in range(len(img)):
    for j in range(len(img[0])):
        if img[i][j] <= 200:
            img[i][j] = 0
        else:
            img[i][j]=255

cv2.imwrite('/home/jacob/Python/compression/maybe.jpg',img)

#im = cv2.imread(path,0)
im2 = cv2.imread('/home/jacob/Python/compression/maybe.jpg',cv2.IMREAD_COLOR)
for i in range(len(im2)):
    while j < len(im2[0]):
        if im2[i][j]==[0,0,0]:
            im2[i][j] = [0,0,255]
            j = 0
            break
        j = j + 1
cv2.imwrite('/home/jacob/Python/compression/contour.jpg',im2)
