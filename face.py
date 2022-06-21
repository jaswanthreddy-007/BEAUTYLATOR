
from typing import Any
import cv2 as cv
import array as arr
from array import *
import numpy as np
import dlib
import math
import adjuster
import keyboard as key


#func to make lines on image
def Addline(image,x1,y1,x2,y2): 
    cv.line(image,(x1,y1),(x2,y2),(0,0,0),2)

def Addline2(image,x1,y1,x2,y2): 
    cv.line(image,(x1,y1),(x2,y2),(0,0,0),1)


face_cascade=cv.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv.imread("bha.jpeg")
im = cv.imread("bha.jpeg")

imgcopy = img.copy()

gray = cv.cvtColor(img,cv.COLOR_BGR2RGB)
faces = face_cascade.detectMultiScale(gray,1.1,4)
co=faces[:]
xcenter=int(faces[0][0]+faces[0][2]//2)
ycenter=int(faces[0][1])


cv.namedWindow("output", cv.WINDOW_NORMAL)
cv.resizeWindow("output", 471, 960)


print(xcenter,ycenter)

xposi = []
yposi = []
gr=1.61803


hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks_GTX.dat")

gray = cv.cvtColor(img,cv.COLOR_BGR2RGB)
faces = hog_face_detector(gray)

for face in faces:
        face_landmarks = dlib_facelandmark(gray, face)
        for n in range(0,68):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            xposi.append(x)
            yposi.append(y)
            #print(n,xposi[n],yposi[n])
            cv.circle(img, (x, y), 2, (0, 0, 255), 2)
            cv.putText(img,str(n),(x,y),cv.FONT_HERSHEY_TRIPLEX,0.4,(0,0,0))

Addline2(imgcopy,xposi[27],yposi[27],xposi[27],yposi[27]-200)

xcenter ,ycenter = adjuster.AdjustFunction(imgcopy,xcenter,ycenter)


#required lines to make facemask

Addline(img,xposi[27],yposi[27],xposi[59],yposi[31])
Addline(img,xposi[27],yposi[27],xposi[55],yposi[35])
Addline(img,xposi[59],yposi[31],xposi[55],yposi[35])
Addline(img,xposi[36],yposi[36],xposi[48],yposi[48])
Addline(img,xposi[45],yposi[45],xposi[54],yposi[54])
Addline(img,xposi[48],yposi[48],xposi[35],yposi[35])
Addline(img,xposi[31],yposi[31],xposi[54],yposi[54])
Addline(img,xposi[48],yposi[48],xposi[51],yposi[51])
Addline(img,xposi[51],yposi[51],xposi[54],yposi[54])
Addline(img,xposi[48],yposi[48],xposi[57],yposi[57])
Addline(img,xposi[57],yposi[57],xposi[54],yposi[54])
Addline(img,xposi[22],yposi[22],xposi[25],yposi[25])
Addline(img,xposi[21],yposi[21],xposi[18],yposi[18])
Addline(img,xposi[36],yposi[36],xposi[45],yposi[45])
Addline(img,xposi[36],yposi[36],xposi[39],yposi[39])
Addline(img,xposi[42],yposi[42],xposi[45],yposi[45])
Addline(img,xposi[39],yposi[39],xposi[42],yposi[42])
Addline(img,xposi[4],yposi[4],xposi[57],yposi[57])
Addline(img,xposi[57],yposi[57],xposi[12],yposi[12])
Addline(img,xposi[22],yposi[22],xposi[26],yposi[26])
Addline(img,xposi[21],yposi[21],xposi[17],yposi[17])
Addline(img,xcenter,ycenter,xposi[8],yposi[8])


#lengths

xmid_lip=(xposi[62]+xposi[66])/2
ymid_lip=(yposi[62]+yposi[66])/2
ymid_lip_chin=abs(ymid_lip-yposi[8])
ynose_midlip=abs(yposi[27]-ymid_lip)
uplip_thickness=abs(yposi[51]-yposi[62])
lowlip_thhickness=abs(yposi[66]-yposi[57])
nose_length=abs(yposi[27]-yposi[30])
nose_width=abs((xposi[59]-xposi[64]))*0.99
nose_chin=abs(yposi[33]-yposi[8])
lip_width=abs(xposi[48]-xposi[54])
eye_width=abs(xposi[42]-xposi[45])
dist_eyes=abs(xposi[42]-xposi[39])
noselength_2 = abs(yposi[27]-yposi[33])
nose_end_chin=abs(yposi[33]-yposi[8])
nose_end_midlip=abs(ymid_lip-yposi[33])
face_length=abs(yposi[8]-ycenter)
face_width=abs(xposi[1]-xposi[15])
eye_mid=abs(xposi[22]+xposi[21])/2
eyemid_noseend=abs(yposi[21]-yposi[33])
topface_eyemid=abs(yposi[21]-ycenter)


#parameters of golden ratio


raw=ynose_midlip/ymid_lip_chin
if raw > gr: 
    para1=(gr/raw)*100*0.5
else:
    para1=(raw/gr)*100*0.5

raw2=nose_end_chin/ymid_lip_chin
if raw2 > gr: 
    para6=(gr/raw2)*100
else:
    para6=(raw2/gr)*100

raw3=nose_width/nose_end_midlip
if raw3 > gr: 
    para7=(gr/raw3)*100
else:
    para7=(raw3/gr)*100

raw4= face_length/face_width
if raw4 > gr: 
    para8=(gr/raw4)*100
else:
    para8=(raw4/gr)*100    
#triple ratio starts here

#ratio 1
raw5 = topface_eyemid/eyemid_noseend
if raw5 > 1:
    para9 = (1/raw5)*100*0.5
else:
    para9 = (raw5)*100*0.5

#ratio2
raw6 = eyemid_noseend/nose_end_chin
if raw6 > 1:
    para10 = (1/raw6)*100
else:
    para10 = (raw6)*100

#ratio3
raw7 = topface_eyemid/nose_end_chin
if raw7 > 1:
    para11 = (1/raw7)*100
else:
    para11 = (raw7)*100
#parameters of ratio 1:1

para2=(min(nose_width,lip_width)/max(nose_width,lip_width))*100
para3=(min(noselength_2,nose_chin)/max(noselength_2,nose_chin))*100
para4=(min(eye_width,dist_eyes)/max(eye_width,dist_eyes))*100
para5=(min(nose_length,nose_width)/max(nose_length,nose_width))*100

#computing avg golden ratio percentage
gr_comp_avg=(para1+para2+para3+para4+para5+para6+para7+para8+para9+para10+para11)/10


print(gr_comp_avg)
print("\n\n")

print(para1)
print(para2)

print(para3)

print(para4)
print(para5)
print(para6)
print(para7)

print(para8)
print(para9)
print(para10)
print(para11)
print("\n\n")

#print(nose_length)
#print(nose_width)

#gr_comp_avg = round(gr_comp_avg,3)
para1 = round(para1, 3)
para2 = round(para2, 3)
para3 = round(para3, 3)
para4 = round(para4, 3)
para5 = round(para5, 3)
para6 = round(para6, 3)
para7 = round(para7, 3)
para8 = round(para8, 3)
para9 = round(para9, 3)
para10 = round(para10, 3)
para11 = round(para11, 3)
gr_comp_avg = round(gr_comp_avg,11)
cv.putText(img,str(gr_comp_avg),(10,60),cv.FONT_HERSHEY_TRIPLEX,1.3,(0,0,0))

cv.imwrite("final.jpg",img)
cv.imshow("output",img)
cv.waitKey(0)
cv.destroyAllWindows()

## printer starts here


# im = cv.imread("tah.jpeg")
im1 = cv.imread("final.jpg")


#img = cv.resize(image,(700,700))


print(im.shape[:])
DESIRED_HEIGHT = 1200
DESIRED_WIDTH = 600


def resize_and_show(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
  return img


def BlankCanvas(length, width):     # blank images maker with given inches
	length = int(length*300)				    # at 300 dpi
	width = int(width*300)
	# 8 bit to represent single pixel value
	blank = np.zeros((length, width, 3), dtype='uint8')
	blank.fill(255)
	return blank


im = resize_and_show(im)
im1 = resize_and_show(im1)
canvas = BlankCanvas(12, 8)
canvas = cv.imread("canvas.jpg")
startIndexX = 400
startIndexY = 150
canvas[startIndexX:im.shape[0]+startIndexX,startIndexY:im.shape[1]+startIndexY] = im
xe = 200
ye = startIndexY
x2 = 200
y2 = im.shape[1]
canvas[startIndexX:im.shape[0]+startIndexX, im.shape[1] +startIndexY+x2:2*im.shape[1]+startIndexY+x2] = im1
canvas[im.shape[0]+startIndexX:]
cv.putText(canvas, str(para1), (915, 2718),cv.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 0))
cv.putText(canvas, str(para2), (915, 2791),cv.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 0))
cv.putText(canvas, str(para3), (915, 2869),
           cv.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 0))
cv.putText(canvas, str(para4), (915, 2946),
           cv.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 0))
cv.putText(canvas, str(para5), (915, 3026),
           cv.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 0))
cv.putText(canvas, str(para6), (915, 3105),
           cv.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 0))
cv.putText(canvas, str(para7), (915, 3180),
           cv.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 0))
cv.putText(canvas, str(para8), (915, 3261),
           cv.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 0))
cv.putText(canvas, str(para9), (915, 3342),
           cv.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 0))
cv.putText(canvas, str(para10), (915, 3418),
           cv.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 0))
cv.putText(canvas, str(para11), (915, 3500),
           cv.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 0))

cv.putText(canvas, str(gr_comp_avg), (1493, 3015),
           cv.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 0))

print(canvas.shape[:])
cv.imwrite("image.jpg", canvas)
print(im.shape[:])
cv.waitKey(0)
cv.destroyAllWindows()














