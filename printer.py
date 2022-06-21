
from typing import Any
import cv2 as cv
import dlib
import math
import numpy as np
from PIL import ImageFont, ImageDraw, Image


image =cv.imread("pan.jpeg")
image1=cv.imread("final.jpg")


#img = cv.resize(image,(700,700))

print(image.shape[:])
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
	length=int(length*300)				    # at 300 dpi 
	width=int(width*300)
	blank=np.zeros((length,width,3),dtype='uint8')    #8 bit to represent single pixel value 
	blank.fill(255)
	return blank

image=resize_and_show(image) 
image1=resize_and_show(image1)
canvas=BlankCanvas(12,8)
canvas=cv.imread("canvas.jpg")
startIndexX=399
startIndexY=150
canvas[startIndexX:image.shape[0]+startIndexX,startIndexY:image.shape[1]+startIndexY]=image1
xe=200
ye=startIndexY
x2=200
y2=image.shape[1]
canvas[startIndexX:image.shape[0]+startIndexX,image.shape[1]+startIndexY+x2:2*image.shape[1]+startIndexY+x2]=image
canvas[image.shape[0]+startIndexX:]

cv.putText(canvas,str('sexy'),(831,2962),cv.FONT_HERSHEY_TRIPLEX,3,(0,0,0))
cv.putText(canvas, str('sexy'), (977, 3051),cv.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 0))
cv.putText(canvas, str('sexy'), (1191, 3135),cv.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 0))
cv.putText(canvas, str('sexy'), (929, 3220),cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))

print(canvas.shape[:])
cv.imwrite("image.jpg",canvas)
print(image.shape[:])
cv.waitKey(0)
cv.destroyAllWindows()

# cv.imshow("ini",image)
# cv.waitKey(0)
# cv.destroyAllWindows()
# cv.imshow("fin",image1)
# cv.waitKey(0)
# cv.destroyAllWindows()