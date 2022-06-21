
from typing import Any
import cv2 as cv
import array as arr
from array import *
import numpy as np
import time as t
import dlib
import keyboard as key 

face_cascade=cv.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv.imread("emm.jpg")
imgcopy = img.copy()
gray = cv.cvtColor(img,cv.COLOR_BGR2RGB)
faces = face_cascade.detectMultiScale(gray,1.1,6)
co=faces[:]
xcenter=int(faces[0][0]+faces[0][2]//2)
ycenter=int(faces[0][1])


def addcircle(image,xcenter,ycenter):               #this function add circle on the face copy image
    cv.circle(image,(xcenter,ycenter),2,(0,255,0),5)

#Editing functions
    
def moveUp(image,xcenter,ycenter):   #update the y co-oridnates to UP position 
    ycenter=ycenter-30
    return xcenter,ycenter

def moveDown(image,xcenter,ycenter):   #update the y co-ordinates to DOWN position
    ycenter=ycenter+15
    return xcenter,ycenter
def moveRight(image,xcenter,ycenter): 
    xcenter=xcenter+15
    return xcenter,ycenter
def moveLeft(image,xcenter,ycenter): 
    xcenter=xcenter-10
    return xcenter,ycenter
#To show img before editing

 

#Editing Image
def AdjustFunction(img,xcenter,ycenter): 
    cv.namedWindow("editoutput", cv.WINDOW_NORMAL)
    cv.resizeWindow("editoutput", 471, 960)
    
    print(co[:])
    addcircle(img,xcenter,ycenter)
    cv.imshow("editoutput",img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    while True: 
        
        a=key.read_key()

        
        #if enter is pressed editing stops
       
        if a == 'enter':  
             break

         #if up is pressed top dot cordinates will be changed and displays image again with the new dot
   
        if a=='up':
            print("up")
            xcenter,ycenter=moveUp(img,xcenter,ycenter)
            addcircle(img,xcenter,ycenter)
            cv.imshow("editoutput",img)
            cv.waitKey(0)
            cv.destroyAllWindows()
            t.sleep(0.5)

    #if down is pressed top dot cordinates shift downwards and image will be redisplayed with new dot

        if a=='down':
            print("down")
            xcenter,ycenter=moveDown(img,xcenter,ycenter)
            addcircle(img,xcenter,ycenter)
            cv.imshow("editoutput",img)
            cv.waitKey(0)
            cv.destroyAllWindows()
            t.sleep(0.5)
        if a == 'left': 
            print('left') 
            xcenter,ycenter=moveLeft(img,xcenter,ycenter)
            addcircle(img,xcenter,ycenter)
            cv.imshow("editoutput",img)
            cv.waitKey(0)
            cv.destroyAllWindows()
            t.sleep(0.5)

        if a == 'right': 
            print('right') 
            xcenter,ycenter=moveRight(img,xcenter,ycenter)
            addcircle(img,xcenter,ycenter)
            cv.imshow("editoutput",img)
            cv.waitKey(0)
            cv.destroyAllWindows()
            t.sleep(0.5)
    return (xcenter,ycenter)
        


#final edited image

cv.waitKey(0)
cv.destroyAllWindows()

#final face top cordinates

print(xcenter,ycenter)



    





