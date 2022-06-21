from fileinput import filename
from io import StringIO
import tkinter as tk
from gui import *
import os 
from PIL import Image,ImageTk,ExifTags
import cv2 as cv
root = tk.Tk()
root.geometry("1920x1080")
lst=excelReader()
#grim=cv.imread('final')



totalRows=len(lst) 
totalCols=len(lst[1])

print(lst[0][2])
def getfileName(filename): 
    #filename=filename[13:]    
    return(filename)

#getfileName("K:\beutylator\images\jas.jpeg")

def getImage(lst):   # sending image path to getImage label
    #fileName=getfileName(lst)
    filename =  lst
    print(filename)
    im1 = Image.open(filename,mode='r')    # IMAGE open  
    #img = Image.open(os.path.join(root,fileName))
    #im1=cv.imread(str(fileName))
    im1=im1.resize((100,100))   # resize image 
    test = ImageTk.PhotoImage(im1)   #creating  tkinter image , becoz can't understand pillow image 
    img_label = tk.Label(image=test)   # this is also creating image label 
    img_label.image = test      # assing label as     
    return img_label

for i in range(totalRows): 
    for j in range(totalCols-1):
        e=tk.Entry(
                root,
                width=30,
                fg="red",
                font=('Poppins',25,'bold')
            )
        e.grid(row=i, column=j) 
        e.insert(j,str(lst[i][j]))
        img_label=getImage(lst[i][2])
        img_label.grid(row=i, column=2)
         # e.insert(img_label.grid(row=i,column=j))

root.mainloop()