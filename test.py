from __future__ import print_function
import cv2
from Classes import RTRecongtion, VideoCapture, person
import tensorflow as tf
from tkinter import *
from PIL import Image
from PIL import ImageTk
import threading
import os
import datetime
import imutils
import pdb
import numpy as np
import dlib


loginValue=[("dilip","python")]
bgclr="#282828"
fgclr="#cecece"
clr='#004a95'

root=Tk()
login=Toplevel()

#login.geometry("400x400")
root.geometry("1280x720")
login.title("login page")
root.title("Main Page")

login.config(bg=bgclr)
root.resizable()
login.resizable()


label_user=Label(login,text="UserName",fg=fgclr,bg=bgclr)
label_password=Label(login,text="Password",fg=fgclr,bg=bgclr)
entry_User=Entry(login)
entry_Password=Entry(login)

button1 = Button(login, text="Login", command=lambda:command1()) #Login button
#warns=Label(bg=bgclr)


def command1():
    if entry_User.get() == "Kustar" and entry_Password.get() == "1234": #Checks whether username and password are correct
        root.deiconify() #Unhides the root window
        login.destroy()
    #else:
        #warns.config(text="Invalid username or Password",fg="red")



panelA=None
global check

def runAnalysis():
    global check
    check=True
    global panelA
    Users={}
    Framecounter=0
    currentFaces=0
    pdb.set_trace()
    while check==True:
        try:
            frame=video.FrameRead()

            #update all registored Faces
            for FID in Users.keys():
                    trackingQuality = Users[ fid ].update(frame)
                    if trackingQuality<7:
                        Users.pop(FID,None)
                    elif Users[ fid ].get_size()<5:
                        tracked_position =  Users[fid].get_bbox()
                        a,UserName=test.AnalysisFrame(frame,x,y,w,h)
                        if a==0:
                           UserName=None
                        Users[fid].set_name(UserName)
            if (Framecounter%5)==0:
                bounding_boxes=test.Facedetect(frame)
                nrof_faces=bounding_boxes.shape[0]
                
                #compare windows
                #error to many variables to unpack 
                for (_x,_y,_w,_h) in bounding_boxes:
                      x=int(_x)
                      y=int(_y)
                      w=int(_w)
                      h=int(_h)
                      x_bar=(x+w)/2 
                      y_bar= (y+h)/2
                      foundObject=None
                      for fid in Users.keys():
                           tracked_position =  Users[fid].get_bbox()
                           t_x = int(tracked_position.left())
                           t_y = int(tracked_position.top())
                           t_w = int(tracked_position.width())
                           t_h = int(tracked_position.height())

                           t_x_bar = (t_x + t_w)/2
                           t_y_bar = (t_y + t_h)/2

                           if ( ( t_x <= x_bar   <= (t_w)) and 
                                 ( t_y <= y_bar   <= (t_h)) and 
                                 ( x   <= t_x_bar <= ( w )) and 
                                 ( y   <= t_y_bar <= ( h ))):
                                    foundObject = fid



                      if foundObject is None:
                             tracker = dlib.correlation_tracker()
                             tracker.start_track(frame,dlib.rectangle(x,y,w,h))
                             a,UserName=test.AnalysisFrame(frame,x,y,w,h)
                             if a==0:
                                 UserName=None
                             Users[currentFaces]=person(tracker,UserName)
                             currentFaces+=1
            for FID in Users.keys():
                 tracked_position =  Users[fid].get_position()
                 t_x = int(tracked_position.left())
                 t_y = int(tracked_position.top())
                 t_w = int(tracked_position.width())
                 t_h = int(tracked_position.height())
                 text_x = t_x
                 text_y =t_h + 20

                 cv2.rectangle(frame, (t_x, t_y),(t_x + t_w , t_y + t_h),rectangleColor ,2)

                 name=None
                 if Users[fid].get_size()<5:
                     for x in range(0,size):
                         name=name+"."
                     cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1, lineType=2)
                 else:
                     cv2.putText(frame, Users[fid].get_name(), (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1, lineType=2)
            #old part
            
                 image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                 image = Image.fromarray(image)
                 image = ImageTk.PhotoImage(image)

                 if panelA is None:
			            # the first panel will store our original image
                    panelA = Label(image=image)
                    panelA.image = image
                    panelA.pack(side="left", padx=10, pady=10)
                 else:
			            # update the pannels
                    panelA.configure(image=image)
                    panelA.image = image
                 var =0
                 root.update()
                 root.after(10)
        except Exception as e: print(e)
    video.exit()
    cv2.destroyAllWindows()

def videoMode():
    global check
    check=True
    global panelA
    while check==True:
        try:
            frame=video.FrameRead()
            bounding_boxes=test.Facedetect(frame)
            nrof_faces = bounding_boxes.shape[0]
            bb = np.zeros((nrof_faces,4), dtype=np.int32)
            det = bounding_boxes[:, 0:4]
            for i in range(nrof_faces):
                x = int(det[i][0])
                y = int(det[i][1])
                w = int(det[i][2])
                h = int(det[i][3])
                a,name=test.AnalysisFrame(frame,x,y,w,h)
                cv2.rectangle(frame, (x,y),(w,h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, (h+20)), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0, 0, 255), thickness=1, lineType=2)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)

            if panelA is None:
                panelA = Label(image=image)
                panelA.image = image
                panelA.pack(side="left", padx=10, pady=10)
            else:  # update the pannels
                panelA.configure(image=image)
                panelA.image = image
            root.update()
            root.after(10)
        except Exception as e: print(e)
    video.exit()

def pictureMode():
     global check
     check=True
     global panelA
     frame=video.FrameRead()
     bounding_boxes=test.Facedetect(frame)
     nrof_faces = bounding_boxes.shape[0]
     det = bounding_boxes[:, 0:4]
     bb = np.zeros((nrof_faces,4), dtype=np.int32)
     if nrof_faces>0:
         i=0
         x = int(det[i][0])
         y = int(det[i][1])
         w = int(det[i][2])
         h = int(det[i][3])
         print(x,y,w,h)
         a,name=test.AnalysisFrame(frame,x,y,w,h)
         cv2.rectangle(frame, (x,y),(w,h), (0, 255, 0), 2)
         cv2.putText(frame, name, (x, (h+20)), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0, 0, 255), thickness=1, lineType=2)
     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
     image = Image.fromarray(image)
     image = ImageTk.PhotoImage(image)

     if panelA is None:
        panelA = Label(image=image)
        panelA.image = image
        panelA.pack(side="left", padx=10, pady=10)
     else:  # update the pannels
        panelA.configure(image=image)
        panelA.image = image
     root.update()
     root.after(10)
     video.exit()

def screenshot():
    frame=video.FrameRead()
    cv2.imwrite('opencv'+str(0)+'.png', frame)


def stopAnalysis():
    global check
    check=False

def threadM():
    thread = threading.Thread(target= videoMode())
    thread.start()

    


btn_Start = Button(root, text="Run", command=threadM)
btn_Start.pack(side="right", expand="yes", padx="10", pady="10")

btn_Start = Button(root, text="Picture", command=pictureMode)
btn_Start.pack(side="right", expand="yes", padx="10", pady="10")

btn_Stop = Button(root, text="Stop", command=stopAnalysis)
btn_Stop.pack(side="right", expand="yes", padx="10", pady="10")

btn_21 = Button(root, text="screenshot", command=screenshot)
btn_21.pack(side="right", expand="yes", padx="10", pady="10")

entry_User.grid(row=0, column=1,padx="10", pady="10") #These pack the elements, this includes the items for the main window
entry_Password.grid(row=1, column=1,padx="10", pady="10")
button1.grid(row=2,column=0,padx="10", pady="10")
label_user.grid(row=0, column=0,padx="10", pady="10")
label_password.grid(row=1, column=0,padx="10", pady="10")
#warns.grid(row=3, column=0, padx="10", pady="10")

video =VideoCapture ()
test =RTRecongtion()

check=True

video.SelectCamera(0)
test.load_classfier()
root.withdraw()
root.mainloop()



