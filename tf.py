# import the necessary packages
from __future__ import print_function
from PIL import Image as m2
from PIL import ImageTk
from tkinter import *
import threading
import datetime
import cv2
import os
from scipy import misc
import argparse
import tensorflow as tf
import numpy as np
import facenet
import detect_face
import random
import pdb


with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess,  os.path.join("Models"))

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    margin = 44
    image_size = 182

    root = Tk()
    global panel
    panel = None
    vs=cv2.VideoCapture(0)
    output=os.path.join("PostProccesed Images")
            # create a button, that when pressed, will take the current
		    # frame and save it to file
    entry_name=Entry(root)
    entry_name.pack(side="bottom", fill="both", expand="yes", padx=10,pady=10)
    global frame
    def videoLoop(vs):
        
		    # DISCLAIMER:
		    # I'm not a GUI developer, nor do I even pretend to be. This
		    # try/except statement is a pretty ugly hack to get around
		    # a RunTime error that Tkinter throws due to threading
            global panel
            global frame
            while True:
			    # keep looping over frames until we are instructed to stop

				    # grab the frame from the video stream and resize it to
				    # have a maximum width of 300 pixels
                    ret, frame = vs.read()
				    #frame = imutils.resize(frame, width=300)
		
				    # OpenCV represents images in BGR order; however PIL
				    # represents images in RGB order, so we need to swap
				    # the channels, then convert to PIL and ImageTk format
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = m2.fromarray(image)
                    image2 = ImageTk.PhotoImage(image)
                    if panel is None:
                        panel = Label(image=image2)
                        panel.image = image2
                        panel.pack(side="left", padx=10, pady=10)
                    else:
                        panel.configure(image=image2)
                        panel.image = image2
                    root.update()


    def takeSnapshot(output):
            global frame
		    # grab the current timestamp and use it to construct the
		    #output path
            img=frame
            #if img.ndim == 2:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print('to_rgb data dimension: ', img.ndim)
            #cv2.imshow(img)
            img = img[:, :, 0:3]
            print('after data dimension: ', img.ndim)
            bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0]
            print('detected_face: %d' % nrof_faces)
            if nrof_faces > 0:
                det = bounding_boxes[:, 0:4]
                img_size = np.asarray(img.shape)[0:2]
                if nrof_faces > 1:
                       bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                       img_center = img_size / 2
                       offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],(det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                       offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                       index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                       det = det[index, :]
                det = np.squeeze(det)
                bb_temp = np.zeros(4, dtype=np.int32)

                bb_temp[0] = det[0]
                bb_temp[1] = det[1]
                bb_temp[2] = det[2]
                bb_temp[3] = det[3]

                cropped_temp = img[bb_temp[1]:bb_temp[3], bb_temp[0]:bb_temp[2], :]
                scaled_temp = misc.imresize(cropped_temp, (image_size, image_size), interp='bilinear')
                ts = datetime.datetime.now()
                filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
                p = os.path.sep.join((output, filename))       
                misc.imsave(p, scaled_temp)
                print("[INFO] saved {}".format(filename))

    def nothing():
        print("enter nothing")
        output_dir_path = os.path.join("PostProccesed Images",entry_name.get()) #C:\Users\kingmkm\Documents\GitHub\real-time-deep-face-recognition-master
        output_dir = os.path.expanduser(output_dir_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        #pdb.set_trace()
        takeSnapshot(output_dir)

    btn = Button(root, text="Snapshot!",command=lambda:nothing())
    btn.pack(side="bottom", fill="both", expand="yes", padx=10,pady=10)

    thread = threading.Thread(target=videoLoop(vs))
    thread.start()
 
		    # set a callback to handle when the window is closed



    root.mainloop()


