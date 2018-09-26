from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tkinter
import tensorflow as tf
from scipy import misc
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import facenet
import detect_face
import os
from os.path import join as pjoin
import sys
import time
import copy
import math
import pickle
from sklearn.svm import SVC
from sklearn.externals import joblib
import pdb


class RTRecongtion:



        #intilize the recongition Variables
      
        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 3
        batch_size = 1000
        image_size = 182
        input_image_size = 160


        def __init__(self):
            with tf.Graph().as_default():
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
                self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
                with self.sess.as_default():
                    self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, os.path.join("Models"))

                    print('Loading feature extraction model')
                    modeldir = os.path.join("Models","20170511-185253","20170511-185253.pb")
                    facenet.load_model(modeldir)

                    self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                    self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                    self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                    self.embedding_size = self.embeddings.get_shape()[1]


        def load_classfier(self):
            classifier_filename = os.path.join("Models","Classifier.pkl")
            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (self.model, self.class_names) = pickle.load(infile)
                self.HumanNames = os.listdir("./PostProccesed Images")
                self.HumanNames.sort()
                print(self.HumanNames)

        def Facedetect(self,frame):
            bounding_boxes, _ = detect_face.detect_face(frame, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)

            return bounding_boxes

        def AnalysisFrame(self,frame,x,y,w,h):
                #pdb.set_trace()
                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                #HumanNames = ['Abdulrahamn','Mohammed']
                frame = frame[:, :, 0:3]
                #bounding_boxes, _ = detect_face.detect_face(frame, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
               # pdb.set_trace()
                #nrof_faces = bounding_boxes.shape[0]
                #print('Detected_FaceNum: %d' % nrof_faces) # change later
                #if nrof_faces > 0:
                    
                    #det = bounding_boxes[:, 0:4]
                self.img_size = np.asarray(frame.shape)[0:2]

                cropped = []
                scaled = []
                scaled_reshape = []
                #bb = np.zeros((nrof_faces,4), dtype=np.int32)

                #for i in range(nrof_faces):
                emb_array = np.zeros((1, self.embedding_size))

                        #bb[i][0] = det[i][0]
                        #bb[i][1] = det[i][1]
                        #bb[i][2] = det[i][2]
                        #bb[i][3] = det[i][3]

                        # inner exception
                if x <= 0 or y <= 0 or w >= len(frame[0]) or h >= len(frame):
                     print('face is inner of range!')
                     return 0

                cropped.append(frame[y:h, x:w, :])
                cropped[0] = facenet.flip(cropped[0], False)
                scaled.append(misc.imresize(cropped[0], (self.image_size, self.image_size), interp='bilinear'))
                scaled[0] = cv2.resize(scaled[0], (self.input_image_size,self.input_image_size),interpolation=cv2.INTER_CUBIC)
                scaled[0] = facenet.prewhiten(scaled[0])
                scaled_reshape.append(scaled[0].reshape(-1,self.input_image_size,self.input_image_size,3))
                feed_dict = {self.images_placeholder: scaled_reshape[0], self.phase_train_placeholder: False}
                emb_array[0, :] = self.sess.run(self.embeddings, feed_dict=feed_dict)
                predictions = self.model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                if best_class_probabilities>0.01:

                    for H_i in self.HumanNames:
                        if self.HumanNames[best_class_indices[0]] == H_i:
                            print(H_i)
                            result_names = self.HumanNames[best_class_indices[0]]
                            return 1,result_names
                                    #cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                               # 1, (0, 0, 255), thickness=1, lineType=2)
                else:
                     return 0
                    
                
                return 0

        def Tracking(self):
            return 0


class VideoCapture:


    def countCameras(self):
        n = 0
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i)
                ret, frame = cap.read()
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cap.release()
                n += 1
            except:
                clearCapture(cap)
                if n==0:
                    return -1
                break
            return n
    
    def SelectCamera(self,n):
        self.video_capture = cv2.VideoCapture(n)

    def FrameRead(self):
        ret, frame = self.video_capture.read()
        #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5) 
        return frame

    def Exit(self):
        self.video_capture.release()

class person:

    name=[]

    def __init__(self,tracker,tempName):
        self.tracker=tracker
        self.name.append(tempName)
        self.size=1
        self.done=True

    def update(self,frame):
        return self.tracker.update(frame)

    def set_name(self,tempName):
        self.name.append(tempName)
        self.size+=1



    def get_name(self):
        if self.done:
            return self.name[0]
    def get_position(self):
        return self.tracker.get_position()

    def get_size(self):
        return self.size
