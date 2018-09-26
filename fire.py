from firebasedata import LiveData
import pyrebase
import json
import random
import string
import os
from scipy import misc
import sys
import argparse
import tensorflow as tf
import numpy as np
import facenet
import detect_face
from time import sleep
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import pickle
from sklearn.svm import SVC






config = {
    "apiKey": "AIzaSyCcOzOdg6GYYfNdY8q_mZkoDhVw7NgBACQ",
    "authDomain": "sdpku-52eb1.firebaseapp.com",
    "databaseURL": "https://sdpku-52eb1.firebaseio.com",
    "projectId": "sdpku-52eb1",
    "storageBucket": "sdpku-52eb1.appspot.com",
    "messagingSenderId": "333036971886",
    "googleCloudVisionAPIKey": "AIzaSyBK3V2v3lLIw9tTk0FnC0zLrn67WaEvkoc"
}

firebase = pyrebase.initialize_app(config)

storage = firebase.storage()
db=firebase.database()

g=0
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
        

def downloadImages(uid,imageURLs,userName):
    

    print("start Downloading")
    count=0
    N=8
    createFolder('./PostProccesed Images/'+ userName)
    for URL in imageURLs:
        pictureName=''.join(random.choices(string.ascii_uppercase + string.digits, k=N))
        count+=1
        print('image',count)
        storage.child(URL).download('PostProccesed Images/'+ userName+'/'+pictureName+'.jpg')
        img=mpimg.imread('PostProccesed Images/'+userName+'/'+pictureName+'.jpg')


        process_img(img,'./PostProccesed Images/'+userName+'/'+pictureName+'.jpg')
        

def veryImagesDownloaded(uid,key):
    name=db.child("/users/"+key).update({"imageDownloaded": "downloaded"})
    classfy()


def getUserName(key):
    name=db.child("/users/"+key+'/name').get()
    print(name.val())
  
    return name.val()

def getImagesNames(uid):
    print(uid)
    imageURLs=[]
    all_images=db.child("/images").order_by_child('uid').equal_to(uid).get()
    for user in all_images.each():
        print(user.val()['imageUrl'])
        imageURLs.append(user.val()['imageUrl'])
    
   
    return imageURLs     

def checkVeryfication(message):
    

    #print(message['data'])
    items= message['data']
    if(g==1):
        for i in items:
            if g==1:
                try:
                    if (message['data'][i]['veryfied']=='true' and message['data'][i]['imageDownloaded']=='start'):
                        uid=message['data'][i]['uid']
                        
                        downloadImages(uid,getImagesNames(uid),getUserName(i))
                        veryImagesDownloaded(uid,i)
                except (RuntimeError, TypeError, NameError,Exception): 
                    print("An error occured")
    else:
        
        data=message['data']        
        key=message['path']
        print(key)
        len(key)
        print(message['data'])
        try: 
            if(message['data']['imageDownloaded']=='start'):
                key=key[1:]
                print(key)
                uid=db.child("/users/"+key+'/uid').get().val()    
                print('start Download')
                downloadImages(uid,getImagesNames(uid),getUserName(key))
                veryImagesDownloaded(uid,key)    
            
            print(message)
              
        except (RuntimeError, TypeError, NameError,Exception): 
            print("An error occured")


         #   downloadImages(uid,getImagesNames(uid),getUserName(key))
          #  veryImagesDownloaded(uid,key)



            

            


def stream_handler(message):
    global g
    g+=1
    
    #print(message['data'])
    checkVeryfication(message)
    

print('Creating networks and loading parameters')
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess,  os.path.join("Models"))


        
        datadir = os.path.join("PostProccesed Images")
        dataset = facenet.get_dataset(datadir)
        paths, labels = facenet.get_image_paths_and_labels(dataset)
        print('Number of classes: %d' % len(dataset))
        print('Number of images: %d' % len(paths))

        print('Loading feature extraction model')
        modeldir =os.path.join("Models","20170511-185253","20170511-185253.pb") #'/Models/20170511-185253/20170511-185253.pb'
        facenet.load_model(modeldir)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        print(embedding_size,'hi')

def classfy():
        print('inside classfy')
        batch_size = 1000
        image_size = 160
        nrof_images = len(paths)
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
        emb_array = np.zeros((nrof_images, embedding_size))
        for i in range(nrof_batches_per_epoch):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, nrof_images)
            paths_batch = paths[start_index:end_index]
            images = facenet.load_data(paths_batch, False, False, image_size)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
        
        print('Training classifier')
        model = SVC(kernel='linear', probability=True)
        model.fit(emb_array, labels)
        print('finish')
        classifier_filename =os.path.join("Models","Classifier.pkl") #'/Models/Classifier.pkl'
        classifier_filename_exp = os.path.expanduser(classifier_filename)


        class_names = [cls.name.replace('_', ' ') for cls in dataset]
        with open(classifier_filename_exp, 'wb') as outfile:
            pickle.dump((model, class_names), outfile)


        
        print('Saved classifier model to file "%s"' % classifier_filename_exp)
       
def process_img(image,output_filename):
    img=image
    if img.ndim < 2:
        print('Unable to align "%s"' % image_path)
        text_file.write('%s\n' % (output_filename))
    else:
        if img.ndim == 2:
            img = facenet.to_rgb(img)
            print('to_rgb data dimension: ', img.ndim)
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
                offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                                    (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
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

            misc.imsave(output_filename, scaled_temp)
                            
            




my_stream = db.child("users").stream(stream_handler)

minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
margin = 44
image_size = 182

while True:
   
    print(stream_handler)
    data = input("[{}] Type exit to disconnect: ".format('?'))
    if data.strip().lower() == 'exit':
        print('Stop Stream Handler')
        if my_stream: my_stream.close()
        break

#j=json.loads(my_stream)


