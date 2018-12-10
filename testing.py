import numpy as np
import cv2
import mxnet as mx
import pandas as pd
import random
import os

from statistics import mode

from keras.models import load_model

from datasets import get_labels

from inference import load_detection_model
from inference import argarr
from preprocessor import preprocess_input


detection_model_path = 'haarcascade_frontalface_default.xml'
emotion_model_path = 'model.hdf5'
emotion_labels = get_labels('fer2013')


frame_window = 10
emotion_offsets = (20, 40)


face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)


emotion_target_size = emotion_classifier.input_shape[1:3]

emotion_window = []




curdir = os.path.abspath(os.path.dirname(__file__))
#filename = 'traindata.csv'
filename = 'testcase3.csv'

filename = os.path.join(curdir,filename)
data = pd.read_csv(filename,delimiter=',',dtype='a')
labels = np.array(data['emotion'],np.int)
# print(labels,'\n',data['emotion'])
    
imagebuffer = np.array(data['pixels'])
images = np.array([np.fromstring(image,np.uint8,sep=' ') for image in imagebuffer])
del imagebuffer
num_shape = int(np.sqrt(images.shape[-1]))
images.shape = (images.shape[0],num_shape,num_shape)
cmatrix=np.zeros([7,7])
# img=images[0];cv2.imshow('test',img);cv2.waitKey(0);cv2.destroyAllWindow();exit()

#print(images.shape)
#exit()
#oo=labels[2]
#print(oo)
#exit()

#print(cmatrix[0][0])
#print(img.size)
#print(emotion_labels)
#exit()

for i in range(0,images.shape[0]-1):
	img=images[i]
	oo=labels[i]
	try:
	    gray_face = cv2.resize(img, (64,64))
	except:
	    print('resize fail')
	gray_face = preprocess_input(gray_face, True)
	gray_face = np.expand_dims(gray_face, 0)
	gray_face = np.expand_dims(gray_face, -1)
	emotion_prediction = emotion_classifier.predict(gray_face)
	emotion_probability = np.max(emotion_prediction)
	emotion_label_arg = np.argmax(emotion_prediction)
	pp=emotion_label_arg
	cmatrix[oo][pp]=cmatrix[oo][pp]+1
	#print(oo)
conmatrix=argarr(cmatrix)
print(conmatrix)
print('Total no. of images',np.sum(conmatrix))
sum=0
for i in range(0,5):
	sum=sum+conmatrix[i][i]
	#print(sum)
print('Total no. of correct emotion',sum)
print('Accuracy:',sum/np.sum(conmatrix))
#print(np.sum(conmatrix))
#exit()
#img=images[2]
#try:
#    gray_face = cv2.resize(img, (64,64))
#except:
#    print('resize fail')
#print(gray_face.size)
#gray_face = preprocess_input(gray_face, True)
#gray_face = np.expand_dims(gray_face, 0)
#gray_face = np.expand_dims(gray_face, -1)
#emotion_prediction = emotion_classifier.predict(gray_face)
#emotion_probability = np.max(emotion_prediction)
#emotion_label_arg = np.argmax(emotion_prediction)
#emotion_text = emotion_labels[emotion_label_arg]
#print(emotion_text)
#cv2.imshow('test',img)
#cv2.waitKey(0)
#cv2.destroyAllWindow()
#print(images.size)
#exit()


