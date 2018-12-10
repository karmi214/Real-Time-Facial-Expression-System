from statistics import mode

import cv2
from keras.models import load_model
import numpy as np

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

# parameters for loading data and images
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)
while True:
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image)

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        gg=gray_face
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue
        try:
            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            print(gg)
            print(y1,y2,x1,x2)
            eyes=gray_image[y1+100:y2-150, x1+50:x2-50]
            cv2.namedWindow("eyes", cv2.WINDOW_FREERATIO)
            #cv2.resizeWindow('windows1', 600,600)
            cv2.imshow("eyes",eyes)
            mouth=gray_image[y1+170:y2-30, x1+50:x2-50]
            cv2.namedWindow("mouth", cv2.WINDOW_FREERATIO)
            #cv2.resizeWindow('windows1', 600,600)
            cv2.imshow("mouth",mouth)
            #yy1=int(y1/2)
            #yy2=int(y2/2)
            #eyes=gg[int(y1+30):int(y2),int(x1):int(x2)]
            #cv2.namedWindow("windows1", cv2.WINDOW_FREERATIO)
            #cv2.imshow("windows1",eyes)
            #eyes2=gg[int(y1/1):int(y2/2),int(x1/1):int(x2/1)]
            #cv2.namedWindow("windows2", cv2.WINDOW_FREERATIO)
            #cv2.imshow("windows2",eyes2)
            #eyes3=gg[int(y1/1):int(y2/1),int(x1/1):int(x2/1)]
            #cv2.namedWindow("windows3", cv2.WINDOW_FREERATIO)
            #cv2.imshow("windows3",eyes3)
            #eyes4=gg[int(y1/1):int(y2/1),int(x1/1):int(x2/1)]
            #cv2.namedWindow("windows4", cv2.WINDOW_FREERATIO)
            #cv2.imshow("windows4",eyes4)
            #cv2.namedWindow("windows", cv2.WINDOW_FREERATIO)
            #cv2.imshow("windows",gg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            continue
        
        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue


    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
