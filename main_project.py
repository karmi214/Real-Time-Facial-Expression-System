from statistics import mode

import cv2
from keras.models import load_model
import numpy as np

from datasets import get_labels
from inference import detect_faces
from inference import draw_text
from inference import draw_bounding_box
from inference import apply_offsets
from inference import load_detection_model
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

#video ko part
cv2.namedWindow('Face expression detection',cv2.WINDOW_NORMAL)
#cv2.namedWindow('Face expression detection')
#cv2.resizeWindow('Face expression detection', 200,200)

video_capture = cv2.VideoCapture(0)
cimage = video_capture.read()[1]
height, width = cimage.shape[:2]
#print(height,width)
cv2.resizeWindow('Face expression detection', width, height)
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
        #print(gray_face.size)
        #cv2.imshow("windows",gg)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        #print(gray_face)
        #cv2.imshow("windows",gg)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 0))
        else:
            color = emotion_probability * np.asarray((0, 0, 0))
        #color = np.asarray((255, 255, 255))
        text = emotion_text
        #text='sad'
        color = color.astype(int)
        color = color.tolist()
        #text=text + " " + str(round(emotion_probability,2))
        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, text,
                  color, 0, -45, 1, 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    cv2.imshow('Face expression detection', bgr_image)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
