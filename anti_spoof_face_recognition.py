# -*- coding: utf-8 -*-
"""Face_recognition.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jdFqlm_VPvZECTHPI-S6r1L1ALiawVlx
"""

# !pip install Pillow

from ctypes import resize
import os
from PIL import Image
import numpy as np

from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator

from matplotlib.pyplot import imread
from matplotlib.pyplot import  imsave
# !wget https://raw.githubusercontent.com/niravnb/Anti-Spoofing-Facial-Recognition/master/dataset.zip

# !unzip dataset.zip

IMG_SIZE = 24

def collect():
	train_datagen = ImageDataGenerator(
			rescale=1./255,
			shear_range=0.2,
			horizontal_flip=True, 
		)

	val_datagen = ImageDataGenerator(
			rescale=1./255,
			shear_range=0.2,
			horizontal_flip=True,		)

	train_generator = train_datagen.flow_from_directory(
	    directory="dataset/train",
	    target_size=(IMG_SIZE, IMG_SIZE),
	    color_mode="grayscale",
	    batch_size=32,
	    class_mode="binary",
	    shuffle=True,
	    seed=42
	)

	val_generator = val_datagen.flow_from_directory(
	    directory="dataset/val",
	    target_size=(IMG_SIZE, IMG_SIZE),
	    color_mode="grayscale",
	    batch_size=32,
	    class_mode="binary",
	    shuffle=True,
	    seed=42
	)
	return train_generator, val_generator

def save_model(model):
  model.save('eye_status_classifier.h5')

from tensorflow.keras.models import load_model

def load_pretrained_model():
    model = load_model('eye_status_classifier.h5')
    model.summary()
    return model
    

def hello(img):
    temp=np.asarray(img,dtype=np.float32)
    temp/=255
    temp=np.reshape(temp,(1,IMG_SIZE,IMG_SIZE,1))
    print("shape=\n\n\n\n",temp.shape)
    return temp
    #return Image.fromarray(temp, 'RGB').convert('L')
def predict(img, model):
	img = Image.fromarray(img, 'RGB').convert('L')
	img =  img.resize((IMG_SIZE,IMG_SIZE),Image.BICUBIC)
	img =hello(img)
	prediction = model.predict(img)
	print("\nprediction=",prediction,"\n")
	if prediction < 0.2:
		prediction = 'closed'
	elif prediction > 0.7:
		prediction = 'open'
	else:
		prediction = 'idk'
	return prediction

def evaluate(X_test, y_test):
	model = load_model()
	print('Evaluate model')
	loss, acc = model.evaluate(X_test, y_test, verbose = 0)
	print(acc * 100)

# train_generator , val_generator = collect()

# train(train_generator,val_generator)



# !sudo apt install cmake

# !pip3 install face_recognition

import os
import cv2
import face_recognition
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from imutils.video import VideoStream

def init():
    face_cascPath = 'haarcascade_frontalface_alt.xml'
    # face_cascPath = 'lbpcascade_frontalface.xml'

    open_eye_cascPath = 'haarcascade_eye_tree_eyeglasses.xml'
    left_eye_cascPath = 'haarcascade_lefteye_2splits.xml'
    right_eye_cascPath ='haarcascade_righteye_2splits.xml'
    dataset = 'faces'

    face_detector = cv2.CascadeClassifier(face_cascPath)
    open_eyes_detector = cv2.CascadeClassifier(open_eye_cascPath)
    left_eye_detector = cv2.CascadeClassifier(left_eye_cascPath)
    right_eye_detector = cv2.CascadeClassifier(right_eye_cascPath)

    model = load_pretrained_model()


    print("[LOG] Collecting images ...")
    images = []
    for direc, _, files in tqdm(os.walk(dataset)):
        for file in files:
            print("file=",file)
            if file.endswith("jpeg") or file.endswith("jpg") or file.endswith("png"):
                images.append(os.path.join(direc,file))
    return (model,face_detector, open_eyes_detector, left_eye_detector,right_eye_detector, images)

def process_and_encode(images):
    # initialize the list of known encodings and known names
    known_encodings = []
    known_names = []
    print("[LOG] Encoding faces ...")
    print("hello world 1")
    for image_path in tqdm(images):
        # Load image
        image = cv2.imread(image_path)
        # Convert it from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
     
        # detect face in the image and get its location (square boxes coordinates)
        boxes = face_recognition.face_locations(image, model='cnn')

        # Encode the face into a 128-d embeddings vector
        encoding = face_recognition.face_encodings(image, boxes)

        # the person's name is the name of the folder where the image comes from
        name = image_path.split(os.path.sep)[-2]
        print("hello world 1")
        if len(encoding) > 0 : 
            known_encodings.append(encoding[0])
            known_names.append(name)
        encodings = {"encodings": known_encodings, "names": known_names}
        np.save('encodings.npy', encodings) 
def isBlinking(history, maxFrames):
    """ @history: A string containing the history of eyes status 
         where a '1' means that the eyes were closed and '0' open.
        @maxFrames: The maximal number of successive frames where an eye is closed """
    for i in range(maxFrames):
        pattern = '1' + '0'*(i+1) + '1'
        if pattern in history:
            return True
    return False

def detect_and_display(model, video_capture, face_detector, open_eyes_detector, left_eye_detector, right_eye_detector, data, eyes_detected):
        frame = video_capture.read()
        # resize the frame
        # frame = cv2.resize(frame, (0, 0), fx=0.9, fy=0.9)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # for each detected face
        for (x,y,w,h) in faces:
            # Encode the face into a 128-d embeddings vector
            encoding = face_recognition.face_encodings(rgb, [(y, x+w, y+h, x)])[0]

            # Compare the vector with all known faces encodings
            matches = face_recognition.compare_faces(data["encodings"], encoding)

            # For now we don't know the person name
            name = "Unknown"

            # If there is at least one match:
            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number of votes
                name = max(counts, key=counts.get)

            face = frame[y:y+h,x:x+w]
            gray_face = gray[y:y+h,x:x+w]

            eyes = []
            
            # Eyes detection
            # check first if eyes are open (with glasses taking into account)
            open_eyes_glasses = open_eyes_detector.detectMultiScale(
                gray_face,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags = cv2.CASCADE_SCALE_IMAGE
            )
            # if open_eyes_glasses detect eyes then they are open 
            if len(open_eyes_glasses) == 2:
                eyes_detected[name]+='1'
                for (ex,ey,ew,eh) in open_eyes_glasses:
                    cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
            # otherwise try detecting eyes using left and right_eye_detector
            # which can detect open and closed eyes                
            else:
                # separate the face into left and right sides
                left_face = frame[y:y+h, x+int(w/2):x+w]
                left_face_gray = gray[y:y+h, x+int(w/2):x+w]

                right_face = frame[y:y+h, x:x+int(w/2)]
                right_face_gray = gray[y:y+h, x:x+int(w/2)]

                # Detect the left eye
                left_eye = left_eye_detector.detectMultiScale(
                    left_face_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags = cv2.CASCADE_SCALE_IMAGE
                )

                # Detect the right eye
                right_eye = right_eye_detector.detectMultiScale(
                    right_face_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags = cv2.CASCADE_SCALE_IMAGE
                )

                eye_status = '1' # we suppose the eyes are open

                # For each eye check wether the eye is closed.
                # If one is closed we conclude the eyes are closed
                for (ex,ey,ew,eh) in right_eye:
                    color = (0,255,0)
                    pred = predict(right_face[ey:ey+eh,ex:ex+ew],model)
                    if pred == 'closed':
                        eye_status='0'
                        color = (0,0,255)
                    cv2.rectangle(right_face,(ex,ey),(ex+ew,ey+eh),color,2)
                for (ex,ey,ew,eh) in left_eye:
                    color = (0,255,0)
                    pred = predict(left_face[ey:ey+eh,ex:ex+ew],model)
                    if pred == 'closed':
                        eye_status='0'
                        color = (0,0,255)
                    cv2.rectangle(left_face,(ex,ey),(ex+ew,ey+eh),color,2)
                eyes_detected[name] += eye_status

            # Each time, we check if the person has blinked
            # If yes, we display its name
            # if len(eyes_detected[name]) < 3:
            #     cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            #     # Display name
            #     y = y - 15 if y - 15 > 15 else y + 15
            #     cv2.putText(frame, 'Processing if '+name+' is real or fake', (x, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (255, 0, 0), 2)
            # else:    
            if isBlinking(eyes_detected[name],3):
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Display name
                y = y - 15 if y - 15 > 15 else y + 15
                cv2.putText(frame, 'Real: '+name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
                # eyes_detected[name] = '111'
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                # Display name
                y = y - 15 if y - 15 > 15 else y + 15
                cv2.putText(frame, 'Fake: '+name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 0, 255), 2)


        return frame


(model, face_detector, open_eyes_detector,left_eye_detector,right_eye_detector, images) = init()

print("images= ",images)

#process_and_encode(images)

data = np.load('encodings.npy',allow_pickle='TRUE').item()


print("[LOG] Opening webcam ...")
video_capture = VideoStream(src=0).start()

eyes_detected = defaultdict(str)
while True:
  frame = detect_and_display(model, video_capture, face_detector, open_eyes_detector,left_eye_detector,right_eye_detector, data, eyes_detected)
  cv2.imshow("Liveness Detection for Facial Recognition", frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cv2.destroyAllWindows()
video_capture.stop()


