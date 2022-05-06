

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

from train_model import train
import getStarted
import processImages
import predictResult 


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




def evaluate(X_test, y_test):
	model = load_model()
	print('Evaluate model')
	loss, acc = model.evaluate(X_test, y_test, verbose = 0)
	print(acc * 100)

#train_generator , val_generator = collect()

#train(train_generator,val_generator)


import os
import cv2
import face_recognition
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from imutils.video import VideoStream




(model, face_detector, open_eyes_detector,left_eye_detector,right_eye_detector, images) =getStarted.invoke()

print("images= ",images)

#processImages.invoke(images)
data = np.load('encodings.npy',allow_pickle='TRUE').item()

video_capture = VideoStream(src=0).start()

eyes_detected = defaultdict(str)
while True:
  frame = predictResult.invoke(model, video_capture, face_detector, open_eyes_detector,left_eye_detector,right_eye_detector, data, eyes_detected)
  cv2.imshow("Liveness", frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cv2.destroyAllWindows()
video_capture.stop()


