import cv2
import face_recognition
import os
from tqdm import tqdm
import numpy as np
def invoke(images):
    known_encodings = []
    known_names = []
    for image_path in tqdm(images):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(image, model='cnn')
        encoding = face_recognition.face_encodings(image, boxes)
        name = image_path.split(os.path.sep)[-2]
        if len(encoding) > 0 : 
            known_encodings.append(encoding[0])
            known_names.append(name)
        encodings = {"encodings": known_encodings, "names": known_names}
        np.save('encodings.npy', encodings) 
