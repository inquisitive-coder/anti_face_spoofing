from train_model import load_pretrained_model
import os
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict
def invoke():
    dataset = 'faces'
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    open_eyes_detector = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
    left_eye_detector = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
    right_eye_detector = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
    model = load_pretrained_model()
    images = []
    for direc, _, files in tqdm(os.walk(dataset)):
        for file in files:
            if file.endswith("jpeg") or file.endswith("jpg") or file.endswith("png"):
                images.append(os.path.join(direc,file))
    return (model,face_detector, open_eyes_detector, left_eye_detector,right_eye_detector, images)