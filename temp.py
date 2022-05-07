from train_model import load_pretrained_model
import os
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
from numpy import asarray
IMG_SIZE = 24
model = load_pretrained_model()
dataset_open = 'dataset/val/open'
dataset_closed='dataset/val/closed'
images1 = []
images2 = []

def helper(img):
    temp=np.asarray(img,dtype=np.float32)
    temp/=255
    temp=np.reshape(temp,(1,IMG_SIZE,IMG_SIZE,1))
    return temp
    
def predict(img, model):
   img = Image.fromarray(img, 'L')
   img =  img.resize((IMG_SIZE,IMG_SIZE),Image.BICUBIC)
   img = helper(img)
   prediction = model.predict(img)
   if prediction < 0.2:
      prediction = 'closed'
   elif prediction > 0.60:
      prediction = 'open'
   else:
      prediction = 'idk'
   return prediction

for direc, _, files in tqdm(os.walk(dataset_open)):
  for file in files:
      if file.endswith("jpeg") or file.endswith("jpg") or file.endswith("png"):
          images1.append(os.path.join(direc,file))

for direc, _, files in tqdm(os.walk(dataset_closed)):
  for file in files:
      if file.endswith("jpeg") or file.endswith("jpg") or file.endswith("png"):
          images2.append(os.path.join(direc,file))
right_count=0
for img_path in images1:
   im = asarray(Image.open(img_path))
   if predict(im,model) =='open':
      right_count+=1

for img_path in images2:
   im = asarray(Image.open(img_path))
   if predict(im,model) =='closed':
      right_count+=1
print("accuracy = ",right_count/(len(images1)+len(images2)))



