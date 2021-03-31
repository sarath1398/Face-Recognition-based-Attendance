import cv2
import os
import time
from keras_facenet import FaceNet
from PIL import Image
import numpy as np
import keras.backend as K
from scipy.spatial.distance import cosine
import tensorflow as tf
import pandas as pd
from datetime import datetime

detector=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
count=0
start = time.time()

model1=FaceNet()
model2=model1

face_detector=model1.mtcnn()
database_embeddings=[]

excel_filename = 'Attendance - {}.xlsx'.format(datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
writer=pd.ExcelWriter(excel_filename)
db_list=os.listdir(r'{}\Database'.format(os.getcwd()))
df=pd.DataFrame({'Name':[names[:-4] for names in db_list],'Status':['Absent' for _ in range(len(db_list))]})

def extract_face(image, required_size=(160, 160)):

  image = image.convert('RGB')
  pixels = np.asarray(image)
  results = face_detector.detect_faces(pixels)
  x, y, w, h = results[0]['box']
  x1, y1 = abs(x), abs(y)
  x2, y2 = x1 + w, y1 + h
  face = pixels[y1:y2, x1:x2]
  image = Image.fromarray(face)
  image = image.resize(required_size)
  face_array = np.asarray(image)
  return face_array

def get_embeddings(model,face):

    face = np.expand_dims(face, axis=0)
    return model.embeddings(face)

def euclidean_distance(y_true, y_pred):
  return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def cosine_similarity(a, b):
  return 1-cosine(a, b)

for images in os.listdir(r'{}\Database'.format(os.getcwd())):

    img=Image.open(r'{}\Database\{}'.format(os.getcwd(),images))
    face_img=extract_face(img)
    database_embeddings.append(get_embeddings(model1,face_img))

def predict_function(face):

    index = -1
    count=0
    distance_list = []
    similarity_list = []

    face_embedded=get_embeddings(model2,face)
    
    for db_embedded in database_embeddings:

        e_distance = euclidean_distance(db_embedded, face_embedded)
        c_similarity = cosine_similarity(db_embedded, face_embedded)
        
        e_distance=tf.keras.backend.get_value(e_distance)
        c_similarity=tf.keras.backend.get_value(c_similarity)

        distance_list.append(e_distance[0])
        similarity_list.append(c_similarity)

        if e_distance < 0.9 and c_similarity>0.7:
            index = count

        count += 1

    database_list=[img[:-4] for img in os.listdir(os.getcwd()+'\Database')]
    
    if index==-1:
      return None

    return database_list[index]

def update_attendance(pred_name,df):

  ind=0
  
  for name in df['Name']:

    if pred_name==name and df['Status'][ind]=='Absent':
      df['Status'][ind]='Present'
    
    ind+=1

  return df

cap = cv2.VideoCapture(0)
webcam_end=time.time()+60*15

while time.time()<webcam_end:

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3,5,minSize=(128,128))

    for x, y, w, h in faces:

        face_rect=cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        face = frame[y:y+h, x:x+w]
        face= np.asarray(Image.fromarray(face).resize((160,160)))
        prediction=predict_function(face)
        df=update_attendance(prediction,df)
        cv2.putText(face_rect,prediction,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)
        
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

df.to_excel(writer)
writer.save()