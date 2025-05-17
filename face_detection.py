# face recognition part II
#IMPORT
import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN

#INITIALIZE
detector = MTCNN()

facenet = FaceNet()
faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)

model = pickle.load(open("svm_f_model_160x160.pkl", 'rb'))

#CAM
cap = cv.VideoCapture(0)

while cap.isOpened():
    _, frame = cap.read()
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = detector.detect_faces(rgb_img)

    for face in faces:
        x, y, w, h = face['box']
        img = rgb_img[y:y+h, x:x+w]
        img = cv.resize(img, (160,160)) # 1x160x160x3
        img = np.expand_dims(img,axis=0)
        ypred = facenet.embeddings(img)
        face_name = model.predict(ypred)
        # final_name = encoder.inverse_transform(face_name)[0]
        pred_label = face_name[0]
        if pred_label in encoder.classes_:
            final_name = encoder.inverse_transform([pred_label])[0]
        else:
            final_name = "face detected"

        cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 10)
        cv.putText(frame, str(final_name), (x,y-10), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0,0,255), 3, cv.LINE_AA)

    cv.imshow("Face Recognition:", frame)
    # if cv.waitKey(1) & ord('q') == 27:
    if cv.waitKey(25) == ord('q'):
        break

#release
cap.release()
cv.destroyAllWindows