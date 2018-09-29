from imutils import face_utils
import dlib
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import time
from graphic.text import  text
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default=r"D:\code_python\Eyguard_cnn\age\age_50.model",
    help="path model")
ap.add_argument("-c", "--camera", default=1,
    help="index camera")
args = vars(ap.parse_args())

cap = cv2.VideoCapture(0)

face_detect = cv2.CascadeClassifier(r"D:\code_python\data\haarcascade_frontalface_default.xml")
# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])

time.sleep(2.0)

while True:
    _, frame = cap.read()
    orig = frame.copy()
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    face = face_detect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in face:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (255, 255, 255), 2)
        face_n = orig[y:y + h, x:x + w]

    # pre-process the image for classification
        face_n = cv2.resize(face_n, (28, 28))
        face_n = face_n.astype("float") / 255.0
        face_n = img_to_array(face_n)
        face_n = np.expand_dims(face_n, axis=0)

    # classify the input image
        (under, older) = model.predict(face_n)[0]

    # build the label
        label = "older40" if older > under else "under40"
        score = older if older > under else under
        label_text = "{}: {:.0f}%".format(label, score * 100)

    # text display
        fontPath = r"D:\code_python\data\Prompt-Regular.ttf"

        if label == "older40":
            orig = text(orig, (x, y+h+7), fontPath, label_text, 20,(102,0,255))
        else:
            orig = text(orig, (x, y+h+7), fontPath, label_text, 20,(255,102,102))


        print("estimate.....{}, {:.2f}".format(label, score))

    cv2.imshow("Frame", orig)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

