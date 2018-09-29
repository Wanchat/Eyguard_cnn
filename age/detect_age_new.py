import dlib
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np

class Age:
    def __init__(self,model=r"D:\code_python\Eyguard_cnn\age\age_50.model"):
        self.madel_age = load_model(model)

    def estimate(self, window_face):
        self.window_face = cv2.resize(window_face, (28, 28))
        self.window_face = face_n.astype("float") / 255.0
        self.window_face = img_to_array(self.window_face)
        self.window_face = np.expand_dims(self.window_face, axis=0)

        (self.under, self.older) = self.madel_age.predict(self.window_face)[0]

        self.label_age = "older40" if self.older > self.under else "under40"
        self.score_age = self.older if self.older > self.under else self.under
        return  self.label_age, self.score_age