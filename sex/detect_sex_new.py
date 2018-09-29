import dlib
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np


class Age:
    def __init__(self, model=r"D:\code_python\Eyguard_cnn\sex\sex_25.model"):
        self.madel_age = load_model(model)

    def estimate(self, window_face):
        self.window_face = cv2.resize(window_face, (28, 28))
        self.window_face = face_n.astype("float") / 255.0
        self.window_face = img_to_array(self.window_face)
        self.window_face = np.expand_dims(self.window_face, axis=0)

        (self.under, self.older) = self.madel_age.predict(self.window_face)[0]

        self.label_sex = "woman" if self.woman > self.man else "man"
        self.score_sex = self.woman if self.woman > self.man else self.man
        return self.label_sex, self.score_sex