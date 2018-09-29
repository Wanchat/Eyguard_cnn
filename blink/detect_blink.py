import cv2
from extend_eye import Extand_eyes
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils

class Blink_cnn:
    def __init__(self, model_blink):
        self.model_blink = model_blink


