from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import dlib
from imutils import face_utils
import imutils
from PIL import ImageFont, ImageDraw, Image
import os
from random import choice

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
args = vars(ap.parse_args())

# haar
face_cascade = cv2.CascadeClassifier('/home/wanchat/Python/data'
                        '/haarcascades/haarcascade_frontalface_default.xml')
# dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/wanchat/Python/data'
                        '/shape_predictor_68_face_landmarks.dat')

# load the image
path = args["image"]
imagePath = [os.path.join(path,f) for f in os.listdir(path)]
image_choice = choice(imagePath)
image = cv2.imread(image_choice)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face = face_cascade.detectMultiScale(gray, 1.3, 5)

rects = detector(gray, 1)

orig = image.copy()

for (i, rect) in enumerate(rects):
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    xy = []
    for num ,(x,y) in enumerate (shape):
        xy.append((x,y))

    # eyeLeft
    eyeLeft = xy[22], xy[23], xy[26], xy[46]
    eyeLeft_1, eyeLeft_2, eyeLeft_3, eyeLeft_4 = eyeLeft

    eyeLeft_1_x, eyeLeft_1_y = eyeLeft_1
    eyeLeft_2_x, eyeLeft_2_y = eyeLeft_2
    eyeLeft_3_x, eyeLeft_3_y = eyeLeft_3
    eyeLeft_4_x, eyeLeft_4_y = eyeLeft_4

    # eyeRight
    eyeRight =  xy[17], xy[19], xy[21], xy[40]
    eyeRight_1, eyeRight_2, eyeRight_3, eyeRight_4 = eyeRight

    eyeRight_1_x, eyeRight_1_y = eyeRight_1
    eyeRight_2_x, eyeRight_2_y = eyeRight_2
    eyeRight_3_x, eyeRight_3_y = eyeRight_3
    eyeRight_4_x, eyeRight_4_y = eyeRight_4

    # nose
    nose = xy[30], xy[31], xy[35], xy[33]
    nose_1, nose_2, nose_3, nose_4 = nose

    nose_1_x, nose_1_y = nose_1
    nose_2_x, nose_2_y = nose_2
    nose_3_x, nose_3_y = nose_3
    nose_4_x, nose_4_y = nose_4

    # mouth
    mouth = xy[48], xy[50], xy[54], xy[57]
    mouth_1, mouth_2, mouth_3, mouth_4 = mouth

    mouth_1_x, mouth_1_y = mouth_1
    mouth_2_x, mouth_2_y = mouth_2
    mouth_3_x, mouth_3_y = mouth_3
    mouth_4_x, mouth_4_y = mouth_4

    # image
    eyeLeft_image = image[eyeLeft_2_y - 10 : eyeLeft_4_y + 30, eyeLeft_1_x - 10 : eyeLeft_3_x + 10]
    eyeRight_image = image[eyeRight_2_y - 10 : eyeRight_4_y + 30, eyeRight_1_x - 10 : eyeRight_3_x + 10]
    nose_image = image[nose_1_y - 20 : nose_4_y + 20, nose_2_x - 15 : nose_3_x + 15]
    mouth_image = image[mouth_1_y - 30 : mouth_4_y + 10, mouth_1_x - 10 : mouth_3_x + 10]

    for (fx,fy,fw,fh) in face:
        cv2.rectangle(image, (fx,fy), (fx+fw, fy+fh), (0,255,0), 2)
        face_n = image[fy:fy+fh, fx:fx+fw]

       # pre-process the image for classification
       # face
        face_n = cv2.resize(face_n, (28, 28))
        face_n = face_n.astype("float") / 255.0
        face_n = img_to_array(face_n)
        face_n = np.expand_dims(face_n, axis=0)

        # eyeRight
        eyeRight_image = cv2.resize(eyeRight_image, (28, 28))
        eyeRight_image = eyeRight_image.astype("float") / 255.0
        eyeRight_image = img_to_array(eyeRight_image)
        eyeRight_image = np.expand_dims(eyeRight_image, axis=0)

        # eyeLeft
        eyeLeft_image = cv2.resize(eyeLeft_image, (28, 28))
        eyeLeft_image = eyeLeft_image.astype("float") / 255.0
        eyeLeft_image = img_to_array(eyeLeft_image)
        eyeLeft_image = np.expand_dims(eyeLeft_image, axis=0)

        # nose
        nose_image = cv2.resize(nose_image, (28, 28))
        nose_image = nose_image.astype("float") / 255.0
        nose_image = img_to_array(nose_image)
        nose_image = np.expand_dims(nose_image, axis=0)

        # mouth
        mouth_image = cv2.resize(mouth_image, (28, 28))
        mouth_image = mouth_image.astype("float") / 255.0
        mouth_image = img_to_array(mouth_image)
        mouth_image = np.expand_dims(mouth_image, axis=0)

        # load the trained convolutional neural network
        print("[INFO] loading image...")

        model_face = load_model("/home/wanchat/Python/THESIS/man/woman_test_d.model")
        model_eyeRight = load_model("/home/wanchat/Python/THESIS/man/eyeright_gender.model")
        model_eyeLeft = load_model("/home/wanchat/Python/THESIS/man/eyeleft_gender.model")
        model_nose = load_model("/home/wanchat/Python/THESIS/man/nose_gender.model")
        model_mouth = load_model("/home/wanchat/Python/THESIS/man/mouth_gender.model")

        # classify the input image
        (man, woman) = model_face.predict(face_n)[0]
        (man_eyeright, woman_eyeright) = model_eyeRight.predict(eyeRight_image)[0]
        (man_eyeleft, woman_eyeleft) = model_eyeLeft.predict(eyeLeft_image)[0]
        (man_nose, woman_nose) = model_nose.predict(nose_image)[0]
        (man_mouth, woman_mouth) = model_mouth.predict(mouth_image)[0]

        # build the label
        proba_face  = woman if woman > man else man
        proba_eyeright  = woman_eyeright if woman_eyeright > man_eyeright else man_eyeright
        proba_eyeleft  = woman_eyeleft if woman_eyeleft > man_eyeleft else man_eyeleft
        proba_nose  = woman_nose if woman_nose > man_nose else man_nose
        proba_mouth  = woman_mouth if woman_mouth > man_mouth else man_mouth

        label_face  = "face woman" if woman > man else "face man"
        label_eyeright  = "eyeright woman" if woman_eyeright > man_eyeright else "eyeright man"
        label_eyeleft  = "eyeleft woman" if woman_eyeleft > man_eyeleft else "eyeleft man "
        label_nose  = "nose woman" if woman_nose > man_nose else "nose man"
        label_mouth  = "mouth woman " if woman_mouth > man_mouth else "mouth man"



        gender_w = (woman+woman_eyeright+woman_eyeleft+woman_nose+woman_mouth)/5
        gender_m = (man+man_eyeright+man_eyeleft+man_nose+man_mouth)/5



        # label = "woman" if gender_w > gender_m else "man"
        label = "woman" if gender_w > gender_m else "man"
        proba = woman if gender_w > gender_m else man

        proba_face = "{:.2f}%".format(proba_face*100)
        proba_eyeright = "{:.2f}%".format(proba_eyeright*100)
        proba_eyeleft = "{:.2f}%".format(proba_eyeleft*100)
        proba_nose = "{:.2f}%".format(proba_nose*100)
        proba_mouth = "{:.2f}%".format(proba_mouth*100)
        proba = "{:.2f}%".format(proba*100)

        print('-'*20)
        print(" file : {}".format(image_choice))
        print('-'*20)
        print(" {} : {}".format(label_face, proba_face))
        print(" {} : {}".format(label_eyeright, proba_eyeright))
        print(" {} : {}".format(label_eyeleft, proba_eyeleft))
        print(" {} : {}".format(label_nose, proba_nose))
        print(" {} : {}".format(label_mouth, proba_mouth))
        print('-'*20)
        print(" {} : {}".format(label, proba))
        print('-'*20)

        # draw text
        cv2.rectangle(image, (fx,fy+fh), (fx+75,fy+fh+45), (0,255,0), -1)
        fontpath = "/home/wanchat/Python/data/font/Roboto/Roboto-Medium.ttf"
        font = ImageFont.truetype(fontpath, 22)
        font2 = ImageFont.truetype(fontpath, 20)
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        draw.text((fx+5, fy+fh), label, font=font2, fill=(255, 255, 255, 0))
        draw.text((fx+5, fy+fh+20), proba, font=font2, fill=(255, 255, 255, 0))
        image = np.array(img_pil)
        cv2.rectangle(image, (fx,fy), (fx+fw, fy+fh), (0,255,0), 2)

    # show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
