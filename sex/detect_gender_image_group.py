from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
from PIL import ImageFont, ImageDraw, Image

ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", required=True,
#     help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
args = vars(ap.parse_args())

face_cascade = cv2.CascadeClassifier('/home/wanchat/Python/data'
                        '/haarcascades/haarcascade_frontalface_default.xml')

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
face = face_cascade.detectMultiScale(gray, 1.3, 5)
num_image = 0

print(" code start ")
print('*'*40)

for (x,y,w,h) in face:
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
    face_n = image[y:y+h, x:x+w]

    face_n = cv2.resize(face_n, (28, 28))
    face_n = face_n.astype("float") / 255.0
    face_n = img_to_array(face_n)
    face_n = np.expand_dims(face_n, axis=0)

    num_image += 1
    print("[INFO] loading image...{}".format(num_image))

    model = load_model("/home/wanchat/Python/THESIS/model/woman_test_d.model")

    (man, woman) = model.predict(face_n)[0]

    label = "WOMAN" if woman > man else "MAN"
    proba = woman if woman > man else man
    proba = "{:.2f}%".format(proba*100)

    fontpath = "/home/wanchat/Python/data/font/Roboto/Roboto-Medium.ttf"
    font = ImageFont.truetype(fontpath, 22)
    font2 = ImageFont.truetype(fontpath, 20)
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x, y+h+5), label, font=font2, fill=(0, 255, 0, 0))
    draw.text((x, y+h+25), proba, font=font, fill=(0, 255, 0, 0))
    image = np.array(img_pil)

print('*'*40)
print(" face image total  {}\n Start Detect Gender\n Goodluck...".format(num_image))

cv2.imshow("IMAGE", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('*'*40)
print(' Thank you for using')
