
import cv2
from extend_eye import Extand_eyes
import argparse
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils

eyes = Extand_eyes()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
    help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
orig = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

eye = eyes.extend(gray)

rightEye = eye['rightEye']
leftEye = eye['leftEye']

right_0_x, right_0_y = rightEye[0]
right_1_x, right_1_y = rightEye[1]
right_3_x, right_3_y = rightEye[3]
right_4_x, right_4_y = rightEye[4]

left_0_x, left_0_y = leftEye[0]
left_1_x, left_1_y = leftEye[1]
left_3_x, left_3_y = leftEye[3]
left_4_x, left_4_y = leftEye[4]

window_right = image[
               right_1_y - 5: right_4_y + 5,
               right_0_x - 5: right_3_x + 5]

window_left = image[
               left_1_y - 5: left_4_y + 5,
               left_0_x - 5: left_3_x + 5]

# pre-process the image for classification
window_right = cv2.resize(window_right, (28, 28))
window_right = window_right.astype("float") / 255.0
window_right = img_to_array(window_right)
window_right = np.expand_dims(window_right, axis=0)

window_left = cv2.resize(window_left, (28, 28))
window_left = window_left.astype("float") / 255.0
window_left = img_to_array(window_left)
window_left = np.expand_dims(window_left, axis=0)

# # load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])
#
# # classify the input image
(closedeyes_right, openeyes_right) = model.predict(window_right)[0]
(closedeyes_left, openeyes_left) = model.predict(window_left)[0]
#
# # build the label
label = "openeyes_right" if openeyes_right > closedeyes_right else \
    "closedeyes_right"
score = openeyes_right if openeyes_right > closedeyes_right else \
    closedeyes_right


label_2 = "openeyes_left" if openeyes_left > closedeyes_left else \
    "closedeyes_left"
score_2 = openeyes_left if openeyes_left > closedeyes_left else closedeyes_left


if label == "openeyes_right" and label_2 == "openeyes_left":
    eye_predict = "openeyes"
else:
    eye_predict = "closedeyes"

label = "{}: {:.2f}%".format(label, score * 100)
label_2 = "{}: {:.2f}%".format(label_2, score_2 * 100)

# # draw the label on the image
output = imutils.resize(orig, width=400)
cv2.putText(output, eye_predict, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
    0.5, (0, 255, 0), 2)

cv2.putText(output, label, (10, 40),  cv2.FONT_HERSHEY_SIMPLEX,
    0.5, (0, 255, 0), 2)

cv2.putText(output, label_2, (10, 60),  cv2.FONT_HERSHEY_SIMPLEX,
    0.5, (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
