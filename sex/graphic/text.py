from PIL import ImageFont, ImageDraw, Image
import numpy as np

def text(frame, roi, fontpath, text, size, color):
    font = ImageFont.truetype(fontpath, size)
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    draw.text(roi, "{}".format(text), font=font, fill=color)
    return  np.array(img_pil)

