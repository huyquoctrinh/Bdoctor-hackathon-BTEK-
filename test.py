from flask import Flask, request, render_template
import cv2
from tensorflow.keras.models import load_model
from Detector import Detector
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from predict import detection_on_image
image_path= "norm.jpeg"
kq= detection_on_image(image_path)
x= kq[0]
print(x)