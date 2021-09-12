
import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

import tensorflow as tf

def get_session():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.compat.v1.Session(config=config)
def load_model():
	keras.backend.tensorflow_backend.set_session(get_session())
	model_path = 'modelblur_inf.h5'    ## replace this with your model path
	model = models.load_model(model_path, backbone_name='resnet50')
	return model
labels_to_names = {0: 'blur'}                    ## replace with your model labels and its index value

# image_path = 'bth.jpg'  ## replace with input image path
# output_path = '1_out.tif'   ## replace with output image path

Count = 0

def detection_on_image(image_path, labels_to_names= {0:'blur'}):
        global Count
        image = cv2.imread(image_path)

        draw = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = preprocess_image(image)
        image, scale = resize_image(image)
        model= load_model()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes /= scale

        def isPath(x, y):
            if (max(x[0], y[0]) <= min(x[2], y[2]) and max(x[1], y[1]) <= min(x[3], y[3])):
                return True
            return False
        
        vector = []
        
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if score < 0.4:
                break
            vector.append(box)
        
        b = [0 for _ in range(len(vector) + 1)]
        d = [[] for _ in range(len(vector) + 10)]
        
        def DFS(u):
            global Count
            b[u] = Count
            d[Count].append(u)
            for v in g[u]:
                if (b[v] == 0):
                    DFS(v)

        g = [[] for _ in range(len(vector) + 1)]
        for i in range(len(vector)):
            for j in range(i + 1, len(vector)):
                if (isPath(vector[i], vector[j])):
                    g[i].append(j)
                    g[j].append(i)

        for i in range(len(vector)):
            if (b[i] == 0):
                Count += 1
                DFS(i)
        
        vectorBoxes = []
        vectorScores = np.array([])
        vectorLabels = np.array([])
        
        for i in range(1, Count + 1):
            minx = 1000000
            maxx = -1000000
            miny = 1000000
            maxy = -1000000
            maxScore = 0
            label = ""
            for j in d[i]:
                label = labels[0][j]
                minx = min(minx, vector[j][0])
                miny = min(miny, vector[j][1])
                maxx = max(maxx, vector[j][2])
                maxy = max(maxy, vector[j][3])
                maxScore = max(maxScore, scores[0][j])
            vectorBoxes.append([minx, miny, maxx, maxy])
            vectorScores = np.append(vectorScores, maxScore)
            vectorLabels = np.append(vectorLabels, label)
        
        for box, score in zip(vectorBoxes, vectorScores):
            box = np.array(box)
            label = 0
            if score < 0.4:
                break

            color = label_color(label)
            b = box.astype(int)
            # draw_box(draw, b, color=color)
            caption = "{} {:.3f}".format(labels_to_names[label], score)
            # draw_caption(draw, b, caption)
        return color,b,caption
        # detected_img =cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)

        # cv2.imwrite(output_path, detected_img)

        # cv2.imshow('Detection',detected_img)
        # cv2.waitKey(0)
# detection_on_image(image_path)