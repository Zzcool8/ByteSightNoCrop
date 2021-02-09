from flask import Flask, request
from flask import Blueprint, jsonify
import numpy as np
import tensorflow as tf
import onnxruntime
from thresholdingfunction import otsuthresholding
#from classify import load_model, load_image_fromnumpy, predict_single
from torchvision import datasets, models, transforms
import torch
import os
from blackandwhiteratios import blackandwhiteratio
from boundingbox import cropImage

from helpers import (load_image, make_square, 
                     augment, pre_process, softmax)
from helper_config import (IMG_HEIGHT, IMG_WIDTH, CLASS_MAP,
                           CHANNELS)

# Usually helps in debugging
print(tf.__version__) # Print the version of tensorflow being used

app = Flask(__name__)

moz = Blueprint('moz', __name__)

prep = pre_process(IMG_WIDTH, IMG_HEIGHT)

@moz.route("/get_label", methods=['GET', 'POST'])
def get_label():
    inf_file = request.files.get('image').read()
    print("Got the file")
    label = run_inference(inf_file)
    return jsonify({
        "genus": label[0],
        "species": label[1],
        "confidence_score": label[2],
        "color_code": label[3]
    })


def color_code(num):
    if(float(num)>0.9):
        return '#4bf542'
    elif (float(num)>0.7):
        return '#f7b17e'
    else:
        return '#f7543b'

def run_inference(inf_file):
    # Preprocessing of the image happens here
    useless, img, status=cropImage(impath, 'm', labelsfile, 21, 0.08)
    print("Image Loaded")
    img = make_square(img)
    img = augment(prep, img)
    print("Transformations done")
    img = img.transpose(-1, 0, 1).astype(np.float32)
    img = img.reshape(-1, CHANNELS, IMG_WIDTH, IMG_HEIGHT)

    # Inferencing starts here
    sess = onnxruntime.InferenceSession("./best_acc.onnx")
    print("The model expects input shape: ", sess.get_inputs()[0].shape)
    print("The shape of the Image is: ", img.shape)
    input_name = sess.get_inputs()[0].name

    result = sess.run(None, {input_name: img})
    prob_array = result[0][0]
    print("Prob Array ", prob_array)
    prob = max(softmax(result[0][0]))
    print("Prob ",prob)
    species = tf.argmax(prob_array.ravel()[:20]).numpy()
    print("Class Label ", species)
    print("Spec ", CLASS_MAP[species][1])
    string_label = CLASS_MAP[species][1].split(" ")
    return (string_label[0], string_label[1], str(prob), color_code(prob))

app.register_blueprint(moz)
