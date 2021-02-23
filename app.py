from flask import Flask, request
from flask import Blueprint, jsonify
import numpy as np
import tensorflow as tf
import onnxruntime
import cv2
#import boto3
#from botocore.exceptions import NoCredentialsError
##import logging
##import boto3
##from botocore.exceptions import ClientError
#import statement for crop, SHOULD NOT BE INCLUDED IN THIS VERSION
##from thresholdingfunction import otsuthresholding
#from classify import load_model, load_image_fromnumpy, predict_single
##from torchvision import datasets, models, transforms
##import torch
###import os
###from blackandwhiteratios import blackandwhiteratio
###from boundingbox import cropImage

from helpers import (load_image, make_square, 
                     augment, pre_process, softmax)
from helper_config import (IMG_HEIGHT, IMG_WIDTH, CLASS_MAP,
                           CHANNELS)

ACCESS_KEY = 'AKIAJA6OT4ISIU6Q4PBA'
SECRET_KEY = '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'

# Usually helps in debugging
print(tf.__version__) # Print the version of tensorflow being used

app = Flask(__name__)

moz = Blueprint('moz', __name__)

prep = pre_process(IMG_WIDTH, IMG_HEIGHT)

@moz.route("/get_label", methods=['GET', 'POST'])
def get_label():
    inf_file = request.files.get('image').read()
    MosqID = request.files.get('MosquitoID').read()
    PicNum = request.files.get('PictureNumber').read()
    print("Got the file")
    label = run_inference(inf_file, MosqID, PicNum)
    
    
    
    return jsonify({
        "genus": label[0],
        "species": label[1],
        "confidence_score": label[2],
        "color_code": label[3]
    })

  
#def upload_to_aws(local_file, bucket, s3_file):
#    s3 = boto3.client('s3', aws_access_key_id= ACCESS_KEY, aws_secret_access_key= SECRET_KEY)
#    try:
#        s3.upload_file(local_file, bucket, s3_file)
#        print("Upload Successful")
#        return True
#    except FileNotFoundError:
#        print("The file was not found")
#        return False
#    except NoCredentialsError:
#        print("Credentials not available")
#        return False


#def upload_file(file_name, bucket, object_name=None):
#    """Upload a file to an S3 bucket
#
#    :param file_name: File to upload
#    :param bucket: Bucket to upload to
#    :param object_name: S3 object name. If not specified then file_name is used
#    :return: True if file was uploaded, else False
#    """

    # If S3 object_name was not specified, use file_name
#    if object_name is None:
#        object_name = file_name

#    # Upload the file
#    s3_client = boto3.client('s3')
#    try:
#        response = s3_client.upload_file(file_name, bucket, object_name)
#    except ClientError as e:
#        logging.error(e)
#        return False
#    return True 
  
  
  
  
  
  
  
  

def color_code(num):
    if(float(num)>0.9):
        return '#4bf542'
    elif (float(num)>0.7):
        return '#f7b17e'
    else:
        return '#f7543b'

def run_inference(inf_file, mosquitoid, picnum):
    # Preprocessing of the image happens here
    img = load_image(inf_file)
    originalimg = img
    #Cropping line is below, SHOULD NOT BE INCLUDED IN THIS VERSION
    #useless, img, status=cropImage(impath, 'm', labelsfile, 21, 0.08)
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
    
    fullfilename = mosquitoid + "_" + picnum + "_" + string_label[0] + "_" + string_label[1]
    fullbucket = 'photostakenduringpilotstudy'
    s3_file = 'PilotStudy'
    #file = cv2.imwrite(fullfilename, originalimg)
    #upload_to_aws(file, fullbucket , s3_file)
    ##upload_file(file, fullbucket)
    return (string_label[0], string_label[1], str(prob), color_code(prob))

app.register_blueprint(moz)
