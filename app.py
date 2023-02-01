from flask import Flask, request, render_template, render_template, url_for, jsonify
# import tensorflow as tf
# from predict import predict_bounding_boxes, extractbox
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from utils import config
from PIL import Image
# import subprocess
# from google.cloud import vision
from preprocess import extract_bbox
import pytesseract
import os
import sys
import cv2



app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if 'image' not in request.files:
        return 'No file found'
    image = request.files['image'][0]

    x1,y1,x2,y2= extract_bbox(preds,image.filename)
    bbox = [x1,y1,x2,y2]

     # load image
    image = Image.open(image.filename)
    # create a cropped image
    # create a cropped image
    cropped = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

    # run OCR on cropped image
    text = pytesseract.image_to_string(cropped)

    return render_template('predict.html',text = text)


if __name__ == '__main__':
    app.run(debug=True)
    # file_path = sys.argv[1]
    # prediction = predict(file_path)
    # print(prediction)
