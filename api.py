import os
from PIL import Image
import shutil 
from os import listdir, mkdir
from os.path import isfile, join, basename, exists
# from tqdm import tqdm
import cv2
import time
import os
from shutil import copyfile
import numpy as np

from flask import Flask, jsonify, request
import json
import requests
import time
from flask_cors import CORS, cross_origin
from random import choice
from io import BytesIO
from flask import send_file, send_from_directory
import os
desktop_agents = [
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/602.2.14 (KHTML, like Gecko) Version/10.0.1 Safari/602.2.14',
    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:50.0) Gecko/20100101 Firefox/50.0']


def random_headers():
    return {'User-Agent': choice(desktop_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'}

def download_image(image_url):
    header = random_headers()

    response = requests.get(image_url, headers=header, stream=True, verify=False, timeout=5)

    image = Image.open(BytesIO(response.content)).convert('RGB')

    return image

def create_query_result(result, error=None):
    if error is not None:
        results = {'result':None, 'error':str(error)}
    else:
        results = {'result':result, 'error':''}
    return results


def jsonify_str(output_list):
    print('output_list: ',output_list)
    with app.app_context():
        with app.test_request_context():
            result = jsonify(output_list)
    return result
    # return json.dumps(output_list)

def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['JSON_AS_ASCII'] = False

import torch
import numpy as np
from experiment import Structure, Experiment
from concern.config import Configurable, Config
import math

from DB import DB



model = DB()
img = cv2.imread('demo_results/000021.jpg')
model.predict_image(img)

@app.route("/scenetext_detection_DB", methods=['GET', 'POST'])
@cross_origin()
def scenetext_detection_DB():
    start_all = time.time()
    if request.method == "POST":
        try:
            print('receive image binary string')
            nparr = np.fromstring(request.data, np.uint8)
            # decode image
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


            # file = request.files['image'].read() ## byte file
            # npimg = np.fromstring(file, np.uint8)
            # img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
            
        except Exception as ex:
            print(ex)
            return jsonify_str(create_query_result("", "Wrong parameter or Can not download image"))
    else:
        try:
            image_url = request.args.get('url', default='', type=str)
            img = download_image(image_url)
            # print('image shape input: ',img.shape)
            img = img.convert('RGB')
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
        except Exception as ex:
            print("Can not download image: ", ex)
            return jsonify_str(create_query_result("", "Wrong parameter or Can not download image"))
    start = time.time()
    pre_shape = img.shape

    # img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
    print("type img: ", type(img))
    print("shape img: ", img.shape)

    predict = scenetext_detection_DB_function(img)
    print("type predict: ", type(predict))
    # print("shape predict: ", predict.shape)
    end = time.time()
    print('all time detection: ', end - start)
    print('all time: ', end - start_all)
    # print('result: ', create_query_result(predict, None))
    return jsonify_str(create_query_result(predict, None))

def scenetext_detection_DB_function(img):
    result = model.predict_image(img)
    return result


if __name__ == '__main__':
      app.run(host='0.0.0.0', port=1411)
