import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

width_ = 160
height_ = 110

HIGH_THROTTLE = False
LOW_THROTTLE = True

@sio.on('telemetry')
def telemetry(sid, data):
#    send_control(0.0, 0.8)
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    transformed_image_array = image_array[None, :, :, :]
    img = cv2.resize(transformed_image_array[0], (width_, height_))    
    img = np.array(img - 128.0) / 128.0
    
    x_img = np.array([img])
    x_driving_state = np.array([[[steering_angle, throttle, speed]]])    
    #print("x_driving_state shape : ", x_driving_state.shape)    
    #print("x_img shape : ", x_img.shape)
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    #predict for god's sake    
    outputs = model.predict([x_driving_state, x_img], batch_size = 1)
    #print("output shape : ", outputs.shape)    
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    steering_angle_new = float(outputs[0][0]) 
    throttle_new = float(outputs[0][1])    
    
    if throttle_new < 0.001:
        throttle_new = 0.15
    if HIGH_THROTTLE == True:
        throttle_new = 0.2
    elif LOW_THROTTLE == True:
        throttle_new = 0.13
    
        
        
    print("New steering: ", steering_angle_new, " New throttle : ", throttle_new)
    
    send_control(steering_angle_new,  throttle_new)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition h5. Model should be on the same path.')
    
    # adding optional argument for throttle control
    parser.add_argument('--throttle', 
                        help = 'Configure throttle control either by the network or set to maximum')
    args = parser.parse_args()
    if (args.throttle == 'raceme'):
        print("Running high throttle!!! Hang on!!!")
        HIGH_THROTTLE = True
        LOW_THROTTLE = False
    elif (args.throttle == 'auto'):
        print("Running on auto-throttle!!! Trust me...")
        HIGH_THROTTLE = False
        LOW_THROTTLE = False
    else:
        print("Playing it safe (default). Running on low throtle...")
        LOW_THROTTLE = True
        HIGH_THROTTLE = False
        
    model = load_model(args.model)

    
    
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)