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
from scipy.misc import imresize

from keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf

tf.python.control_flow_ops = tf

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None
throttle = 0.2


@sio.on('telemetry')
def telemetry(sid, data):

    start_time = time.clock()
    # The current steering angle of the car
    # steering_angle = data["steering_angle"]
    # The current throttle of the car
    # current_throttle = data["throttle"]
    # The current speed of the car
    # speed = data["speed"]
    # The current image from the center camera of the car
    img_string = data["image"]
    image = Image.open(BytesIO(base64.b64decode(img_string)))
    image_array = np.asarray(image)
    # transformed_image_array = (image_array[None, :, 1:-1, :] / 127.5) - 1
    transformed_image_array = np.asarray([(imresize(image_array, (80, 160, 3)) / 127.5) - 1.])

    # print('angle: {}, throttle: {}, speed: {}, img: {}'.format(steering_angle, throttle,
    #                                                            speed, transformed_image_array.shape))

    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    try:
        steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    except Exception as ex:
        print(ex)
        raise

    end_time = time.clock()
    print('At {}, processed in {}, steering {}, throttle {}'.format(
        end_time, end_time - start_time, steering_angle, throttle))
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, _throttle):
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': _throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
                        help='Path to model definition h5. Model should be on the same path.')
    parser.add_argument('--throttle', '-t', type=float, default=0.2,
                        help='Path to model definition h5. Model should be on the same path.')
    args = parser.parse_args()

    model = load_model(args.model)
    throttle = args.throttle

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
