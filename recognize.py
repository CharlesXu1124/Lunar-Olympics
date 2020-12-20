#!/usr/bin/python
# -*- coding: utf-8 -*-
## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2019 Intel Corporation. All Rights Reserved.
# Python 2/3 compatibility
from __future__ import print_function

import re
from sklearn import svm, metrics
from skimage import io, feature, filters, exposure, color
from skimage.transform import rescale, resize, downscale_local_mean

# importing extra packages
# utilized tensorflow-gpu for training
# import tensorflow as tf
import tensorflow as tf

from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from joblib import load
import imutils
from keras import backend
from keras.models import Sequential
from keras.models import Model
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.layers import BatchNormalization
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import concatenate
from joblib import dump
import statistics

# tf.disable_v2_behavior()

# First import the library
import pyrealsense2 as rs

# Import OpenCV and numpy
import cv2
import numpy as np
from math import tan, pi

classifier = load('clf.joblib')
label_arr = ['five', 'none', 'one', 'seven']
img_window = ['none', 'none', 'none', 'none', 'none']

class ImageClassifier:

    def __init__(self):
        self.classifier = None

    def imread_convert(self, f):
        return io.imread(f).astype(np.uint8)

    def load_data_from_folder(self, dir):
        # read all images into an image collection
        ic = io.ImageCollection(dir+"*.jpg", load_func=self.imread_convert)

        # create one large array of image data
        data = io.concatenate_images(ic)

        # extract labels from image names
        labels = np.array(ic.files)
        for i, f in enumerate(labels):
            m = re.search("_", f)
            labels[i] = f[len(dir):m.start()]

        return(data, labels)

    def extract_image_features(self, data):
        # Please do not modify the header above

        # extract feature vector from image data

        ########################
        # YOUR CODE HERE
        # load the image, preprocess it, and store it in the data list
        # initialize the processed image array
        featured_data = []
        for image in data:
            # scale into 32X32 arrays
            image = rescale(image,32,32)
            image = img_to_array(image)
            featured_data.append(image)
        # normalization
        featured_data = np.array(featured_data, dtype="float") / 255.0
        ########################
        # Please do not modify the return type below
        return(featured_data)

    def train_classifier(self, train_data, train_labels):
        # Please do not modify the header above

        # train model and save the trained model to self.classifier

        ########################
        # YOUR CODE HERE
        # split the training dataset into training and validation set
        # 10% for validation set
        (train_x, test_x, train_y, test_y) = train_test_split(train_data, train_labels, test_size=0.10, random_state=42)
        # Convert the labels from integers to vectors
        lb = LabelBinarizer().fit(train_y)
        train_y = lb.transform(train_y)
        test_y = lb.transform(test_y)
        # Initialize the model
        # use stochastic gradient descent for optimizer
        opt= SGD(lr = INIT_LR,momentum=0.9)
        # build the model
        model = MiniGoogLeNet.build(width=32,height = 32,depth=3,classes = 4)
        # compile the model
        model.compile(loss = 'categorical_crossentropy',optimizer=opt, metrics = ['accuracy'])
        # use nesterov acceleration due to relatively small dataset
        optimizer = SGD(lr=0.01, decay=1e-8, momentum=0.9, nesterov=True)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        # Train the network
        print("[INFO]: Training....")
        H = model.fit(np.array(train_x), np.array(train_y), validation_data=(np.array(test_x), np.array(test_y)), batch_size=32, epochs=NUM_EPOCHS, verbose=1)
        filepath = 'model'

        dump(model, 'clf.joblib')
        # store the model file
        self.classifier = model
        ########################

    def predict_labels(self, data):
        # Please do not modify the header

        # predict labels of test data using trained model in self.classifier
        # the code below expects output to be stored in predicted_labels

        ########################
        # YOUR CODE HERE
        
        predicted_labels = []
        for image in data:
            # note: -1 is added since it expects input dimension of 4
            # preprocess the picture
            image = image.reshape(-1,224,224,3)
            result = self.classifier.predict(image)
            result = result[0]
            result = result.tolist()
            idx = result.index(max(result))
            print(idx)
            label = label_arr[idx]
            predicted_labels.append(label)
            print(label)
        ########################
        # Please do not modify the return type below
        return predicted_labels


# helper method, for rescaling images
def rescale(image, width, height):
    # Grab the dimensions of the image, then initialize the padding values
    (h, w) = image.shape[:2]

    # If the width is greater than the height then resize along the width
    if w > h:
        image = imutils.resize(image, width=width)
    # Otherwise, the height is greater than the width so resize along the height
    else:
        image = imutils.resize(image, height=height)

    # Determine the padding values for the width and height to obtain the target dimensions
    pad_w = int((width - image.shape[1]) / 2.0)
    pad_h = int((height - image.shape[0]) / 2.0)

    # Pad the image then apply one more resizing to handle any rounding issues
    image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    # image_resized = resize(image, (224, 224, 3), anti_aliasing=True)
    # Return the pre-processed image
    return image


"""
In this section, we will set up the functions that will translate the camera
intrinsics and extrinsics from librealsense into parameters that can be used
with OpenCV.
The T265 uses very wide angle lenses, so the distortion is modeled using a four
parameter distortion model known as Kanalla-Brandt. OpenCV supports this
distortion model in their "fisheye" module, more details can be found here:
https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html
"""

"""
Returns R, T transform from src to dst
"""
def get_extrinsics(src, dst):
    extrinsics = src.get_extrinsics_to(dst)
    R = np.reshape(extrinsics.rotation, [3,3]).T
    T = np.array(extrinsics.translation)
    return (R, T)

"""
Returns a camera matrix K from librealsense intrinsics
"""
def camera_matrix(intrinsics):
    return np.array([[intrinsics.fx,             0, intrinsics.ppx],
                     [            0, intrinsics.fy, intrinsics.ppy],
                     [            0,             0,              1]])

"""
Returns the fisheye distortion from librealsense intrinsics
"""
def fisheye_distortion(intrinsics):
    return np.array(intrinsics.coeffs[:4])

# Set up a mutex to share data between threads 
from threading import Lock
frame_mutex = Lock()
frame_data = {"left"  : None,
              "right" : None,
              "timestamp_ms" : None
              }

"""
This callback is called on a separate thread, so we must use a mutex
to ensure that data is synchronized properly. We should also be
careful not to do much work on this thread to avoid data backing up in the
callback queue.
"""
def callback(frame):
    global frame_data
    if frame.is_frameset():
        frameset = frame.as_frameset()
        f1 = frameset.get_fisheye_frame(1).as_video_frame()
        f2 = frameset.get_fisheye_frame(2).as_video_frame()
        left_data = np.asanyarray(f1.get_data())
        right_data = np.asanyarray(f2.get_data())
        ts = frameset.get_timestamp()
        frame_mutex.acquire()
        frame_data["left"] = left_data
        frame_data["right"] = right_data
        frame_data["timestamp_ms"] = ts
        frame_mutex.release()

# Declare RealSense pipeline, encapsulating the actual device and sensors
pipe = rs.pipeline()

# Build config object and stream everything
cfg = rs.config()

# Start streaming with our callback
pipe.start(cfg, callback)

try:
    # Set up an OpenCV window to visualize the results
    WINDOW_TITLE = 'Realsense'
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    # Configure the OpenCV stereo algorithm. See
    # https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html for a
    # description of the parameters
    window_size = 5
    min_disp = 0
    # must be divisible by 16
    num_disp = 112 - min_disp
    max_disp = min_disp + num_disp
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
                                   numDisparities = num_disp,
                                   blockSize = 16,
                                   P1 = 8*3*window_size**2,
                                   P2 = 32*3*window_size**2,
                                   disp12MaxDiff = 1,
                                   uniquenessRatio = 10,
                                   speckleWindowSize = 100,
                                   speckleRange = 32)

    # Retreive the stream and intrinsic properties for both cameras
    profiles = pipe.get_active_profile()
    streams = {"left"  : profiles.get_stream(rs.stream.fisheye, 1).as_video_stream_profile(),
               "right" : profiles.get_stream(rs.stream.fisheye, 2).as_video_stream_profile()}
    intrinsics = {"left"  : streams["left"].get_intrinsics(),
                  "right" : streams["right"].get_intrinsics()}

    # Print information about both cameras
    print("Left camera:",  intrinsics["left"])
    print("Right camera:", intrinsics["right"])

    # Translate the intrinsics from librealsense into OpenCV
    K_left  = camera_matrix(intrinsics["left"])
    D_left  = fisheye_distortion(intrinsics["left"])
    K_right = camera_matrix(intrinsics["right"])
    D_right = fisheye_distortion(intrinsics["right"])
    (width, height) = (intrinsics["left"].width, intrinsics["left"].height)

    # Get the relative extrinsics between the left and right camera
    (R, T) = get_extrinsics(streams["left"], streams["right"])

    # We need to determine what focal length our undistorted images should have
    # in order to set up the camera matrices for initUndistortRectifyMap.  We
    # could use stereoRectify, but here we show how to derive these projection
    # matrices from the calibration and a desired height and field of view

    # We calculate the undistorted focal length:
    #
    #         h
    # -----------------
    #  \      |      /
    #    \    | f  /
    #     \   |   /
    #      \ fov /
    #        \|/
    stereo_fov_rad = 90 * (pi/180)  # 90 degree desired fov
    stereo_height_px = 300          # 300x300 pixel stereo output
    stereo_focal_px = stereo_height_px/2 / tan(stereo_fov_rad/2)

    # We set the left rotation to identity and the right rotation
    # the rotation between the cameras
    R_left = np.eye(3)
    R_right = R

    # The stereo algorithm needs max_disp extra pixels in order to produce valid
    # disparity on the desired output region. This changes the width, but the
    # center of projection should be on the center of the cropped image
    stereo_width_px = stereo_height_px + max_disp
    stereo_size = (stereo_width_px, stereo_height_px)
    stereo_cx = (stereo_height_px - 1)/2 + max_disp
    stereo_cy = (stereo_height_px - 1)/2

    # Construct the left and right projection matrices, the only difference is
    # that the right projection matrix should have a shift along the x axis of
    # baseline*focal_length
    P_left = np.array([[stereo_focal_px, 0, stereo_cx, 0],
                       [0, stereo_focal_px, stereo_cy, 0],
                       [0,               0,         1, 0]])
    P_right = P_left.copy()
    P_right[0][3] = T[0]*stereo_focal_px

    # Construct Q for use with cv2.reprojectImageTo3D. Subtract max_disp from x
    # since we will crop the disparity later
    Q = np.array([[1, 0,       0, -(stereo_cx - max_disp)],
                  [0, 1,       0, -stereo_cy],
                  [0, 0,       0, stereo_focal_px],
                  [0, 0, -1/T[0], 0]])

    # Create an undistortion map for the left and right camera which applies the
    # rectification and undoes the camera distortion. This only has to be done
    # once
    m1type = cv2.CV_32FC1
    (lm1, lm2) = cv2.fisheye.initUndistortRectifyMap(K_left, D_left, R_left, P_left, stereo_size, m1type)
    (rm1, rm2) = cv2.fisheye.initUndistortRectifyMap(K_right, D_right, R_right, P_right, stereo_size, m1type)
    undistort_rectify = {"left"  : (lm1, lm2),
                         "right" : (rm1, rm2)}

    mode = "stack"

    counter = 0
    while True:
        counter += 1
        # Check if the camera has acquired any frames
        frame_mutex.acquire()
        valid = frame_data["timestamp_ms"] is not None
        frame_mutex.release()

        # If frames are ready to process
        if valid:
            # Hold the mutex only long enough to copy the stereo frames
            frame_mutex.acquire()
            frame_copy = {"left"  : frame_data["left"].copy(),
                          "right" : frame_data["right"].copy()}
            frame_mutex.release()

            # Undistort and crop the center of the frames
            center_undistorted = {"left" : cv2.remap(src = frame_copy["left"],
                                          map1 = undistort_rectify["left"][0],
                                          map2 = undistort_rectify["left"][1],
                                          interpolation = cv2.INTER_LINEAR),
                                  "right" : cv2.remap(src = frame_copy["right"],
                                          map1 = undistort_rectify["right"][0],
                                          map2 = undistort_rectify["right"][1],
                                          interpolation = cv2.INTER_LINEAR)}

            # compute the disparity on the center of the frames and convert it to a pixel disparity (divide by DISP_SCALE=16)
            disparity = stereo.compute(center_undistorted["left"], center_undistorted["right"]).astype(np.float32) / 16.0

            # re-crop just the valid part of the disparity
            disparity = disparity[:,max_disp:]

            # convert disparity to 0-255 and color it
            disp_vis = 255*(disparity - min_disp)/ num_disp
            disp_color = cv2.applyColorMap(cv2.convertScaleAbs(disp_vis,1), cv2.COLORMAP_JET)
            color_image = cv2.cvtColor(center_undistorted["left"][:,max_disp:], cv2.COLOR_GRAY2RGB)

            new_image = np.array(disp_color)

            # rescale the image into 32 X 32 X 3 array
            image = rescale(new_image,32,32)
            image = img_to_array(image)

            # normalize the image array
            img = np.array(image, dtype="float") / 255.0

            img = img.reshape(-1, 32, 32, 3)
            result = classifier.predict(img)

            # convert the prediction result to array
            result = result[0].tolist()

            # extract the index for the maximum label
            idx = result.index(max(result))

            # extract the label
            label = label_arr[idx]
            for i in range(1, len(img_window)):
                img_window[i] = img_window[i - 1]
            img_window[0] = label
            majority_label = statistics.mode(img_window)
            print(majority_label)

            if mode == "stack":
                cv2.imshow(WINDOW_TITLE, np.hstack((color_image, disp_color)))

                # filename = 'data/none/none_%d.jpg' % counter
                # cv2.imwrite(filename, disp_color)
            if mode == "overlay":
                ind = disparity >= min_disp
                color_image[ind, 0] = disp_color[ind, 0]
                color_image[ind, 1] = disp_color[ind, 1]
                color_image[ind, 2] = disp_color[ind, 2]
                cv2.imshow(WINDOW_TITLE, color_image)
            # time.sleep(0.1)
        key = cv2.waitKey(1)
        if key == ord('s'): mode = "stack"
        if key == ord('o'): mode = "overlay"
        if key == ord('q') or cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
            break
finally:
    pipe.stop()