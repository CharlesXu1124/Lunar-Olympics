import numpy as np
import re
from sklearn import svm, metrics
from skimage import io, feature, filters, exposure, color

# importing extra packages
# utilized tensorflow-gpu for training
import tensorflow as tf

from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
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
import cv2

# tf.disable_v2_behavior()


# define the total number of epochs to train for along with the
# initial learning rate
NUM_EPOCHS = 25
INIT_LR = 5e-3

# polynomial decay of learning rate
def poly_decay(epoch):
    # initialize the maximum number of epochs, base learning rate,
    # and power of the polynomial
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0
    
    # compute the new learning rate based on polynomial decay
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

    # return the new learning rate
    return alpha

# helper class, structuring MiniGoogLeNet
# over extensive testing I found that MiniGoogLeNet is a nice tradeoff between training time as well as accuracy
class MiniGoogLeNet:
    @staticmethod
    def conv_module(x,K,kX,kY,stride,chanDim,padding='same'):
        # define a CONV => BN => RELU pattern
        x = Conv2D(K,(kX,kY),strides = stride,padding=padding)(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Activation('relu')(x)

        # return the block
        return x
    
    @staticmethod
    def inception_module(x,numK1x1,numK3x3,chanDim):
        # define two CONV modules, then concatenate across the
        # channel dimension
        conv_1x1 = MiniGoogLeNet.conv_module(x,numK1x1,1,1,(1,1),chanDim)
        conv_3x3 = MiniGoogLeNet.conv_module(x,numK3x3,3,3,(1,1),chanDim)
        x = concatenate([conv_1x1,conv_3x3],axis = chanDim)

        # return the block
        return x
    
    @staticmethod
    def downsample_module(x,K,chanDim):
        # defines the CONV module and POOL, then concatenate
        # across the channel dimension
        conv_3x3 = MiniGoogLeNet.conv_module(x,K,3,3,(2,2),chanDim,padding='valid')
        pool = MaxPooling2D((3,3),strides=(2,2))(x)
        x = concatenate([conv_3x3,pool],axis = chanDim)

        # returnt the block
        return x
    
    @staticmethod
    def build(width,height,depth,classes):
        # initialize the input shape to be "channels last" and the
        # channels dimension itself
        inputShape =(height,width,depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth,height,width)
            chanDim = 1
        
        # define the model input and the first CONV layer
        inputs= Input(shape = inputShape)
        x = MiniGoogLeNet.conv_module(inputs,96,3,3,(1,1),chanDim)
        
        # two Inception modules followed by a downsample module
        x = MiniGoogLeNet.inception_module(x,32,32,chanDim) # 64 filters
        x = MiniGoogLeNet.inception_module(x,32,48,chanDim) # 80 filters
        x = MiniGoogLeNet.downsample_module(x,80,chanDim) # reduces input volume
        
        # four Inception modules followed by a downsample module
        x = MiniGoogLeNet.inception_module(x,112,48,chanDim)
        x = MiniGoogLeNet.inception_module(x,96,64,chanDim)
        x = MiniGoogLeNet.inception_module(x,80,80,chanDim)
        x = MiniGoogLeNet.inception_module(x,48,96,chanDim)
        x = MiniGoogLeNet.downsample_module(x,96,chanDim)
        
        # two Inception modules followed by global POOL and dropout
        x = MiniGoogLeNet.inception_module(x,176,160,chanDim)
        x = MiniGoogLeNet.inception_module(x,176,160,chanDim)
        x = AveragePooling2D((7,7))(x)
        x = Dropout(0.5)(x)
        
        # softmax classifier
        x = Flatten()(x)
        x = Dense(classes)(x)
        x = Activation('softmax')(x)
        
        # create the model
        model = Model(inputs,x,name='googlenet')

        # return the constructed network architecture
        return model

# helper method that rescales the images
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

    # Return the pre-processed image
    return image

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
        label_arr = ['five',
                     'none',
                     'one',
                     'seven'
                     ]
        predicted_labels = []
        for image in data:
            # note: -1 is added since it expects input dimension of 4
            # preprocess the picture
            image = image.reshape(-1,32, 32, 3)
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

def main():

    img_clf = ImageClassifier()

    # load images
    (train_raw, train_labels) = img_clf.load_data_from_folder('./train/')
    (test_raw, test_labels) = img_clf.load_data_from_folder('./test/')

    # convert images into features
    train_data = img_clf.extract_image_features(train_raw)
    test_data = img_clf.extract_image_features(test_raw)

    # train model and test on training data
    img_clf.train_classifier(train_data, train_labels)
    predicted_labels = img_clf.predict_labels(train_data)
    print("\nTraining results")
    print("=============================")
    print("Confusion Matrix:\n", metrics.confusion_matrix(
        train_labels, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(train_labels, predicted_labels))
    print("F1 score: ", metrics.f1_score(
        train_labels, predicted_labels, average='micro'))

    # test model
    predicted_labels = img_clf.predict_labels(test_data)
    print("\nTest results")
    print("=============================")
    print("Confusion Matrix:\n", metrics.confusion_matrix(
        test_labels, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(test_labels, predicted_labels))
    print("F1 score: ", metrics.f1_score(
        test_labels, predicted_labels, average='micro'))


if __name__ == "__main__":
    main()
