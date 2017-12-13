
import sys

req_link = '/usr/local/lib/python2.7/site-packages/'
sys.path.append(req_link)

from keras import backend as K
K.set_image_dim_ordering('th')

import numpy as np
np.random.seed(123)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import model_from_json
import glob
import cv2

def train_model():
    # 1. Load pre-shuffled MNIST data into train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print (X_train.shape)
    # 2. Preprocess input data
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # 3. Preprocess class labels
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    print X_test[0]
    # 4. Define model architecture
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(1,28,28)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    # 5. Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # 6. Fit model on training data
    model.fit(X_train, Y_train, 
              batch_size=32, epochs=10, verbose=1)

    # 7. serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # 8. serialize weights to HDF5
    model.save_weights("model.h5", overwrite=True)
    print("Saved model to disk")
    print ("hi")
    return model

def load_model():
    # 9. load json and create model
    json_file = open('model.json', 'r')
    loadmodel = json_file.read()
    json_file.close()
    model = model_from_json(loadmodel)

    # 10. load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk") 

    # 11. Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print("Compiled loaded model") 
    
    # 11. Evaluate model on test data
    score = model.evaluate(X_test, Y_test, verbose=0)
    print score
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return model

# 12. predict

def processImage(path):
    x = cv2.imread(path)
    x = cv2.resize(x,(28, 28))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    x = x.reshape(1, 1, 28, 28)
    x = x.astype('float32')
    x /= 255
    return x


def getLabel(label):
    
    pattern = ([0]*label)+[1]+([0]*(9-label))
    print pattern
    label = np.array(pattern) 
    
    label = label.reshape(1, 10)
    return label

def predict(image, label, model):
    
    label = getLabel(label)
    
    score = model.evaluate(image, label, verbose=0)
    pred_array = model.predict(image)
    pred_class = model.predict_classes(image)

    prediction = (score, pred_array, pred_class)
    
    return prediction

def select(number):
    try:
        x = processImage(str(number)+".png")
    except:
        x = processImage(str(number)+".jpeg")
    return x

from pprint import pprint
model = load_model()

pprint (predict(select("six"), 6, model))
