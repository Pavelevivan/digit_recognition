import keras
import sys
import numpy as np
import argparse
from keras.preprocessing import image
from keras.models import model_from_json
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import (
    Dense,
    Dropout,
    Conv2D,
    MaxPooling2D,
    Flatten
)
from keras.utils import np_utils


def built_cnn():
    # image size

    img_rows, img_cols = 28, 28
    (X_train, Y_train), (X_test, Y_test) = get_mnist_data(img_cols, img_rows)
    input_shape = (img_rows, img_cols, 1)
    # setting seed for random values
    np.random.seed(42)

    # creating model
    model = Sequential()

    model.add(Conv2D(75, kernel_size=(5, 5),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(  Conv2D(100, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    # compiling model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    print(model.summary())

    # train network
    model.fit(X_train, Y_train, batch_size=200, epochs=5, validation_split=0.2, verbose=2)

    # rate accuracy
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("Точность работы на тестовых данных: %.2f%%" % (scores[1] * 100))
    save_model(model)


def get_mnist_data(img_cols, img_rows):
    # loading data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # converting data dimensions
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

    # normalize data
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # convert marks to categorically
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    return (X_train, Y_train), (X_test, Y_test)


def save_model(model):
    # saving model and weights
    model_json = model.to_json()
    with open('cnn_model.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('cnn_weights.h5')


def predict_image(img_path):
    # specifying path
    model_file = 'cnn_model.json'
    weights_file = 'cnn_weights.h5'

    try:
        # loading weights
        with open(model_file, 'r') as json_file:
            loaded_model = model_from_json(json_file.read())
        loaded_model.load_weights(weights_file)
        #
        # loading image and converting it to 28x28 size and grayscale
        image_data = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
    except OSError as e:
        print(e)
        print(f"Check if file {e.filename} exist and can be reached")
        sys.exit(0)

    # convert image to array and normalize
    image_arr = image.img_to_array(image_data)
    image_arr = 255 - image_arr
    image_arr /= 255
    image_arr = np.expand_dims(image_arr, axis=0)

    # compiling model
    loaded_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


    # predicting image
    prediction = loaded_model.predict(image_arr)
    prediction = np.argmax(prediction, axis=1)
    print("CNN predicted it's number {} on image {}".format(prediction[0], img_path))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', dest='built', help='built artificial neural network')
    parser.add_argument('-i', dest='image_path', type=str, help='provides path to image that will be predicted')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    if args.built:
        built_cnn()
    elif args.image_path:
        predict_image(args.image_path)
