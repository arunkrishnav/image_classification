import math
import os

import numpy as np
from keras import applications
from keras.layers import Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# dimensions of our images.
img_width, img_height = 224, 224
identifier = "car_vgg16"

top_model_weights_path = BASE_DIR + '/image_classification/models/model_%s.h5' % identifier
train_data_dir = BASE_DIR + "/image_classification/car_images/train_data"
validation_data_dir = BASE_DIR + "/image_classification/car_images/validation_data"
feature_train_path = BASE_DIR + '/image_classification/bottleneck_features/train_%s.npy' % identifier
feature_validation_path = BASE_DIR + '/image_classification/bottleneck_features/validation_%s.npy' % identifier
class_indices_path = BASE_DIR + '/image_classification/models/class_indices_%s.npy' % identifier


epochs = 500
batch_size = 16


def save_bottleneck_features():
    model = applications.VGG16(include_top=False, weights='imagenet')

    datagen = ImageDataGenerator(rescale=1. / 255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)

    generator = datagen.flow_from_directory(train_data_dir,
                                            target_size=(img_width, img_height),
                                            batch_size=batch_size,
                                            shuffle=False)

    no_train_samples = len(generator.filenames)

    predict_size_train = int(math.ceil(no_train_samples / batch_size))
    bottleneck_features_train = model.predict_generator(generator, predict_size_train)
    np.save(feature_train_path, bottleneck_features_train)

    datagen = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_directory(validation_data_dir,
                                            target_size=(img_width, img_height),
                                            batch_size=batch_size,
                                            class_mode=None,
                                            shuffle=False)

    no_validation_samples = len(generator.filenames)

    predict_size_validation = int(math.ceil(no_validation_samples / batch_size))
    bottleneck_features_validation = model.predict_generator(generator, predict_size_validation)
    np.save(feature_validation_path, bottleneck_features_validation)
    train_top_model()


def train_top_model():
    datagen_top = ImageDataGenerator(rescale=1. / 255)

    generator_top = datagen_top.flow_from_directory(train_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=False)

    num_classes = len(generator_top.class_indices)

    # save the class indices to use later in predictions
    np.save(class_indices_path, generator_top.class_indices)

    # get the class labels for the training data, in the original order
    train_labels = generator_top.classes
    # convert the training labels to categorical vectors
    train_labels = to_categorical(train_labels, num_classes=num_classes)

    generator_top = datagen_top.flow_from_directory(validation_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode=None,
                                                    shuffle=False)

    validation_labels = generator_top.classes
    validation_labels = to_categorical(validation_labels, num_classes=num_classes)

    # load the bottleneck features saved earlier
    train_data = np.load(feature_train_path)
    validation_data = np.load(feature_validation_path)

    # build the model
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))

    model.save_weights(top_model_weights_path)

    eval_loss, eval_accuracy = model.evaluate(validation_data, validation_labels,
                                              batch_size=batch_size, verbose=1)
    print(eval_loss, eval_accuracy)
