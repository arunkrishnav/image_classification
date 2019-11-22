import numpy as np
from keras import applications
from keras.layers import Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img


def predict(image_path, class_indices_path, top_model_weights_path):
    class_dictionary = np.load(class_indices_path).item()

    num_classes = len(class_dictionary)

    print('Loading and pre-processing image')
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image / 255
    image = np.expand_dims(image, axis=0)

    model = applications.VGG16(include_top=False, weights='imagenet')

    bottleneck_prediction = model.predict(image)

    # build top model
    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.load_weights(top_model_weights_path)

    # use the bottleneck prediction on the top model to get the final classification
    class_predicted = model.predict_classes(bottleneck_prediction)

    probabilities = model.predict_proba(bottleneck_prediction)
    print(probabilities)

    in_id = class_predicted[0]

    inv_map = {v: k for k, v in class_dictionary.items()}

    label = inv_map[in_id]

    return label
