import tensorflow as tf
import numpy as np
from keras.models import load_model
from keras.models import Model

MODEL_PATH = './facenet/keras/facenet_keras.h5'


def get_face_net(model_path=MODEL_PATH, hot_start=True):
    model = load_model(model_path)

    avgPool_layer_model = Model(inputs=model.input, outputs=model.get_layer('AvgPool').output)

    if hot_start:
        input_batch = np.zeros((75, 160, 160, 3), dtype='float')
        output_batch = avgPool_layer_model.predict(input_batch)

    return avgPool_layer_model
