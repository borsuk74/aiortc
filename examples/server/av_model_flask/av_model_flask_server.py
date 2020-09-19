import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import time
import tensorflow as tf
from av_model.loss import audio_discriminate_loss2 as audio_loss
from keras.models import load_model
import json
from json import JSONEncoder
import base64
from flask_ngrok import run_with_ngrok



class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

num_speakers = 2
def generate_warmup_batch(num_speakers=2):
    audio_sz = (1, 298, 257, 2)
    video_sz = (1, 75, 1, 1792, num_speakers)
    sample_audio = np.random.rand(*audio_sz)
    sample_video = np.random.rand(*video_sz)
    batch = [sample_audio, sample_video]
    return batch


app = Flask(__name__)
#run_with_ngrok(app)
model_path = '../av_model/AVmodel-2p-020-0.46028.h5'
#with tf.device('/device:gpu:1'):
with tf.device('/cpu'):
    loss = audio_loss(gamma=0.1, beta=0.2, people_num=num_speakers)
    av_model = load_model(model_path, custom_objects={'tf': tf, 'loss_func': loss})
    cRMs = av_model.predict(generate_warmup_batch(num_speakers))



#@app.route('/')
#def home():
#    return render_template('index.html')

# @app.route('/predict',methods=['POST'])
# def predict():
#
#    int_features = [int(x) for x in request.form.values()]
#    final_features = [np.array(int_features)]
#    cRMs = av_model.predict(final_features)
#    #cRMs are of the shape (298,257,2,2)
#
#    output = round(cRMs[0], 2)
#
#    return render_template('index.html', prediction_text='Sales should be $ {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    start = time.time()
    print("Request arrived at {}".format(start))
    #data = request.get_json(force=True)
    data = request.json
    params = data['params']

    video_sz = (1, 75, 1, 1792, 2)
    video = np.ndarray(shape=video_sz, dtype=np.float32,
                              buffer=bytes(data['video'], encoding='ISO-8859-1'))
    audio_sz = (1, 298, 257, 2)

    #audio = np.ndarray(shape=audio_sz, dtype=np.float64,
                              #buffer=base64.b64decode(data['video'].encode('ascii')))
    audio = np.ndarray(shape=audio_sz, dtype=np.float32,
                       buffer=bytes(data['audio'], encoding='ISO-8859-1'))
    print("Server deserialization took {} sec".format(time.time() -start))
    start = time.time()
    cRMs = av_model.predict([audio, video])
    print("Prediction took {} secs".format(time.time() - start))
    numpyData = str(cRMs[0].astype(np.float32).tobytes(), encoding='ISO-8859-1')
    #encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)
    #return encodedNumpyData
    return numpyData
    #ret_data = {'params': params, 'crms': cRMs[0].tolist()}

    # return jsonify(
    #     message="Success",
    #     category="success",
    #     data="ret_data",
    #    status=200
    # )
    #return ret_data
    #return "Success"

if __name__ == "__main__":
    #app.run(debug=True)
    app.run()