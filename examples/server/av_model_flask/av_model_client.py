import requests
import numpy as np
import time
import json
import base64

num_speakers = 2
url = 'http://localhost:5000/results'
audio_sz = (1, 298, 257, 2)
video_sz = (1, 75, 1, 1792, num_speakers)
sample_audio = np.random.rand(*audio_sz) # float64
sample_audio = sample_audio.astype(np.float32)
sample_video = np.random.rand(*video_sz)  # float64
sample_video = sample_video.astype(np.float32)
print(sample_video.dtype)
params = {'param0': 'param0', 'param1': 'param1'}
for i in range(10):
    start = time.time()
    # data = {'params': params, 'video': base64.b64encode(sample_video.tobytes().decode('ascii')),
    #         'audio': base64.b64encode(sample_audio.tobytes()).decode('ascii')}
    data = {'params': params, 'video': str(sample_video.tobytes(), encoding='ISO-8859-1'),
            'audio': str(sample_audio.tobytes(), encoding='ISO-8859-1')}
    print("serilization took {} secs".format(time.time() - start))
    start = time.time()
    print(" Request sent at {} ".format(start))
    response = requests.post(url, json=data)
    print("post request took {} secs".format(time.time() - start))
    #print(response.text)
    print("Decode JSON serialized NumPy array")
    start = time.time()
    crms_sz = (298, 257, 2, 2)

    crms = np.ndarray(shape=crms_sz, dtype=np.float32,
                       buffer=bytes(response.text, encoding='ISO-8859-1'))
    #finalNumpyArray = np.asarray(decodedArrays["array"])
    print(crms.shape)
    print("deserialization took {} secs".format(time.time() - start))