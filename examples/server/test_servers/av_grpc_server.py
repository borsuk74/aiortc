"""Server which would show how to deserialize stream of batches, sent from client
Works with av_grpc_client. """
import numpy as np
import grpc
from test_servers import av_batch_request_pb2, av_batch_request_pb2_grpc
from concurrent import futures
import time
import tensorflow as tf
from av_model.loss import audio_discriminate_loss2 as audio_loss
from keras.models import load_model

import sys
sys.path.append('../util')
from util.audio_utils import stft, fast_stft, istft, fast_istft, fast_icRM
# Python client
# $ pip3 install -U grpcio grpcio-tools
# $ python3 -m grpc_tools.protoc -I protobuf/ --python_out=. --grpc_python_out=. protobuf/av_batch_request.proto


class AVBatchService(av_batch_request_pb2_grpc.AVBatchServiceServicer):
    def __init__(self, model_path, num_speakers=2):
        self._model_path = model_path
        self._num_speakers = num_speakers
        self._loss = audio_loss(gamma=0.1, beta=0.2, people_num=self._num_speakers)
        # with tf.device('/device:cpu'):
        self._av_model = load_model(model_path, custom_objects={'tf': tf, 'loss_func': self._loss})
        self._av_model.predict(self._generate_warmup_batch())

    def SendAVBatch(self, request_iterator, context):
        for req in request_iterator:
            # need to deserialize requests here
            print("Request arrived!")
            start = time.time()
            num_speakers = req.numSpeakers
            # each server will be  specialized
            # for specific type of requests, i.e. which differs by number of speakers
            assert (self._num_speakers == num_speakers)
            audio_sz = (298, 257, 2)
            video_sz = (75, 1, 1792, num_speakers)

            result_audio = np.ndarray(shape=audio_sz, dtype=np.float64,
                                      buffer=req.audioContent)

            result_video = np.ndarray(shape=video_sz, dtype=np.float64,
                                      buffer=req.videoContent)

            batch = [np.expand_dims(result_audio, axis=0),
                     np.expand_dims(result_video, axis=0)]

            # send them to the model
            # with tf.device('/device:cpu'):?
            cRMs = self._av_model.predict(batch)
            print(cRMs.shape)

            # prepare number of channels equal to number of speakers
            cRMs = cRMs[0]  # batch contains one sample
            # loop through number of speakers
            speech_channels = []
            for i in range(cRMs.shape[-1]):
                cRM = cRMs[:, :, :, i]
                assert cRM.shape == (298, 257, 2)
                F = fast_icRM(result_audio, cRM)
                print(F.shape)
                T = fast_istft(F, power=False)  # default was false
                speech_channels.append(T)

            # send results to GST API, etc
            print(len(speech_channels))
            print("Server run for: ")
            print(time.time() - start)

            # imitation of stream of text, returned from GST API
            yield av_batch_request_pb2.AVBatchResponse(num=req.timeStamp)

    def _generate_warmup_batch(self):
        audio_sz = (1, 298, 257, 2)
        video_sz = (1, 75, 1, 1792, self._num_speakers)
        sample_audio = np.random.rand(*audio_sz)
        sample_video = np.random.rand(*video_sz)
        batch = [sample_audio, sample_video]
        return batch


def serve():
    model_path = '../av_model/AVmodel-2p-020-0.46028.h5'
    num_speakers = 2
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    av_batch_request_pb2_grpc.add_AVBatchServiceServicer_to_server(
        AVBatchService(model_path, num_speakers), server)
    server.add_insecure_port('[::]:50050')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    try:
        print('Server running on port: 50050')
        serve()
    except KeyboardInterrupt:
        pass
