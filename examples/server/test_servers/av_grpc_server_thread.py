"""Server which would show how to deserialize stream of batches, sent from client.
The batches will be placed in queue, processed on  a separate thread and sent
to speech request, modeled here as  primefactor grpc service, implemented in
test_grpc_server.py.
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
from util.audio_utils import stft, fast_stft, istft, fast_istft, fast_icRM, convert_toPyAudio
from threading import Thread, Lock, Condition
import test_servers.primefactor_pb2
import test_servers.primefactor_pb2_grpc
import speech_response_pb2
import speech_response_pb2_grpc

import socket
# Python client
# $ pip3 install -U grpcio grpcio-tools
# $ python3 -m grpc_tools.protoc -I protobuf/ --python_out=. --grpc_python_out=. protobuf/av_batch_request.proto
MAX_QUEUE_LEN = 20
CLIENT_PORT = 12345

class AVBatchService(av_batch_request_pb2_grpc.AVBatchServiceServicer):
    def __init__(self, model_path, num_speakers=2):
        self._model_path = model_path
        self._num_speakers = num_speakers
        self._loss = audio_loss(gamma=0.1, beta=0.2, people_num=self._num_speakers)
        # with tf.device('/device:cpu'):
        with tf.device('/device:gpu:1'):
            self._av_model = load_model(model_path, custom_objects={'tf': tf, 'loss_func': self._loss})
            self._av_model.predict(self._generate_warmup_batch())
        self._batch_queue = [] #  Prepared batches which will be serverd on other thread
        self._channel = grpc.insecure_channel('localhost:50091', options=(('grpc.enable_http_proxy', 0),))
        self._speech_back_stub = speech_response_pb2_grpc.SpeechBackServiceStub(self._channel)
        self._condition = Condition()
        self._thread = Thread(target=self.run, args=(self.gen_request,))
        self._thread.start()
        time.sleep(2.0)


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

            # performance of the conversion from 2 mono channels to stereo channels
            #  is significant : 0.45 sec
            start1 = time.time()
            ch1 = convert_toPyAudio(speech_channels[0])
            # ch1 = np.int16(speech_channels[0]*32767.0)
            # ch2 = np.int16(speech_channels[1]*32767.0)
            ch2 = convert_toPyAudio(speech_channels[1])
            stereo_ch = bytearray()
            for i in range(len(speech_channels[0])):
                data1 = ch1[i].tobytes()
                data2 = ch2[i].tobytes()
                stereo_ch += data1 + data2

            print("Stereo conversion run for: {} secs".format(time.time() - start1))

            print(time.time() - start)
            # send results to GST API, etc
            #print(len(speech_channels))
            print("Server run for: ")
            print(time.time() - start)

            # imitation of stream of text, returned from GST API
            yield av_batch_request_pb2.AVBatchResponse(num=req.timeStamp)

    def SendAVOne(self, request, context):
        # need to deserialize requests here
        print("Request arrived!")
        start = time.time()
        num_speakers = request.numSpeakers
        # each server will be  specialized
        # for specific type of requests, i.e. which differs by number of speakers
        assert (self._num_speakers == num_speakers)
        audio_sz = (298, 257, 2)
        video_sz = (75, 1, 1792, num_speakers)

        result_audio = np.ndarray(shape=audio_sz, dtype=np.float64,
                                  buffer=request.audioContent)

        result_video = np.ndarray(shape=video_sz, dtype=np.float64,
                                  buffer=request.videoContent)

        #batch = [np.expand_dims(result_audio, axis=0),
                 #np.expand_dims(result_video, axis=0)]
        self._condition.acquire()
        if len(self._batch_queue) == MAX_QUEUE_LEN:
            print("Queue full, producer is waiting")
            self._condition.wait()
        self._batch_queue.append((np.expand_dims(result_audio, axis=0), np.expand_dims(result_video, axis=0), request.timeStamp))
        self._condition.notify()
        self._condition.release()

        return av_batch_request_pb2.AVBatchResponse(num=request.timeStamp)

    def _generate_warmup_batch(self):
        audio_sz = (1, 298, 257, 2)
        video_sz = (1, 75, 1, 1792, self._num_speakers)
        sample_audio = np.random.rand(*audio_sz)
        sample_video = np.random.rand(*video_sz)
        batch = [sample_audio, sample_video]
        return batch

    def gen_request(self,):
        #global request_queue
        while True:
            self._condition.acquire()
            if not self._batch_queue:
                print("Nothing in queue, consumer is waiting")
                self._condition.wait()
                print("Producer added something to queue and notified the consumer")
            batch = self._batch_queue.pop(0)
            # process a batch by av network here
            with tf.device('/device:gpu:1'):
                cRMs = self._av_model.predict((batch[0], batch[1]))

            #  from here I demonstrate a generation of next streaming request down the pipeline
            #  it should be call to the next server which processes two sources and
            #  sends them to GST API
            print("Time after av model : {} for timestamp {}".format(time.time(), batch[2]))
            yield test_servers.primefactor_pb2.Request(num=batch[2])
            self._condition.notify()
            self._condition.release()

    def run(self, generator_fn):
        """ Run on a separate thread, part of double stream grpc service"""
        channel = grpc.insecure_channel('localhost:50089')
        stub = test_servers.primefactor_pb2_grpc.FactorsStub(channel)
        # it = stub.PrimeFactors(gen())
        it = stub.PrimeFactors(generator_fn())
        try:
            for r in it:
                # we need to send this response further up the pipeline,
                # to deliver to the client eventually somehow.
                # Probably client should listen to it on separate socket.
                request = speech_response_pb2.SpeechRequest(
                    timeStamp=r.result
                )
                self._speech_back_stub.SendSpeechOne(request)
                #print(f"Prime factor = {r.result}")
                # print(f"Prime factor = {r.result}")
        except grpc._channel._Rendezvous as err:
            print(err)

def serve():
    model_path = '../av_model/AVmodel-2p-020-0.46028.h5'
    num_speakers = 2
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    av_batch_request_pb2_grpc.add_AVBatchServiceServicer_to_server(
        AVBatchService(model_path, num_speakers), server)
    server.add_insecure_port('[::]:50031')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    try:
        print('Server running on port: 50031')
        serve()
    except KeyboardInterrupt:
        pass
