"""Represents a client which encodes AV batch to e message and sends request to
the av_grpc_server."""
import numpy as np
import grpc
from test_servers import av_batch_request_pb2, av_batch_request_pb2_grpc

num_speakers = 2
audio_sz = (298, 257, 2)

video_sz = (75, 1, 1792, num_speakers)


def gen_request():
    timestamp = 0
    while True:
        sample_audio = np.random.rand(*audio_sz)
        serialized_audio_bytes = sample_audio.tobytes()
        sample_video = np.random.rand(*video_sz)
        serialized_video_bytes = sample_video.tobytes()
        yield av_batch_request_pb2.AVBatchRequest(
            timeStamp=timestamp,
            numSpeakers=num_speakers,
            videoContent=serialized_video_bytes,
            audioContent=serialized_audio_bytes
        )
        timestamp += 1


def run(generator_fn):
    """ Run on a separate thread, part of double stream grpc service"""
    channel = grpc.insecure_channel('localhost:50050', options=(('grpc.enable_http_proxy', 0),))
    stub = av_batch_request_pb2_grpc.AVBatchServiceStub(channel)
    it = stub.SendAVBatch(generator_fn())
    try:
        for r in it:
            print(f"Number from server = {r.num}")
    except grpc._channel._Rendezvous as err:
        print(err)


if __name__ == '__main__':
    try:
        run(gen_request)
    except KeyboardInterrupt:
        pass
