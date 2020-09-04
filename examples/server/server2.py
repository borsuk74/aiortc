import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid

import cv2
from aiohttp import web
#from aiohttp_requests import requests
from av import VideoFrame, AudioFrame
from av.frame import Frame
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
import time
# import fractions

# from mtcnn.mtcnn import MTCNN

import torch
from fast_mtcnn import FastMTCNN
from face_emb import get_face_net
import tensorflow as tf
import numpy as np
from util.resampler import Resampler
from util.audio_utils import stft, fast_stft, istft, fast_istft
import base64
import requests

""""This version doesn't work, because loop = asyncio.get_event_loop() 
generates runtime error when executed not in the MainThread, which is the case here."""


AUDIO_PTIME = 0.020  # 20ms audio packetization

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")

pcs = []


class BatchCommunicator:
    '''Used to share batch data from Audio stream processor to video stream processor'''

    def __init__(self, audio_steps=50):
        self.audio_steps = audio_steps
        self.is_batch_ready = False
        self.audio_array = None
        self.right_window_time = 0.0
        self.left_window_time = 0.0

def speech_callback(speech_data):
    ''' Hack that uses 'protected' fields to send
    results back to the client  over the data channel '''
    global pcs
    pc = pcs[0]
    if pc is not None:
        # I need to send data back, here is a hack
        channel = next(iter(pc.sctp._data_channels.values()))
        if channel is not None:
            channel.send(speech_data)


#https://stackoverflow.com/questions/52526353/asynchronous-python-fire-and-forget-http-request
def background(f):
    from functools import wraps
    @wraps(f)
    def wrapped(*args, **kwargs):
        loop = asyncio.get_event_loop()
        if callable(f):
            return loop.run_in_executor(None, f, *args, **kwargs)
            #return asyncio.run(f(*args, **kwargs))
        else:
            raise TypeError('Task must be a callable')
    return wrapped

@background
def send_and_forget(host_name, data, callback):
    response = requests.post(host_name, data=data)
    response.raise_for_status()
    prediction = response.text  # for testing
    callback(str(prediction))




class MediaStreamError(Exception):
    pass


class AudioTransformTrack(MediaStreamTrack):
    """
    Process audio input
    """

    kind = "audio"

    _start: float
    _timestamp: int

    def __init__(self, track, comm, max_audio_interval=3.0):
        super().__init__()  # don't forget this!
        self.track = track
        self.communicator = comm
        self.left_window_time = 0.0
        self.right_window_time = 0.0
        self.curr_batch_array = None
        self.max_audio_interval = max_audio_interval

    async def recv(self) -> Frame:
        """
        Receive the next :class:`~av.audio.frame.AudioFrame`.
        The base implementation just reads silence, subclass
        :class:`AudioStreamTrack` to provide a useful implementation.
        """
        if self.readyState != "live":
            raise MediaStreamError

        frame = await self.track.recv()
        audio = frame.to_ndarray()

        if self.curr_batch_array is None:
            self.curr_batch_array = audio
        else:
            # need to  be optimized
            self.curr_batch_array = np.concatenate((self.curr_batch_array, audio), axis=None)

        self.right_window_time = frame.time
        if self.right_window_time - self.left_window_time >= self.max_audio_interval:  # 3 sec of audio
            self.communicator.is_batch_ready = True
            # 96000 is ugly, will fix later
            safe_arr_len = min(len(self.curr_batch_array), 96000)
            self.communicator.audio_array = self.curr_batch_array[-safe_arr_len:].copy()
            self.curr_batch_array = None
            self.communicator.left_window_time = self.left_window_time
            self.communicator.right_window_time = self.right_window_time
            self.left_window_time = self.right_window_time

        print("Audio input: {:6.2f}".format(frame.time))

        return frame

        # sample_rate = 8000
        # samples = int(AUDIO_PTIME * sample_rate)

        # if hasattr(self, "_timestamp"):
        # self._timestamp += samples
        # wait = self._start + (self._timestamp / sample_rate) - time.time()
        # await asyncio.sleep(wait)
        # else:
        # self._start = time.time()
        # self._timestamp = 0

        # frame = AudioFrame(format="s16", layout="mono", samples=samples)
        # for p in frame.planes:
        # p.update(bytes(p.buffer_size))
        # frame.pts = self._timestamp
        # frame.sample_rate = sample_rate
        # frame.time_base = fractions.Fraction(1, sample_rate)
        # return frame


class VideoTransformTrack(MediaStreamTrack):

    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, comm, transform, speech_container=None):
        super().__init__()  # don't forget this!
        self.track = track
        self.communicator = comm
        self.transform = transform
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.detector = FastMTCNN(
            stride=4,
            resize=1,
            margin=14,
            factor=0.6,
            keep_all=True,
            device=self.device
            # ,post_process=False
        )  # MTCNN()
        with tf.device('/device:gpu:1'):
            self.face_net = get_face_net()
        # memory pre-allocation
        self.face_net_batch = np.zeros((100, 160, 160, 3), dtype='float')
        self.video_timestamps = []
        self.frame_counter = 0
        #self.speech_container = speech_container

    async def recv(self):
        frame = await self.track.recv()
        print("Video input: {:6.2f}".format(frame.time))
        self.frame_counter += 1
        if self.transform == "cartoon":
            img = frame.to_ndarray(format="bgr24")

            # prepare color
            img_color = cv2.pyrDown(cv2.pyrDown(img))
            for _ in range(6):
                img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
            img_color = cv2.pyrUp(cv2.pyrUp(img_color))

            # prepare edges
            img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_edges = cv2.adaptiveThreshold(
                cv2.medianBlur(img_edges, 7),
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                9,
                2,
            )
            img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

            # combine color and edges
            img = cv2.bitwise_and(img_color, img_edges)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.transform == "edges":
            # perform edge detection
            img = frame.to_ndarray(format="bgr24")
            img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.transform == "rotate":
            # rotate image
            img = frame.to_ndarray(format="bgr24")
            rows, cols, _ = img.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
            img = cv2.warpAffine(img, M, (cols, rows))

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.transform == "face_detect":
            print("Here I need BlazeFace, ideally which run as a separate microservice at 25fps")
            img = frame.to_ndarray(format="bgr24")
            start = time.time()
            faces = self.detector.detect_faces(img)
            print(time.time() - start)
            if len(faces) > 0:
                print("Some face detected")
                bounding_box = faces[0]['box']
                crop_img = img[bounding_box[1]:bounding_box[1] + bounding_box[3],
                           bounding_box[0]:bounding_box[0] + bounding_box[2]]
                crop_img = cv2.resize(crop_img, (160, 160))
                # cv2.imwrite('%s/frame_' % '.jpg', crop_img)
                new_frame = VideoFrame.from_ndarray(crop_img, format="bgr24")
                new_frame.pts = frame.pts
                new_frame.time_base = frame.time_base
                return new_frame
            else:
                return frame
            # check if detected faces
            # face = await blazeFaceService(frame)
        elif self.transform == "fast_face_detect":
            frames = []
            img = frame.to_ndarray(format="bgr24")
            # Need to investigate what format face detector is expecting ,
            # what it returns and what format FaceNet is expecting
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print("Image converted.")
            frames.append(img)
            start = time.time()
            # here one frame is sent for detection, but detector is optimized specifically
            # for multiple frames.
            faces = self.detector(frames)
            print("Detector run")
            # Here just one face is taken into account, but multiple faces
            # could be potentially concatenated into single batch
            if len(faces) > 0 and all(faces[0].shape) > 0:
                print("Show faces!")
                print(faces[0].shape)
                faces[0] = cv2.resize(faces[0], (160, 160))
                self.video_timestamps.append(frame.time)
                ind = len(self.video_timestamps) - 1
                self.face_net_batch[ind, :, :, :] = faces[0]
                if self.communicator.is_batch_ready:
                    print("Batch processing triggered.")
                    # do batch embedding and resampling here
                    with tf.device('/device:gpu:1'):
                        # output_batch = self.face_net.predict(np.stack(self.face_net_batch, axis=0))
                        # BTW, we can integrate multiple faces into one batch, just keep indexes
                        # of their positions
                        output_emb = self.face_net.predict(self.face_net_batch)
                        output_emb = output_emb[:ind + 1]
                    # do resampling with output_batch here
                    sampler = Resampler(np.array(self.video_timestamps), output_emb.T)
                    sampling_grid = np.linspace(self.communicator.left_window_time,
                                                self.communicator.right_window_time, 75)
                    resampled_emb = sampler.resample(sampling_grid).T
                    print(resampled_emb.shape)
                    # grab audio, transform it to stft and send everything
                    # to the next step down the line
                    audio = fast_stft(self.communicator.audio_array)
                    print("Shape of the audio and video:")
                    print(audio.shape)
                    # send a fake request to remote model, which could represent the rest of pipeline
                    #SERVER_URL = 'http://localhost:8501/v1/models/half_plus_two:predict'
                    SERVER_URL = 'http://localhost:5000/api'# for testing
                    predict_request = '{"instances": [' + str(self.frame_counter) + ']}'
                    send_and_forget(SERVER_URL, predict_request, speech_callback)
                    #response = await requests.post(SERVER_URL, data=predict_request)
                    #response = requests.post(SERVER_URL, data=predict_request)
                    #response.raise_for_status()
                    #prediction = response.json()['predictions'][0]
                    #prediction = await response.text()# for testing
                    #prediction = response.text  # for testing
                    #response is sent to the client
                    #speech_callback(str(prediction))
                    #self.speech_container.text = str(prediction)
                    print(len(self.video_timestamps))
                    self.video_timestamps = []
                    self.communicator.is_batch_ready = False
                    # self.speech_container.text = "Hello" + str(self.frame_counter)

                # rebuild a VideoFrame, preserving timing information
                new_frame = VideoFrame.from_ndarray(faces[0], format="bgr24")
                new_frame.pts = frame.pts
                new_frame.time_base = frame.time_base
                print(time.time() - start)
                return new_frame
            else:
                return frame
        else:
            return frame


#class SpeechContainer:
    #def __init__(self, text="PONG"):
        #self.text = text

    #def clear(self):
        #self.text = ""





#speech_container = SpeechContainer()


communicator = BatchCommunicator()


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)




async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.append(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # prepare local media
    player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))
    if args.write_audio:
        recorder = MediaRecorder(args.write_audio)
    else:
        recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            pass
            #if isinstance(message, str) and message.startswith("ping"):
                #channel.send(speech_container.text)
                #speech_container.clear()

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        log_info("ICE connection state is %s", pc.iceConnectionState)
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.pop(0)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            local_audio = AudioTransformTrack(track, communicator)
            # pc.addTrack(player.audio)
            # recorder.addTrack(track)
            pc.addTrack(local_audio)
        elif track.kind == "video":
            local_video = VideoTransformTrack(
                track, communicator,
                transform=params["video_transform"]
                #,speech_container=speech_container
            )
            pc.addTrack(local_video)

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    parser.add_argument("--write-audio", help="Write received audio to a file")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
