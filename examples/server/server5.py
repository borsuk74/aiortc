"""Refactored previous version which sends batch to the av model.
Similar to av_grpc_client.py as in run_one_call() function,
 the server used for testing should be av_grpc_server_model.py"""
import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid

import cv2
from aiohttp import web
from av import VideoFrame, AudioFrame, AudioResampler
from av.frame import Frame
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
import time

import torch
from fast_mtcnn import FastMTCNN
from face_emb import get_face_net
import tensorflow as tf
import numpy as np
from util.resampler import Resampler
from util.audio_utils import stft, fast_stft, istft, fast_istft

# New way of doing things
import grpc
#  servers used for testing
import test_servers.primefactor_pb2
import test_servers.primefactor_pb2_grpc
#  production code
from test_servers import av_batch_request_pb2, \
    av_batch_request_pb2_grpc  # , speech_response_pb2, speech_response_pb2_grpc

from threading import Thread, Lock, Condition
import speech_response_pb2, speech_response_pb2_grpc

from concurrent import futures

"""Setup logging"""
#logger = logging.getLogger("pc")
# Gets or creates a logger
logger = logging.getLogger(__name__)

# set log level
logger.setLevel(logging.INFO)

# define file handler and set formatter
file_handler = logging.FileHandler('server5.log')
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(funcName)s : %(message)s')
file_handler.setFormatter(formatter)

# add file handler to logger
logger.addHandler(file_handler)

# Logging example
#logger.debug('A debug message')
#logger.info('An info message')
#logger.warning('Something is not right.')
#logger.error('A Major error has happened.')
#logger.critical('Fatal error. Cannot continue')
""""This version works, the only deficiency is pull request from the client."""

AUDIO_PTIME = 0.020  # 20ms audio packetization

ROOT = os.path.dirname(__file__)



pcs = []

"""Demonstrates Producer/Consumer design with Consumer running on a separate thread."""

request_lock = Lock()
global_speech_lock = Lock()
condition = Condition()
# For simplicity queues will contain numbers
request_queue = []
text_response_queue = []
MAX_QUEUE_LEN = 20  # it should never happen
MIN_FACE_WIDTH = 50
MIN_FACE_HEIGHT = 50


def _is_face_valid(face):
    #  restriction on face size
    if all(face.shape) > 0 and face is not None \
            and face.shape[0] > MIN_FACE_WIDTH \
            and face.shape[1] > MIN_FACE_HEIGHT:
        return True
    else:
        return False


class SpeechBackService(speech_response_pb2_grpc.SpeechBackServiceServicer):
    """GRPC Service, which is called from av_grpc_server_model when text from speech was produced.
    It listens to such messages on a separate thread and populates a response_queue,
    which is eventually polled by WebRTC on_datachannel callback"""

    def __init__(self, global_queue, speech_lock):
        self._speech_lock = speech_lock
        self._response_queue = global_queue

    def SendSpeechStream(self, request_iterator, context):
        start = time.time()
        for req in request_iterator:
            print("Request for stream speech received!")
            # factors = prime_factors(req.num)
            # for fac in factors:
            # yield primefactor_pb2.Response(result=fac)
        print("Server run for: ")
        print(time.time() - start)

    def SendSpeechOne(self, request, context):
        #  need to put this values into response queue,
        #  data channel thread will pull it(
        #print("Request for one speech received!")
        logger.info("Data for batch timestamp {} received".format(request.timeStamp))
        #print("Time for timestamp {} after it was sent to server : {}".format(request.timeStamp, time.time()))
        self._speech_lock.acquire()
        logger.info("Speech lock acquired")
        #  here we want to store data returned by SpeechBack service,
        # clear is done just to prevent its growth, onchannel callback should do it
        self._response_queue.clear()
        self._response_queue.append(request.timeStamp)
        self._speech_lock.release()
        logger.info("Speech lock released")
        #print("Before speech response sent.")
        return speech_response_pb2.SpeechResponse(num=42)


def serve_back_speech():
    global text_response_queue
    global global_speech_lock
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    speech_response_pb2_grpc.add_SpeechBackServiceServicer_to_server(SpeechBackService(text_response_queue,
                                                                                       global_speech_lock), server)
    server.add_insecure_port('[::]:50095')
    server.start()
    logger.info('SpeechBackService started on port 50095')
    server.wait_for_termination()


class BatchProcessor:
    '''Used to prepare batch data '''

    def __init__(self, max_audio_interval=3.0):

        self.is_batch_ready = False
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        #######AUDIO##################
        self.audio_batch_array = None
        self.right_window_time = 0.0
        self.left_window_time = 0.0
        self.batch_left_window_time = 0.0
        self.batch_right_window_time = 0.0
        self.curr_audio_array = None
        self.max_audio_interval = max_audio_interval
        ####VIDEO##################
        self.face_net_batch = np.zeros((100, 160, 160, 3), dtype='float')  # used to store faces for a batch
        self.first_face_batch_index = 0  # will contain the end of first face in the batch
        self.second_face_batch_index = 50  # will contain the end of second face in the batch
        # self.frames = []  # images are temporarily stored here and run through face detector periodically
        # self.face_detector_input_size = face_detector_input_size  # number of frames to process by face detector
        # times when valid video frames arrived, used for resampling later
        self.video_timestamps_one = []
        self.video_timestamps_two = []
        self.detector = FastMTCNN(
            stride=4,
            resize=1,
            margin=14,
            factor=0.6,
            keep_all=True,
            device=self.device
        )
        # run warm up for the detector
        self.face_net = get_face_net()

    def add_video_frame(self, frame):
        """Will append a frame, when size is large it will send
        them as a batch through face_detector network"""
        start_vid = time.time()
        frames = []
        img = frame.to_ndarray(format="bgr24")
        logger.debug("Adding video frame of shape {} ".format(img.shape))
        #print("Shape of video_frame is {} ".format(img.shape))
        frames.append(img)
        # run detection on one frame
        faces = self.detector(frames)
        number_of_faces = len(faces)
        faces_valid = [_is_face_valid(faces[i]) for i in range(number_of_faces)]
        logger.debug("Detected {} faces".format(number_of_faces))
        #  print("Detected {} faces".format(number_of_faces))
        # when no valid faces detected, return here
        if not any(faces_valid):
            logger.warning("No valid faces detected")
            return
        #  when one valid face detected, append it to the batch
        if faces_valid[0]:
            faces[0] = cv2.resize(faces[0], (160, 160))
            self.face_net_batch[self.first_face_batch_index, :, :, :] = faces[0]
            self.first_face_batch_index += 1
            self.video_timestamps_one.append(frame.time)
        #  when two were detected and second one is valid, append it to the batch
        if number_of_faces == 2 and faces_valid[1]:
            faces[1] = cv2.resize(faces[1], (160, 160))
            self.face_net_batch[self.second_face_batch_index, :, :, :] = faces[1]
            self.second_face_batch_index += 1
            self.video_timestamps_two.append(frame.time)
        logger.info("Add video frame took {} sec".format(time.time() - start_vid))
        #   print("Add video frame took {} sec".format(time.time() - start_vid))
        # if self.is_batch_ready, create a thread which would run
        # something like process batch on separate thread

    def add_audio_frame(self, frame):
        start_au = time.time()
        audio = frame.to_ndarray()
        if self.curr_audio_array is None:
            self.curr_audio_array = audio
        else:
            # need to  be optimized
            self.curr_audio_array = np.concatenate((self.curr_audio_array, audio), axis=None)
        # Need to refactor when array reaches certain size need to capture
        # left time, right time and copy audio_array
        self.right_window_time = frame.time
        logger.debug("Regular audio frame processed in {} sec".format(time.time() - start_au))
        #print("Regular audio frame processed in {} sec".format(time.time() - start_au))
        if self.curr_audio_array.shape[0] >= self.max_audio_interval * 16000:
            safe_arr_len = min(len(self.curr_audio_array), int(self.max_audio_interval * 16000))
            self.audio_batch_array = self.curr_audio_array[-safe_arr_len:].copy()
            self.curr_audio_array = None
            self.batch_left_window_time = self.left_window_time
            self.batch_right_window_time = self.right_window_time
            self.left_window_time = self.right_window_time
            self.is_batch_ready = True
            logger.debug("Batch audio frame processed in {} sec".format(time.time() - start_au))
            #print("Batch audio frame processed in {} sec".format(time.time() - start_au))

    def get_batch(self, min_face_num=5):
        """Will do batch preparation according to the logic in video.recv.is_batch_ready"""
        start_batch = time.time()
        # Prepare batch here, including resampling
        first_present, second_present = False, False
        if not self.is_batch_ready:
            return None
        # run embedding on video frames
        with tf.device('/device:gpu:1'):
            # output_batch = self.face_net.predict(np.stack(self.face_net_batch, axis=0))
            # BTW, we can integrate multiple faces into one batch, just keep indexes
            # of their positions
            output_emb = self.face_net.predict(self.face_net_batch)
        logger.info("Generation of embedding took : {} sec".format(time.time() - start_batch))
        #print("Generation of embedding took : {} sec".format(time.time() - start_batch))

        sampling_grid = np.linspace(self.batch_left_window_time,
                                    self.batch_right_window_time, 75)
        # we need to have at least 10 frames in 3 sec to pay attention to such faces
        resampled_emb_1 = None
        resampled_emb_2 = None
        logger.info("Face 1 detected in {} cases, face 2 in {} cases".format(self.first_face_batch_index,
                                                                             self.second_face_batch_index - 50))
        #print("First face detected in {} cases.".format(self.first_face_batch_index))
        #print("Second face detected in {} cases.".format(self.second_face_batch_index - 50))
        if self.first_face_batch_index > 0 + min_face_num:
            first_present = True
            output_emb_1 = output_emb[:self.first_face_batch_index]
            # run resampling
            sampler = Resampler(np.array(self.video_timestamps_one), output_emb_1.T)
            resampled_emb_1 = sampler.resample(sampling_grid).T
            logger.debug("Face_1 embedding generated with shape : {} from {} samples ".format(resampled_emb_1.shape,
                                                                                              self.first_face_batch_index))
            #print("Face_1 embedding generated with shape : {} ".format(resampled_emb_1.shape))
            #print("Original batch for face  : {} ".format(output_emb_1.shape))
        if self.second_face_batch_index > 50 + min_face_num:
            second_present = True
            output_emb_2 = output_emb[50:self.second_face_batch_index]
            # run resampling
            sampler = Resampler(np.array(self.video_timestamps_two), output_emb_2.T)
            resampled_emb_2 = sampler.resample(sampling_grid).T
            logger.debug("Face_2 embedding generated with shape : {} from {} samples ".format(resampled_emb_2.shape,
                                                                                              self.second_face_batch_index - 50))
            #print("Face_2 embedding generated with shape : {} ".format(resampled_emb_2.shape))

        # need to figure out how to combine both faces:
        #if resampled_emb_1 is not None:
            #logger.debug("Shape of resampled embedding_1 before combining {}".format(resampled_emb_1.shape))
            #print("Shape of resampled embedding_1 before combining {}".format(resampled_emb_1.shape))
        combined_emb = np.stack([resampled_emb_1, resampled_emb_2], axis=2) \
            if first_present and second_present else np.stack([resampled_emb_1, resampled_emb_1], axis=2)
        # np.expand_dims(resampled_emb_1, axis=2)

        # run fast_stft on audio
        # we need to resample here, we are getting audio sampled at 96000, while
        # for AV_network just 16000 is required.
        audio = fast_stft(self.audio_batch_array)
        logger.debug("Shape of the audio mix: {}".format(audio.shape))
        #print("Shape of the audio mix: {}".format(audio.shape))
        # batch indexes reset
        self.first_face_batch_index = 0  # will contain the end of first face in the batch
        self.second_face_batch_index = 50  # will contain the end of second face in the batch
        # reset video timestamps
        self.video_timestamps_one.clear()
        self.video_timestamps_two.clear()
        # c_comb_emb = combined_emb.copy()
        # c_audio = audio.copy()

        self.is_batch_ready = False
        logger.info("Preparation of the batch took {} sec".format(time.time() - start_batch))
        #print("Preparation of the batch took {} sec".format(time.time() - start_batch))

        return audio, combined_emb


def speech_callback_enqueue(speech_data):
    global text_response_queue
    global global_speech_lock
    global_speech_lock.acquire()
    text_response_queue.append(speech_data)
    global_speech_lock.release()


def run_batch_requests(generator_fn):
    """ Run on a separate thread, part of double stream grpc service"""
    #  Grpc channel, used for communication with AV_model
    channel = grpc.insecure_channel('localhost:50031', options=(('grpc.enable_http_proxy', 0),))
    #  object for remote call of AV_model
    stub = av_batch_request_pb2_grpc.AVBatchServiceStub(channel)
    # it = stub.PrimeFactors(gen())
    # it = stub.PrimeFactors(generator_fn())
    it = stub.SendAVBatch(generator_fn())
    try:
        for response in it:
            #  here we just print result, which indicates a status
            # speech_callback_enqueue(f"Response from AV_model = {response.result}")
            print(f"Response from AV_model = {response.num}")
            logger.info(f"Response from AV_model = {response.num}")

    except grpc._channel._Rendezvous as err:
        #print(err)
        logger.error(err)


batch_time_stamp = 1


def gen_batch_request():
    global request_queue
    global batch_time_stamp
    while True:
        condition.acquire()
        if not request_queue:
            logger.info("Nothing in batch queue, blocked on condition")
            #print("Nothing in queue, consumer is waiting")
            condition.wait()
            logger.info("Batch added, proceed after condition")
            #print("Producer added something to queue and notified the consumer")
        video_batch, audio_batch = request_queue.pop(0)
        batch_time_stamp += 1
        #print("Time for timestamp {} before it was sent to server {}".format(batch_time_stamp, time.time()))
        logger.info("Batch with timestamp {} before it was sent to server".format(batch_time_stamp))
        serialized_audio_bytes = audio_batch.tobytes()
        # shape of video_batch is (batch, 75,1,1792, num_speakers)
        num_speakers = video_batch.shape[-1]
        serialized_video_bytes = video_batch.tobytes()

        request = av_batch_request_pb2.AVBatchRequest(
            timeStamp=batch_time_stamp,
            numSpeakers=num_speakers,
            videoContent=serialized_video_bytes,
            audioContent=serialized_audio_bytes
        )
        logger.info("Before yield request")
        yield request
        logger.info("After yield request")
        condition.notify()
        logger.info("After condition notified")
        condition.release()
        logger.info("After condition released")


class MediaStreamError(Exception):
    pass


class AudioTransformTrack(MediaStreamTrack):
    """
    Process audio input
    """
    kind = "audio"

    _start: float
    _timestamp: int

    def __init__(self, track, comm):
        super().__init__()  # don't forget this!
        self.track = track
        self.communicator = comm
        self.left_window_time = 0.0
        self.right_window_time = 0.0
        self.curr_batch_array = None
        self.resampler = AudioResampler(format='s16', layout='mono', rate=16000)

    async def recv(self) -> Frame:
        """
        Receive the next :class:`~av.audio.frame.AudioFrame`.
        The base implementation just reads silence, subclass
        :class:`AudioStreamTrack` to provide a useful implementation.
        """
        if self.readyState != "live":
            raise MediaStreamError

        frame = await self.track.recv()
        print("Audio input: {:6.2f}".format(frame.time))
        # print(frame.sample_rate)
        # print(frame.format)
        # print(frame.samples)
        # print(frame.layout)
        start = time.time()
        pts = frame.pts
        frame.pts = None
        new_frame = self.resampler.resample(frame)
        new_frame.pts = pts
        new_frame.time_base = frame.time_base
        self.communicator.add_audio_frame(new_frame)
        # audio = frame.to_ndarray()

        # if self.curr_batch_array is None:
        # self.curr_batch_array = audio
        # else:
        # need to  be optimized
        # self.curr_batch_array = np.concatenate((self.curr_batch_array, audio), axis=None)

        # self.right_window_time = frame.time
        # if self.right_window_time - self.left_window_time >= self.max_audio_interval:  # 3 sec of audio
        # self.communicator.is_batch_ready = True
        # 96000 is ugly, will fix later
        # safe_arr_len = min(len(self.curr_batch_array), 96000)
        # self.communicator.audio_array = self.curr_batch_array[-safe_arr_len:].copy()
        # self.curr_batch_array = None
        # self.communicator.left_window_time = self.left_window_time
        # self.communicator.right_window_time = self.right_window_time
        # self.left_window_time = self.right_window_time
        # rebuild a VideoFrame, preserving timing information
        #  To test resampling
        # audio = frame.to_ndarray()
        # new_frame = AudioFrame.from_ndarray(audio[::2])
        # new_frame.pts = frame.pts
        # new_frame.time_base = frame.time_base

        # new_frame = self.resampler.resample(frame)
        # new_frame.pts = frame.pts
        # new_frame.time_base = frame.time_base
        print("Processing audio took {} sec".format(time.time() - start))
        # print(new_frame.sample_rate)
        # print(new_frame.format)
        # print(new_frame.samples)
        # print(new_frame.layout)
        return new_frame

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
        self.device = 'cpu'  # 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # self.detector = FastMTCNN(
        # stride=4,
        # resize=1,
        # margin=14,
        # factor=0.6,
        # keep_all=True,
        # device=self.device
        # )
        with tf.device('/device:gpu:1'):
            self.face_net = get_face_net()
        # memory pre-allocation for the batch
        # self.face_net_batch = np.zeros((160, 160, 160, 3), dtype='float')
        # self.video_timestamps = []
        self.video_frame_counter = 0

    async def recv(self):
        frame = await self.track.recv()
        print("Video input: {:6.2f}".format(frame.time))
        self.video_frame_counter += 1
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

            if self.video_frame_counter % 3 == 0:
                self.communicator.add_video_frame(frame)

            if self.communicator.is_batch_ready:
                audio_batch, video_batch = self.communicator.get_batch()
                logger.info("Batch prepared")
                condition.acquire()
                logger.info("Condition acquired.")
                if len(request_queue) == MAX_QUEUE_LEN:
                    logger.warning("Batch queue is full, producer is waiting on condition")
                    condition.wait()
                    logger.warning("Producer allowed to continue.")
                # in reality should be batch data
                request_queue.append((video_batch, audio_batch))
                condition.notify()
                logger.info("After condition notified")
                condition.release()
                logger.info("After condition released")
            return frame
        else:
            return frame


communicator = BatchProcessor()


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
            global text_response_queue
            # it will grab all available results from the queue
            # and send them to the client. It will be pull, not push(
            # but on the right thread
            global_speech_lock.acquire()
            logger.info("Speech lock acquired")
            while len(text_response_queue) > 0:
                result = text_response_queue.pop(0)
                logger.debug("Result from the text_response_queue : {}".format(str(result)))
                channel.send("Here what result I got: {} ".format(str(result)))
            global_speech_lock.release()
            logger.info("Speech lock released")

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
            pc.addTrack(local_audio)
        elif track.kind == "video":

            local_video = VideoTransformTrack(
                track, communicator,
                transform=params["video_transform"]
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
    thread_back_speech = Thread(target=serve_back_speech)
    thread_back_speech.start()

    time.sleep(1.0)

    batch_request_thread = Thread(target=run_batch_requests, args=(gen_batch_request,))
    batch_request_thread.start()

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
    batch_request_thread.join()
    thread_back_speech.join()
