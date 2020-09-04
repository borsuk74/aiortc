Audio, video and data channel server
====================================

This example illustrates establishing audio, video and a data channel with a
browser. It also performs some image processing on the video frames using
OpenCV.

Running
-------

First install the required packages:

.. code-block:: console

    $ pip install aiohttp aiortc opencv-python

When you start the example, it will create an HTTP server which you
can connect to from your browser:

.. code-block:: console

    $ python server.py

You can then browse to the following page with your browser:

http://127.0.0.1:8080

Once you click `Start` the browser will send the audio and video from its
webcam to the server.

The server will play a pre-recorded audio clip and send the received video back
to the browser, optionally applying a transform to it.

In parallel to media streams, the browser sends a 'ping' message over the data
channel, and the server replies with 'pong'.

Additional options
------------------

If you want to enable verbose logging, run:

.. code-block:: console

    $ python server.py -v

Generate gRPC code for server and client
----------------------------------------

We are going to use gRPC to generate libraries for Go and Python 3.
To generate the Go code, you'll need to install  protoc_.

.. _protoc: https://github.com/google/protobuf/#protocol-compiler-installation

.. code-block:: bash

 # Python client
 $ pip3 install -U grpcio grpcio-tools
 $ python3 -m grpc_tools.protoc -I protobuf/ --python_out=. --grpc_python_out=. protobuf/primefactor.proto
 # Go
 $ protoc -I protobuf/ --go_out=plugins=grpc:protobuf/ protobuf/primefactor.proto


Credits
-------

The audio file "demo-instruct.wav" was borrowed from the Asterisk
project. It is licensed as Creative Commons Attribution-Share Alike 3.0:

https://wiki.asterisk.org/wiki/display/AST/Voice+Prompts+and+Music+on+Hold+License
