"""
Requirements
1. Numpy
2. Pympler or a recursive sys.getsizeof()
3. PIL
"""

import numpy as np
from pympler.asizeof import asizeof
import json
import io
import base64
from PIL import Image
import time

# Lambda function to represent memory in Mb
get_size = lambda x: asizeof(x) / 10 ** 6

# Sample array - Represents a HD Image , 3 Channels , 256 bit
np_arr = np.random.randint(0, 256, (1920, 1080, 3), dtype=np.uint8)

#video_sz = (1, 75, 1, 1792, 2)
#@np_arr = np.random.rand(*video_sz)
# Encode and transmit Numpy Array in bytes
def encode_and_transmit_numpy_array_in_bytes(numpy_array: np.array) -> str:
    # Create a Byte Stream Pointer
    compressed_file = io.BytesIO()

    # Use PIL JPEG reduction to save the image to bytes
    Image.fromarray(numpy_array).save(compressed_file, format="JPEG")

    # Set index to start position
    compressed_file.seek(0)

    # Convert the byte representation to base 64 representation for REST Post
    return json.dumps(base64.b64encode(compressed_file.read()).decode())


# Receive and decode bytes to numpy array
def receive_and_decode_bytes_to_numpy_array(j_dumps: str) -> np.array:
    # Convert Base 64 representation to byte representation
    compressed_data = base64.b64decode(j_dumps)

    # Read byte array to an Image
    im = Image.open(io.BytesIO(compressed_data))

    # Return Image to numpy array format
    return np.array(im)


print('Original Image Size: {} Mb'.format(get_size(np_arr)))
print('List Transfer: {} Mb'.format(get_size(json.dumps(np_arr.tolist()))))
start = time.time()
j_dumps = encode_and_transmit_numpy_array_in_bytes(np_arr)
print('Transmitted Image Size: {} Mb'.format(get_size(j_dumps)))
print("Encoding took {} secs".format(time.time() - start))
start = time.time()
im = receive_and_decode_bytes_to_numpy_array(j_dumps)
print("Decoding took {} secs".format(time.time() - start))
print('Reloaded Image Size: {} Mb'.format(get_size(im)))