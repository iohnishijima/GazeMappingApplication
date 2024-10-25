# receiver.py

"""
This module handles receiving frames via ZeroMQ in a separate thread.

Functions:
- receive_frames(zmq_address): Receives frames and updates shared data.

Data Sent:
- zmq_address: ZeroMQ address to connect to.

Data Returned:
- Updates shared_data in config.py with the latest frame and associated data.
"""

import zmq
import base64
import cv2
import numpy as np
from config import frame_lock, frame_available, shared_data

def receive_frames(zmq_address):
    # ZeroMQ setup (as a subscriber)
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(zmq_address)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    while True:
        message = socket.recv_pyobj()
        frame_num = message['frame']
        gaze_x = message['gaze_x']
        gaze_y = message['gaze_y']
        score_right = message.get('score_right', 0)
        score_left = message.get('score_left', 0)
        system_time = message.get('system_time', None)
        encoded_image = message['image']

        # Decode image data
        decoded_image = base64.b64decode(encoded_image)
        np_image = np.frombuffer(decoded_image, dtype=np.uint8)
        frame_temp = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        with frame_lock:
            shared_data['frame'] = frame_temp.copy()
            shared_data['gaze_x'] = gaze_x
            shared_data['gaze_y'] = gaze_y
            shared_data['frame_num'] = frame_num  # Use as PicNum
            shared_data['score_right'] = score_right
            shared_data['score_left'] = score_left
            shared_data['system_time'] = system_time
        frame_available.set()
