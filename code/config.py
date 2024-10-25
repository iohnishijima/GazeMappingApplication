# config.py

"""
This module contains global configuration variables and shared data structures used across the application.

Data Sent:
- N/A

Data Returned:
- Shared variables and data structures for configuration and threading.
"""

import threading

# Global variables
camera_matrix = None
dist_coeffs = None
ref_image = None
ref_gray = None
ref_keypoints = None
ref_descriptors = None
orb = None
flann = None
map1_frame = None
map2_frame = None
roi_frame = None
new_camera_mtx_frame = None

# Lock and event for threading
frame_lock = threading.Lock()
frame_available = threading.Event()
shared_data = {
    'frame': None,
    'gaze_x': None,
    'gaze_y': None,
    'frame_num': None,
    'score_right': None,
    'score_left': None,
    'system_time': None
}
