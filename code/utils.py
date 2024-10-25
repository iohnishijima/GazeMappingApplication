# utils.py

"""
This module provides utility functions used across the application.

Functions:
- precompute_undistort_map(image_shape): Precomputes the undistortion map for a given image shape.
- parse_system_time(system_time_str): Parses a system time string into a timestamp.

Data Sent:
- precompute_undistort_map: image_shape (tuple of image dimensions).
- parse_system_time: system_time_str (string in 'YYYY:MM:DD:HH:MM:SS:MS' format).

Data Returned:
- precompute_undistort_map: map1, map2 (undistortion maps), roi (region of interest), new_camera_mtx (new camera matrix).
- parse_system_time: timestamp (float representing the time in seconds since the epoch).
"""

import cv2
import numpy as np
import datetime
from config import camera_matrix, dist_coeffs

def precompute_undistort_map(image_shape, camera_matrix, dist_coeffs):
    h, w = image_shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=0, centerPrincipalPoint=1)
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_camera_mtx, (w, h), cv2.CV_16SC2)
    return map1, map2, roi, new_camera_mtx

def parse_system_time(system_time_str):
    # Parses system time in the format 'YYYY:MM:DD:HH:MM:SS:MS'
    try:
        parts = system_time_str.split(':')
        year, month, day, hour, minute, second, millisecond = map(int, parts)
        dt = datetime.datetime(year, month, day, hour, minute, second, millisecond * 1000)
        timestamp = dt.timestamp()
        return timestamp
    except Exception:
        return datetime.datetime.now().timestamp()
