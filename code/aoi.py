# aoi.py

"""
This module defines the AOI (Area of Interest) class used to manage areas on the image.

Classes:
- AOI: Represents an Area of Interest on the image.

Data Sent:
- rect: QRectF defining the area.
- name: Optional name for the AOI.

Data Returned:
- Keeps track of hit counts, dwell time, and whether gaze is inside the AOI.
"""

from PyQt5.QtCore import QRectF

class AOI:
    def __init__(self, rect, name=''):
        self.rect = rect  # QRectF
        self.hit_count = 0
        self.dwell_time = 0.0  # Dwell time in seconds
        self.entry_time = None  # Time when gaze entered the AOI (system_time)
        self.is_gaze_inside = False  # Whether the gaze is inside the AOI
        self.name = name  # Name of the AOI
