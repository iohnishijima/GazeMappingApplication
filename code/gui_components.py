# gui_components.py

"""
This module contains custom GUI components used in the application.

Classes:
- Communicate(QObject): For inter-thread communication via PyQt signals.
- CollapsibleBox(QWidget): A collapsible group box widget.
- TimeAxisItem(pg.AxisItem): Custom axis item for displaying time on graphs.

Data Sent and Returned:
- N/A
"""

from PyQt5.QtWidgets import QWidget, QToolButton, QSizePolicy, QVBoxLayout
from PyQt5.QtCore import Qt, QPropertyAnimation, pyqtSignal, QObject
import pyqtgraph as pg
import datetime

class Communicate(QObject):
    update_image = pyqtSignal()

class CollapsibleBox(QWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.toggle_button = QToolButton(text=title, checkable=True, checked=False)
        self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.RightArrow)
        self.toggle_button.clicked.connect(self.on_toggle)

        self.content_area = QWidget()
        self.content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.content_area.setMinimumHeight(0)
        self.content_area.setMaximumHeight(0)

        self.animation = QPropertyAnimation(self.content_area, b"maximumHeight")
        self.animation.setDuration(200)
        self.animation.setStartValue(0)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(0)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(self.toggle_button)
        self.main_layout.addWidget(self.content_area)

    def on_toggle(self):
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        if checked:
            # Expand: compute content height
            content_height = self.content_area.layout().sizeHint().height()
            self.animation.setEndValue(content_height)
        else:
            self.animation.setEndValue(0)
        self.animation.start()

    def setContentLayout(self, layout):
        self.content_area.setLayout(layout)
        # Recompute content height
        content_height = layout.sizeHint().height()
        self.animation.setEndValue(content_height)

class TimeAxisItem(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tickStrings(self, values, scale, spacing):
        # Convert timestamps to 'HH:MM:SS' format
        return [datetime.datetime.fromtimestamp(value).strftime('%H:%M:%S') for value in values]
