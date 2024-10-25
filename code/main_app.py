# main_app.py

"""
This module defines the main application class GazeApp which is the core of the GUI application.

Classes:
- GazeApp(QMainWindow): The main application window.

Data Sent:
- Receives user inputs and events, processes frames, and updates the GUI accordingly.

Data Returned:
- Displays the processed images, graphs, and handles recording and statistics.

Dependencies:
- Imports functions and classes from utils, receiver, gui_components, aoi, and config modules.
"""

import sys
import os
import time
import threading
import numpy as np
import cv2
import json
import csv
import datetime
from collections import deque
import ast

from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget,
                             QSlider, QColorDialog, QPushButton, QCheckBox, QHBoxLayout, QSizePolicy,
                             QFileDialog, QLineEdit, QMessageBox, QTextEdit,
                             QSplitter, QAction, QScrollArea, QInputDialog, QMenu, QActionGroup, QComboBox)
from PyQt5.QtGui import QImage, QPixmap, QColor, QIcon
from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF, QTranslator

import pyqtgraph as pg

from utils import precompute_undistort_map, parse_system_time
from receiver import receive_frames
from gui_components import Communicate, CollapsibleBox, TimeAxisItem
from aoi import AOI
import config


class GazeApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize translator for language support
        self.translator = QTranslator()
        QApplication.instance().installTranslator(self.translator)
        self.current_language = 'ja'  # Default language is Japanese

        # Initialize variables
        self.frame_undistorted = None
        self.ref_image_display = None
        self.gaze_point_size = 10
        self.gaze_point_color = (0, 0, 255)  # Red color (BGR)
        self.gaze_point_opacity = 1.0        # Opacity (1.0: opaque, 0.0: transparent)
        self.show_fps = True
        self.previous_time = time.time()
        self.fps = 0
        self.overlay_scene = False           # Flag for overlaying scene camera
        self.scene_opacity = 0.5             # Opacity for scene camera
        self.aoi_list = []                   # List of AOIs
        self.drawing_aoi = False             # Whether an AOI is being drawn
        self.aoi_start_point = None          # Start point of AOI
        self.is_configured = False           # Flag indicating configuration is complete
        self.reset_requested = False         # Flag for resetting counts
        self.previous_frame_shape = None

        # Path to GazeVisualizeSoftware folder in user's Documents
        self.base_directory = os.path.join(os.path.expanduser('~/Documents'), 'GazeVisualizeSoftware')
        os.makedirs(self.base_directory, exist_ok=True)

        # Get user list
        self.users = self.get_user_list()
        self.current_user = None
        self.current_session = None

        # Gaze data history
        self.max_history = 100               # Default history frame count
        self.gaze_history = deque(maxlen=self.max_history)  # Gaze coordinate history
        self.heatmap_opacity = 0.5           # Heatmap opacity

        # Recording related
        self.is_recording = False
        self.recorded_data = []
        self.frame_counter = 0  # Frame number in software
        self.csv_filename = "recorded_data.csv"

        # Data buffers for graph
        self.graph_data_right = []
        self.graph_data_left = []
        self.graph_time = []

        # Application start time
        self.start_time = None  # Initialized in apply_settings

        # Setup UI
        self.init_ui()

        # Signals for inter-thread communication
        self.comm = Communicate()
        self.comm.update_image.connect(self.update_frame)

    def init_ui(self):
        self.setWindowTitle(self.tr('視線ポイントビューア'))

        # メインウィジェットとレイアウト
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # メインレイアウトにQSplitterを使用
        self.splitter = QSplitter(Qt.Horizontal)
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.splitter)
        main_widget.setLayout(main_layout)

        # サイドバー (左側)
        self.sidebar_widget = QWidget()
        self.sidebar_layout = QVBoxLayout()
        self.sidebar_widget.setLayout(self.sidebar_layout)

        # スクロール可能なサイドバー
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.sidebar_widget)

        # 各設定グループを作成
        self.create_initial_settings_group()
        self.create_other_settings_group()
        self.create_heatmap_settings_group()
        self.create_record_settings_group()
        self.create_statistics_group()
        self.create_graph_group()

        # サイドバーにグループを追加
        self.sidebar_layout.addWidget(self.initial_settings_group)
        self.sidebar_layout.addWidget(self.other_settings_group)
        self.sidebar_layout.addWidget(self.heatmap_settings_group)
        self.sidebar_layout.addWidget(self.record_settings_group)
        self.sidebar_layout.addWidget(self.statistics_group)
        self.sidebar_layout.addWidget(self.graph_group)
        self.sidebar_layout.addStretch()

        # 画像表示エリア
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.mousePressEvent = self.image_mouse_press_event
        self.image_label.mouseMoveEvent = self.image_mouse_move_event
        self.image_label.mouseReleaseEvent = self.image_mouse_release_event
        self.image_label.mouseDoubleClickEvent = self.image_mouse_double_click_event  # ダブルクリックイベント

        # サイドバーと画像ラベルをスプリッタに追加
        self.splitter.addWidget(self.scroll_area)
        self.splitter.addWidget(self.image_label)
        self.splitter.setStretchFactor(1, 1)  # 画像表示部分を伸縮可能に設定

        # 最初はサイドバーを開く
        self.sidebar_widget.setVisible(True)

        # メニューアクションを作成
        self.create_menu()

    def get_user_list(self):
        if os.path.exists(self.base_directory):
            # Get list of user folder names
            return [name for name in os.listdir(self.base_directory)
                    if os.path.isdir(os.path.join(self.base_directory, name))]
        else:
            return []

    def create_menu(self):
        # Create menu bar
        menubar = self.menuBar()
        self.file_menu = menubar.addMenu(self.tr('ファイル'))

        # Save AOI
        self.save_aoi_action = QAction(self.tr('AOIを保存'), self)
        self.save_aoi_action.triggered.connect(self.save_aoi)
        self.file_menu.addAction(self.save_aoi_action)

        # Load AOI
        self.load_aoi_action = QAction(self.tr('AOIを読み込み'), self)
        self.load_aoi_action.triggered.connect(self.load_aoi)
        self.file_menu.addAction(self.load_aoi_action)

        self.view_menu = menubar.addMenu(self.tr('表示'))

        # Action to toggle sidebar visibility
        self.toggle_sidebar_action = QAction(self.tr('サイドバーを表示'), self, checkable=True)
        self.toggle_sidebar_action.setChecked(True)
        self.toggle_sidebar_action.triggered.connect(self.toggle_sidebar)
        self.view_menu.addAction(self.toggle_sidebar_action)

        # Add language menu
        self.language_menu = menubar.addMenu(self.tr('言語'))
        self.language_action_group = QActionGroup(self)
        self.language_action_group.setExclusive(True)

        # Japanese action
        self.ja_action = QAction('日本語', self, checkable=True)
        self.ja_action.setChecked(True)
        self.ja_action.triggered.connect(lambda: self.change_language('ja'))
        self.language_action_group.addAction(self.ja_action)
        self.language_menu.addAction(self.ja_action)

        # English action
        self.en_action = QAction('English', self, checkable=True)
        self.en_action.triggered.connect(lambda: self.change_language('en'))
        self.language_action_group.addAction(self.en_action)
        self.language_menu.addAction(self.en_action)

    def change_language(self, language_code):
        if language_code == 'ja':
            self.translator.load('../data/ja.qm')
            self.current_language = 'ja'
        elif language_code == 'en':
            self.translator.load('../data/en.qm')
            self.current_language = 'en'
        QApplication.instance().installTranslator(self.translator)
        self.retranslate_ui()

    def retranslate_ui(self):
        self.setWindowTitle(self.tr('視線ポイントビューア'))
        # Retranslate menus
        self.file_menu.setTitle(self.tr('ファイル'))
        self.save_aoi_action.setText(self.tr('AOIを保存'))
        self.load_aoi_action.setText(self.tr('AOIを読み込み'))
        self.view_menu.setTitle(self.tr('表示'))
        self.toggle_sidebar_action.setText(self.tr('サイドバーを表示'))
        self.language_menu.setTitle(self.tr('言語'))

        # Reset titles of collapsible group boxes
        self.initial_settings_group.toggle_button.setText(self.tr('初期設定'))
        self.other_settings_group.toggle_button.setText(self.tr('その他の設定'))
        self.heatmap_settings_group.toggle_button.setText(self.tr('ヒートマップ設定'))
        self.record_settings_group.toggle_button.setText(self.tr('レコード設定'))
        self.statistics_group.toggle_button.setText(self.tr('統計情報'))
        self.graph_group.toggle_button.setText(self.tr('リアルタイムグラフ'))

        # Widgets in initial settings group
        self.image_browse_button.setText(self.tr('参照'))
        self.image_label_text.setText(self.tr('基準画像ファイル:'))
        self.zmq_label.setText(self.tr('ZMQアドレス:'))
        self.camera_matrix_label.setText(self.tr('カメラ行列 (3x3):'))
        self.camera_matrix_text.setPlaceholderText(self.tr('例: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]'))
        self.dist_coeffs_label.setText(self.tr('歪み係数 (5つ):'))
        self.dist_coeffs_text.setPlaceholderText(self.tr('例: [k1, k2, p1, p2, k3]'))
        self.configure_button.setText(self.tr('設定完了'))

        # Widgets in other settings group
        self.size_label.setText(self.tr('視線ポイントのサイズ:'))
        self.opacity_label.setText(self.tr('視線ポイントの透明度:'))
        self.color_button.setText(self.tr('視線ポイントの色を選択'))
        self.fps_checkbox.setText(self.tr('FPSを表示'))
        self.overlay_checkbox.setText(self.tr('シーンカメラを重ねて表示'))
        self.scene_opacity_label.setText(self.tr('シーンカメラの透明度:'))
        self.reset_button.setText(self.tr('カウントリセット'))

        # Widgets in heatmap settings group
        self.heatmap_checkbox.setText(self.tr('ヒートマップを表示'))
        self.heatmap_opacity_label.setText(self.tr('ヒートマップの透明度:'))
        self.history_label.setText(self.tr('履歴フレーム数:'))

        # Widgets in record settings group
        self.csv_label.setText(self.tr('CSVファイル名:'))
        self.record_start_button.setText(self.tr('レコード開始'))
        self.record_stop_button.setText(self.tr('レコード停止'))

        self.session_start_button.setText(self.tr('セッション開始'))
        self.session_end_button.setText(self.tr('セッション終了'))
        self.user_label.setText(self.tr('ユーザー名:'))

    def toggle_sidebar(self, state):
        self.sidebar_widget.setVisible(state)
        self.scroll_area.setVisible(state)

    def user_finished_editing(self):
        user_name = self.user_combobox.currentText().strip()
        if user_name:
            if user_name not in self.users:
                self.user_changed(user_name)
                self.user_combobox.addItem(user_name)
                self.users.append(user_name)
                index = self.user_combobox.findText(user_name)
                self.user_combobox.setCurrentIndex(index)
            else:
                # If existing user name is entered, select that user
                index = self.user_combobox.findText(user_name)
                if index >= 0:
                    self.user_combobox.setCurrentIndex(index)
        else:
            QMessageBox.warning(self, self.tr("エラー"), self.tr("ユーザー名を入力してください。"))

    def user_selected(self, index):
        if index >= 0:
            user_name = self.user_combobox.itemText(index).strip()
            self.user_changed(user_name)

    def user_changed(self, user_name):
        self.current_user = user_name

        user_directory = os.path.join(self.base_directory, self.current_user)
        os.makedirs(user_directory, exist_ok=True)

        # Get session list
        self.sessions = self.get_session_list()
        # Update session-related UI if necessary

    def start_session(self):
        if not self.current_user:
            QMessageBox.warning(self, self.tr("エラー"), self.tr("ユーザー名を入力してください。"))
            return

        # Input or select session name
        existing_sessions = self.get_session_list()
        session_name, ok = QInputDialog.getItem(self, self.tr('セッション名'),
                                                self.tr('セッションを選択するか新しいセッション名を入力してください:'),
                                                existing_sessions, editable=True)
        if ok and session_name:
            self.current_session = session_name.strip()
            session_directory = os.path.join(self.base_directory, self.current_user, self.current_session)
            os.makedirs(session_directory, exist_ok=True)
            self.session_start_button.setEnabled(False)
            self.session_end_button.setEnabled(True)
            self.record_start_button.setEnabled(True)
            self.record_stop_button.setEnabled(False)
        else:
            QMessageBox.warning(self, self.tr("エラー"), self.tr("セッション名を入力してください。"))

    def end_session(self):
        # Stop recording
        self.stop_recording()
        self.current_session = None
        self.session_start_button.setEnabled(True)
        self.session_end_button.setEnabled(False)
        self.record_start_button.setEnabled(False)
        self.record_stop_button.setEnabled(False)

    def get_session_list(self):
        if self.current_user:
            user_directory = os.path.join(self.base_directory, self.current_user)
            if os.path.exists(user_directory):
                return [name for name in os.listdir(user_directory)
                        if os.path.isdir(os.path.join(user_directory, name))]
        return []

    def create_initial_settings_group(self):
        # Initial settings group
        self.initial_settings_group = CollapsibleBox(self.tr('初期設定'))
        layout = QVBoxLayout()

        # Image file selection
        image_layout = QHBoxLayout()
        self.image_path_edit = QLineEdit()
        self.image_browse_button = QPushButton(self.tr('参照'))
        self.image_browse_button.clicked.connect(self.browse_image)
        self.image_label_text = QLabel(self.tr('基準画像ファイル:'))
        image_layout.addWidget(self.image_label_text)
        image_layout.addWidget(self.image_path_edit)
        image_layout.addWidget(self.image_browse_button)
        layout.addLayout(image_layout)

        # ZMQ address input
        zmq_layout = QHBoxLayout()
        self.zmq_address_edit = QLineEdit("tcp://localhost:5555")
        self.zmq_label = QLabel(self.tr('ZMQアドレス:'))
        zmq_layout.addWidget(self.zmq_label)
        zmq_layout.addWidget(self.zmq_address_edit)
        layout.addLayout(zmq_layout)

        # Camera matrix input
        self.camera_matrix_text = QTextEdit()
        self.camera_matrix_text.setPlaceholderText(self.tr('例: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]'))
        self.camera_matrix_label = QLabel(self.tr('カメラ行列 (3x3):'))
        layout.addWidget(self.camera_matrix_label)
        layout.addWidget(self.camera_matrix_text)

        # Distortion coefficients input
        self.dist_coeffs_text = QTextEdit()
        self.dist_coeffs_text.setPlaceholderText(self.tr('例: [k1, k2, p1, p2, k3]'))
        self.dist_coeffs_label = QLabel(self.tr('歪み係数 (5つ):'))
        layout.addWidget(self.dist_coeffs_label)
        layout.addWidget(self.dist_coeffs_text)

        # User name input
        user_layout = QHBoxLayout()
        self.user_label = QLabel(self.tr('ユーザー名:'))
        self.user_combobox = QComboBox()
        self.user_combobox.setEditable(True)
        self.user_combobox.setPlaceholderText(self.tr('ユーザー名を入力または選択'))
        self.user_combobox.addItems(self.users)

        # Connect signals
        self.user_combobox.currentIndexChanged.connect(self.user_selected)
        self.user_combobox.lineEdit().returnPressed.connect(self.user_finished_editing)

        user_layout.addWidget(self.user_label)
        user_layout.addWidget(self.user_combobox)
        layout.addLayout(user_layout)

        # Configure button
        self.configure_button = QPushButton(self.tr('設定完了'))
        self.configure_button.clicked.connect(self.apply_settings)
        layout.addWidget(self.configure_button)

        self.initial_settings_group.setContentLayout(layout)

    def create_other_settings_group(self):
        # Other settings group
        self.other_settings_group = CollapsibleBox(self.tr('その他の設定'))
        layout = QVBoxLayout()

        # Gaze point size slider
        size_layout = QHBoxLayout()
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMinimum(1)
        self.size_slider.setMaximum(50)
        self.size_slider.setValue(self.gaze_point_size)
        self.size_slider.setTickPosition(QSlider.TicksBelow)
        self.size_slider.setTickInterval(5)
        self.size_slider.valueChanged.connect(self.change_point_size)
        self.size_label = QLabel(self.tr('視線ポイントのサイズ:'))
        self.size_value_label = QLabel(str(self.gaze_point_size))
        size_layout.addWidget(self.size_label)
        size_layout.addWidget(self.size_slider)
        size_layout.addWidget(self.size_value_label)
        layout.addLayout(size_layout)

        # Gaze point opacity slider
        opacity_layout = QHBoxLayout()
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(int(self.gaze_point_opacity * 100))
        self.opacity_slider.setTickPosition(QSlider.TicksBelow)
        self.opacity_slider.setTickInterval(10)
        self.opacity_slider.valueChanged.connect(self.change_opacity)
        self.opacity_label = QLabel(self.tr('視線ポイントの透明度:'))
        self.opacity_value_label = QLabel(str(int(self.gaze_point_opacity * 100)))
        opacity_layout.addWidget(self.opacity_label)
        opacity_layout.addWidget(self.opacity_slider)
        opacity_layout.addWidget(self.opacity_value_label)
        layout.addLayout(opacity_layout)

        # Gaze point color selection button
        color_layout = QHBoxLayout()
        self.color_button = QPushButton(self.tr('視線ポイントの色を選択'))
        self.color_button.clicked.connect(self.select_color)
        layout.addWidget(self.color_button)

        # FPS display checkbox
        fps_layout = QHBoxLayout()
        self.fps_checkbox = QCheckBox(self.tr('FPSを表示'))
        self.fps_checkbox.setChecked(self.show_fps)
        self.fps_checkbox.stateChanged.connect(self.toggle_fps)
        fps_layout.addWidget(self.fps_checkbox)
        layout.addLayout(fps_layout)

        # Scene camera overlay checkbox
        overlay_layout = QHBoxLayout()
        self.overlay_checkbox = QCheckBox(self.tr('シーンカメラを重ねて表示'))
        self.overlay_checkbox.setChecked(self.overlay_scene)
        self.overlay_checkbox.stateChanged.connect(self.toggle_overlay)
        overlay_layout.addWidget(self.overlay_checkbox)
        layout.addLayout(overlay_layout)

        # Scene camera opacity slider
        scene_opacity_layout = QHBoxLayout()
        self.scene_opacity_slider = QSlider(Qt.Horizontal)
        self.scene_opacity_slider.setMinimum(0)
        self.scene_opacity_slider.setMaximum(100)
        self.scene_opacity_slider.setValue(int(self.scene_opacity * 100))
        self.scene_opacity_slider.setTickPosition(QSlider.TicksBelow)
        self.scene_opacity_slider.setTickInterval(10)
        self.scene_opacity_slider.valueChanged.connect(self.change_scene_opacity)
        self.scene_opacity_label = QLabel(self.tr('シーンカメラの透明度:'))
        self.scene_opacity_value_label = QLabel(str(int(self.scene_opacity * 100)))
        scene_opacity_layout.addWidget(self.scene_opacity_label)
        scene_opacity_layout.addWidget(self.scene_opacity_slider)
        scene_opacity_layout.addWidget(self.scene_opacity_value_label)
        layout.addLayout(scene_opacity_layout)

        # Reset counts button
        reset_layout = QHBoxLayout()
        self.reset_button = QPushButton(self.tr('カウントリセット'))
        self.reset_button.clicked.connect(self.reset_counts)
        layout.addWidget(self.reset_button)

        self.other_settings_group.setContentLayout(layout)

    def create_heatmap_settings_group(self):
        # Heatmap settings group
        self.heatmap_settings_group = CollapsibleBox(self.tr('ヒートマップ設定'))
        layout = QVBoxLayout()

        # Enable/disable heatmap
        self.heatmap_checkbox = QCheckBox(self.tr('ヒートマップを表示'))
        self.heatmap_checkbox.setChecked(False)
        layout.addWidget(self.heatmap_checkbox)

        # Heatmap opacity slider
        heatmap_opacity_layout = QHBoxLayout()
        self.heatmap_opacity_slider = QSlider(Qt.Horizontal)
        self.heatmap_opacity_slider.setMinimum(0)
        self.heatmap_opacity_slider.setMaximum(100)
        self.heatmap_opacity_slider.setValue(int(self.heatmap_opacity * 100))
        self.heatmap_opacity_slider.setTickPosition(QSlider.TicksBelow)
        self.heatmap_opacity_slider.setTickInterval(10)
        self.heatmap_opacity_slider.valueChanged.connect(self.change_heatmap_opacity)
        self.heatmap_opacity_label = QLabel(self.tr('ヒートマップの透明度:'))
        self.heatmap_opacity_value_label = QLabel(str(int(self.heatmap_opacity * 100)))
        heatmap_opacity_layout.addWidget(self.heatmap_opacity_label)
        heatmap_opacity_layout.addWidget(self.heatmap_opacity_slider)
        heatmap_opacity_layout.addWidget(self.heatmap_opacity_value_label)
        layout.addLayout(heatmap_opacity_layout)

        # Heatmap history frame count
        history_layout = QHBoxLayout()
        self.history_slider = QSlider(Qt.Horizontal)
        self.history_slider.setMinimum(1)
        self.history_slider.setMaximum(1000)
        self.history_slider.setValue(self.max_history)
        self.history_slider.setTickPosition(QSlider.TicksBelow)
        self.history_slider.setTickInterval(100)
        self.history_slider.valueChanged.connect(self.change_history)
        self.history_label = QLabel(self.tr('履歴フレーム数:'))
        self.history_value_label = QLabel(str(self.max_history))
        history_layout.addWidget(self.history_label)
        history_layout.addWidget(self.history_slider)
        history_layout.addWidget(self.history_value_label)
        layout.addLayout(history_layout)

        self.heatmap_settings_group.setContentLayout(layout)

    def create_record_settings_group(self):
        # レコード設定グループ
        self.record_settings_group = CollapsibleBox(self.tr('レコード設定'))
        layout = QVBoxLayout()

        # CSVファイル名入力
        csv_layout = QHBoxLayout()
        self.csv_filename_edit = QLineEdit(self.csv_filename)
        self.csv_label = QLabel(self.tr('CSVファイル名:'))
        csv_layout.addWidget(self.csv_label)
        csv_layout.addWidget(self.csv_filename_edit)
        layout.addLayout(csv_layout)

        # セッション開始/終了ボタン
        session_layout = QHBoxLayout()
        self.session_start_button = QPushButton(self.tr('セッション開始'))
        self.session_start_button.clicked.connect(self.start_session)
        self.session_end_button = QPushButton(self.tr('セッション終了'))
        self.session_end_button.clicked.connect(self.end_session)
        self.session_end_button.setEnabled(False)  # セッション終了ボタンは初期状態で無効化
        session_layout.addWidget(self.session_start_button)
        session_layout.addWidget(self.session_end_button)
        layout.addLayout(session_layout)

        # レコード開始/停止ボタン
        record_layout = QHBoxLayout()
        self.record_start_button = QPushButton(self.tr('レコード開始'))
        self.record_start_button.clicked.connect(self.start_recording)
        self.record_stop_button = QPushButton(self.tr('レコード停止'))
        self.record_stop_button.clicked.connect(self.stop_recording)
        self.record_start_button.setEnabled(False)  # 初期状態で無効化
        self.record_stop_button.setEnabled(False)   # 初期状態で無効化
        record_layout.addWidget(self.record_start_button)
        record_layout.addWidget(self.record_stop_button)
        layout.addLayout(record_layout)

        self.record_settings_group.setContentLayout(layout)

    def create_statistics_group(self):
        # Statistics group
        self.statistics_group = CollapsibleBox(self.tr('統計情報'))
        layout = QVBoxLayout()

        # Layout for displaying AOI statistics
        self.statistics_layout = QVBoxLayout()
        layout.addLayout(self.statistics_layout)

        self.statistics_group.setContentLayout(layout)

    def create_graph_group(self):
        # Graph display group
        self.graph_group = CollapsibleBox(self.tr('リアルタイムグラフ'))
        layout = QVBoxLayout()

        # Create plot widget
        time_axis = TimeAxisItem(orientation='bottom')
        self.plot_widget = pg.PlotWidget(axisItems={'bottom': time_axis})
        self.plot_widget.setBackground('w')
        self.plot_widget.addLegend()
        self.plot_widget.showGrid(x=True, y=True)

        # Create data lines
        self.score_right_line = self.plot_widget.plot(pen='r', name='Score Right')
        self.score_left_line = self.plot_widget.plot(pen='b', name='Score Left')

        layout.addWidget(self.plot_widget)
        self.graph_group.setContentLayout(layout)

    def change_heatmap_opacity(self, value):
        self.heatmap_opacity = value / 100.0
        self.heatmap_opacity_value_label.setText(str(value))

    def change_history(self, value):
        self.max_history = value
        self.history_value_label.setText(str(value))
        # Update maxlen of gaze_history
        self.gaze_history = deque(self.gaze_history, maxlen=self.max_history)

    def reset_counts(self):
        for aoi in self.aoi_list:
            aoi.hit_count = 0
            aoi.dwell_time = 0.0
            aoi.entry_time = None
        # Update statistics
        self.update_statistics()

    def browse_image(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(
            self, self.tr("基準画像を選択"), "", self.tr("画像ファイル (*.png *.jpg *.jpeg);;全てのファイル (*)"), options=options)
        if filename:
            self.image_path_edit.setText(filename)

    def apply_settings(self):
        # Load reference image
        image_path = self.image_path_edit.text()
        if not image_path:
            QMessageBox.warning(self, self.tr("エラー"), self.tr("基準画像ファイルを選択してください。"))
            return

        config.ref_image = cv2.imread(image_path)
        if config.ref_image is None:
            QMessageBox.warning(self, self.tr("エラー"), self.tr("基準画像の読み込みに失敗しました。"))
            return

        # Get camera matrix
        try:
            camera_matrix_str = self.camera_matrix_text.toPlainText()
            camera_matrix_values = ast.literal_eval(camera_matrix_str)
            config.camera_matrix = np.array(camera_matrix_values, dtype=np.float64)
            if config.camera_matrix.shape != (3, 3):
                raise ValueError
        except Exception:
            QMessageBox.warning(self, self.tr("エラー"), self.tr("カメラ行列の値を正しく入力してください。"))
            return

        # Get distortion coefficients
        try:
            dist_coeffs_str = self.dist_coeffs_text.toPlainText()
            dist_coeffs_values = ast.literal_eval(dist_coeffs_str)
            config.dist_coeffs = np.array(dist_coeffs_values, dtype=np.float64)
            if config.dist_coeffs.shape[0] != 5:
                raise ValueError
        except Exception:
            QMessageBox.warning(self, self.tr("エラー"), self.tr("歪み係数の値を正しく入力してください。"))
            return

        if not self.current_user:
            QMessageBox.warning(self, self.tr("エラー"), self.tr("ユーザー名を入力してください。"))
            return

        # Initialize ORB feature detector
        config.orb = cv2.ORB_create(nfeatures=300, fastThreshold=7, scaleFactor=1.2,
                                 nlevels=8, edgeThreshold=31, patchSize=31)

        # Compute keypoints and descriptors for reference image
        config.ref_gray = cv2.cvtColor(config.ref_image, cv2.COLOR_BGR2GRAY)
        config.ref_keypoints, config.ref_descriptors = config.orb.detectAndCompute(config.ref_gray, None)

        # Create matcher (FlannBasedMatcher)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        search_params = dict(checks=50)
        config.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Precompute undistortion map for frames (with dummy frame size)
        dummy_frame = np.zeros_like(config.ref_image)
        config.map1_frame, config.map2_frame, config.roi_frame, config.new_camera_mtx_frame = precompute_undistort_map(
            dummy_frame.shape, config.camera_matrix, config.dist_coeffs)

        # Get ZMQ address
        zmq_address = self.zmq_address_edit.text()

        # Start frame receiving thread
        self.receive_thread = threading.Thread(target=receive_frames, args=(zmq_address,))
        self.receive_thread.daemon = True
        self.receive_thread.start()

        # After configuration, enable control panel
        self.is_configured = True
        # Enable menu bar
        self.menuBar().setEnabled(True)

        # Start timer for frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16)  # Update at approximately 60 FPS

        # Set application start time
        self.start_time = time.time()

    def change_point_size(self, value):
        self.gaze_point_size = value
        self.size_value_label.setText(str(value))

    def change_opacity(self, value):
        self.gaze_point_opacity = value / 100.0
        self.opacity_value_label.setText(str(value))

    def change_scene_opacity(self, value):
        self.scene_opacity = value / 100.0
        self.scene_opacity_value_label.setText(str(value))

    def select_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            # Convert QColor to BGR tuple
            self.gaze_point_color = (
                color.blue(), color.green(), color.red())

    def toggle_fps(self, state):
        self.show_fps = state == Qt.Checked

    def toggle_overlay(self, state):
        self.overlay_scene = state == Qt.Checked

    def save_aoi(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(
            self, self.tr("AOIを保存"), "", self.tr("AOIファイル (*.aoi);;全てのファイル (*)"), options=options)
        if filename:
            try:
                aoi_data = []
                for aoi in self.aoi_list:
                    aoi_data.append({
                        'name': aoi.name,
                        'rect': [aoi.rect.left(), aoi.rect.top(), aoi.rect.width(), aoi.rect.height()]
                    })
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(aoi_data, f, ensure_ascii=False, indent=4)
                QMessageBox.information(self, self.tr("保存完了"), self.tr("AOIを保存しました。"))
            except Exception as e:
                QMessageBox.warning(self, self.tr("エラー"), f"{self.tr('AOIの保存中にエラーが発生しました:')} {e}")

    def load_aoi(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(
            self, self.tr("AOIを読み込み"), "", self.tr("AOIファイル (*.aoi);;全てのファイル (*)"), options=options)
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    aoi_data = json.load(f)
                self.aoi_list.clear()
                for item in aoi_data:
                    name = item['name']
                    left, top, width, height = item['rect']
                    rect = QRectF(left, top, width, height)
                    aoi = AOI(rect, name)
                    self.aoi_list.append(aoi)
                self.update_frame()
                QMessageBox.information(self, self.tr("読み込み完了"), self.tr("AOIを読み込みました。"))
            except Exception as e:
                QMessageBox.warning(self, self.tr("エラー"), f"{self.tr('AOIの読み込み中にエラーが発生しました:')} {e}")

    def image_mouse_press_event(self, event):
        if event.button() == Qt.LeftButton and self.is_configured:
            # Convert click position to coordinates in reference image
            pos = event.pos()
            pixmap = self.image_label.pixmap()
            if pixmap:
                label_rect = self.image_label.contentsRect()
                label_width = label_rect.width()
                label_height = label_rect.height()
                pixmap_width = pixmap.width()
                pixmap_height = pixmap.height()
                scaled_w = pixmap_width * min(label_width / pixmap_width, label_height / pixmap_height)
                scaled_h = pixmap_height * min(label_width / pixmap_width, label_height / pixmap_height)
                offset_x = (label_width - scaled_w) / 2
                offset_y = (label_height - scaled_h) / 2
                scale_x = pixmap_width / scaled_w
                scale_y = pixmap_height / scaled_h
                x = (pos.x() - offset_x) * scale_x
                y = (pos.y() - offset_y) * scale_y
                # Check if click is inside an existing AOI
                for aoi in reversed(self.aoi_list):
                    if aoi.rect.contains(QPointF(x, y)):
                        # If inside existing AOI, do not start new AOI
                        return
            # Start creating new AOI
            self.drawing_aoi = True
            self.aoi_start_point = event.pos()

    def image_mouse_move_event(self, event):
        if self.drawing_aoi and self.is_configured:
            self.aoi_end_point = event.pos()
            self.update_frame(draw_aoi_preview=True)

    def image_mouse_release_event(self, event):
        if event.button() == Qt.LeftButton and self.is_configured and self.drawing_aoi:
            self.drawing_aoi = False
            self.aoi_end_point = event.pos()
            # Calculate AOI rectangle
            start = self.aoi_start_point
            end = self.aoi_end_point
            # Scale coordinates to reference image
            pixmap = self.image_label.pixmap()
            if pixmap:
                label_rect = self.image_label.contentsRect()
                label_width = label_rect.width()
                label_height = label_rect.height()
                pixmap_width = pixmap.width()
                pixmap_height = pixmap.height()
                scaled_w = pixmap_width * min(label_width / pixmap_width, label_height / pixmap_height)
                scaled_h = pixmap_height * min(label_width / pixmap_width, label_height / pixmap_height)
                offset_x = (label_width - scaled_w) / 2
                offset_y = (label_height - scaled_h) / 2
                scale_x = pixmap_width / scaled_w
                scale_y = pixmap_height / scaled_h
                start_x = (start.x() - offset_x) * scale_x
                start_y = (start.y() - offset_y) * scale_y
                end_x = (end.x() - offset_x) * scale_x
                end_y = (end.y() - offset_y) * scale_y
                rect = QRectF(QPointF(start_x, start_y), QPointF(end_x, end_y))
                aoi = AOI(rect.normalized())
                self.aoi_list.append(aoi)
                self.update_frame()

    def image_mouse_double_click_event(self, event):
        if self.is_configured:
            # Convert click position to coordinates in reference image
            pos = event.pos()
            pixmap = self.image_label.pixmap()
            if pixmap:
                label_rect = self.image_label.contentsRect()
                label_width = label_rect.width()
                label_height = label_rect.height()
                pixmap_width = pixmap.width()
                pixmap_height = pixmap.height()
                scaled_w = pixmap_width * min(label_width / pixmap_width, label_height / pixmap_height)
                scaled_h = pixmap_height * min(label_width / pixmap_width, label_height / pixmap_height)
                offset_x = (label_width - scaled_w) / 2
                offset_y = (label_height - scaled_h) / 2
                scale_x = pixmap_width / scaled_w
                scale_y = pixmap_height / scaled_h
                x = (pos.x() - offset_x) * scale_x
                y = (pos.y() - offset_y) * scale_y
                # Check if click is inside an AOI
                for aoi in reversed(self.aoi_list):
                    if aoi.rect.contains(QPointF(x, y)):
                        # Show dialog to change AOI name
                        text, ok = QInputDialog.getText(self, self.tr('AOIの名前を設定'), self.tr('AOIの名前:'), text=aoi.name)
                        if ok:
                            aoi.name = text
                        break

    def contextMenuEvent(self, event):
        if not self.is_configured:
            return
        pos = event.pos()
        if self.image_label.geometry().contains(pos):
            # Convert click position to image coordinates
            img_pos = self.image_label.mapFromParent(pos)
            pixmap = self.image_label.pixmap()
            if pixmap:
                label_rect = self.image_label.contentsRect()
                label_width = label_rect.width()
                label_height = label_rect.height()
                pixmap_width = pixmap.width()
                pixmap_height = pixmap.height()
                scaled_w = pixmap_width * min(label_width / pixmap_width, label_height / pixmap_height)
                scaled_h = pixmap_height * min(label_width / pixmap_width, label_height / pixmap_height)
                offset_x = (label_width - scaled_w) / 2
                offset_y = (label_height - scaled_h) / 2
                scale_x = pixmap_width / scaled_w
                scale_y = pixmap_height / scaled_h
                x = (img_pos.x() - offset_x) * scale_x
                y = (img_pos.y() - offset_y) * scale_y
                # Check if click is inside an AOI
                for aoi in reversed(self.aoi_list):
                    if aoi.rect.contains(QPointF(x, y)):
                        # Create context menu
                        menu = QMenu(self)
                        rename_action = menu.addAction(self.tr('AOIの名前を変更'))
                        delete_action = menu.addAction(self.tr('AOIを削除'))
                        action = menu.exec_(self.mapToGlobal(pos))
                        if action == rename_action:
                            text, ok = QInputDialog.getText(self, self.tr('AOIの名前を設定'), self.tr('AOIの名前:'), text=aoi.name)
                            if ok:
                                aoi.name = text
                        elif action == delete_action:
                            self.aoi_list.remove(aoi)
                            self.update_frame()
                        break

    def start_recording(self):
        if not self.current_session:
            QMessageBox.warning(self, self.tr("エラー"), self.tr("セッションを開始してください。"))
            return
        self.is_recording = True
        self.recorded_data = []
        self.frame_counter = 0

        # Set save directory
        self.session_directory = os.path.join(self.base_directory, self.current_user, self.current_session)
        os.makedirs(self.session_directory, exist_ok=True)

        # Determine CSV filename
        base_csv_filename = "recorded_data.csv"
        csv_filename = os.path.join(self.session_directory, base_csv_filename)
        index = 1
        while os.path.exists(csv_filename):
            csv_filename = os.path.join(self.session_directory, f"recorded_data({index}).csv")
            index += 1
        self.csv_filename = csv_filename

        self.csv_filename_edit.setEnabled(False)
        # Toggle button states
        self.record_start_button.setEnabled(False)
        self.record_stop_button.setEnabled(True)

    def stop_recording(self):
        self.is_recording = False
        # Save data to CSV file
        try:
            with open(self.csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Frame', 'PicNum', 'GazeX', 'GazeY', 'AOI', 'ScoreRight', 'ScoreLeft', 'SystemTime']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for data in self.recorded_data:
                    writer.writerow(data)
            QMessageBox.information(self, self.tr("保存完了"), f"{self.csv_filename} {self.tr('にデータを保存しました。')}")
        except Exception as e:
            QMessageBox.warning(self, self.tr("エラー"), f"{self.tr('データの保存中にエラーが発生しました:')} {e}")

        self.csv_filename_edit.setEnabled(True)
        # Toggle button states
        self.record_start_button.setEnabled(True)
        self.record_stop_button.setEnabled(False)

    def update_statistics(self):
        # Clear statistics layout
        for i in reversed(range(self.statistics_layout.count())):
            widget_to_remove = self.statistics_layout.itemAt(i).widget()
            self.statistics_layout.removeWidget(widget_to_remove)
            widget_to_remove.setParent(None)

        # Display statistics for each AOI
        for aoi in self.aoi_list:
            name = aoi.name if aoi.name else self.tr('無名')
            hit_count = aoi.hit_count
            dwell_time = aoi.dwell_time
            if aoi.is_gaze_inside and aoi.entry_time is not None:
                # Add current dwell time
                current_dwell = self.current_system_time - aoi.entry_time
            else:
                current_dwell = 0.0
            total_dwell_time = dwell_time + current_dwell
            label_text = f"{name} - {self.tr('ヒット数')}: {hit_count}, {self.tr('滞在時間')}: {total_dwell_time:.2f}s"
            label = QLabel(label_text)
            self.statistics_layout.addWidget(label)

    def update_frame(self, draw_aoi_preview=False):
        if not self.is_configured:
            return

        if config.frame_available.is_set() or draw_aoi_preview:
            if not draw_aoi_preview:
                config.frame_available.clear()

            with config.frame_lock:
                frame_proc = config.shared_data['frame']
                gaze_x = config.shared_data['gaze_x']
                gaze_y = config.shared_data['gaze_y']
                pic_num = config.shared_data['frame_num']
                score_right = config.shared_data['score_right']
                score_left = config.shared_data['score_left']
                system_time_str = config.shared_data['system_time']

            if frame_proc is None:
                return

            # Check if frame size has changed; recompute undistortion map if necessary
            if self.previous_frame_shape != frame_proc.shape[:2]:
                # config.map1_frame, config.map2_frame, config.roi_frame, config.new_camera_mtx_frame = precompute_undistort_map(frame_proc.shape)
                config.map1_frame, config.map2_frame, config.roi_frame, config.new_camera_mtx_frame = precompute_undistort_map(frame_proc.shape, config.camera_matrix, config.dist_coeffs)
                self.previous_frame_shape = frame_proc.shape[:2]

            # Scale gaze coordinates to image size
            h_frame, w_frame = frame_proc.shape[:2]
            gaze_x = gaze_x * w_frame
            gaze_y = gaze_y * h_frame

            # Undistort the frame
            frame_undistorted = cv2.remap(
                frame_proc, config.map1_frame, config.map2_frame, cv2.INTER_LINEAR)
            x, y, w, h = config.roi_frame
            frame_undistorted = frame_undistorted[y:y+h, x:x+w]

            # Undistort gaze coordinates
            gaze_point = np.array([[gaze_x, gaze_y]], dtype=np.float32).reshape(-1, 1, 2)
            gaze_point_undistorted = cv2.undistortPoints(
                gaze_point, config.camera_matrix, config.dist_coeffs, P=config.new_camera_mtx_frame)
            gaze_x_ud, gaze_y_ud = gaze_point_undistorted[0][0]
            gaze_x_ud -= x  # Adjust for cropping
            gaze_y_ud -= y  # Adjust for cropping

            # Convert frame to grayscale
            frame_gray = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)

            # Compute keypoints and descriptors for the frame
            frame_keypoints, frame_descriptors = config.orb.detectAndCompute(frame_gray, None)

            if frame_descriptors is not None and len(frame_descriptors) > 0:
                # Perform matching
                matches = config.flann.knnMatch(config.ref_descriptors, frame_descriptors, k=2)

                # Apply ratio test
                good_matches = []
                for m_n in matches:
                    if len(m_n) == 2:
                        m, n = m_n
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)

                if len(good_matches) > 10:
                    src_pts = np.float32(
                        [config.ref_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32(
                        [frame_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    # Compute homography matrix
                    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

                    # Transform gaze point to reference image coordinates
                    if M is not None:
                        gaze_point_ref = cv2.perspectiveTransform(
                            np.array([[[gaze_x_ud, gaze_y_ud]]], dtype=np.float32), M)
                        x_ref, y_ref = gaze_point_ref[0][0]

                        # Add gaze data to history
                        self.gaze_history.append((int(x_ref), int(y_ref)))
                        if len(self.gaze_history) > self.max_history:
                            self.gaze_history.popleft()

                        # Copy reference image for display
                        self.ref_image_display = config.ref_image.copy()

                        # Overlay scene camera if enabled
                        if self.overlay_scene:
                            h_ref, w_ref = config.ref_image.shape[:2]
                            warped_scene = cv2.warpPerspective(frame_undistorted, M, (w_ref, h_ref))
                            # Apply scene camera opacity
                            cv2.addWeighted(warped_scene, self.scene_opacity,
                                            self.ref_image_display, 1 - self.scene_opacity, 0, self.ref_image_display)

                        # Apply heatmap if enabled
                        if self.heatmap_checkbox.isChecked() and len(self.gaze_history) > 0:
                            heatmap = np.zeros((config.ref_image.shape[0], config.ref_image.shape[1]), dtype=np.float32)
                            for point in self.gaze_history:
                                cv2.circle(heatmap, point, self.gaze_point_size, 1, -1)
                            heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=15, sigmaY=15)
                            heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
                            heatmap_color = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)

                            # Create alpha mask based on intensity
                            alpha_mask = heatmap / 255.0 * self.heatmap_opacity
                            alpha_mask = cv2.merge([alpha_mask, alpha_mask, alpha_mask])

                            # Apply heatmap
                            self.ref_image_display = self.ref_image_display.astype(np.float32) / 255.0
                            heatmap_color = heatmap_color.astype(np.float32) / 255.0
                            self.ref_image_display = (1 - alpha_mask) * self.ref_image_display + alpha_mask * heatmap_color
                            self.ref_image_display = (self.ref_image_display * 255).astype(np.uint8)

                        # Apply gaze point opacity
                        overlay = self.ref_image_display.copy()
                        color = (*self.gaze_point_color,)
                        cv2.circle(overlay, (int(x_ref), int(y_ref)),
                                self.gaze_point_size, color, -1)
                        cv2.addWeighted(overlay, self.gaze_point_opacity,
                                        self.ref_image_display, 1 - self.gaze_point_opacity, 0, self.ref_image_display)

                        # Hold current system time
                        self.current_system_time = parse_system_time(system_time_str)

                        # AOI processing
                        gaze_aoi_name = ''
                        for aoi in self.aoi_list:
                            # Get AOI rectangle
                            rect = aoi.rect
                            # Check if gaze is inside AOI
                            gaze_inside = rect.contains(QPointF(x_ref, y_ref))

                            if gaze_inside:
                                if not aoi.is_gaze_inside:
                                    # Gaze entered AOI
                                    aoi.hit_count += 1
                                    aoi.entry_time = self.current_system_time
                                aoi.is_gaze_inside = True
                                gaze_aoi_name = aoi.name
                            else:
                                if aoi.is_gaze_inside:
                                    # Gaze exited AOI
                                    if aoi.entry_time is not None:
                                        aoi.dwell_time += self.current_system_time - aoi.entry_time
                                        aoi.entry_time = None
                                aoi.is_gaze_inside = False

                            # Draw AOI
                            rect_top_left = (int(rect.left()), int(rect.top()))
                            rect_bottom_right = (int(rect.right()), int(rect.bottom()))
                            if aoi.is_gaze_inside:
                                # Gaze is inside AOI (red color)
                                rect_color = (0, 0, 255)
                            else:
                                # Default color (green)
                                rect_color = (0, 255, 0)
                            cv2.rectangle(self.ref_image_display, rect_top_left, rect_bottom_right, rect_color, 1)

                            # Display hit count or name on AOI
                            if aoi.name:
                                display_text = f'{aoi.name}: {aoi.hit_count}'
                            else:
                                display_text = f'{self.tr("無名")}: {aoi.hit_count}'
                            text_size, baseline = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                            text_x = rect_top_left[0]
                            text_y = rect_top_left[1] - 5
                            if text_y < 0:
                                text_y = rect_bottom_right[1] + text_size[1] + 5
                            cv2.putText(self.ref_image_display, display_text, (text_x, text_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, rect_color, 1)

                        # Update statistics
                        self.update_statistics()

                        # If drawing AOI and preview is enabled
                        if self.drawing_aoi and draw_aoi_preview:
                            start = self.aoi_start_point
                            end = self.aoi_end_point
                            # Scale coordinates to reference image
                            pixmap = self.image_label.pixmap()
                            if pixmap:
                                label_rect = self.image_label.contentsRect()
                                label_width = label_rect.width()
                                label_height = label_rect.height()
                                pixmap_width = pixmap.width()
                                pixmap_height = pixmap.height()
                                scaled_w = pixmap_width * min(
                                    label_width / pixmap_width, label_height / pixmap_height)
                                scaled_h = pixmap_height * min(
                                    label_width / pixmap_width, label_height / pixmap_height)
                                offset_x = (label_width - scaled_w) / 2
                                offset_y = (label_height - scaled_h) / 2
                                scale_x = pixmap_width / scaled_w
                                scale_y = pixmap_height / scaled_h
                                start_x = (start.x() - offset_x) * scale_x
                                start_y = (start.y() - offset_y) * scale_y
                                end_x = (end.x() - offset_x) * scale_x
                                end_y = (end.y() - offset_y) * scale_y
                                cv2.rectangle(self.ref_image_display, (int(start_x), int(start_y)),
                                            (int(end_x), int(end_y)), (255, 0, 0), 1)

                        # Calculate FPS
                        current_time = time.time()
                        self.fps = 1 / (current_time - self.previous_time)
                        self.previous_time = current_time

                        # Display FPS if enabled
                        if self.show_fps:
                            fps_text = f"FPS: {self.fps:.2f}"
                            cv2.putText(self.ref_image_display, fps_text, (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                        # Display image
                        self.display_image(self.ref_image_display)

                        # Update graph data
                        current_time = self.current_system_time
                        self.graph_time.append(current_time)
                        self.graph_data_right.append(score_right)
                        self.graph_data_left.append(score_left)

                        # Limit buffer size
                        max_points = 100  # Maximum number of points to display
                        if len(self.graph_time) > max_points:
                            self.graph_time = self.graph_time[-max_points:]
                            self.graph_data_right = self.graph_data_right[-max_points:]
                            self.graph_data_left = self.graph_data_left[-max_points:]

                        # Update graph plots
                        self.score_right_line.setData(self.graph_time, self.graph_data_right)
                        self.score_left_line.setData(self.graph_time, self.graph_data_left)

                        # If recording is active, record data
                        if self.is_recording:
                            self.frame_counter += 1
                            data = {
                                'Frame': self.frame_counter,
                                'PicNum': pic_num,
                                'GazeX': x_ref,
                                'GazeY': y_ref,
                                'AOI': gaze_aoi_name,
                                'ScoreRight': score_right,
                                'ScoreLeft': score_left,
                                'SystemTime': system_time_str
                            }
                            self.recorded_data.append(data)
                    else:
                        # If homography could not be computed
                        pass
                else:
                    # Not enough good matches
                    pass

    def display_image(self, img):
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap)

    def resizeEvent(self, event):
        # Update AOI preview when window size changes
        if self.is_configured:
            self.update_frame(draw_aoi_preview=True)
        super().resizeEvent(event)

# Main function to run the application
def main():
    app = QApplication(sys.argv)
    gaze_app = GazeApp()
    gaze_app.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

