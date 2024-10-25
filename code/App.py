import sys
import zmq
import base64
import cv2
import numpy as np
import threading
import time
import ast
import csv
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget,
    QSlider, QColorDialog, QPushButton, QCheckBox, QHBoxLayout, QSizePolicy,
    QFileDialog, QLineEdit, QMessageBox, QTextEdit,
    QSplitter, QAction, QScrollArea, QToolButton, QInputDialog, QMenu, QActionGroup
)
from PyQt5.QtGui import QImage, QPixmap, QColor, QIcon, QContextMenuEvent, QCloseEvent
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QRectF, QPointF, QPropertyAnimation, QTranslator

# グローバル変数の初期化
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

# スレッド間で共有する変数
frame_lock = threading.Lock()
frame_available = threading.Event()
shared_data = {'frame': None, 'gaze_x': None, 'gaze_y': None, 'frame_num': None}

# 歪み補正マップの事前計算関数
def precompute_undistort_map(image_shape):
    h, w = image_shape[:2]
    # alpha=0で黒い部分がなくなるように調整
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), alpha=0, centerPrincipalPoint=1)
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_mtx, (w,h), cv2.CV_16SC2)
    return map1, map2, roi, new_camera_mtx

# GUIスレッドと通信するためのシグナルクラス
class Communicate(QObject):
    update_image = pyqtSignal()

# フレームを連続的に受信する関数
def receive_frames(zmq_address):
    global shared_data
    # ZeroMQの設定（サブスクライバーとして設定）
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(zmq_address)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    while True:
        message = socket.recv_pyobj()
        frame_num = message['frame']
        gaze_x = message['gaze_x']
        gaze_y = message['gaze_y']
        encoded_image = message['image']

        # 画像データをデコード
        decoded_image = base64.b64decode(encoded_image)
        np_image = np.frombuffer(decoded_image, dtype=np.uint8)
        frame_temp = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        with frame_lock:
            shared_data['frame'] = frame_temp.copy()
            shared_data['gaze_x'] = gaze_x
            shared_data['gaze_y'] = gaze_y
            shared_data['frame_num'] = frame_num  # PicNumとして使用
        frame_available.set()

# AOIを管理するクラス
class AOI:
    def __init__(self, rect, name=''):
        self.rect = rect  # QRectF
        self.hit_count = 0
        self.is_gaze_inside = False  # 視線が内側にあるか
        self.name = name  # AOIの名前

# 折りたたみ可能なグループボックス
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
            # 展開時にコンテンツの高さを計算
            content_height = self.content_area.layout().sizeHint().height()
            self.animation.setEndValue(content_height)
        else:
            self.animation.setEndValue(0)
        self.animation.start()

    def setContentLayout(self, layout):
        self.content_area.setLayout(layout)
        # コンテンツの高さを再計算
        content_height = layout.sizeHint().height()
        self.animation.setEndValue(content_height)

# PyQtアプリケーション
class GazeApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # 翻訳機能の初期化
        self.translator = QTranslator()
        QApplication.instance().installTranslator(self.translator)
        self.current_language = 'ja'  # デフォルトは日本語

        # 変数の初期化
        self.frame_undistorted = None
        self.ref_image_display = None
        self.gaze_point_size = 10
        self.gaze_point_color = (0, 0, 255)  # 赤色（BGR）
        self.gaze_point_opacity = 1.0        # 透明度（1.0:不透明、0.0:透明）
        self.show_fps = True
        self.previous_time = time.time()
        self.fps = 0
        self.overlay_scene = False           # シーンカメラのオーバーレイ表示フラグ
        self.scene_opacity = 0.5             # シーンカメラの透明度
        self.aoi_list = []                   # AOIのリスト
        self.drawing_aoi = False             # AOIを描画中かどうか
        self.aoi_start_point = None          # AOIの開始点
        self.is_configured = False           # 設定完了フラグ
        self.reset_requested = False         # カウントリセット要求フラグ

        # 視線データの保持
        self.gaze_history = []               # 視線座標の履歴
        self.max_history = 100               # デフォルトの履歴フレーム数
        self.heatmap_opacity = 0.5           # ヒートマップの透明度

        # レコード関連
        self.is_recording = False
        self.recorded_data = []
        self.frame_counter = 0  # ソフトウェア内のフレーム番号
        self.csv_filename = "recorded_data.csv"

        # UIのセットアップ
        self.init_ui()

        # スレッド間通信のためのシグナル
        self.comm = Communicate()
        self.comm.update_image.connect(self.update_frame)

    def init_ui(self):
        self.setWindowTitle(self.tr('視線ポイントビューア'))

        # メインウィジェットとレイアウト
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # メインレイアウトとしてQSplitterを使用
        self.splitter = QSplitter(Qt.Horizontal)
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.splitter)
        main_widget.setLayout(main_layout)

        # サイドバー（左側）の作成
        self.sidebar_widget = QWidget()
        self.sidebar_layout = QVBoxLayout()
        self.sidebar_widget.setLayout(self.sidebar_layout)

        # スクロールエリアでサイドバーを包む
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.sidebar_widget)

        # 初期設定グループの作成
        self.create_initial_settings_group()

        # その他の設定グループの作成
        self.create_other_settings_group()

        # ヒートマップ設定グループの作成
        self.create_heatmap_settings_group()

        # レコード設定グループの作成
        self.create_record_settings_group()

        # サイドバーにグループを追加
        self.sidebar_layout.addWidget(self.initial_settings_group)
        self.sidebar_layout.addWidget(self.other_settings_group)
        self.sidebar_layout.addWidget(self.heatmap_settings_group)
        self.sidebar_layout.addWidget(self.record_settings_group)
        self.sidebar_layout.addStretch()

        # 画像表示エリア
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.mousePressEvent = self.image_mouse_press_event
        self.image_label.mouseMoveEvent = self.image_mouse_move_event
        self.image_label.mouseReleaseEvent = self.image_mouse_release_event
        self.image_label.mouseDoubleClickEvent = self.image_mouse_double_click_event  # ダブルクリックイベント

        # サイドバーと画像ラベルをスプリッターに追加
        self.splitter.addWidget(self.scroll_area)
        self.splitter.addWidget(self.image_label)
        self.splitter.setStretchFactor(1, 1)  # 画像表示部分が伸縮するように設定

        # サイドバーの初期状態を開いた状態に設定
        self.sidebar_widget.setVisible(True)

        # メニューアクションの作成
        self.create_menu()

    def create_menu(self):
        # メニューバーの作成
        menubar = self.menuBar()
        self.file_menu = menubar.addMenu(self.tr('ファイル'))

        # AOIを保存
        self.save_aoi_action = QAction(self.tr('AOIを保存'), self)
        self.save_aoi_action.triggered.connect(self.save_aoi)
        self.file_menu.addAction(self.save_aoi_action)

        # AOIを読み込み
        self.load_aoi_action = QAction(self.tr('AOIを読み込み'), self)
        self.load_aoi_action.triggered.connect(self.load_aoi)
        self.file_menu.addAction(self.load_aoi_action)

        self.view_menu = menubar.addMenu(self.tr('表示'))

        # サイドバーの表示/非表示を切り替えるアクション
        self.toggle_sidebar_action = QAction(self.tr('サイドバーを表示'), self, checkable=True)
        self.toggle_sidebar_action.setChecked(True)
        self.toggle_sidebar_action.triggered.connect(self.toggle_sidebar)
        self.view_menu.addAction(self.toggle_sidebar_action)

        # 言語メニューの追加
        self.language_menu = menubar.addMenu(self.tr('言語'))
        self.language_action_group = QActionGroup(self)
        self.language_action_group.setExclusive(True)

        # 日本語のアクション
        self.ja_action = QAction('日本語', self, checkable=True)
        self.ja_action.setChecked(True)
        self.ja_action.triggered.connect(lambda: self.change_language('ja'))
        self.language_action_group.addAction(self.ja_action)
        self.language_menu.addAction(self.ja_action)

        # 英語のアクション
        self.en_action = QAction('English', self, checkable=True)
        self.en_action.triggered.connect(lambda: self.change_language('en'))
        self.language_action_group.addAction(self.en_action)
        self.language_menu.addAction(self.en_action)

    def change_language(self, language_code):
        if language_code == 'ja':
            self.translator.load('ja.qm')
            self.current_language = 'ja'
        elif language_code == 'en':
            self.translator.load('en.qm')
            self.current_language = 'en'
        QApplication.instance().installTranslator(self.translator)
        self.retranslate_ui()

    def retranslate_ui(self):
        self.setWindowTitle(self.tr('視線ポイントビューア'))
        # メニューの再翻訳
        self.file_menu.setTitle(self.tr('ファイル'))
        self.save_aoi_action.setText(self.tr('AOIを保存'))
        self.load_aoi_action.setText(self.tr('AOIを読み込み'))
        self.view_menu.setTitle(self.tr('表示'))
        self.toggle_sidebar_action.setText(self.tr('サイドバーを表示'))
        self.language_menu.setTitle(self.tr('言語'))

        # 折りたたみ可能なグループボックスのタイトルを再設定
        self.initial_settings_group.toggle_button.setText(self.tr('初期設定'))
        self.other_settings_group.toggle_button.setText(self.tr('その他の設定'))
        self.heatmap_settings_group.toggle_button.setText(self.tr('ヒートマップ設定'))
        self.record_settings_group.toggle_button.setText(self.tr('レコード設定'))

        # 初期設定グループ内のウィジェット
        self.image_browse_button.setText(self.tr('参照'))
        self.image_label_text.setText(self.tr('基準画像ファイル:'))
        self.zmq_label.setText(self.tr('ZMQアドレス:'))
        self.camera_matrix_label.setText(self.tr('カメラ行列 (3x3):'))
        self.camera_matrix_text.setPlaceholderText(self.tr('例: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]'))
        self.dist_coeffs_label.setText(self.tr('歪み係数 (5つ):'))
        self.dist_coeffs_text.setPlaceholderText(self.tr('例: [k1, k2, p1, p2, k3]'))
        self.configure_button.setText(self.tr('設定完了'))

        # その他の設定グループ内のウィジェット
        self.size_label.setText(self.tr('視線ポイントのサイズ:'))
        self.opacity_label.setText(self.tr('視線ポイントの透明度:'))
        self.color_button.setText(self.tr('視線ポイントの色を選択'))
        self.fps_checkbox.setText(self.tr('FPSを表示'))
        self.overlay_checkbox.setText(self.tr('シーンカメラを重ねて表示'))
        self.scene_opacity_label.setText(self.tr('シーンカメラの透明度:'))
        self.reset_button.setText(self.tr('カウントリセット'))

        # ヒートマップ設定グループ内のウィジェット
        self.heatmap_checkbox.setText(self.tr('ヒートマップを表示'))
        self.heatmap_opacity_label.setText(self.tr('ヒートマップの透明度:'))
        self.history_label.setText(self.tr('履歴フレーム数:'))

        # レコード設定グループ内のウィジェット
        self.csv_label.setText(self.tr('CSVファイル名:'))
        self.record_start_button.setText(self.tr('レコード開始'))
        self.record_stop_button.setText(self.tr('レコード停止'))

    def toggle_sidebar(self, state):
        self.sidebar_widget.setVisible(state)
        self.scroll_area.setVisible(state)

    def create_initial_settings_group(self):
        # 初期設定グループ
        self.initial_settings_group = CollapsibleBox(self.tr('初期設定'))
        layout = QVBoxLayout()

        # 画像ファイル選択
        image_layout = QHBoxLayout()
        self.image_path_edit = QLineEdit()
        self.image_browse_button = QPushButton(self.tr('参照'))
        self.image_browse_button.clicked.connect(self.browse_image)
        self.image_label_text = QLabel(self.tr('基準画像ファイル:'))
        image_layout.addWidget(self.image_label_text)
        image_layout.addWidget(self.image_path_edit)
        image_layout.addWidget(self.image_browse_button)
        layout.addLayout(image_layout)

        # ZMQアドレス入力
        zmq_layout = QHBoxLayout()
        self.zmq_address_edit = QLineEdit("tcp://localhost:5555")
        self.zmq_label = QLabel(self.tr('ZMQアドレス:'))
        zmq_layout.addWidget(self.zmq_label)
        zmq_layout.addWidget(self.zmq_address_edit)
        layout.addLayout(zmq_layout)

        # カメラ行列入力
        self.camera_matrix_text = QTextEdit()
        self.camera_matrix_text.setPlaceholderText(self.tr('例: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]'))
        self.camera_matrix_label = QLabel(self.tr('カメラ行列 (3x3):'))
        layout.addWidget(self.camera_matrix_label)
        layout.addWidget(self.camera_matrix_text)

        # 歪み係数入力
        self.dist_coeffs_text = QTextEdit()
        self.dist_coeffs_text.setPlaceholderText(self.tr('例: [k1, k2, p1, p2, k3]'))
        self.dist_coeffs_label = QLabel(self.tr('歪み係数 (5つ):'))
        layout.addWidget(self.dist_coeffs_label)
        layout.addWidget(self.dist_coeffs_text)

        # 設定完了ボタン
        self.configure_button = QPushButton(self.tr('設定完了'))
        self.configure_button.clicked.connect(self.apply_settings)
        layout.addWidget(self.configure_button)

        self.initial_settings_group.setContentLayout(layout)

    def create_other_settings_group(self):
        # その他の設定グループ
        self.other_settings_group = CollapsibleBox(self.tr('その他の設定'))
        layout = QVBoxLayout()

        # 視線ポイントのサイズ調整スライダー
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

        # 視線ポイントの透明度調整スライダー
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

        # 視線ポイントの色選択ボタン
        color_layout = QHBoxLayout()
        self.color_button = QPushButton(self.tr('視線ポイントの色を選択'))
        self.color_button.clicked.connect(self.select_color)
        layout.addWidget(self.color_button)

        # FPS表示のチェックボックス
        fps_layout = QHBoxLayout()
        self.fps_checkbox = QCheckBox(self.tr('FPSを表示'))
        self.fps_checkbox.setChecked(self.show_fps)
        self.fps_checkbox.stateChanged.connect(self.toggle_fps)
        fps_layout.addWidget(self.fps_checkbox)
        layout.addLayout(fps_layout)

        # シーンカメラのオーバーレイ表示チェックボックス
        overlay_layout = QHBoxLayout()
        self.overlay_checkbox = QCheckBox(self.tr('シーンカメラを重ねて表示'))
        self.overlay_checkbox.setChecked(self.overlay_scene)
        self.overlay_checkbox.stateChanged.connect(self.toggle_overlay)
        overlay_layout.addWidget(self.overlay_checkbox)
        layout.addLayout(overlay_layout)

        # シーンカメラの透明度調整スライダー
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

        # カウントリセットボタン
        reset_layout = QHBoxLayout()
        self.reset_button = QPushButton(self.tr('カウントリセット'))
        self.reset_button.clicked.connect(self.reset_counts)
        layout.addWidget(self.reset_button)

        self.other_settings_group.setContentLayout(layout)

    def create_heatmap_settings_group(self):
        # ヒートマップ設定グループ
        self.heatmap_settings_group = CollapsibleBox(self.tr('ヒートマップ設定'))
        layout = QVBoxLayout()

        # ヒートマップの有効/無効
        self.heatmap_checkbox = QCheckBox(self.tr('ヒートマップを表示'))
        self.heatmap_checkbox.setChecked(False)
        layout.addWidget(self.heatmap_checkbox)

        # ヒートマップの透明度調整スライダー
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

        # ヒートマップの履歴フレーム数
        history_layout = QHBoxLayout()
        self.history_slider = QSlider(Qt.Horizontal)
        self.history_slider.setMinimum(1)
        self.history_slider.setMaximum(500)
        self.history_slider.setValue(self.max_history)
        self.history_slider.setTickPosition(QSlider.TicksBelow)
        self.history_slider.setTickInterval(50)
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

        # CSVファイル名の入力
        csv_layout = QHBoxLayout()
        self.csv_filename_edit = QLineEdit(self.csv_filename)
        self.csv_label = QLabel(self.tr('CSVファイル名:'))
        csv_layout.addWidget(self.csv_label)
        csv_layout.addWidget(self.csv_filename_edit)
        layout.addLayout(csv_layout)

        # レコード開始・停止ボタン
        record_layout = QHBoxLayout()
        self.record_start_button = QPushButton(self.tr('レコード開始'))
        self.record_start_button.clicked.connect(self.start_recording)
        self.record_stop_button = QPushButton(self.tr('レコード停止'))
        self.record_stop_button.clicked.connect(self.stop_recording)
        self.record_stop_button.setEnabled(False)  # 初期状態では停止ボタンは無効
        record_layout.addWidget(self.record_start_button)
        record_layout.addWidget(self.record_stop_button)
        layout.addLayout(record_layout)

        self.record_settings_group.setContentLayout(layout)

    def change_heatmap_opacity(self, value):
        self.heatmap_opacity = value / 100.0
        self.heatmap_opacity_value_label.setText(str(value))

    def change_history(self, value):
        self.max_history = value
        self.history_value_label.setText(str(value))
        # 履歴を新しい最大値に合わせてトリミング
        if len(self.gaze_history) > self.max_history:
            self.gaze_history = self.gaze_history[-self.max_history:]

    def reset_counts(self):
        for aoi in self.aoi_list:
            aoi.hit_count = 0

    def browse_image(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(
            self, self.tr("基準画像を選択"), "", self.tr("画像ファイル (*.png *.jpg *.jpeg);;全てのファイル (*)"), options=options)
        if filename:
            self.image_path_edit.setText(filename)

    def apply_settings(self):
        global camera_matrix, dist_coeffs, ref_image, ref_gray, ref_keypoints, ref_descriptors, orb, flann, map1_frame, map2_frame, roi_frame, new_camera_mtx_frame

        # 基準画像の読み込み
        image_path = self.image_path_edit.text()
        if not image_path:
            QMessageBox.warning(self, self.tr("エラー"), self.tr("基準画像ファイルを選択してください。"))
            return

        ref_image = cv2.imread(image_path)
        if ref_image is None:
            QMessageBox.warning(self, self.tr("エラー"), self.tr("基準画像の読み込みに失敗しました。"))
            return

        # カメラ行列の取得
        try:
            camera_matrix_str = self.camera_matrix_text.toPlainText()
            camera_matrix_values = ast.literal_eval(camera_matrix_str)
            camera_matrix = np.array(camera_matrix_values, dtype=np.float64)
            if camera_matrix.shape != (3, 3):
                raise ValueError
        except Exception:
            QMessageBox.warning(self, self.tr("エラー"), self.tr("カメラ行列の値を正しく入力してください。"))
            return

        # 歪み係数の取得
        try:
            dist_coeffs_str = self.dist_coeffs_text.toPlainText()
            dist_coeffs_values = ast.literal_eval(dist_coeffs_str)
            dist_coeffs = np.array(dist_coeffs_values, dtype=np.float64)
            if dist_coeffs.shape[0] != 5:
                raise ValueError
        except Exception:
            QMessageBox.warning(self, self.tr("エラー"), self.tr("歪み係数の値を正しく入力してください。"))
            return

        # ORB特徴量検出器の初期化
        orb = cv2.ORB_create(nfeatures=500, fastThreshold=7, scaleFactor=1.2,
                             nlevels=8, edgeThreshold=31, patchSize=31)

        # 基準画像の特徴点と記述子を計算
        ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
        ref_keypoints, ref_descriptors = orb.detectAndCompute(ref_gray, None)

        # マッチャーの作成（FlannBasedMatcher）
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # フレームの歪み補正マップの事前計算（仮のフレームサイズで計算）
        # ここでdummy_frameをref_imageと同じサイズにしておく
        dummy_frame = np.zeros_like(ref_image)
        map1_frame, map2_frame, roi_frame, new_camera_mtx_frame = precompute_undistort_map(dummy_frame.shape)

        # ZMQアドレスの取得
        zmq_address = self.zmq_address_edit.text()

        # フレーム受信スレッドの開始
        self.receive_thread = threading.Thread(target=receive_frames, args=(zmq_address,))
        self.receive_thread.daemon = True
        self.receive_thread.start()

        # 設定完了後にコントロールパネルを有効化
        self.is_configured = True
        # サイドバーを閉じることができるようにメニューを有効化
        self.menuBar().setEnabled(True)

        # フレーム更新用のタイマー
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 約33FPSで更新

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
            # QColorをBGRのタプルに変換
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
            # クリック位置を基準画像の座標に変換
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
                # クリック位置が既存のAOI内にあるかチェック
                for aoi in reversed(self.aoi_list):
                    if aoi.rect.contains(QPointF(x, y)):
                        # 既存のAOI内をクリックした場合、新しいAOIの作成を開始しない
                        return
            # 新しいAOIの作成を開始
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
            # AOIの矩形を計算
            start = self.aoi_start_point
            end = self.aoi_end_point
            # 座標を基準画像のスケールに合わせる
            pixmap = self.image_label.pixmap()
            if pixmap:
                label_rect = self.image_label.contentsRect()
                label_width = label_rect.width()
                label_height = label_rect.height()
                pixmap_width = pixmap.width()
                pixmap_height = pixmap.height()
                # 画像がラベル内でどのように配置されているかを計算
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
            # クリック位置を基準画像の座標に変換
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
                # AOIのリストを逆順にチェック（最後に追加されたものが最前面）
                for aoi in reversed(self.aoi_list):
                    if aoi.rect.contains(QPointF(x, y)):
                        # 名前を変更するダイアログを表示
                        text, ok = QInputDialog.getText(self, self.tr('AOIの名前を設定'), self.tr('AOIの名前:'), text=aoi.name)
                        if ok:
                            aoi.name = text
                        break

    def contextMenuEvent(self, event):
        if not self.is_configured:
            return
        pos = event.pos()
        if self.image_label.geometry().contains(pos):
            # クリック位置を画像上の座標に変換
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
                # クリック位置がAOI内かチェック
                for aoi in reversed(self.aoi_list):
                    if aoi.rect.contains(QPointF(x, y)):
                        # コンテキストメニューの作成
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
        if not self.is_configured:
            QMessageBox.warning(self, self.tr("エラー"), self.tr("設定を完了してください。"))
            return
        self.is_recording = True
        self.recorded_data = []
        self.frame_counter = 0
        self.csv_filename = self.csv_filename_edit.text()

        self.csv_filename_edit.setEnabled(False)
        # ボタンの有効・無効を切り替え
        self.record_start_button.setEnabled(False)
        self.record_stop_button.setEnabled(True)

    def stop_recording(self):
        self.is_recording = False
        # CSVファイルにデータを保存
        try:
            with open(self.csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Frame', 'PicNum', 'GazeX', 'GazeY', 'AOI']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for data in self.recorded_data:
                    writer.writerow(data)
            QMessageBox.information(self, self.tr("保存完了"), f"{self.csv_filename} {self.tr('にデータを保存しました。')}")
        except Exception as e:
            QMessageBox.warning(self, self.tr("エラー"), f"{self.tr('データの保存中にエラーが発生しました:')} {e}")

        self.csv_filename_edit.setEnabled(True)
        # ボタンの有効・無効を切り替え
        self.record_start_button.setEnabled(True)
        self.record_stop_button.setEnabled(False)

    def reset_counts(self):
        for aoi in self.aoi_list:
            aoi.hit_count = 0

    def update_frame(self, draw_aoi_preview=False):
        if not self.is_configured:
            return

        if frame_available.is_set() or draw_aoi_preview:
            if not draw_aoi_preview:
                frame_available.clear()

            with frame_lock:
                frame_proc = shared_data['frame']
                gaze_x = shared_data['gaze_x']
                gaze_y = shared_data['gaze_y']
                pic_num = shared_data['frame_num']

            if frame_proc is None:
                return

            # フレームサイズが変わった可能性があるので、歪み補正マップを再計算
            global map1_frame, map2_frame, roi_frame, new_camera_mtx_frame
            map1_frame, map2_frame, roi_frame, new_camera_mtx_frame = precompute_undistort_map(frame_proc.shape)

            # 視線座標を画像サイズにスケーリング
            h_frame, w_frame = frame_proc.shape[:2]
            gaze_x = gaze_x * w_frame
            gaze_y = gaze_y * h_frame

            # フレームの歪み補正
            frame_undistorted = cv2.remap(
                frame_proc, map1_frame, map2_frame, cv2.INTER_LINEAR)
            x, y, w, h = roi_frame
            frame_undistorted = frame_undistorted[y:y+h, x:x+w]

            # 視線座標の歪み補正
            gaze_point = np.array([[gaze_x, gaze_y]], dtype=np.float32).reshape(-1, 1, 2)
            gaze_point_undistorted = cv2.undistortPoints(
                gaze_point, camera_matrix, dist_coeffs, P=new_camera_mtx_frame)
            gaze_x_ud, gaze_y_ud = gaze_point_undistorted[0][0]
            gaze_x_ud -= x  # クロップした分を調整
            gaze_y_ud -= y  # クロップした分を調整

            # グレースケール変換
            frame_gray = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)

            # フレームの特徴点と記述子を計算
            frame_keypoints, frame_descriptors = orb.detectAndCompute(frame_gray, None)

            if frame_descriptors is not None and len(frame_descriptors) > 0:
                # マッチングを実行
                matches = flann.knnMatch(ref_descriptors, frame_descriptors, k=2)

                # 比率テストを適用
                good_matches = []
                for m_n in matches:
                    if len(m_n) == 2:
                        m, n = m_n
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)

                if len(good_matches) > 10:
                    src_pts = np.float32(
                        [ref_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32(
                        [frame_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    # ホモグラフィ行列の計算
                    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

                    # 視線位置を基準画像の座標系に変換
                    if M is not None:
                        gaze_point_ref = cv2.perspectiveTransform(
                            np.array([[[gaze_x_ud, gaze_y_ud]]], dtype=np.float32), M)
                        x_ref, y_ref = gaze_point_ref[0][0]

                        # 視線データを履歴に追加
                        self.gaze_history.append((int(x_ref), int(y_ref)))
                        if len(self.gaze_history) > self.max_history:
                            self.gaze_history.pop(0)

                        # 基準画像上に視線位置を描画
                        ref_image_display = ref_image.copy()

                        # シーンカメラのオーバーレイが有効な場合
                        if self.overlay_scene:
                            # シーンカメラのフレームを基準画像の座標系に変換
                            h_ref, w_ref = ref_image.shape[:2]
                            warped_scene = cv2.warpPerspective(frame_undistorted, M, (w_ref, h_ref))
                            # シーンカメラの透明度を適用
                            cv2.addWeighted(warped_scene, self.scene_opacity,
                                            ref_image_display, 1 - self.scene_opacity, 0, ref_image_display)

                        # ヒートマップの作成と適用
                        if self.heatmap_checkbox.isChecked() and len(self.gaze_history) > 0:
                            heatmap = np.zeros((ref_image.shape[0], ref_image.shape[1]), dtype=np.float32)
                            for point in self.gaze_history:
                                cv2.circle(heatmap, point, self.gaze_point_size, 1, -1)
                            heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=15, sigmaY=15)
                            heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
                            heatmap_color = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)

                            # 強度に基づいたアルファマスクを作成
                            alpha_mask = heatmap / 255.0 * self.heatmap_opacity
                            alpha_mask = cv2.merge([alpha_mask, alpha_mask, alpha_mask])

                            # ヒートマップを適用
                            ref_image_display = ref_image_display.astype(np.float32) / 255.0
                            heatmap_color = heatmap_color.astype(np.float32) / 255.0
                            ref_image_display = (1 - alpha_mask) * ref_image_display + alpha_mask * heatmap_color
                            ref_image_display = (ref_image_display * 255).astype(np.uint8)

                        # 視線ポイントの透明度を適用
                        overlay = ref_image_display.copy()
                        color = (*self.gaze_point_color,)
                        cv2.circle(overlay, (int(x_ref), int(y_ref)),
                                   self.gaze_point_size, color, -1)
                        cv2.addWeighted(overlay, self.gaze_point_opacity,
                                        ref_image_display, 1 - self.gaze_point_opacity, 0, ref_image_display)

                        # AOIの処理
                        gaze_aoi_name = ''
                        for aoi in self.aoi_list:
                            # AOIの矩形を取得
                            rect = aoi.rect
                            # 視線がAOI内にあるかチェック
                            if rect.contains(QPointF(x_ref, y_ref)):
                                aoi.is_gaze_inside = True
                                aoi.hit_count += 1
                                gaze_aoi_name = aoi.name
                            else:
                                aoi.is_gaze_inside = False

                            # AOIの描画
                            rect_top_left = (int(rect.left()), int(rect.top()))
                            rect_bottom_right = (int(rect.right()), int(rect.bottom()))
                            if aoi.is_gaze_inside:
                                # 視線が内側にある場合の色（例：赤）
                                rect_color = (0, 0, 255)
                            else:
                                # デフォルトの色（例：緑）
                                rect_color = (0, 255, 0)
                            cv2.rectangle(ref_image_display, rect_top_left, rect_bottom_right, rect_color, 1)

                            # ヒットカウントまたは名前をAOIの上に表示
                            if aoi.name:
                                display_text = f'{aoi.name}: {aoi.hit_count}'
                            else:
                                display_text = f'{self.tr("無名")}: {aoi.hit_count}'
                            text_size, baseline = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                            text_x = rect_top_left[0]
                            text_y = rect_top_left[1] - 5
                            if text_y < 0:
                                text_y = rect_bottom_right[1] + text_size[1] + 5
                            cv2.putText(ref_image_display, display_text, (text_x, text_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, rect_color, 1)

                        # AOIを描画中の場合、プレビューを表示
                        if self.drawing_aoi and draw_aoi_preview:
                            start = self.aoi_start_point
                            end = self.aoi_end_point
                            # 座標を基準画像のスケールに合わせる
                            pixmap = self.image_label.pixmap()
                            if pixmap:
                                label_rect = self.image_label.contentsRect()
                                label_width = label_rect.width()
                                label_height = label_rect.height()
                                pixmap_width = pixmap.width()
                                pixmap_height = pixmap.height()
                                # 画像がラベル内でどのように配置されているかを計算
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
                                cv2.rectangle(ref_image_display, (int(start_x), int(start_y)),
                                              (int(end_x), int(end_y)), (255, 0, 0), 1)

                        # FPSの計算
                        current_time = time.time()
                        self.fps = 1 / (current_time - self.previous_time)
                        self.previous_time = current_time

                        # FPSを表示する場合
                        if self.show_fps:
                            fps_text = f"FPS: {self.fps:.2f}"
                            cv2.putText(ref_image_display, fps_text, (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                        # 画像をQt形式に変換して表示
                        self.display_image(ref_image_display)

                        # レコード中の場合、データを記録
                        if self.is_recording:
                            self.frame_counter += 1
                            data = {
                                'Frame': self.frame_counter,
                                'PicNum': pic_num,
                                'GazeX': x_ref,
                                'GazeY': y_ref,
                                'AOI': gaze_aoi_name
                            }
                            self.recorded_data.append(data)

    def display_image(self, img):
        # BGRからRGBに変換
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap)

    def resizeEvent(self, event):
        # ウィンドウサイズが変更されたときにAOIプレビューを更新
        if self.is_configured:
            self.update_frame(draw_aoi_preview=True)
        super().resizeEvent(event)

# PyQtアプリケーションの実行
def main():
    app = QApplication(sys.argv)
    gaze_app = GazeApp()
    gaze_app.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
