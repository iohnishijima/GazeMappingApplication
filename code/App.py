import sys
import zmq
import base64
import cv2
import numpy as np
import threading
import time
import ast
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget,
    QSlider, QColorDialog, QPushButton, QCheckBox, QHBoxLayout, QSizePolicy,
    QFileDialog, QLineEdit, QGridLayout, QMessageBox, QTextEdit, QGroupBox,
    QSplitter, QAction
)
from PyQt5.QtGui import QImage, QPixmap, QColor, QIcon
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject

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
shared_data = {'frame': None, 'gaze_x': None, 'gaze_y': None}

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
        frame_available.set()

# PyQtアプリケーション
class GazeApp(QMainWindow):
    def __init__(self):
        super().__init__()

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

        # 設定完了フラグ
        self.is_configured = False

        # UIのセットアップ
        self.init_ui()

        # スレッド間通信のためのシグナル
        self.comm = Communicate()
        self.comm.update_image.connect(self.update_frame)

    def init_ui(self):
        self.setWindowTitle('視線ポイントビューア')

        # メインウィジェットとレイアウト
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # メインレイアウトとしてQSplitterを使用
        self.splitter = QSplitter(Qt.Horizontal)
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.splitter)
        main_widget.setLayout(main_layout)

        # サイドバー（左側）の作成
        self.sidebar = QWidget()
        self.sidebar_layout = QVBoxLayout()
        self.sidebar.setLayout(self.sidebar_layout)

        # 初期設定グループの作成
        self.create_initial_settings_group()

        # その他の設定グループの作成
        self.create_other_settings_group()

        # サイドバーにグループを追加
        self.sidebar_layout.addWidget(self.initial_settings_group)
        self.sidebar_layout.addWidget(self.other_settings_group)
        self.sidebar_layout.addStretch()

        # 画像表示ラベル
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # サイドバーと画像ラベルをスプリッターに追加
        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(self.image_label)
        self.splitter.setStretchFactor(1, 1)  # 画像表示部分が伸縮するように設定

        # サイドバーの初期状態を開いた状態に設定
        self.sidebar.setVisible(True)

        # メニューアクションの作成
        self.create_menu()

    def create_menu(self):
        # メニューバーの作成
        menubar = self.menuBar()
        view_menu = menubar.addMenu('表示')

        # サイドバーの表示/非表示を切り替えるアクション
        toggle_sidebar_action = QAction('サイドバーを表示', self, checkable=True)
        toggle_sidebar_action.setChecked(True)
        toggle_sidebar_action.triggered.connect(self.toggle_sidebar)
        view_menu.addAction(toggle_sidebar_action)

    def toggle_sidebar(self, state):
        self.sidebar.setVisible(state)

    def create_initial_settings_group(self):
        # 初期設定グループ
        self.initial_settings_group = QGroupBox('初期設定')
        layout = QVBoxLayout()

        # 画像ファイル選択
        image_layout = QHBoxLayout()
        self.image_path_edit = QLineEdit()
        image_browse_button = QPushButton('参照')
        image_browse_button.clicked.connect(self.browse_image)
        image_layout.addWidget(QLabel('基準画像ファイル:'))
        image_layout.addWidget(self.image_path_edit)
        image_layout.addWidget(image_browse_button)
        layout.addLayout(image_layout)

        # ZMQアドレス入力
        zmq_layout = QHBoxLayout()
        self.zmq_address_edit = QLineEdit("tcp://localhost:5555")
        zmq_layout.addWidget(QLabel('ZMQアドレス:'))
        zmq_layout.addWidget(self.zmq_address_edit)
        layout.addLayout(zmq_layout)

        # カメラ行列入力
        self.camera_matrix_text = QTextEdit()
        self.camera_matrix_text.setPlaceholderText('例: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]')
        layout.addWidget(QLabel('カメラ行列 (3x3):'))
        layout.addWidget(self.camera_matrix_text)

        # 歪み係数入力
        self.dist_coeffs_text = QTextEdit()
        self.dist_coeffs_text.setPlaceholderText('例: [k1, k2, p1, p2, k3]')
        layout.addWidget(QLabel('歪み係数 (5つ):'))
        layout.addWidget(self.dist_coeffs_text)

        # 設定完了ボタン
        configure_button = QPushButton('設定完了')
        configure_button.clicked.connect(self.apply_settings)
        layout.addWidget(configure_button)

        self.initial_settings_group.setLayout(layout)

    def create_other_settings_group(self):
        # その他の設定グループ
        self.other_settings_group = QGroupBox('その他の設定')
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
        size_layout.addWidget(QLabel('視線ポイントのサイズ:'))
        size_layout.addWidget(self.size_slider)
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
        opacity_layout.addWidget(QLabel('視線ポイントの透明度:'))
        opacity_layout.addWidget(self.opacity_slider)
        layout.addLayout(opacity_layout)

        # 視線ポイントの色選択ボタン
        color_layout = QHBoxLayout()
        self.color_button = QPushButton('視線ポイントの色を選択')
        self.color_button.clicked.connect(self.select_color)
        color_layout.addWidget(self.color_button)
        layout.addLayout(color_layout)

        # FPS表示のチェックボックス
        fps_layout = QHBoxLayout()
        self.fps_checkbox = QCheckBox('FPSを表示')
        self.fps_checkbox.setChecked(self.show_fps)
        self.fps_checkbox.stateChanged.connect(self.toggle_fps)
        fps_layout.addWidget(self.fps_checkbox)
        layout.addLayout(fps_layout)

        # シーンカメラのオーバーレイ表示チェックボックス
        overlay_layout = QHBoxLayout()
        self.overlay_checkbox = QCheckBox('シーンカメラを重ねて表示')
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
        scene_opacity_layout.addWidget(QLabel('シーンカメラの透明度:'))
        scene_opacity_layout.addWidget(self.scene_opacity_slider)
        layout.addLayout(scene_opacity_layout)

        self.other_settings_group.setLayout(layout)

    def browse_image(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(
            self, "基準画像を選択", "", "画像ファイル (*.png *.jpg *.jpeg);;全てのファイル (*)", options=options)
        if filename:
            self.image_path_edit.setText(filename)

    def apply_settings(self):
        global camera_matrix, dist_coeffs, ref_image, ref_gray, ref_keypoints, ref_descriptors, orb, flann, map1_frame, map2_frame, roi_frame, new_camera_mtx_frame

        # 基準画像の読み込み
        image_path = self.image_path_edit.text()
        if not image_path:
            QMessageBox.warning(self, "エラー", "基準画像ファイルを選択してください。")
            return

        ref_image = cv2.imread(image_path)
        if ref_image is None:
            QMessageBox.warning(self, "エラー", "基準画像の読み込みに失敗しました。")
            return

        # カメラ行列の取得
        try:
            camera_matrix_str = self.camera_matrix_text.toPlainText()
            camera_matrix_values = ast.literal_eval(camera_matrix_str)
            camera_matrix = np.array(camera_matrix_values, dtype=np.float64)
            if camera_matrix.shape != (3, 3):
                raise ValueError
        except Exception:
            QMessageBox.warning(self, "エラー", "カメラ行列の値を正しく入力してください。")
            return

        # 歪み係数の取得
        try:
            dist_coeffs_str = self.dist_coeffs_text.toPlainText()
            dist_coeffs_values = ast.literal_eval(dist_coeffs_str)
            dist_coeffs = np.array(dist_coeffs_values, dtype=np.float64)
            if dist_coeffs.shape[0] != 5:
                raise ValueError
        except Exception:
            QMessageBox.warning(self, "エラー", "歪み係数の値を正しく入力してください。")
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

    def change_opacity(self, value):
        self.gaze_point_opacity = value / 100.0

    def change_scene_opacity(self, value):
        self.scene_opacity = value / 100.0

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

    def update_frame(self):
        if not self.is_configured:
            return

        if frame_available.is_set():
            frame_available.clear()

            with frame_lock:
                frame_proc = shared_data['frame']
                gaze_x = shared_data['gaze_x']
                gaze_y = shared_data['gaze_y']

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

                        # 視線ポイントの透明度を適用
                        overlay = ref_image_display.copy()
                        color = (*self.gaze_point_color,)
                        cv2.circle(overlay, (int(x_ref), int(y_ref)),
                                   self.gaze_point_size, color, -1)
                        cv2.addWeighted(overlay, self.gaze_point_opacity,
                                        ref_image_display, 1 - self.gaze_point_opacity, 0, ref_image_display)

                        # FPSの計算
                        current_time = time.time()
                        self.fps = 1 / (current_time - self.previous_time)
                        self.previous_time = current_time

                        # FPSを表示する場合
                        if self.show_fps:
                            fps_text = "FPS: {:.2f}".format(self.fps)
                            cv2.putText(ref_image_display, fps_text, (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                        # 画像をQt形式に変換して表示
                        self.display_image(ref_image_display)

    def display_image(self, img):
        # BGRからRGBに変換
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap)

# PyQtアプリケーションの実行
def main():
    app = QApplication(sys.argv)
    gaze_app = GazeApp()
    gaze_app.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
