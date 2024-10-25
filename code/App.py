import zmq
import base64
import cv2
import numpy as np
import threading

# OpenCVの最適化設定
cv2.setUseOptimized(True)
cv2.setNumThreads(cv2.getNumberOfCPUs())

# カメラの内部パラメータ
camera_matrix = np.array([[500.3346336673587, 0, 286.4275647277839],
                          [0, 497.3594260665638, 251.3027090657917],
                          [0, 0, 1]])

dist_coeffs = np.array([-5.070020e-01, 2.356477e-01, -1.024952e-04, 4.798940e-03, -4.303462e-02])

# 歪み補正マップの事前計算関数
def precompute_undistort_map(image_shape):
    h, w = image_shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_mtx, (w,h), cv2.CV_16SC2)
    return map1, map2, roi

# 基準画像の読み込み
ref_image = cv2.imread('./shop.jpg')

# グレースケール変換
ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)

# ORB特徴量検出器の初期化
orb = cv2.ORB_create(nfeatures=500, fastThreshold=7, scaleFactor=1.2, nlevels=8, edgeThreshold=31, patchSize=31)

# 基準画像の特徴点と記述子を計算
ref_keypoints, ref_descriptors = orb.detectAndCompute(ref_gray, None)

# マッチャーの作成（FlannBasedMatcher）
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,  # 12
                    key_size=12,     # 20
                    multi_probe_level=1)  #2
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# ZeroMQの設定（サブスクライバーとして設定）
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5555")
socket.setsockopt_string(zmq.SUBSCRIBE, "")

# フレームサイズを取得するために、最初のメッセージを受信
message = socket.recv_pyobj()
frame_num = message['frame']
gaze_x = message['gaze_x']
gaze_y = message['gaze_y']
encoded_image = message['image']

# 画像データをデコード
decoded_image = base64.b64decode(encoded_image)
np_image = np.frombuffer(decoded_image, dtype=np.uint8)
frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

# フレームの歪み補正マップを事前計算
map1_frame, map2_frame, roi_frame = precompute_undistort_map(frame.shape)

# スレッド間で共有する変数
frame_lock = threading.Lock()
frame_available = threading.Event()

# フレームと視線データを保持する変数
shared_data = {'frame': frame, 'gaze_x': gaze_x, 'gaze_y': gaze_y}

# フレームを連続的に受信する関数
def receive_frames():
    global shared_data
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

# スレッドの開始
receive_thread = threading.Thread(target=receive_frames)
receive_thread.daemon = True
receive_thread.start()

while True:
    # 新しいフレームが受信されるのを待つ
    frame_available.wait()
    frame_available.clear()
    
    with frame_lock:
        frame_proc = shared_data['frame'].copy()
        gaze_x = shared_data['gaze_x']
        gaze_y = shared_data['gaze_y']
    
    # フレームの歪み補正
    frame_undistorted = cv2.remap(frame_proc, map1_frame, map2_frame, cv2.INTER_LINEAR)
    
    # グレースケール変換
    frame_gray = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)
    
    # フレームの特徴点と記述子を計算
    frame_keypoints, frame_descriptors = orb.detectAndCompute(frame_gray, None)
    
    if frame_descriptors is not None and len(frame_descriptors) > 0:
        # マッチングを実行
        matches = flann.knnMatch(ref_descriptors, frame_descriptors, k=2)
        
        # 比率テストを適用
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        
        if len(good_matches) > 10:
            src_pts = np.float32([ref_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
            dst_pts = np.float32([frame_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
            
            # ホモグラフィ行列の計算
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            
            # 視線データ（フレーム座標系での視線位置）
            gaze_point_frame = np.array([[gaze_x, gaze_y]], dtype=np.float32).reshape(-1,1,2)
            
            # 視線位置を基準画像の座標系に変換
            if M is not None:
                gaze_point_ref = cv2.perspectiveTransform(gaze_point_frame, M)
                x_ref, y_ref = gaze_point_ref[0][0]
                
                # 基準画像上に視線位置を描画
                ref_image_display = ref_image.copy()
                cv2.circle(ref_image_display, (int(x_ref), int(y_ref)), 10, (0,0,255), -1)
                
                # 結果を表示
                cv2.imshow('Reference Image with Gaze Point', ref_image_display)
    
    # 'q'キーで終了
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
