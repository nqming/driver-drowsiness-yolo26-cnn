"""
Luồng xử lý Thị giác Máy tính & Học sâu 2 Giai đoạn thời gian thực (backend/ai_engine.py):
- Giai đoạn 1: YOLO26 (yolo26n.pt NMS-Free) phát hiện người lái + MediaPipe FaceLandmarker (478 điểm 3D,
  tự động dự phòng OpenCV Haar Cascade) định vị Mắt trái, Mắt phải, Miệng & góc cúi đầu (Head Pitch/Yaw).
  Hỗ trợ bộ lọc giả lập Camera Hồng ngoại Ban đêm (Night/IR Mode).
- Giai đoạn 2: Mạng học sâu MobileNetV3-Small (models/cnn_driver_state.pth) tính xác suất Softmax
  P(Nhắm mắt) & P(Ngáp) kết hợp chỉ số hình học EAR/MAR (Cơ chế Zero-Crash không bao giờ văng lỗi).
- Bộ lọc ngữ cảnh xe đỗ (Cần số P/N/D & Vận tốc km/h) + Cảnh báo đa tầng 3 mức.
"""

import base64
import math
import os
import threading
import time
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.database import SNAPSHOTS_DIR, insert_violation, update_telegram_status
from backend.notifier import TelegramNotifier
from training.model_arch import (
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    load_trained_cnn,
)

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
CNN_WEIGHTS_PATH = MODELS_DIR / "cnn_driver_state.pth"
YOLO_WEIGHTS_PATH = MODELS_DIR / "yolo26n.pt"
FACE_LANDMARKER_PATH = MODELS_DIR / "face_landmarker.task"
DEMO_VIDEO_PATH = DATA_DIR / "demo_driver_test.mp4"

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# Chỉ số điểm mốc chuẩn trên lưới khuôn mặt 478 điểm của MediaPipe
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_OUTER_IDX = [61, 291, 13, 14, 78, 308, 81, 178, 311, 402]


def _sigmoid(x: float) -> float:
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _encode_roi_base64(bgr_crop: Optional[np.ndarray], size: int = 96) -> str:
    """Mã hóa ảnh vùng ROI sang chuỗi Data URI JPEG để hiển thị trực tiếp trên bảng Giai đoạn 2."""
    if bgr_crop is None or bgr_crop.size == 0:
        canvas = np.full((size, size, 3), 24, dtype=np.uint8)
        cv2.putText(canvas, "N/A", (22, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (110, 125, 145), 2)
    else:
        canvas = cv2.resize(bgr_crop, (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _draw_vietnamese_banner(
    bgr_img: np.ndarray,
    text: str,
    box_xyxy: Tuple[int, int, int, int],
    bg_color_rgb: Tuple[int, int, int],
    text_color_rgb: Tuple[int, int, int] = (255, 255, 255),
    font_size: int = 19,
) -> np.ndarray:
    """Vẽ băng rôn chữ tiếng Việt có dấu sắc nét lên khung hình OpenCV bằng Pillow."""
    x1, y1, x2, y2 = box_xyxy
    overlay = bgr_img.copy()
    bg_bgr = (bg_color_rgb[2], bg_color_rgb[1], bg_color_rgb[0])
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_bgr, -1)
    blended = cv2.addWeighted(overlay, 0.84, bgr_img, 0.16, 0)

    pil_img = Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    font = None
    for font_candidate in [
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/tahomabd.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]:
        if os.path.exists(font_candidate):
            try:
                font = ImageFont.truetype(font_candidate, font_size)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = x1 + max(10, ((x2 - x1) - tw) // 2)
    ty = y1 + max(4, ((y2 - y1) - th) // 2 - 2)
    draw.text((tx, ty), text, font=font, fill=text_color_rgb)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def ensure_demo_video() -> Path:
    """
    Tạo tự động video kiểm thử thực tế (data/demo_driver_test.mp4) từ chính các khung hình
    người lái ô tô thực tế trong khoang lái (tập dữ liệu hợp nhất) theo kịch bản 3 giai đoạn:
    0-5s: Tỉnh táo (Mức 0) -> 5-9.5s: Ngáp kéo dài (Mức 1) -> 9.5-15s: Nhắm mắt & Cúi đầu ngủ gật (Mức 2).
    """
    if DEMO_VIDEO_PATH.exists() and DEMO_VIDEO_PATH.stat().st_size > 80_000:
        return DEMO_VIDEO_PATH

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    width, height, fps = 640, 480, 15
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(DEMO_VIDEO_PATH), fourcc, fps, (width, height))

    awake_img: Optional[np.ndarray] = None
    yawn_img: Optional[np.ndarray] = None
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    merged_root = DATA_DIR / "merged_dataset"
    for split in ["test", "val", "train"]:
        ny_dir = merged_root / split / "no_yawn"
        yw_dir = merged_root / split / "yawn"
        if awake_img is None and ny_dir.exists():
            for p in sorted(ny_dir.glob("*.jpg"))[:25]:
                im = cv2.imread(str(p))
                if im is not None and im.shape[0] >= 180:
                    gray_c = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    if len(cascade.detectMultiScale(gray_c, 1.1, 4, minSize=(50, 50))) > 0:
                        awake_img = cv2.resize(im, (width, height))
                        break
        if yawn_img is None and yw_dir.exists():
            for p in sorted(yw_dir.glob("*.jpg"))[:25]:
                im = cv2.imread(str(p))
                if im is not None and im.shape[0] >= 180:
                    gray_c = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    if len(cascade.detectMultiScale(gray_c, 1.1, 4, minSize=(50, 50))) > 0:
                        yawn_img = cv2.resize(im, (width, height))
                        break

    # Chuẩn bị phiên bản nhắm mắt ngủ gật tự nhiên trên khung hình tài xế thật
    drowsy_img: Optional[np.ndarray] = None
    if awake_img is not None:
        drowsy_img = awake_img.copy()
        gray_aw = cv2.cvtColor(drowsy_img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray_aw, 1.1, 4, minSize=(60, 60))
        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
            forehead_patch = drowsy_img[max(0, fy + int(fh * 0.10)):fy + int(fh * 0.22), fx + int(fw * 0.25):fx + int(fw * 0.75)]
            skin_bgr = forehead_patch.mean(axis=(0, 1)).astype(np.uint8).tolist() if forehead_patch.size > 0 else [150, 170, 195]
            ey1, ey2 = fy + int(fh * 0.28), fy + int(fh * 0.48)
            for ex1_r, ex2_r in [(0.15, 0.46), (0.54, 0.85)]:
                ex1, ex2 = fx + int(fw * ex1_r), fx + int(fw * ex2_r)
                cv2.ellipse(drowsy_img, ((ex1 + ex2) // 2, (ey1 + ey2) // 2), ((ex2 - ex1) // 2, (ey2 - ey1) // 2), 0, 0, 360, skin_bgr, -1)
                mid_y = (ey1 + ey2) // 2 + 3
                cv2.ellipse(drowsy_img, ((ex1 + ex2) // 2, mid_y - 3), ((ex2 - ex1) // 2 - 3, 5), 0, 15, 165, (32, 28, 30), 2, cv2.LINE_AA)

    total_frames = fps * 15
    for f_idx in range(total_frames):
        sec = f_idx / fps
        if sec < 5.0 and awake_img is not None:
            frame = awake_img.copy()
            # Chuyển động vi mô tự nhiên của camera khoang lái
            dx = int(math.sin(f_idx * 0.25) * 3)
            M = np.float32([[1, 0, dx], [0, 1, 0]])
            frame = cv2.warpAffine(frame, M, (width, height), borderMode=cv2.BORDER_REFLECT)
        elif 5.0 <= sec < 9.5 and yawn_img is not None:
            frame = yawn_img.copy()
            dx = int(math.sin(f_idx * 0.2) * 3)
            M = np.float32([[1, 0, dx], [0, 1, 0]])
            frame = cv2.warpAffine(frame, M, (width, height), borderMode=cv2.BORDER_REFLECT)
        elif drowsy_img is not None:
            nod_px = min(46.0, (sec - 9.5) * 18.0)
            M = np.float32([[1, 0, 0], [0, 1, nod_px]])
            frame = cv2.warpAffine(drowsy_img, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
        else:
            frame = np.full((height, width, 3), (32, 38, 48), dtype=np.uint8)
            cv2.ellipse(frame, (width // 2, 210), (80, 100), 0, 0, 360, (180, 200, 220), -1)

        writer.write(frame)

    writer.release()
    return DEMO_VIDEO_PATH


class DriverMonitoringEngine:
    """
    Bộ máy AI trung tâm xử lý luồng video thời gian thực 2 giai đoạn (YOLO26 + CNN)
    kèm quản lý trạng thái xe (Vận tốc, Cần số P/N/D) và Cảnh báo đa tầng 3 mức.
    """

    def __init__(self, notifier: Optional[TelegramNotifier] = None) -> None:
        self.lock = threading.Lock()
        self.notifier = notifier or TelegramNotifier()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            torch.set_num_threads(max(4, min(12, os.cpu_count() or 8)))

        # Trạng thái vận hành xe & cấu hình cảnh báo
        self.speed_kmh: float = 60.0
        self.gear: str = "D"  # 'P', 'N', 'D'
        self.drowsy_threshold_sec: float = 2.0
        self.ir_mode: bool = False

        # Quản lý nguồn phát video ('stopped', 'webcam', 'video', 'demo')
        self.source_mode: str = "stopped"
        self.video_path: Optional[str] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._cap_lock = threading.Lock()

        # Tải mô hình Giai đoạn 1 (YOLO26 + FaceLandmarker + OpenCV Fallback)
        self.yolo_model = None
        self.yolo_ready: bool = False
        self.face_landmarker = None
        self.mediapipe_ready: bool = False
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

        # Tải mô hình Giai đoạn 2 (DriverStateCNN MobileNetV3-Small)
        self.cnn_model = None
        self.cnn_ready: bool = False
        self.cnn_class_names: List[str] = CLASS_NAMES
        self.cnn_img_size: int = IMG_SIZE
        self._cnn_mtime: float = 0.0
        self._mean_arr = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
        self._std_arr = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)

        # Biến theo dõi thời gian liên tục & lịch sử ngáp
        self._last_tick: float = time.perf_counter()
        self.eye_closed_duration: float = 0.0
        self.yawn_duration: float = 0.0
        self.head_nod_duration: float = 0.0
        self._yawn_active_flag: bool = False
        self._yawn_events: deque = deque(maxlen=30)
        self._last_violation_log_ts: float = 0.0
        self._ema_p_closed: float = 8.0
        self._ema_p_yawn: float = 5.0
        self._frame_counter: int = 0
        self._cached_driver_box: Optional[Tuple[int, int, int, int]] = None
        self._cached_yolo_conf: float = 0.0
        self._clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8))

        # Telemetry mới nhất phục vụ Frontend Dashboard & biểu đồ 30s
        self.latest_telemetry: Dict = self._default_telemetry()
        self.latest_jpeg: Optional[bytes] = None

        self._initialize_models()

    def _default_telemetry(self) -> Dict:
        return {
            "active": False,
            "source_mode": "stopped",
            "ir_mode": False,
            "driver_detected": False,
            "yolo_conf": 0.0,
            "fps": 0.0,
            "latency_ms": 0.0,
            "p_closed": 0.0,
            "p_yawn": 0.0,
            "ear": 0.30,
            "mar": 0.18,
            "head_pitch": 0.0,
            "head_yaw": 0.0,
            "eye_closed_duration": 0.0,
            "yawn_duration": 0.0,
            "head_nod_duration": 0.0,
            "yawns_per_min": 0,
            "speed_kmh": self.speed_kmh,
            "gear": self.gear,
            "drowsy_threshold_sec": self.drowsy_threshold_sec,
            "standby_mode": False,
            "alert_level": 0,
            "alert_code": "NORMAL",
            "alert_title": "HỆ THỐNG SẴN SÀNG",
            "alert_subtitle": "Chọn nguồn Camera hoặc Video kiểm thử để bắt đầu giám sát",
            "voice_message": "",
            "siren_active": False,
            "cnn_ready": False,
            "yolo_ready": False,
            "landmark_backend": "MediaPipe FaceMesh",
            "roi_left_eye": _encode_roi_base64(None),
            "roi_right_eye": _encode_roi_base64(None),
            "roi_mouth": _encode_roi_base64(None),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }

    def _initialize_models(self) -> None:
        """Khởi tạo an toàn YOLO26, MediaPipe FaceLandmarker và MobileNetV3-Small CNN (Zero-Crash)."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Tải YOLO26n
        try:
            from ultralytics import YOLO
            yolo_target = str(YOLO_WEIGHTS_PATH) if YOLO_WEIGHTS_PATH.exists() else "yolo26n.pt"
            self.yolo_model = YOLO(yolo_target)
            self.yolo_ready = True
        except Exception as exc:
            print(f"[AI Engine] Cảnh báo khởi tạo YOLO26n (chuyển sang chế độ dự phòng): {exc}")
            self.yolo_ready = False

        # 2. Tải MediaPipe FaceLandmarker (478 3D Landmarks)
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            if not FACE_LANDMARKER_PATH.exists():
                try:
                    urllib.request.urlretrieve(FACE_LANDMARKER_URL, str(FACE_LANDMARKER_PATH))
                except Exception:
                    pass

            if FACE_LANDMARKER_PATH.exists() and FACE_LANDMARKER_PATH.stat().st_size > 500_000:
                base_opts = mp_python.BaseOptions(model_asset_path=str(FACE_LANDMARKER_PATH))
                options = mp_vision.FaceLandmarkerOptions(
                    base_options=base_opts,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=True,
                    num_faces=1,
                    min_face_detection_confidence=0.35,
                    min_face_presence_confidence=0.35,
                    min_tracking_confidence=0.35,
                )
                self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
                self.mediapipe_ready = True
        except Exception as exc:
            print(f"[AI Engine] Dùng bộ định vị khuôn mặt OpenCV dự phòng ({exc})")
            self.mediapipe_ready = False

        # 3. Tải mạng học sâu CNN MobileNetV3-Small
        self.reload_cnn_if_updated()

    def reload_cnn_if_updated(self) -> bool:
        """Tự động nạp hoặc làm mới trọng số models/cnn_driver_state.pth ngay khi huấn luyện xong."""
        if not CNN_WEIGHTS_PATH.exists():
            return False
        mtime = CNN_WEIGHTS_PATH.stat().st_mtime
        if self.cnn_ready and abs(mtime - self._cnn_mtime) < 1e-3:
            return True
        try:
            model, cls_names, img_size, _ = load_trained_cnn(str(CNN_WEIGHTS_PATH), device=self.device)
            self.cnn_model = model
            self.cnn_class_names = cls_names
            self.cnn_img_size = img_size
            self._cnn_mtime = mtime
            self.cnn_ready = True
            print(f"[AI Engine] Đã nạp thành công trọng số CNN: {CNN_WEIGHTS_PATH.name}")
            return True
        except Exception as exc:
            print(f"[AI Engine] Chưa nạp được CNN (sử dụng chế độ dự phòng EAR/MAR): {exc}")
            self.cnn_ready = False
            return False

    def update_vehicle_context(
        self,
        speed_kmh: Optional[float] = None,
        gear: Optional[str] = None,
        drowsy_threshold_sec: Optional[float] = None,
        ir_mode: Optional[bool] = None,
    ) -> Dict:
        with self.lock:
            if speed_kmh is not None:
                self.speed_kmh = max(0.0, min(140.0, float(speed_kmh)))
            if gear is not None and gear.upper() in {"P", "N", "D"}:
                self.gear = gear.upper()
            if drowsy_threshold_sec is not None:
                self.drowsy_threshold_sec = max(1.0, min(5.0, float(drowsy_threshold_sec)))
            if ir_mode is not None:
                self.ir_mode = bool(ir_mode)

            self.latest_telemetry["speed_kmh"] = self.speed_kmh
            self.latest_telemetry["gear"] = self.gear
            self.latest_telemetry["drowsy_threshold_sec"] = self.drowsy_threshold_sec
            self.latest_telemetry["ir_mode"] = self.ir_mode
            self.latest_telemetry["standby_mode"] = (self.gear != "D") or (self.speed_kmh <= 5.0)
        return self.get_telemetry()

    def start_stream(self, mode: str, video_path: Optional[str] = None) -> Dict:
        """Chuyển đổi nguồn luồng video: 'webcam', 'video' (.mp4 tải lên) hoặc 'demo'."""
        self.stop_stream()
        with self._cap_lock:
            if mode == "webcam":
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    # Tự động chuyển sang video mô phỏng thực tế nếu máy không cắm webcam vật lý
                    demo_file = ensure_demo_video()
                    cap = cv2.VideoCapture(str(demo_file))
                    self.source_mode = "demo"
                    self.video_path = str(demo_file)
                else:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.source_mode = "webcam"
                    self.video_path = None
                self._cap = cap

            elif mode == "video" and video_path and os.path.exists(video_path):
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise RuntimeError("Không thể giải mã tệp video đã tải lên.")
                self._cap = cap
                self.source_mode = "video"
                self.video_path = video_path

            else:
                demo_file = ensure_demo_video()
                cap = cv2.VideoCapture(str(demo_file))
                self._cap = cap
                self.source_mode = "demo"
                self.video_path = str(demo_file)

            self._reset_timers()
            return {"status": "ok", "source_mode": self.source_mode, "video_path": self.video_path}

    def stop_stream(self) -> Dict:
        with self._cap_lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
            self.source_mode = "stopped"
            self._reset_timers()
            self.latest_telemetry = self._default_telemetry()
            self.latest_telemetry["cnn_ready"] = self.cnn_ready
            self.latest_telemetry["yolo_ready"] = self.yolo_ready
        return {"status": "stopped"}

    def _reset_timers(self) -> None:
        self._last_tick = time.perf_counter()
        self.eye_closed_duration = 0.0
        self.yawn_duration = 0.0
        self.head_nod_duration = 0.0
        self._yawn_active_flag = False
        self._yawn_events.clear()
        self._ema_p_closed = 8.0
        self._ema_p_yawn = 5.0

    def _apply_night_ir_filter(self, bgr_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mô phỏng camera hồng ngoại NIR 850nm trong khoang lái ban đêm:
        - Trả về (display_ir_bgr, pure_nir_3ch_bgr) để vừa trực quan hóa trên HUD vừa khớp
          đặc trưng ảnh xám 3 kênh của bộ dữ liệu MRL Eye Dataset.
        """
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        enhanced = self._clahe.apply(gray)
        pure_nir_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        # Tạo tông màu bạc-lục nhạt đặc trưng của cảm biến hồng ngoại ô tô ban đêm trên màn hình hiển thị
        b_ch = np.clip(enhanced.astype(np.float32) * 0.92, 0, 255).astype(np.uint8)
        g_ch = np.clip(enhanced.astype(np.float32) * 1.03, 0, 255).astype(np.uint8)
        r_ch = np.clip(enhanced.astype(np.float32) * 0.88, 0, 255).astype(np.uint8)
        display_ir = cv2.merge([b_ch, g_ch, r_ch])
        return display_ir, pure_nir_bgr

    @staticmethod
    def _eye_aspect_ratio(pts: np.ndarray) -> float:
        # pts: 6 điểm [p1, p2, p3, p4, p5, p6]
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        horiz = max(np.linalg.norm(pts[0] - pts[3]), 1e-4)
        return float((v1 + v2) / (2.0 * horiz))

    @staticmethod
    def _mouth_aspect_ratio(pts: np.ndarray) -> float:
        # pts: [61, 291, 13, 14, 78, 308, 81, 178, 311, 402]
        horiz = max(np.linalg.norm(pts[0] - pts[1]), 1e-4)
        v_center = np.linalg.norm(pts[2] - pts[3])
        v_left = np.linalg.norm(pts[6] - pts[7])
        v_right = np.linalg.norm(pts[8] - pts[9])
        return float((v_center + v_left + v_right) / (3.0 * horiz))

    @staticmethod
    def _crop_with_margin(img: np.ndarray, pts: np.ndarray, margin_ratio: float = 0.45, square: bool = True) -> Optional[np.ndarray]:
        h, w = img.shape[:2]
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        bw = max(12.0, float(x_max - x_min))
        bh = max(12.0, float(y_max - y_min))

        cx = (x_min + x_max) * 0.5
        cy = (y_min + y_max) * 0.5
        if square:
            side = max(bw, bh) * (1.0 + 2.0 * margin_ratio)
            half_w = half_h = side * 0.5
        else:
            half_w = bw * (0.5 + margin_ratio)
            half_h = bh * (0.5 + margin_ratio)

        x1 = max(0, int(round(cx - half_w)))
        y1 = max(0, int(round(cy - half_h)))
        x2 = min(w, int(round(cx + half_w)))
        y2 = min(h, int(round(cy + half_h)))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        return img[y1:y2, x1:x2].copy()

    def _extract_landmarks_and_rois(
        self,
        analysis_bgr: np.ndarray,
        display_bgr: np.ndarray,
        person_box: Optional[Tuple[int, int, int, int]],
    ) -> Dict:
        """
        Trích xuất 3 vùng ROI (Mắt trái, Mắt phải, Miệng), khuôn mặt tổng thể và góc cúi đầu Pitch/Yaw.
        Ưu tiên MediaPipe FaceLandmarker 478 điểm, tự động dự phòng sang OpenCV Haar Cascade.
        """
        h, w = analysis_bgr.shape[:2]
        result = {
            "found": False,
            "backend": "MediaPipe FaceMesh",
            "left_eye_roi": None,
            "right_eye_roi": None,
            "mouth_roi": None,
            "face_roi": None,
            "ear": 0.30,
            "mar": 0.18,
            "pitch": 0.0,
            "yaw": 0.0,
        }

        # Nhánh 1: MediaPipe FaceLandmarker 478 điểm 3D
        if self.mediapipe_ready and self.face_landmarker is not None:
            try:
                import mediapipe as mp
                rgb_mp = cv2.cvtColor(analysis_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_mp)
                mp_res = self.face_landmarker.detect(mp_image)

                if mp_res.face_landmarks and len(mp_res.face_landmarks) > 0:
                    lm_list = mp_res.face_landmarks[0]
                    pts_2d = np.array([[lm.x * w, lm.y * h] for lm in lm_list], dtype=np.float32)

                    left_pts = pts_2d[LEFT_EYE_IDX]
                    right_pts = pts_2d[RIGHT_EYE_IDX]
                    mouth_pts = pts_2d[MOUTH_OUTER_IDX]

                    ear_l = self._eye_aspect_ratio(left_pts)
                    ear_r = self._eye_aspect_ratio(right_pts)
                    ear = (ear_l + ear_r) * 0.5
                    mar = self._mouth_aspect_ratio(mouth_pts)

                    # Tính góc cúi đầu (Pitch > 0 khi gục đầu xuống) & góc quay đầu (Yaw)
                    pitch_deg = 0.0
                    yaw_deg = 0.0
                    if mp_res.facial_transformation_matrixes and len(mp_res.facial_transformation_matrixes) > 0:
                        mat = np.array(mp_res.facial_transformation_matrixes[0])
                        rot = mat[:3, :3]
                        pitch_rad = math.atan2(-rot[1, 2], rot[2, 2])
                        yaw_rad = math.atan2(rot[0, 2], math.sqrt(rot[1, 2] ** 2 + rot[2, 2] ** 2))
                        pitch_deg = float(math.degrees(pitch_rad))
                        yaw_deg = float(math.degrees(yaw_rad))
                    else:
                        # Ước lượng hình học từ chóp mũi (1), trán (10), cằm (152)
                        forehead = pts_2d[10]
                        nose = pts_2d[1]
                        chin = pts_2d[152]
                        face_h = max(float(chin[1] - forehead[1]), 1.0)
                        nose_ratio = float(nose[1] - forehead[1]) / face_h
                        pitch_deg = (nose_ratio - 0.53) * 145.0

                    # Bổ sung hiệu chuẩn gục đầu theo vị trí dọc của mũi-mắt để nhạy với tư thế cúi gục thực tế
                    eye_mid = (left_pts.mean(axis=0) + right_pts.mean(axis=0)) * 0.5
                    chin_pt = pts_2d[152]
                    nose_pt = pts_2d[1]
                    eye_to_chin = max(float(chin_pt[1] - eye_mid[1]), 1.0)
                    geom_pitch = ((float(nose_pt[1] - eye_mid[1]) / eye_to_chin) - 0.36) * 135.0
                    if abs(pitch_deg) > 75.0:
                        pitch_deg = geom_pitch
                    else:
                        pitch_deg = 0.55 * (-pitch_deg if pitch_deg < -12 and geom_pitch > 12 else pitch_deg) + 0.45 * geom_pitch

                    left_roi = self._crop_with_margin(analysis_bgr, left_pts, margin_ratio=0.55, square=True)
                    right_roi = self._crop_with_margin(analysis_bgr, right_pts, margin_ratio=0.55, square=True)
                    mouth_roi = self._crop_with_margin(analysis_bgr, mouth_pts, margin_ratio=0.45, square=True)
                    face_roi = self._crop_with_margin(analysis_bgr, pts_2d, margin_ratio=0.12, square=True)

                    # Vẽ lưới mắt & miệng tinh tế lên khung hiển thị
                    cv2.polylines(display_bgr, [left_pts.astype(np.int32)], True, (56, 239, 125), 1, cv2.LINE_AA)
                    cv2.polylines(display_bgr, [right_pts.astype(np.int32)], True, (56, 239, 125), 1, cv2.LINE_AA)
                    cv2.polylines(display_bgr, [mouth_pts[:4].astype(np.int32)], True, (0, 212, 255), 1, cv2.LINE_AA)
                    for pt in np.vstack([left_pts, right_pts, mouth_pts[:6]]):
                        cv2.circle(display_bgr, (int(pt[0]), int(pt[1])), 2, (255, 240, 100), -1, cv2.LINE_AA)

                    result.update({
                        "found": True,
                        "backend": "MediaPipe FaceMesh (478D)",
                        "left_eye_roi": left_roi,
                        "right_eye_roi": right_roi,
                        "mouth_roi": mouth_roi,
                        "face_roi": face_roi,
                        "ear": round(float(ear), 3),
                        "mar": round(float(mar), 3),
                        "pitch": round(float(pitch_deg), 1),
                        "yaw": round(float(yaw_deg), 1),
                    })
                    return result
            except Exception:
                pass

        # Nhánh 2: Dự phòng OpenCV Haar Cascade + Ước lượng vùng hình học trên khung YOLO26
        gray = cv2.cvtColor(analysis_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(60, 60))

        fx, fy, fw, fh = 0, 0, 0, 0
        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
        elif person_box is not None:
            px1, py1, px2, py2 = person_box
            pw, ph = px2 - px1, py2 - py1
            fw = int(pw * 0.46)
            fh = int(ph * 0.44)
            fx = px1 + (pw - fw) // 2
            fy = py1 + int(ph * 0.06)

        if fw >= 40 and fh >= 40:
            fx, fy = max(0, fx), max(0, fy)
            fw, fh = min(w - fx, fw), min(h - fy, fh)
            face_roi = analysis_bgr[fy:fy + fh, fx:fx + fw].copy()

            # Tọa độ giải phẫu chuẩn Mắt trái, Mắt phải và Miệng
            lx1, ly1 = fx + int(fw * 0.14), fy + int(fh * 0.24)
            lx2, ly2 = fx + int(fw * 0.46), fy + int(fh * 0.48)
            rx1, ry1 = fx + int(fw * 0.54), fy + int(fh * 0.24)
            rx2, ry2 = fx + int(fw * 0.86), fy + int(fh * 0.48)
            mx1, my1 = fx + int(fw * 0.25), fy + int(fh * 0.62)
            mx2, my2 = fx + int(fw * 0.75), fy + int(fh * 0.90)

            left_roi = analysis_bgr[ly1:ly2, lx1:lx2].copy()
            right_roi = analysis_bgr[ry1:ry2, rx1:rx2].copy()
            mouth_roi = analysis_bgr[my1:my2, mx1:mx2].copy()

            cv2.rectangle(display_bgr, (lx1, ly1), (lx2, ly2), (56, 239, 125), 1)
            cv2.rectangle(display_bgr, (rx1, ry1), (rx2, ry2), (56, 239, 125), 1)
            cv2.rectangle(display_bgr, (mx1, my1), (mx2, my2), (0, 212, 255), 1)

            eyes_detected = self.eye_cascade.detectMultiScale(gray[fy:fy + int(fh * 0.6), fx:fx + fw], 1.1, 3)
            est_ear = 0.29 if len(eyes_detected) >= 1 else 0.16
            pitch_est = max(0.0, ((fy + fh * 0.5) / h - 0.44) * 140.0)

            result.update({
                "found": True,
                "backend": "OpenCV Landmarks Fallback",
                "left_eye_roi": left_roi,
                "right_eye_roi": right_roi,
                "mouth_roi": mouth_roi,
                "face_roi": face_roi,
                "ear": round(est_ear, 3),
                "mar": 0.22,
                "pitch": round(pitch_est, 1),
                "yaw": 0.0,
            })

        return result

    def _infer_cnn_probabilities(
        self,
        left_eye_bgr: Optional[np.ndarray],
        right_eye_bgr: Optional[np.ndarray],
        mouth_bgr: Optional[np.ndarray],
        face_bgr: Optional[np.ndarray],
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Suy luận Giai đoạn 2 bằng mạng học sâu MobileNetV3-Small trên Batch các vùng ROI:
        - Mắt trái & Mắt phải -> Xác suất có điều kiện P(Closed | Eye)
        - Miệng & Khuôn mặt -> Xác suất có điều kiện P(yawn | Mouth/Face)
        """
        self.reload_cnn_if_updated()
        if not self.cnn_ready or self.cnn_model is None:
            return None, None

        crops = []
        roles = []
        for role_name, crop in [
            ("left_eye", left_eye_bgr),
            ("right_eye", right_eye_bgr),
            ("mouth", mouth_bgr),
            ("face", face_bgr),
        ]:
            if crop is not None and crop.size > 0:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (self.cnn_img_size, self.cnn_img_size), interpolation=cv2.INTER_LINEAR)
                crops.append(resized)
                roles.append(role_name)

        if not crops:
            return None, None

        try:
            batch_np = np.stack(crops, axis=0).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            batch_np = (batch_np - self._mean_arr) / self._std_arr
            batch_t = torch.from_numpy(batch_np).to(self.device)

            with torch.inference_mode():
                logits = self.cnn_model(batch_t).cpu().numpy()

            idx_closed = self.cnn_class_names.index("Closed") if "Closed" in self.cnn_class_names else 0
            idx_open = self.cnn_class_names.index("Open") if "Open" in self.cnn_class_names else 1
            idx_no_yawn = self.cnn_class_names.index("no_yawn") if "no_yawn" in self.cnn_class_names else 2
            idx_yawn = self.cnn_class_names.index("yawn") if "yawn" in self.cnn_class_names else 3

            eye_probs = []
            yawn_probs = []
            for r_idx, role_name in enumerate(roles):
                z = logits[r_idx]
                if role_name in {"left_eye", "right_eye"}:
                    pair = np.array([z[idx_closed], z[idx_open]], dtype=np.float64)
                    pair_exp = np.exp(pair - pair.max())
                    p_closed_cond = float(pair_exp[0] / pair_exp.sum())
                    eye_probs.append(p_closed_cond)
                elif role_name in {"face", "mouth"}:
                    pair_y = np.array([z[idx_yawn], z[idx_no_yawn]], dtype=np.float64)
                    pair_y_exp = np.exp(pair_y - pair_y.max())
                    p_yawn_cond = float(pair_y_exp[0] / pair_y_exp.sum())
                    # Ảnh yawn/no_yawn trong Bộ 1 là ảnh vùng mặt/miệng nên ưu tiên trọng số face + mouth
                    weight = 0.65 if role_name == "face" else 0.35
                    yawn_probs.append((p_yawn_cond, weight))

            p_closed_cnn = float(np.mean(eye_probs) * 100.0) if eye_probs else None
            if yawn_probs:
                total_w = sum(w for _, w in yawn_probs)
                p_yawn_cnn = float(sum(p * w for p, w in yawn_probs) / max(total_w, 1e-6) * 100.0)
            else:
                p_yawn_cnn = None
            return p_closed_cnn, p_yawn_cnn
        except Exception:
            return None, None

    def process_single_frame(self, raw_bgr: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Xử lý trọn vẹn 1 khung hình qua Pipeline 2 giai đoạn + Bộ lọc ngữ cảnh xe + Cảnh báo 3 mức."""
        t_start = time.perf_counter()
        now_perf = t_start
        dt = max(0.01, min(0.35, now_perf - self._last_tick))
        self._last_tick = now_perf

        frame_bgr = cv2.resize(raw_bgr, (640, 480)) if (raw_bgr.shape[1] != 640 or raw_bgr.shape[0] != 480) else raw_bgr.copy()

        if self.ir_mode:
            display_bgr, analysis_bgr = self._apply_night_ir_filter(frame_bgr)
        else:
            display_bgr = frame_bgr.copy()
            analysis_bgr = frame_bgr

        # ==================== GIAI ĐOẠN 1: YOLO26n PHÁT HIỆN TÀI XẾ ====================
        self._frame_counter += 1
        driver_box: Optional[Tuple[int, int, int, int]] = self._cached_driver_box
        yolo_conf: float = self._cached_yolo_conf

        if self.yolo_ready and self.yolo_model is not None and (self._frame_counter % 3 == 1 or driver_box is None):
            try:
                preds = self.yolo_model.predict(analysis_bgr, imgsz=320, classes=[0], conf=0.30, verbose=False)
                if preds and len(preds[0].boxes) > 0:
                    boxes = preds[0].boxes
                    best_i = int(boxes.conf.argmax().item())
                    xyxy = boxes.xyxy[best_i].cpu().numpy().astype(int).tolist()
                    yolo_conf = float(boxes.conf[best_i].item())
                    driver_box = (xyxy[0], xyxy[1], xyxy[2], xyxy[3])
                    self._cached_driver_box = driver_box
                    self._cached_yolo_conf = yolo_conf
                else:
                    self._cached_driver_box = None
                    self._cached_yolo_conf = 0.0
                    driver_box = None
            except Exception:
                driver_box = None

        # Định vị Face Landmarks & cắt 3 vùng ROI
        lm_info = self._extract_landmarks_and_rois(analysis_bgr, display_bgr, driver_box)
        driver_detected = bool(driver_box is not None or lm_info["found"])

        if driver_box is not None:
            x1, y1, x2, y2 = driver_box
            box_color = (235, 165, 35) if self.ir_mode else (246, 130, 59)
            cv2.rectangle(display_bgr, (x1, y1), (x2, y2), box_color, 2)
            label_txt = f"YOLO26n Driver {yolo_conf*100:.0f}%"
            cv2.rectangle(display_bgr, (x1, max(0, y1 - 24)), (min(639, x1 + 195), y1), box_color, -1)
            cv2.putText(display_bgr, label_txt, (x1 + 6, max(16, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

        # ==================== GIAI ĐOẠN 2: MẠNG HỌC SÂU CNN + KẾT HỢP EAR/MAR ====================
        ear = lm_info["ear"]
        mar = lm_info["mar"]
        head_pitch = lm_info["pitch"]
        head_yaw = lm_info["yaw"]

        if lm_info["found"]:
            p_closed_cnn, p_yawn_cnn = self._infer_cnn_probabilities(
                lm_info["left_eye_roi"],
                lm_info["right_eye_roi"],
                lm_info["mouth_roi"],
                lm_info["face_roi"],
            )
            p_closed_geo = _sigmoid((0.215 - ear) * 28.0) * 100.0
            p_yawn_geo = _sigmoid((mar - 0.46) * 14.5) * 100.0

            raw_p_closed = (0.68 * p_closed_cnn + 0.32 * p_closed_geo) if p_closed_cnn is not None else p_closed_geo
            raw_p_yawn = (0.62 * p_yawn_cnn + 0.38 * p_yawn_geo) if p_yawn_cnn is not None else p_yawn_geo

            self._ema_p_closed = 0.58 * raw_p_closed + 0.42 * self._ema_p_closed
            self._ema_p_yawn = 0.58 * raw_p_yawn + 0.42 * self._ema_p_yawn
        else:
            self._ema_p_closed = max(0.0, self._ema_p_closed * 0.7)
            self._ema_p_yawn = max(0.0, self._ema_p_yawn * 0.7)

        p_closed = round(float(self._ema_p_closed), 1)
        p_yawn = round(float(self._ema_p_yawn), 1)

        # Cập nhật bộ đếm thời gian trạng thái mắt, ngáp và cúi đầu
        if lm_info["found"] and (p_closed >= 55.0 or ear < 0.195):
            self.eye_closed_duration += dt
        else:
            self.eye_closed_duration = max(0.0, self.eye_closed_duration - dt * 2.5)

        if lm_info["found"] and (p_yawn >= 56.0 or mar > 0.50):
            self.yawn_duration += dt
            if self.yawn_duration >= 1.3 and not self._yawn_active_flag:
                self._yawn_active_flag = True
                self._yawn_events.append(now_perf)
        else:
            self.yawn_duration = 0.0
            self._yawn_active_flag = False

        while self._yawn_events and (now_perf - self._yawn_events[0] > 60.0):
            self._yawn_events.popleft()
        yawns_per_min = len(self._yawn_events)

        if lm_info["found"] and head_pitch > 25.0:
            self.head_nod_duration += dt
        else:
            self.head_nod_duration = max(0.0, self.head_nod_duration - dt * 2.0)

        # ==================== BỘ LỌC NGỮ CẢNH XE ĐỖ & CẢNH BÁO 3 MỨC ====================
        with self.lock:
            speed = self.speed_kmh
            gear = self.gear
            threshold_sec = self.drowsy_threshold_sec

        is_parked_or_slow = (gear == "P") or (speed <= 5.0)
        is_neutral = (gear == "N") and not is_parked_or_slow
        standby_mode = is_parked_or_slow or is_neutral

        alert_level = 0
        alert_code = "LEVEL_0_AWAKE"
        alert_title = "MỨC 0: TỈNH TÁO - LÁI XE AN TOÀN"
        alert_subtitle = f"Vận tốc {speed:.0f} km/h (Số {gear}) • Nháy mắt tự nhiên (< 0.8s)"
        voice_message = ""
        siren_active = False
        banner_rgb = (16, 185, 129)  # Xanh lá

        if is_parked_or_slow:
            alert_level = 0
            alert_code = "STANDBY_PARKED"
            alert_title = "CHẾ ĐỘ CHỜ - XE ĐANG ĐỖ (Tạm ngắt còi cảnh báo)"
            alert_subtitle = f"Cần số {gear} | Vận tốc {speed:.0f} km/h ≤ 5 km/h • Đang nhận diện nhưng không phát còi"
            banner_rgb = (71, 85, 105)
        elif is_neutral:
            alert_level = 0
            alert_code = "STANDBY_NEUTRAL"
            alert_title = "CHẾ ĐỘ CHỜ - CẦN SỐ N (Tạm ngắt còi cảnh báo)"
            alert_subtitle = f"Cần số N | Vận tốc {speed:.0f} km/h • Chỉ kích hoạt cảnh báo khi vào số D"
            banner_rgb = (71, 85, 105)
        else:
            # Xe đang di chuyển ở Số D và Vận tốc > 5 km/h
            is_drowsy_eyes = self.eye_closed_duration >= threshold_sec
            is_drowsy_nod = (head_pitch > 25.0) and (self.head_nod_duration >= 2.0)

            if is_drowsy_eyes or is_drowsy_nod:
                alert_level = 2
                alert_code = "LEVEL_2_DANGER"
                alert_title = "NGUY HIỂM: TÀI XẾ NGỦ GẬT!"
                if is_drowsy_eyes and is_drowsy_nod:
                    reason = f"Nhắm mắt {self.eye_closed_duration:.1f}s & Cúi gục đầu {head_pitch:.1f}° ({self.head_nod_duration:.1f}s)"
                elif is_drowsy_eyes:
                    reason = f"Nhắm mắt liên tục {self.eye_closed_duration:.1f} giây (Ngưỡng {threshold_sec:.1f}s)"
                else:
                    reason = f"Cúi gục đầu {head_pitch:.1f}° liên tục {self.head_nod_duration:.1f} giây"

                alert_subtitle = reason
                voice_message = "Nguy hiểm! Phát hiện tài xế ngủ gật! Vui lòng tỉnh táo lại ngay lập tức!"
                siren_active = True
                banner_rgb = (220, 38, 38)  # Đỏ khẩn cấp

            elif (
                self.yawn_duration > 1.8
                or yawns_per_min >= 3
                or (1.0 <= self.eye_closed_duration < threshold_sec)
            ):
                alert_level = 1
                alert_code = "LEVEL_1_FATIGUE"
                alert_title = "MỨC 1: DẤU HIỆU MỆT MỎI"
                if self.yawn_duration > 1.8:
                    alert_subtitle = f"Phát hiện ngáp kéo dài {self.yawn_duration:.1f} giây (> 1.8s)"
                elif yawns_per_min >= 3:
                    alert_subtitle = f"Tần suất ngáp cao: {yawns_per_min} lần/phút (≥ 3 lần/phút)"
                else:
                    alert_subtitle = f"Mắt nhắm lờ đờ {self.eye_closed_duration:.1f} giây"
                voice_message = "Chú ý: Phát hiện dấu hiệu mệt mỏi, vui lòng tập trung lái xe!"
                siren_active = False
                banner_rgb = (217, 119, 6)  # Vàng hổ phách

        # Đo tốc độ FPS & độ trễ hoàn chỉnh
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        fps = min(60.0, 1000.0 / max(latency_ms, 1.0))

        # Vẽ băng rôn tiếng Việt & HUD thông số kỹ thuật lên khung hình
        display_bgr = _draw_vietnamese_banner(display_bgr, alert_title, (10, 10, 630, 50), banner_rgb, font_size=18)

        hud_bar = display_bgr.copy()
        cv2.rectangle(hud_bar, (10, 438), (630, 472), (15, 23, 42), -1)
        display_bgr = cv2.addWeighted(hud_bar, 0.80, display_bgr, 0.20, 0)
        mode_tag = "IR-NIGHT" if self.ir_mode else "RGB-DAY"
        hud_text = (
            f"FPS: {fps:4.1f} | {latency_ms:4.1f}ms | {mode_tag} | "
            f"EAR: {ear:.2f} MAR: {mar:.2f} | Pitch: {head_pitch:+.1f} deg | Gear: {gear} ({speed:.0f}km/h)"
        )
        cv2.putText(display_bgr, hud_text, (18, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (226, 232, 240), 1, cv2.LINE_AA)

        if alert_level == 2 and int(now_perf * 4) % 2 == 0:
            cv2.rectangle(display_bgr, (2, 2), (637, 477), (38, 38, 239), 5)

        # Lưu ảnh chụp vi phạm & Ghi SQLite + Gửi Telegram khi Mức 2 (giãn cách 6 giây giữa 2 lần ghi liên tiếp)
        if alert_level == 2 and (now_perf - self._last_violation_log_ts >= 6.0):
            self._last_violation_log_ts = now_perf
            SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
            snap_name = f"violation_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}.jpg"
            snap_file = SNAPSHOTS_DIR / snap_name
            cv2.imwrite(str(snap_file), display_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            snap_url = f"/snapshots/{snap_name}"

            rec_id = insert_violation(
                alert_level=2,
                alert_label=alert_title,
                violation_reason=alert_subtitle,
                speed_kmh=speed,
                gear=gear,
                p_closed=p_closed,
                p_yawn=p_yawn,
                ear=ear,
                mar=mar,
                head_pitch=head_pitch,
                head_yaw=head_yaw,
                ir_mode=self.ir_mode,
                snapshot_url=snap_url,
                telegram_sent=False,
            )

            self.notifier.send_drowsiness_alert_async(
                caption_info={
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "reason": alert_subtitle,
                    "speed_kmh": speed,
                    "gear": gear,
                    "p_closed": p_closed,
                    "p_yawn": p_yawn,
                    "head_pitch": head_pitch,
                    "ir_mode": self.ir_mode,
                },
                snapshot_file=snap_file,
                on_complete=lambda sent, rid=rec_id: update_telegram_status(rid, sent),
            )

        telemetry = {
            "active": True,
            "source_mode": self.source_mode,
            "ir_mode": self.ir_mode,
            "driver_detected": driver_detected,
            "yolo_conf": round(yolo_conf * 100.0, 1),
            "fps": round(fps, 1),
            "latency_ms": round(latency_ms, 1),
            "p_closed": p_closed,
            "p_yawn": p_yawn,
            "ear": ear,
            "mar": mar,
            "head_pitch": head_pitch,
            "head_yaw": head_yaw,
            "eye_closed_duration": round(self.eye_closed_duration, 2),
            "yawn_duration": round(self.yawn_duration, 2),
            "head_nod_duration": round(self.head_nod_duration, 2),
            "yawns_per_min": yawns_per_min,
            "speed_kmh": speed,
            "gear": gear,
            "drowsy_threshold_sec": threshold_sec,
            "standby_mode": standby_mode,
            "alert_level": alert_level,
            "alert_code": alert_code,
            "alert_title": alert_title,
            "alert_subtitle": alert_subtitle,
            "voice_message": voice_message,
            "siren_active": siren_active,
            "cnn_ready": self.cnn_ready,
            "yolo_ready": self.yolo_ready,
            "landmark_backend": lm_info["backend"],
            "roi_left_eye": _encode_roi_base64(lm_info["left_eye_roi"]),
            "roi_right_eye": _encode_roi_base64(lm_info["right_eye_roi"]),
            "roi_mouth": _encode_roi_base64(lm_info["mouth_roi"]),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }

        with self.lock:
            self.latest_telemetry = telemetry

        return display_bgr, telemetry

    def read_and_process_next(self) -> Optional[bytes]:
        """Đọc khung hình kế tiếp từ nguồn phát hiện tại và trả về JPEG bytes cho luồng MJPEG."""
        with self._cap_lock:
            if self._cap is None or self.source_mode == "stopped":
                return None
            ok, frame = self._cap.read()
            if not ok or frame is None:
                if self.source_mode in {"video", "demo"} and self.video_path:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = self._cap.read()
                if not ok or frame is None:
                    return None

        annotated, _ = self.process_single_frame(frame)
        ok_jpg, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
        if not ok_jpg:
            return None
        jpg_bytes = buf.tobytes()
        self.latest_jpeg = jpg_bytes
        return jpg_bytes

    def get_standby_jpeg(self) -> bytes:
        """Tạo khung hình chờ sắc nét khi người dùng tạm dừng Camera."""
        canvas = np.full((480, 640, 3), (15, 23, 42), dtype=np.uint8)
        cv2.rectangle(canvas, (20, 20), (620, 460), (51, 65, 85), 1)
        canvas = _draw_vietnamese_banner(
            canvas,
            "BUỒNG LÁI THÔNG MINH AI - ĐANG TẠM DỪNG CAMERA",
            (50, 185, 590, 235),
            (30, 58, 138),
            font_size=18,
        )
        canvas = _draw_vietnamese_banner(
            canvas,
            "Nhấn 'Webcam Trực tiếp', 'Video Mô phỏng' hoặc 'Tải Video (.mp4)' để bắt đầu",
            (40, 245, 600, 285),
            (30, 41, 59),
            font_size=14,
        )
        _, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        return buf.tobytes()

    def get_telemetry(self) -> Dict:
        with self.lock:
            data = dict(self.latest_telemetry)
            data["cnn_ready"] = self.cnn_ready
            data["yolo_ready"] = self.yolo_ready
            return data
