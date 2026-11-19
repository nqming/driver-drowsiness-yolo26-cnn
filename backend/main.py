"""
Máy chủ FastAPI trung tâm (backend/main.py) cho hệ thống:
"Nhận dạng trạng thái buồn ngủ của người điều khiển phương tiện ô tô
 sử dụng kết hợp mô hình YOLO26 và mạng học sâu CNN"

Hỗ trợ MJPEG Video Streaming thời gian thực, WebSockets Telemetry, tải lên video kiểm thử (.mp4),
mô phỏng thông số xe (Vận tốc, Cần số P/N/D), Chế độ Hồng ngoại Ban đêm (IR Mode),
lưu trữ SQLite, xuất báo cáo CSV và gửi cảnh báo khẩn cấp qua Telegram Bot.
"""

import asyncio
import json
import shutil
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ai_engine import DriverMonitoringEngine
from backend.database import (
    SNAPSHOTS_DIR,
    clear_violations,
    export_violations_csv,
    get_recent_violations,
    get_violation_stats,
    init_db,
)
from backend.notifier import TelegramNotifier

FRONTEND_DIR = PROJECT_ROOT / "frontend"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

init_db()
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Hệ thống Nhận dạng Buồn ngủ Tài xế Ô tô (YOLO26 + CNN)",
    description="Đồ án Đại học: Kết hợp YOLO26 NMS-Free End-to-End và Mạng MobileNetV3-Small trên Đa bộ dữ liệu",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

notifier = TelegramNotifier()
engine = DriverMonitoringEngine(notifier=notifier)

# Gắn kết các thư mục tĩnh
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
app.mount("/snapshots", StaticFiles(directory=str(SNAPSHOTS_DIR)), name="snapshots")
app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")


class CameraStartRequest(BaseModel):
    mode: str = "webcam"  # 'webcam' | 'demo'
    ir_mode: Optional[bool] = None


class VehicleUpdateRequest(BaseModel):
    speed_kmh: Optional[float] = None
    gear: Optional[str] = None
    drowsy_threshold_sec: Optional[float] = None
    ir_mode: Optional[bool] = None


class TelegramConfigRequest(BaseModel):
    bot_token: str = ""
    chat_id: str = ""
    enabled: bool = True


@app.get("/")
async def serve_dashboard():
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Chưa tìm thấy giao diện frontend/index.html")
    return FileResponse(str(index_file))


@app.get("/api/health")
async def health_check():
    engine.reload_cnn_if_updated()
    return {
        "status": "healthy",
        "yolo26_loaded": engine.yolo_ready,
        "cnn_loaded": engine.cnn_ready,
        "mediapipe_loaded": engine.mediapipe_ready,
        "source_mode": engine.source_mode,
    }


def _mjpeg_generator():
    """Bộ sinh luồng MJPEG thời gian thực không gián đoạn."""
    while True:
        t0 = time.perf_counter()
        if engine.source_mode == "stopped":
            jpg_bytes = engine.get_standby_jpeg()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n"
            )
            time.sleep(0.35)
        else:
            jpg_bytes = engine.read_and_process_next()
            if jpg_bytes is None:
                jpg_bytes = engine.get_standby_jpeg()
                time.sleep(0.15)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n"
            )
            elapsed = time.perf_counter() - t0
            if elapsed < 0.045:
                time.sleep(0.045 - elapsed)


@app.get("/api/video_feed")
async def video_feed():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/api/telemetry")
async def get_telemetry():
    return engine.get_telemetry()


@app.websocket("/ws/telemetry")
async def websocket_telemetry(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            payload = engine.get_telemetry()
            await ws.send_json(payload)
            await asyncio.sleep(0.18)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


@app.post("/api/camera/start")
async def start_camera(req: CameraStartRequest):
    if req.ir_mode is not None:
        engine.update_vehicle_context(ir_mode=req.ir_mode)
    info = engine.start_stream(mode=req.mode)
    return info


@app.post("/api/camera/stop")
async def stop_camera():
    return engine.stop_stream()


@app.post("/api/camera/upload_video")
async def upload_test_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Vui lòng chọn tệp video hợp lệ (.mp4, .avi, .webm).")
    ext = Path(file.filename).suffix.lower()
    if ext not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        raise HTTPException(status_code=400, detail="Định dạng không hỗ trợ. Vui lòng tải lên tệp .mp4, .avi hoặc .webm.")

    safe_name = f"test_video_{int(time.time())}{ext}"
    dest_path = UPLOADS_DIR / safe_name
    with open(dest_path, "wb") as out_fp:
        shutil.copyfileobj(file.file, out_fp)

    info = engine.start_stream(mode="video", video_path=str(dest_path))
    info["uploaded_filename"] = file.filename
    return info


@app.post("/api/vehicle/update")
async def update_vehicle(req: VehicleUpdateRequest):
    return engine.update_vehicle_context(
        speed_kmh=req.speed_kmh,
        gear=req.gear,
        drowsy_threshold_sec=req.drowsy_threshold_sec,
        ir_mode=req.ir_mode,
    )


@app.get("/api/model_info")
async def get_model_and_dataset_info():
    engine.reload_cnn_if_updated()
    dataset_summary = {}
    training_metrics = {}
    benchmark_metrics = {}

    ds_file = DATA_DIR / "dataset_summary.json"
    tr_file = MODELS_DIR / "training_metrics.json"
    bm_file = MODELS_DIR / "benchmark_metrics.json"

    if ds_file.exists():
        dataset_summary = json.loads(ds_file.read_text(encoding="utf-8"))
    if tr_file.exists():
        training_metrics = json.loads(tr_file.read_text(encoding="utf-8"))
    if bm_file.exists():
        benchmark_metrics = json.loads(bm_file.read_text(encoding="utf-8"))

    figure_items = [
        {"file": "multi_dataset_distribution.png", "title": "Phân bố Đa Bộ Dữ Liệu (Yawn-Eye + MRL Eye)"},
        {"file": "dataset_samples_comparison.png", "title": "Đối chiếu Ảnh mẫu RGB Ngày & Hồng ngoại Đêm"},
        {"file": "loss_accuracy_curves.png", "title": "Đường cong Huấn luyện Loss & Accuracy"},
        {"file": "confusion_matrix.png", "title": "Ma trận Nhầm lẫn Tập Test Tổng hợp"},
        {"file": "cross_dataset_accuracy.png", "title": "So sánh Accuracy & F1 trên Bộ 1, Bộ 2 & Tập Gộp"},
        {"file": "roc_curve.png", "title": "Đường cong ROC & Chỉ số AUC Đa lớp"},
        {"file": "yolo26_benchmark.png", "title": "Hiệu năng Thời gian thực YOLO26n + CNN"},
    ]
    available_figures = [
        {**item, "url": f"/figures/{item['file']}"}
        for item in figure_items
        if (FIGURES_DIR / item["file"]).exists()
    ]

    return {
        "yolo_ready": engine.yolo_ready,
        "cnn_ready": engine.cnn_ready,
        "mediapipe_ready": engine.mediapipe_ready,
        "dataset_summary": dataset_summary,
        "training_metrics": training_metrics,
        "benchmark_metrics": benchmark_metrics,
        "figures": available_figures,
    }


@app.get("/api/logs")
async def list_violation_logs(limit: int = 50):
    return {
        "stats": get_violation_stats(),
        "items": get_recent_violations(limit=limit),
    }


@app.delete("/api/logs")
async def delete_all_logs():
    clear_violations()
    return {"status": "cleared"}


@app.get("/api/logs/export_csv")
async def download_violations_csv():
    csv_data = export_violations_csv()
    filename = f"bao_cao_vi_pham_ngu_gat_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        content=csv_data.encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/telegram/status")
async def telegram_status():
    return notifier.get_status()


@app.post("/api/telegram/config")
async def configure_telegram(req: TelegramConfigRequest):
    return notifier.configure(bot_token=req.bot_token, chat_id=req.chat_id, enabled=req.enabled)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
