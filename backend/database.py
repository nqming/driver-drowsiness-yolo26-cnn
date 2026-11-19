"""
Quản lý cơ sở dữ liệu SQLite (logs/drowsiness_logs.db) lưu trữ lịch sử vi phạm ngủ gật,
ảnh chụp hiện trường (snapshots) và xuất báo cáo định dạng CSV chuẩn tiếng Việt (UTF-8 BOM).
"""

import csv
import io
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
SNAPSHOTS_DIR = LOGS_DIR / "snapshots"
DB_PATH = LOGS_DIR / "drowsiness_logs.db"

_db_lock = threading.Lock()


def init_db() -> None:
    """Khởi tạo thư mục logs/snapshots và bảng SQLite nếu chưa tồn tại."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS violation_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    alert_level INTEGER NOT NULL,
                    alert_label TEXT NOT NULL,
                    violation_reason TEXT NOT NULL,
                    speed_kmh REAL NOT NULL,
                    gear TEXT NOT NULL,
                    p_closed REAL NOT NULL,
                    p_yawn REAL NOT NULL,
                    ear REAL NOT NULL,
                    mar REAL NOT NULL,
                    head_pitch REAL NOT NULL,
                    head_yaw REAL NOT NULL,
                    ir_mode INTEGER NOT NULL DEFAULT 0,
                    snapshot_url TEXT,
                    telegram_sent INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.commit()
        finally:
            conn.close()


def insert_violation(
    alert_level: int,
    alert_label: str,
    violation_reason: str,
    speed_kmh: float,
    gear: str,
    p_closed: float,
    p_yawn: float,
    ear: float,
    mar: float,
    head_pitch: float,
    head_yaw: float,
    ir_mode: bool = False,
    snapshot_url: Optional[str] = None,
    telegram_sent: bool = False,
) -> int:
    """Ghi nhận một sự kiện vi phạm ngủ gật vào cơ sở dữ liệu SQLite."""
    init_db()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO violation_logs (
                    timestamp, alert_level, alert_label, violation_reason,
                    speed_kmh, gear, p_closed, p_yawn, ear, mar,
                    head_pitch, head_yaw, ir_mode, snapshot_url, telegram_sent
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    int(alert_level),
                    alert_label,
                    violation_reason,
                    round(float(speed_kmh), 1),
                    gear,
                    round(float(p_closed), 1),
                    round(float(p_yawn), 1),
                    round(float(ear), 3),
                    round(float(mar), 3),
                    round(float(head_pitch), 1),
                    round(float(head_yaw), 1),
                    1 if ir_mode else 0,
                    snapshot_url or "",
                    1 if telegram_sent else 0,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()


def update_telegram_status(record_id: int, sent: bool) -> None:
    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute(
                "UPDATE violation_logs SET telegram_sent = ? WHERE id = ?",
                (1 if sent else 0, int(record_id)),
            )
            conn.commit()
        finally:
            conn.close()


def get_recent_violations(limit: int = 50) -> List[Dict]:
    """Truy vấn danh sách các vi phạm gần nhất."""
    init_db()
    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM violation_logs ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def get_violation_stats() -> Dict:
    """Thống kê tổng số vi phạm phục vụ bảng điều khiển."""
    init_db()
    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        try:
            total = conn.execute("SELECT COUNT(*) FROM violation_logs").fetchone()[0]
            ir_count = conn.execute("SELECT COUNT(*) FROM violation_logs WHERE ir_mode = 1").fetchone()[0]
            avg_speed = conn.execute("SELECT COALESCE(AVG(speed_kmh), 0) FROM violation_logs").fetchone()[0]
            return {
                "total_violations": int(total),
                "night_ir_violations": int(ir_count),
                "avg_violation_speed": round(float(avg_speed), 1),
            }
        finally:
            conn.close()


def export_violations_csv() -> str:
    """Xuất toàn bộ lịch sử vi phạm ra chuỗi CSV có BOM UTF-8 để mở chuẩn tiếng Việt trên Excel."""
    records = get_recent_violations(limit=5000)
    output = io.StringIO()
    output.write("\ufeff")  # UTF-8 BOM cho Microsoft Excel
    writer = csv.writer(output)
    writer.writerow([
        "Mã vi phạm",
        "Thời gian",
        "Mức cảnh báo",
        "Trạng thái cảnh báo",
        "Nguyên nhân chi tiết",
        "Vận tốc xe (km/h)",
        "Cần số",
        "Xác suất Nhắm mắt (%)",
        "Xác suất Ngáp (%)",
        "Chỉ số EAR",
        "Chỉ số MAR",
        "Góc cúi đầu Pitch (độ)",
        "Góc quay đầu Yaw (độ)",
        "Chế độ Camera",
        "Đường dẫn ảnh chụp",
        "Đã gửi Telegram",
    ])
    for r in records:
        writer.writerow([
            r["id"],
            r["timestamp"],
            f"Mức {r['alert_level']}",
            r["alert_label"],
            r["violation_reason"],
            r["speed_kmh"],
            r["gear"],
            r["p_closed"],
            r["p_yawn"],
            r["ear"],
            r["mar"],
            r["head_pitch"],
            r["head_yaw"],
            "Hồng ngoại Ban đêm (IR)" if r["ir_mode"] else "RGB Tiêu chuẩn",
            r["snapshot_url"],
            "Có" if r["telegram_sent"] else "Không",
        ])
    return output.getvalue()


def clear_violations() -> None:
    """Xóa lịch sử vi phạm khi người dùng yêu cầu làm mới phiên."""
    init_db()
    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute("DELETE FROM violation_logs")
            conn.commit()
        finally:
            conn.close()
