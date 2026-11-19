"""
Đánh giá hiệu năng thời gian thực (FPS, Độ trễ ms/frame, Số lượng tham số & Dung lượng mô hình)
của Pipeline 2 giai đoạn: YOLO26 (yolo26n.pt NMS-Free End-to-End) + Mạng học sâu CNN (MobileNetV3-Small).
Xuất biểu đồ báo cáo 300 DPI: reports/figures/yolo26_benchmark.png
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.model_arch import CLASS_NAMES, IMG_SIZE, DriverStateCNN, load_trained_cnn

MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
BENCHMARK_FIG = FIGURES_DIR / "yolo26_benchmark.png"
BENCHMARK_JSON = MODELS_DIR / "benchmark_metrics.json"


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "Tahoma", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["savefig.dpi"] = 300


def ensure_yolo26_weights() -> Path:
    """Tải tự động trọng số yolo26n.pt ở lần chạy đầu tiên và lưu trữ đồng bộ vào models/yolo26n.pt."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target_path = MODELS_DIR / "yolo26n.pt"
    root_path = PROJECT_ROOT / "yolo26n.pt"

    if target_path.exists() and target_path.stat().st_size > 100_000:
        return target_path

    if root_path.exists() and root_path.stat().st_size > 100_000:
        shutil.copy2(root_path, target_path)
        return target_path

    print("[*] Đang tải trọng số thế hệ mới YOLO26 (yolo26n.pt - NMS-Free End-to-End)...")
    model = YOLO("yolo26n.pt")
    # Sao chép về thư mục models/ để quản lý tập trung
    ckpt_src = Path(getattr(model, "ckpt_path", "yolo26n.pt"))
    if ckpt_src.exists():
        shutil.copy2(ckpt_src, target_path)
    elif root_path.exists():
        shutil.copy2(root_path, target_path)
    return target_path if target_path.exists() else Path("yolo26n.pt")


def create_synthetic_driver_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Tạo khung hình mô phỏng buồng lái thực tế để đo tốc độ suy luận chuẩn xác."""
    frame = np.full((height, width, 3), 42, dtype=np.uint8)
    # Vẽ phông nền khoang lái & hình dáng tài xế giả lập
    cv2.rectangle(frame, (0, int(height * 0.65)), (width, height), (28, 32, 38), -1)
    cx, cy = width // 2, int(height * 0.42)
    cv2.ellipse(frame, (cx, int(height * 0.82)), (int(width * 0.24), int(height * 0.28)), 0, 180, 360, (70, 90, 120), -1)
    cv2.ellipse(frame, (cx, cy), (int(width * 0.11), int(height * 0.17)), 0, 0, 360, (175, 195, 215), -1)
    cv2.circle(frame, (cx - int(width * 0.04), cy - int(height * 0.03)), 8, (35, 35, 35), -1)
    cv2.circle(frame, (cx + int(width * 0.04), cy - int(height * 0.03)), 8, (35, 35, 35), -1)
    cv2.ellipse(frame, (cx, cy + int(height * 0.06)), (20, 8), 0, 0, 180, (60, 50, 130), 3)
    return frame


def run_benchmark(warmup_runs: int = 8, timed_runs: int = 35) -> Dict:
    configure_matplotlib()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(4, min(16, os.cpu_count() or 8)))

    yolo_path = ensure_yolo26_weights()
    yolo_model = YOLO(str(yolo_path))
    yolo_params = sum(p.numel() for p in yolo_model.model.parameters())
    yolo_size_mb = yolo_path.stat().st_size / (1024 * 1024) if yolo_path.exists() else 5.5

    cnn_ckpt = MODELS_DIR / "cnn_driver_state.pth"
    if cnn_ckpt.exists():
        cnn_model, _, img_size, _ = load_trained_cnn(str(cnn_ckpt), device=device)
        cnn_size_mb = cnn_ckpt.stat().st_size / (1024 * 1024)
    else:
        img_size = IMG_SIZE
        cnn_model = DriverStateCNN(num_classes=len(CLASS_NAMES), pretrained=False).to(device)
        cnn_model.eval()
        cnn_size_mb = 6.1

    cnn_params = sum(p.numel() for p in cnn_model.parameters())

    frame = create_synthetic_driver_frame(640, 480)
    roi_batch = torch.randn(3, 3, img_size, img_size, device=device)

    resolutions = [320, 480, 640]
    bench_records: List[Dict] = []

    print("=" * 78)
    print(f"ĐO HIỆU NĂNG PIPELINE YOLO26 (yolo26n.pt) + MOBILENETV3-SMALL ({device.type.upper()})")
    print(f"- YOLO26n Params        : {yolo_params:,} ({yolo_size_mb:.2f} MB)")
    print(f"- MobileNetV3-Small CNN : {cnn_params:,} ({cnn_size_mb:.2f} MB)")
    print("=" * 78)

    for imgsz in resolutions:
        # Warm-up
        for _ in range(warmup_runs):
            _ = yolo_model.predict(frame, imgsz=imgsz, classes=[0], conf=0.35, verbose=False)
            with torch.no_grad():
                _ = cnn_model(roi_batch)

        yolo_times_ms = []
        roi_times_ms = []
        cnn_times_ms = []
        total_times_ms = []

        for _ in range(timed_runs):
            t0 = time.perf_counter()
            _ = yolo_model.predict(frame, imgsz=imgsz, classes=[0], conf=0.35, verbose=False)
            t1 = time.perf_counter()

            # Mô phỏng bước cắt 3 ROI + chuẩn hóa tensor
            crop_l = cv2.resize(frame[160:210, 260:310], (img_size, img_size))
            crop_r = cv2.resize(frame[160:210, 330:380], (img_size, img_size))
            crop_m = cv2.resize(frame[220:280, 285:355], (img_size, img_size))
            stacked = np.stack([crop_l, crop_r, crop_m], axis=0).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            batch_t = torch.from_numpy(stacked).to(device)
            t2 = time.perf_counter()

            with torch.no_grad():
                logits = cnn_model(batch_t)
                _ = torch.softmax(logits, dim=1)
            t3 = time.perf_counter()

            y_ms = (t1 - t0) * 1000.0
            r_ms = (t2 - t1) * 1000.0
            c_ms = (t3 - t2) * 1000.0
            tot_ms = (t3 - t0) * 1000.0

            yolo_times_ms.append(y_ms)
            roi_times_ms.append(r_ms)
            cnn_times_ms.append(c_ms)
            total_times_ms.append(tot_ms)

        avg_yolo = float(np.median(yolo_times_ms))
        avg_roi = float(np.median(roi_times_ms))
        avg_cnn = float(np.median(cnn_times_ms))
        avg_total = float(np.median(total_times_ms))
        fps = 1000.0 / max(avg_total, 0.1)

        rec = {
            "imgsz": imgsz,
            "yolo_ms": round(avg_yolo, 2),
            "roi_ms": round(avg_roi, 2),
            "cnn_ms": round(avg_cnn, 2),
            "total_ms": round(avg_total, 2),
            "fps": round(fps, 1),
        }
        bench_records.append(rec)
        print(
            f"[imgsz={imgsz}px] YOLO26n: {avg_yolo:5.2f} ms | ROI Prep: {avg_roi:4.2f} ms | "
            f"CNN (3 ROIs): {avg_cnn:5.2f} ms | Tổng trễ: {avg_total:5.2f} ms -> {fps:5.1f} FPS"
        )

    # Xuất biểu đồ 3 thành phần chuyên nghiệp 300 DPI
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
    x_labels = [f"{r['imgsz']}×{r['imgsz']} px" for r in bench_records]
    x = np.arange(len(bench_records))

    # Panel 1: Phân rã độ trễ từng giai đoạn (Stacked Bar)
    yolo_vals = np.array([r["yolo_ms"] for r in bench_records])
    roi_vals = np.array([r["roi_ms"] for r in bench_records])
    cnn_vals = np.array([r["cnn_ms"] for r in bench_records])
    tot_vals = yolo_vals + roi_vals + cnn_vals

    ax1 = axes[0]
    ax1.bar(x, yolo_vals, width=0.52, label="GĐ 1: YOLO26n (NMS-Free)", color="#2563EB")
    ax1.bar(x, roi_vals, width=0.52, bottom=yolo_vals, label="Cắt & Chuẩn hóa 3 vùng ROI", color="#F59E0B")
    ax1.bar(x, cnn_vals, width=0.52, bottom=yolo_vals + roi_vals, label="GĐ 2: MobileNetV3-Small (Batch 3)", color="#8B5CF6")
    for i, tv in enumerate(tot_vals):
        ax1.text(x[i], tv + max(tot_vals) * 0.025, f"{tv:.1f} ms", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax1.set_title("Độ trễ Suy luận theo Giai đoạn (ms/frame)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Độ trễ trung vị (ms)", fontsize=10.5, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, fontweight="bold")
    ax1.set_ylim(0, max(tot_vals) * 1.22)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend(fontsize=8.8, loc="upper left")

    # Panel 2: Tốc độ khung hình thời gian thực (FPS)
    ax2 = axes[1]
    fps_vals = [r["fps"] for r in bench_records]
    bars_fps = ax2.bar(x, fps_vals, width=0.52, color=["#10B981", "#0EA5E9", "#3B82F6"], edgecolor="white")
    ax2.axhline(24.0, color="#EF4444", linestyle="--", linewidth=1.8, label="Chuẩn thời gian thực (24 FPS)")
    for bar, fv in zip(bars_fps, fps_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, fv + max(fps_vals) * 0.025, f"{fv:.1f} FPS", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax2.set_title("Tốc độ Xử lý Khung hình Toàn cục (FPS)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Số khung hình / giây (FPS)", fontsize=10.5, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, fontweight="bold")
    ax2.set_ylim(0, max(max(fps_vals) * 1.22, 32))
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.legend(fontsize=9.2, loc="upper right")

    # Panel 3: Số lượng tham số (Triệu tham số - M Params)
    ax3 = axes[2]
    comp_names = ["YOLO26n\n(yolo26n.pt)", "MobileNetV3-Small\n(DriverStateCNN)", "Tổng Pipeline\nKết hợp"]
    param_millions = [
        yolo_params / 1e6,
        cnn_params / 1e6,
        (yolo_params + cnn_params) / 1e6,
    ]
    bars_p = ax3.bar(
        np.arange(3),
        param_millions,
        width=0.52,
        color=["#2563EB", "#8B5CF6", "#0F172A"],
        edgecolor="white",
    )
    for bar, pm in zip(bars_p, param_millions):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            pm + max(param_millions) * 0.025,
            f"{pm:.2f}M",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )
    ax3.set_title("Độ Phức tạp Tham số Mô hình (Million Params)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Số lượng tham số (Triệu - Millions)", fontsize=10.5, fontweight="bold")
    ax3.set_xticks(np.arange(3))
    ax3.set_xticklabels(comp_names, fontsize=9.5, fontweight="bold")
    ax3.set_ylim(0, max(param_millions) * 1.24)
    ax3.grid(axis="y", linestyle="--", alpha=0.35)

    for ax in axes:
        sns.despine(ax=ax)

    fig.suptitle(
        f"Báo cáo Hiệu năng Hệ thống YOLO26 + CNN trên Thiết bị Thực tế ({device.type.upper()})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(BENCHMARK_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "device": device.type.upper(),
        "cpu_cores": os.cpu_count() or 1,
        "yolo_model": "yolo26n.pt",
        "yolo_params": int(yolo_params),
        "yolo_size_mb": round(float(yolo_size_mb), 2),
        "cnn_model": "MobileNetV3-Small",
        "cnn_params": int(cnn_params),
        "cnn_size_mb": round(float(cnn_size_mb), 2),
        "total_params": int(yolo_params + cnn_params),
        "resolutions": bench_records,
    }
    BENCHMARK_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[*] Đã lưu biểu đồ hiệu năng YOLO26 tại: {BENCHMARK_FIG}")
    return summary


if __name__ == "__main__":
    run_benchmark()
