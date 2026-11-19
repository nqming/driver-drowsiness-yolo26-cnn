"""
Huấn luyện mạng học sâu CNN (MobileNetV3-Small Transfer Learning) trên Đa bộ dữ liệu hợp nhất
(Bộ 1: Drowsiness Yawn-Eye RGB + Bộ 2: MRL Eye NIR Hồng ngoại ban đêm)
và xuất trọn bộ 6 biểu đồ học thuật 300 DPI + báo cáo phân loại bằng tiếng Việt.
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.download_datasets import MERGED_DIR, METADATA_CSV, prepare_merged_dataset
from training.model_arch import (
    CLASS_LABELS_VI,
    CLASS_NAMES,
    IMG_SIZE,
    DriverStateCNN,
    get_eval_transforms,
    get_train_transforms,
)

MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
MODEL_SAVE_PATH = MODELS_DIR / "cnn_driver_state.pth"
METRICS_JSON_PATH = MODELS_DIR / "training_metrics.json"


def configure_vietnamese_matplotlib() -> None:
    """Cấu hình font chữ tiếng Việt chuẩn sắc nét cho biểu đồ học thuật 300 DPI."""
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "Tahoma", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 300


class MetadataImageDataset(Dataset):
    """Dataset đọc trực tiếp từ metadata.csv để theo dõi cả nhãn lớp lẫn nguồn dữ liệu gốc."""

    def __init__(self, df: pd.DataFrame, class_to_idx: Dict[str, int], transform=None):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = PROJECT_ROOT / row["filepath"]
        with Image.open(img_path) as img:
            rgb_img = img.convert("RGB")
        if self.transform is not None:
            tensor = self.transform(rgb_img)
        else:
            tensor = rgb_img
        label_idx = self.class_to_idx[row["label"]]
        source_ds = row["source_dataset"]
        return tensor, label_idx, source_ds


def plot_multi_dataset_distribution(meta_df: pd.DataFrame, out_path: Path) -> None:
    """1. Biểu đồ cột chồng so sánh số lượng mẫu đóng góp từ Bộ 1 và Bộ 2 ở từng lớp."""
    pivot = (
        meta_df.groupby(["label", "source_dataset"])
        .size()
        .unstack(fill_value=0)
        .reindex(CLASS_NAMES, fill_value=0)
    )
    ds1_counts = pivot.get("Dataset1_YawnEye", pd.Series([0] * len(CLASS_NAMES), index=CLASS_NAMES)).values
    ds2_counts = pivot.get("Dataset2_MRLEye", pd.Series([0] * len(CLASS_NAMES), index=CLASS_NAMES)).values

    x_labels = [
        "Nhắm mắt\n(Closed)",
        "Mở mắt\n(Open)",
        "Bình thường\n(no_yawn)",
        "Ngáp\n(yawn)",
    ]
    x = np.arange(len(CLASS_NAMES))
    width = 0.56

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(
        x,
        ds1_counts,
        width,
        label="Bộ dữ liệu 1: Drowsiness Yawn-Eye (RGB Ban ngày)",
        color="#2563EB",
        edgecolor="white",
        linewidth=1.2,
    )
    bars2 = ax.bar(
        x,
        ds2_counts,
        width,
        bottom=ds1_counts,
        label="Bộ dữ liệu 2: MRL Eye (Hồng ngoại NIR Ban đêm)",
        color="#7C3AED",
        edgecolor="white",
        linewidth=1.2,
    )

    for i in range(len(x)):
        if ds1_counts[i] > 0:
            ax.text(
                x[i],
                ds1_counts[i] / 2,
                f"{ds1_counts[i]:,}",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=10,
            )
        if ds2_counts[i] > 0:
            ax.text(
                x[i],
                ds1_counts[i] + ds2_counts[i] / 2,
                f"{ds2_counts[i]:,}",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=10,
            )
        total_val = ds1_counts[i] + ds2_counts[i]
        ax.text(
            x[i],
            total_val + max(ds1_counts + ds2_counts) * 0.02,
            f"Tổng: {total_val:,}",
            ha="center",
            va="bottom",
            color="#1E293B",
            fontweight="bold",
            fontsize=10.5,
        )

    ax.set_title(
        "Phân bố Dữ liệu Hợp nhất Đa Nguồn (Multi-Dataset Stratified Distribution)",
        fontsize=14,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("Nhãn trạng thái tài xế (4 lớp chuẩn hóa)", fontsize=11.5, fontweight="bold")
    ax.set_ylabel("Số lượng mẫu ảnh (RGB 3 kênh)", fontsize=11.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10.5)
    ax.set_ylim(0, max(ds1_counts + ds2_counts) * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    sns.despine(ax=ax)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_dataset_samples_comparison(meta_df: pd.DataFrame, out_path: Path) -> None:
    """2. Hình trực quan hóa ảnh mẫu RGB từ Bộ 1 và ảnh hồng ngoại ban đêm từ Bộ 2."""
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.8))
    fig.suptitle(
        "Trực quan hóa Mẫu Ảnh Thực tế: Bộ 1 (Yawn-Eye RGB Ban ngày) vs Bộ 2 (MRL Eye NIR Ban đêm)",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )

    # Hàng 1: 4 lớp từ Bộ dữ liệu 1
    ds1_df = meta_df[meta_df["source_dataset"] == "Dataset1_YawnEye"]
    for col_idx, cls_name in enumerate(CLASS_NAMES):
        ax = axes[0, col_idx]
        sub = ds1_df[ds1_df["label"] == cls_name]
        if len(sub) > 0:
            sample_row = sub.iloc[min(3, len(sub) - 1)]
            with Image.open(PROJECT_ROOT / sample_row["filepath"]) as im:
                ax.imshow(im.convert("RGB"))
        ax.set_title(f"Bộ 1 (RGB) - {CLASS_LABELS_VI[cls_name]}", fontsize=9.8, fontweight="bold", color="#1D4ED8")
        ax.axis("off")

    # Hàng 2: 2 mẫu Closed + 2 mẫu Open từ Bộ dữ liệu 2 (Hồng ngoại NIR, bao gồm đeo kính/thiếu sáng)
    ds2_df = meta_df[meta_df["source_dataset"] == "Dataset2_MRLEye"]
    ds2_specs = [
        ("Closed", 0, "Bộ 2 (NIR Đêm) - Nhắm mắt #1"),
        ("Closed", 7, "Bộ 2 (NIR Đêm) - Nhắm mắt #2"),
        ("Open", 0, "Bộ 2 (NIR Đêm) - Mở mắt #1"),
        ("Open", 7, "Bộ 2 (NIR Đêm) - Mở mắt #2"),
    ]
    for col_idx, (cls_name, pick_idx, title_vi) in enumerate(ds2_specs):
        ax = axes[1, col_idx]
        sub = ds2_df[ds2_df["label"] == cls_name]
        if len(sub) > 0:
            sample_row = sub.iloc[min(pick_idx, len(sub) - 1)]
            with Image.open(PROJECT_ROOT / sample_row["filepath"]) as im:
                ax.imshow(im.convert("RGB"))
        ax.set_title(title_vi, fontsize=9.8, fontweight="bold", color="#6D28D9")
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_loss_accuracy_curves(history: Dict[str, List[float]], best_epoch: int, out_path: Path) -> None:
    """3. Đường cong quá trình huấn luyện Train/Val Loss & Accuracy."""
    epochs_range = list(range(1, len(history["train_loss"]) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))

    ax1.plot(epochs_range, history["train_loss"], "o-", color="#2563EB", linewidth=2.2, label="Hàm mất mát Huấn luyện (Train Loss)")
    ax1.plot(epochs_range, history["val_loss"], "s--", color="#DC2626", linewidth=2.2, label="Hàm mất mát Kiểm định (Val Loss)")
    ax1.axvline(best_epoch, color="#059669", linestyle=":", linewidth=1.8, label=f"Mô hình tốt nhất (Epoch {best_epoch})")
    ax1.set_title("Biến thiên Hàm Mất mát (Cross-Entropy Loss)", fontsize=12.5, fontweight="bold")
    ax1.set_xlabel("Chu kỳ huấn luyện (Epoch)", fontsize=11)
    ax1.set_ylabel("Giá trị Loss", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(fontsize=9.5)

    train_acc_pct = [v * 100 for v in history["train_acc"]]
    val_acc_pct = [v * 100 for v in history["val_acc"]]
    ax2.plot(epochs_range, train_acc_pct, "o-", color="#2563EB", linewidth=2.2, label="Độ chính xác Huấn luyện (Train Acc)")
    ax2.plot(epochs_range, val_acc_pct, "s--", color="#059669", linewidth=2.2, label="Độ chính xác Kiểm định (Val Acc)")
    ax2.axvline(best_epoch, color="#D97706", linestyle=":", linewidth=1.8, label=f"Đỉnh Val Acc: {max(val_acc_pct):.2f}%")
    ax2.set_title("Biến thiên Độ Chính xác (Accuracy %)", fontsize=12.5, fontweight="bold")
    ax2.set_xlabel("Chu kỳ huấn luyện (Epoch)", fontsize=11)
    ax2.set_ylabel("Độ chính xác (%)", fontsize=11)
    ax2.set_ylim(max(50.0, min(train_acc_pct + val_acc_pct) - 5), 101.0)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(fontsize=9.5)

    fig.suptitle("Quá trình Huấn luyện Mạng MobileNetV3-Small trên Đa Bộ Dữ Liệu", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix_vi(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    """4. Ma trận nhầm lẫn trên Tập Test Tổng hợp."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    cm_pct = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100.0

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)"

    labels_display = ["Nhắm mắt\n(Closed)", "Mở mắt\n(Open)", "Bình thường\n(no_yawn)", "Ngáp\n(yawn)"]
    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=labels_display,
        yticklabels=labels_display,
        cbar_kws={"label": "Số lượng mẫu dự đoán"},
        linewidths=1.0,
        linecolor="white",
        ax=ax,
        annot_kws={"size": 10.5, "weight": "bold"},
    )
    ax.set_title("Ma trận Nhầm lẫn (Confusion Matrix) - Tập Kiểm thử Tổng hợp", fontsize=13.5, fontweight="bold", pad=14)
    ax.set_xlabel("Nhãn Dự đoán bởi MobileNetV3-Small", fontsize=11.5, fontweight="bold", labelpad=10)
    ax.set_ylabel("Nhãn Thực tế (Ground Truth)", fontsize=11.5, fontweight="bold", labelpad=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_cross_dataset_accuracy(eval_summary: Dict[str, Dict[str, float]], out_path: Path) -> None:
    """5. Biểu đồ so sánh Accuracy, Precision, Recall, F1-Score trên Bộ 1, Bộ 2 và Tập Gộp."""
    groups = [
        ("Dataset1_YawnEye", "Tập Test Bộ 1\n(Yawn-Eye RGB Ban ngày)"),
        ("Dataset2_MRLEye", "Tập Test Bộ 2\n(MRL Eye NIR Ban đêm)"),
        ("Combined_All", "Tập Test Tổng hợp\n(Đa bộ dữ liệu Ngày + Đêm)"),
    ]
    metrics_keys = [
        ("accuracy", "Accuracy (%)", "#2563EB"),
        ("precision_macro", "Precision (%)", "#0EA5E9"),
        ("recall_macro", "Recall (%)", "#10B981"),
        ("f1_macro", "F1-Score (%)", "#8B5CF6"),
    ]

    x = np.arange(len(groups))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10.8, 6.2))

    for m_idx, (m_key, m_label, color) in enumerate(metrics_keys):
        vals = [eval_summary[g_key][m_key] * 100.0 for g_key, _ in groups]
        offset = (m_idx - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=m_label, color=color, edgecolor="white", linewidth=1.0)
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.6,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.8,
                fontweight="bold",
                color="#1E293B",
            )

    ax.set_title(
        "Đánh giá Chéo Độ Chính xác & F1-Score trên Từng Bộ Dữ liệu Kiểm thử",
        fontsize=13.5,
        fontweight="bold",
        pad=15,
    )
    ax.set_ylabel("Điểm hiệu năng (%)", fontsize=11.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([g_label for _, g_label in groups], fontsize=10.8, fontweight="bold")
    ax.set_ylim(0, 109)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    sns.despine(ax=ax)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_roc_curves(y_true: np.ndarray, y_probs: np.ndarray, out_path: Path) -> Dict[str, float]:
    """6. Đường cong ROC đa lớp (One-vs-Rest) và diện tích dưới đường cong AUC."""
    n_classes = len(CLASS_NAMES)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fpr, tpr, roc_auc = {}, {}, {}
    colors = ["#EF4444", "#10B981", "#3B82F6", "#F59E0B"]

    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    for i, cls_name in enumerate(CLASS_NAMES):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc[cls_name] = float(auc(fpr[i], tpr[i]))
        ax.plot(
            fpr[i],
            tpr[i],
            color=colors[i],
            lw=2.2,
            label=f"{CLASS_LABELS_VI[cls_name]} (AUC = {roc_auc[cls_name]:.4f})",
        )

    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_probs.ravel())
    auc_micro = float(auc(fpr_micro, tpr_micro))
    roc_auc["micro_avg"] = auc_micro
    ax.plot(
        fpr_micro,
        tpr_micro,
        color="#1E293B",
        linestyle="--",
        lw=2.4,
        label=f"Trung bình vi mô - Micro-average (AUC = {auc_micro:.4f})",
    )

    ax.plot([0, 1], [0, 1], color="#94A3B8", linestyle=":", lw=1.5, label="Ngẫu nhiên (AUC = 0.5000)")
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.03])
    ax.set_xlabel("Tỷ lệ Dương tính Giả (False Positive Rate - FPR)", fontsize=11.5, fontweight="bold")
    ax.set_ylabel("Tỷ lệ Dương tính Thật (True Positive Rate - TPR)", fontsize=11.5, fontweight="bold")
    ax.set_title("Đường cong ROC & Chỉ số AUC Đa lớp (MobileNetV3-Small CNN)", fontsize=13.5, fontweight="bold", pad=14)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=9.8, frameon=True)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return roc_auc


def compute_subset_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[Dict[str, float], str]:
    """Tính toán chỉ số đánh giá chính xác theo các lớp hiện diện thực tế trong từng bộ dữ liệu."""
    present_labels = sorted(list(set(y_true.tolist())))
    present_names = [CLASS_LABELS_VI[CLASS_NAMES[i]] for i in present_labels]

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, labels=present_labels, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, labels=present_labels, average="macro", zero_division=0))
    f1_mac = float(f1_score(y_true, y_pred, labels=present_labels, average="macro", zero_division=0))
    f1_w = float(f1_score(y_true, y_pred, labels=present_labels, average="weighted", zero_division=0))

    rep_text = classification_report(
        y_true,
        y_pred,
        labels=present_labels,
        target_names=present_names,
        digits=4,
        zero_division=0,
    )
    return {
        "samples": int(len(y_true)),
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1_mac,
        "f1_weighted": f1_w,
    }, rep_text


def train_and_evaluate(epochs: int = 12, batch_size: int = 64, lr: float = 1.2e-3, patience: int = 4) -> Dict:
    configure_vietnamese_matplotlib()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Tự động tải & chuẩn bị dữ liệu nếu chưa tồn tại
    if not MERGED_DIR.exists() or not METADATA_CSV.exists():
        print("[*] Chưa phát hiện data/merged_dataset/, tự động kích hoạt download_datasets.py...")
        meta_df = prepare_merged_dataset(force_rebuild=False)
    else:
        meta_df = pd.read_csv(METADATA_CSV)
        if len(meta_df) < 500:
            meta_df = prepare_merged_dataset(force_rebuild=True)

    # Xuất ngay 2 biểu đồ phân tích dữ liệu đầu vào
    print("[*] Đang xuất biểu đồ phân bố đa bộ dữ liệu và ảnh mẫu đối chiếu...")
    plot_multi_dataset_distribution(meta_df, FIGURES_DIR / "multi_dataset_distribution.png")
    plot_dataset_samples_comparison(meta_df, FIGURES_DIR / "dataset_samples_comparison.png")

    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(CLASS_NAMES)}
    train_df = meta_df[meta_df["split"] == "train"].reset_index(drop=True)
    val_df = meta_df[meta_df["split"] == "val"].reset_index(drop=True)
    test_df = meta_df[meta_df["split"] == "test"].reset_index(drop=True)

    train_ds = MetadataImageDataset(train_df, class_to_idx, transform=get_train_transforms(IMG_SIZE))
    val_ds = MetadataImageDataset(val_df, class_to_idx, transform=get_eval_transforms(IMG_SIZE))
    test_ds = MetadataImageDataset(test_df, class_to_idx, transform=get_eval_transforms(IMG_SIZE))

    # Tận dụng tối đa đa luồng CPU/GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        num_threads = max(4, min(16, os.cpu_count() or 8))
        torch.set_num_threads(num_threads)

    num_workers = 0 if os.name == "nt" else 4
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Trọng số cân bằng lớp (Inverse Class Frequency làm mượt căn bậc hai)
    train_counts = np.array([max(1, int((train_df["label"] == c).sum())) for c in CLASS_NAMES], dtype=np.float32)
    inv_weights = np.sqrt(train_counts.max() / train_counts)
    inv_weights = inv_weights / inv_weights.mean()
    class_weights_tensor = torch.tensor(inv_weights, dtype=torch.float32, device=device)

    model = DriverStateCNN(num_classes=len(CLASS_NAMES), pretrained=True, dropout=0.25).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.04)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    total_params = sum(p.numel() for p in model.parameters())
    print("=" * 78)
    print(f"HUẤN LUYỆN MOBILENETV3-SMALL TRÊN ĐA BỘ DỮ LIỆU ({device.type.upper()})")
    print(f"- Kích thước ảnh ROI : {IMG_SIZE}x{IMG_SIZE} RGB | Số lớp: {len(CLASS_NAMES)} {CLASS_NAMES}")
    print(f"- Tổng tham số CNN   : {total_params:,} parameters")
    print(f"- Số mẫu Train/Val/Test: {len(train_ds)} / {len(val_ds)} / {len(test_ds)}")
    print(f"- Trọng số cân bằng  : {[round(float(w), 3) for w in inv_weights]}")
    print("=" * 78)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 1
    no_improve_epochs = 0
    start_train_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]", leave=False)
        for imgs, targets, _ in pbar:
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            running_loss += loss.item() * imgs.size(0)
            running_correct += (preds == targets).sum().item()
            running_total += imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        train_loss = running_loss / max(1, running_total)
        train_acc = running_correct / max(1, running_total)

        # Kiểm định trên tập Validation
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, targets, _ in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                logits = model(imgs)
                loss = criterion(logits, targets)
                preds = logits.argmax(dim=1)
                val_loss_sum += loss.item() * imgs.size(0)
                val_correct += (preds == targets).sum().item()
                val_total += imgs.size(0)

        val_loss = val_loss_sum / max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))

        ep_dt = time.time() - ep_t0
        improved_marker = ""
        if (val_acc > best_val_acc + 1e-4) or (abs(val_acc - best_val_acc) <= 1e-4 and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve_epochs = 0
            improved_marker = " ★ [Lưu Best Model]"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "class_to_idx": class_to_idx,
                    "img_size": IMG_SIZE,
                    "best_epoch": best_epoch,
                    "best_val_acc": float(best_val_acc),
                    "architecture": "MobileNetV3-Small",
                },
                MODEL_SAVE_PATH,
            )
        else:
            no_improve_epochs += 1

        print(
            f"Epoch {epoch:02d}/{epochs:02d} ({ep_dt:4.1f}s) | "
            f"Train Loss: {train_loss:.4f} - Acc: {train_acc*100:6.2f}% | "
            f"Val Loss: {val_loss:.4f} - Acc: {val_acc*100:6.2f}%{improved_marker}"
        )

        if epoch >= 8 and no_improve_epochs >= patience:
            print(f"[*] Kích hoạt Early Stopping tại Epoch {epoch} (không cải thiện sau {patience} epochs).")
            break

    total_train_seconds = time.time() - start_train_time
    plot_loss_accuracy_curves(history, best_epoch, FIGURES_DIR / "loss_accuracy_curves.png")

    # Nạp trọng số tốt nhất để đánh giá độc lập trên 3 tập Test
    print("\n[*] Đang nạp trọng số tốt nhất từ models/cnn_driver_state.pth để đánh giá chéo...")
    best_ckpt = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    all_true, all_pred, all_probs, all_sources = [], [], [], []
    with torch.no_grad():
        for imgs, targets, sources in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.append(probs)
            all_pred.extend(preds.tolist())
            all_true.extend(targets.numpy().tolist())
            all_sources.extend(list(sources))

    y_true = np.array(all_true, dtype=int)
    y_pred = np.array(all_pred, dtype=int)
    y_probs = np.vstack(all_probs)
    sources_arr = np.array(all_sources)

    # Đánh giá trên 3 tập: Bộ 1, Bộ 2 và Tập Tổng hợp
    mask_ds1 = sources_arr == "Dataset1_YawnEye"
    mask_ds2 = sources_arr == "Dataset2_MRLEye"

    ds1_metrics, ds1_report = compute_subset_metrics(y_true[mask_ds1], y_pred[mask_ds1])
    ds2_metrics, ds2_report = compute_subset_metrics(y_true[mask_ds2], y_pred[mask_ds2])
    comb_metrics, comb_report = compute_subset_metrics(y_true, y_pred)

    eval_summary = {
        "Dataset1_YawnEye": ds1_metrics,
        "Dataset2_MRLEye": ds2_metrics,
        "Combined_All": comb_metrics,
    }

    # Xuất các biểu đồ đánh giá còn lại ở độ phân giải 300 DPI
    plot_confusion_matrix_vi(y_true, y_pred, FIGURES_DIR / "confusion_matrix.png")
    plot_cross_dataset_accuracy(eval_summary, FIGURES_DIR / "cross_dataset_accuracy.png")
    roc_auc_dict = plot_roc_curves(y_true, y_probs, FIGURES_DIR / "roc_curve.png")

    # Ghi báo cáo phân loại chi tiết bằng tiếng Việt
    report_content = (
        "====================================================================================\n"
        "BÁO CÁO ĐÁNH GIÁ MÔ HÌNH HỌC SÂU CNN (MOBILENETV3-SMALL TRANSFER LEARNING)\n"
        "Đề tài: Nhận dạng trạng thái buồn ngủ của người điều khiển phương tiện ô tô\n"
        "        sử dụng kết hợp mô hình YOLO26 và mạng học sâu CNN\n"
        "====================================================================================\n\n"
        f"- Kiến trúc mô hình     : MobileNetV3-Small ({total_params:,} tham số)\n"
        f"- Kích thước đầu vào    : {IMG_SIZE}x{IMG_SIZE}x3 (RGB chuẩn hóa ImageNet)\n"
        f"- Epoch tốt nhất        : {best_epoch} (Val Accuracy = {best_val_acc*100:.2f}%)\n"
        f"- Tổng thời gian train  : {total_train_seconds:.1f} giây\n\n"
        "------------------------------------------------------------------------------------\n"
        "1. KẾT QUẢ TRÊN TẬP TEST BỘ DỮ LIỆU 1: DROWSINESS YAWN-EYE DATASET (RGB BAN NGÀY)\n"
        f"   Số mẫu kiểm thử: {ds1_metrics['samples']} | Accuracy: {ds1_metrics['accuracy']*100:.2f}% | "
        f"Macro F1-Score: {ds1_metrics['f1_macro']*100:.2f}%\n"
        "------------------------------------------------------------------------------------\n"
        f"{ds1_report}\n"
        "------------------------------------------------------------------------------------\n"
        "2. KẾT QUẢ TRÊN TẬP TEST BỘ DỮ LIỆU 2: MRL EYE DATASET (HỒNG NGOẠI NIR BAN ĐÊM)\n"
        f"   Số mẫu kiểm thử: {ds2_metrics['samples']} | Accuracy: {ds2_metrics['accuracy']*100:.2f}% | "
        f"Macro F1-Score: {ds2_metrics['f1_macro']*100:.2f}%\n"
        "------------------------------------------------------------------------------------\n"
        f"{ds2_report}\n"
        "------------------------------------------------------------------------------------\n"
        "3. KẾT QUẢ TRÊN TẬP TEST TỔNG HỢP ĐA BỘ DỮ LIỆU (NGÀY + ĐÊM)\n"
        f"   Số mẫu kiểm thử: {comb_metrics['samples']} | Accuracy: {comb_metrics['accuracy']*100:.2f}% | "
        f"Macro F1-Score: {comb_metrics['f1_macro']*100:.2f}% | Micro ROC-AUC: {roc_auc_dict['micro_avg']:.4f}\n"
        "------------------------------------------------------------------------------------\n"
        f"{comb_report}\n"
    )

    rep_fig_path = FIGURES_DIR / "classification_report.txt"
    rep_fig_path.write_text(report_content, encoding="utf-8")
    shutil.copyfile(rep_fig_path, PROJECT_ROOT / "reports" / "classification_report.txt")

    # Cập nhật metadata đầy đủ vào checkpoint và file JSON cho Dashboard
    best_ckpt["eval_summary"] = eval_summary
    best_ckpt["roc_auc"] = roc_auc_dict
    best_ckpt["total_params"] = int(total_params)
    torch.save(best_ckpt, MODEL_SAVE_PATH)

    metrics_payload = {
        "architecture": "MobileNetV3-Small (Transfer Learning)",
        "yolo_backbone": "yolo26n.pt (NMS-Free End-to-End)",
        "img_size": IMG_SIZE,
        "class_names": CLASS_NAMES,
        "total_params": int(total_params),
        "epochs_trained": len(history["train_loss"]),
        "best_epoch": int(best_epoch),
        "best_val_acc": round(float(best_val_acc * 100), 2),
        "train_seconds": round(float(total_train_seconds), 1),
        "dataset_splits": {
            "train": int(len(train_ds)),
            "val": int(len(val_ds)),
            "test": int(len(test_ds)),
            "total": int(len(meta_df)),
        },
        "eval_summary": {
            k: {mk: round(mv * 100, 2) if mk != "samples" else mv for mk, mv in v.items()}
            for k, v in eval_summary.items()
        },
        "roc_auc": {k: round(v, 4) for k, v in roc_auc_dict.items()},
    }
    METRICS_JSON_PATH.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("HOÀN TẤT HUẤN LUYỆN & XUẤT TRỌN BỘ BIỂU ĐỒ ĐÁNH GIÁ (300 DPI)")
    print("=" * 78)
    print(f"- Trọng số mô hình      : {MODEL_SAVE_PATH}")
    print(f"- Accuracy Bộ 1 (RGB)   : {ds1_metrics['accuracy']*100:.2f}% (F1: {ds1_metrics['f1_macro']*100:.2f}%)")
    print(f"- Accuracy Bộ 2 (NIR)   : {ds2_metrics['accuracy']*100:.2f}% (F1: {ds2_metrics['f1_macro']*100:.2f}%)")
    print(f"- Accuracy Tập Tổng hợp : {comb_metrics['accuracy']*100:.2f}% (F1: {comb_metrics['f1_macro']*100:.2f}%)")
    print(f"- Thư mục biểu đồ       : {FIGURES_DIR}")
    return metrics_payload


if __name__ == "__main__":
    train_and_evaluate()
