"""
Module tự động tải, chuẩn hóa kênh màu RGB, lấy mẫu phân tầng (Stratified Sampling)
và hợp nhất 2 bộ dữ liệu thực tế chuẩn quốc tế qua kagglehub:
1. Bộ dữ liệu 1: serenaraju/yawn-eye-dataset-new (Drowsiness Yawn-Eye Dataset - RGB ban ngày, 4 lớp)
2. Bộ dữ liệu 2: prasadvpatil/mrl-dataset (fallback: tauilabdelilah/mrl-eye-dataset - MRL Eye NIR hồng ngoại ban đêm)
"""

import hashlib
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import kagglehub
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MERGED_DIR = DATA_DIR / "merged_dataset"
METADATA_CSV = DATA_DIR / "metadata.csv"
SUMMARY_JSON = DATA_DIR / "dataset_summary.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MRL_MAX_PER_CLASS = 1500
RANDOM_SEED = 42

# Ánh xạ tên thư mục từ nhiều bộ dữ liệu về 4 nhãn chuẩn
FOLDER_LABEL_MAP: Dict[str, str] = {
    # Lớp Open
    "open": "Open",
    "open_eyes": "Open",
    "open-eyes": "Open",
    "openeyes": "Open",
    "awake": "Open",
    "alert": "Open",
    # Lớp Closed
    "closed": "Closed",
    "close": "Closed",
    "close_eyes": "Closed",
    "closed_eyes": "Closed",
    "close-eyes": "Closed",
    "closed-eyes": "Closed",
    "closeeyes": "Closed",
    "closedeyes": "Closed",
    "sleepy": "Closed",
    "drowsy": "Closed",
    # Lớp yawn
    "yawn": "yawn",
    "yawning": "yawn",
    # Lớp no_yawn
    "no_yawn": "no_yawn",
    "no-yawn": "no_yawn",
    "noyawn": "no_yawn",
}


def compute_sha1(filepath: Path) -> str:
    """Tính mã băm SHA1 để loại bỏ ảnh trùng lặp vật lý (ngăn rò rỉ dữ liệu Train/Test)."""
    h = hashlib.sha1()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def infer_label_and_sub_stratum(filepath: Path) -> Tuple[Optional[str], str]:
    """
    Xác định nhãn chuẩn (Open, Closed, yawn, no_yawn) và nhóm phân tầng phụ (sub-stratum)
    từ cấu trúc cây thư mục hoặc quy ước đặt tên chuẩn của bộ dữ liệu MRL Eye.
    """
    parts_lower = [p.lower() for p in filepath.parts[:-1]]
    # Kiểm tra từ thư mục cha gần nhất ngược lên
    detected_label = None
    for folder_name in reversed(parts_lower):
        cleaned = folder_name.strip().replace(" ", "_")
        if cleaned in FOLDER_LABEL_MAP:
            detected_label = FOLDER_LABEL_MAP[cleaned]
            break

    stem = filepath.stem
    stem_parts = stem.split("_")

    # Hỗ trợ thêm quy ước tên file gốc của MRL Eye:
    # s0001_00001_gender_glasses_eyestate_reflections_lighting_sensor
    mrl_stratum = "general"
    if len(stem_parts) >= 7 and stem_parts[0].lower().startswith("s") and stem_parts[4] in {"0", "1"}:
        if detected_label is None:
            detected_label = "Open" if stem_parts[4] == "1" else "Closed"
        glasses = f"glass{stem_parts[3]}"
        lighting = f"light{stem_parts[6]}"
        mrl_stratum = f"{glasses}_{lighting}"
    else:
        mrl_stratum = filepath.parent.name.lower()

    return detected_label, mrl_stratum


def scan_dataset_directory(
    root_dir: Path,
    source_name: str,
    allowed_labels: Optional[set] = None,
) -> List[Dict]:
    """Duyệt cây thư mục os.walk để thu thập danh sách ảnh hợp lệ và khử trùng lặp."""
    records: List[Dict] = []
    seen_hashes = set()

    print(f"[*] Đang quét cây thư mục của {source_name} tại: {root_dir}")
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue
            fpath = Path(dirpath) / fname
            label, sub_stratum = infer_label_and_sub_stratum(fpath)
            if label is None:
                continue
            if allowed_labels is not None and label not in allowed_labels:
                continue

            try:
                fhash = compute_sha1(fpath)
            except OSError:
                continue

            if fhash in seen_hashes:
                continue
            seen_hashes.add(fhash)

            records.append({
                "original_path": str(fpath),
                "label": label,
                "source_dataset": source_name,
                "sub_stratum": sub_stratum,
            })

    counts = defaultdict(int)
    for r in records:
        counts[r["label"]] += 1
    print(f"    -> Thu thập được {len(records)} ảnh duy nhất từ {source_name}: {dict(counts)}")
    return records


def stratified_sample_mrl(records: List[Dict], max_per_class: int = MRL_MAX_PER_CLASS, seed: int = RANDOM_SEED) -> List[Dict]:
    """
    Lấy mẫu phân tầng thông minh (Stratified Sampling) cho MRL Eye Dataset:
    Đảm bảo tối đa `max_per_class` mỗi lớp (Open, Closed) đồng thời giữ tỷ lệ cân bằng
    giữa các điều kiện đeo kính / ánh sáng hồng ngoại ban đêm.
    """
    rng = random.Random(seed)
    sampled_records: List[Dict] = []

    by_label: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)

    for label, items in sorted(by_label.items()):
        if len(items) <= max_per_class:
            sampled_records.extend(items)
            continue

        # Nhóm theo sub_stratum (ví dụ: đeo kính / không kính / điều kiện sáng NIR)
        strata_groups: Dict[str, List[Dict]] = defaultdict(list)
        for item in items:
            strata_groups[item["sub_stratum"]].append(item)

        total = len(items)
        selected_for_label: List[Dict] = []
        leftovers: List[Dict] = []

        for s_key, s_items in sorted(strata_groups.items()):
            rng.shuffle(s_items)
            quota = int(round(max_per_class * (len(s_items) / total)))
            quota = min(quota, len(s_items))
            selected_for_label.extend(s_items[:quota])
            leftovers.extend(s_items[quota:])

        rng.shuffle(leftovers)
        if len(selected_for_label) < max_per_class:
            needed = max_per_class - len(selected_for_label)
            selected_for_label.extend(leftovers[:needed])
        elif len(selected_for_label) > max_per_class:
            rng.shuffle(selected_for_label)
            selected_for_label = selected_for_label[:max_per_class]

        sampled_records.extend(selected_for_label)

    counts = defaultdict(int)
    for r in sampled_records:
        counts[r["label"]] += 1
    print(f"    -> Sau Stratified Sampling ({max_per_class}/lớp): {len(sampled_records)} ảnh {dict(counts)}")
    return sampled_records


def download_kaggle_datasets() -> Tuple[Path, Path, str]:
    """Tải tự động 2 bộ dữ liệu công khai qua kagglehub kèm cơ chế fallback tự động."""
    print("=" * 75)
    print("BƯỚC 1: TẢI TỰ ĐỘNG 2 BỘ DỮ LIỆU CHUẨN QUỐC TẾ QUA KAGGLEHUB")
    print("=" * 75)

    ds1_slug = "serenaraju/yawn-eye-dataset-new"
    print(f"\n[1/2] Đang tải Bộ dữ liệu 1: {ds1_slug} (RGB Ban ngày - 4 lớp)...")
    ds1_path = Path(kagglehub.dataset_download(ds1_slug))
    print(f"      Hoàn tất tải Bộ 1 tại: {ds1_path}")

    ds2_primary = "prasadvpatil/mrl-dataset"
    ds2_fallback = "tauilabdelilah/mrl-eye-dataset"
    ds2_used_slug = ds2_primary

    print(f"\n[2/2] Đang tải Bộ dữ liệu 2: {ds2_primary} (MRL Eye Hồng ngoại NIR Ban đêm)...")
    try:
        ds2_path = Path(kagglehub.dataset_download(ds2_primary))
        # Kiểm tra xem có ảnh hợp lệ bên trong không
        test_scan = scan_dataset_directory(ds2_path, "Dataset2_MRLEye_Check", allowed_labels={"Open", "Closed"})
        if len(test_scan) < 100:
            raise RuntimeError(f"Số lượng ảnh hợp lệ từ {ds2_primary} quá ít ({len(test_scan)}), chuyển sang fallback.")
        print(f"      Hoàn tất tải Bộ 2 ({ds2_primary}) tại: {ds2_path}")
    except Exception as exc:
        print(f"      [Cảnh báo] Không thể dùng {ds2_primary} ({exc}). Đang tự động chuyển sang fallback: {ds2_fallback}...")
        ds2_used_slug = ds2_fallback
        ds2_path = Path(kagglehub.dataset_download(ds2_fallback))
        print(f"      Hoàn tất tải Bộ 2 fallback ({ds2_fallback}) tại: {ds2_path}")

    return ds1_path, ds2_path, ds2_used_slug


def prepare_merged_dataset(force_rebuild: bool = False) -> pd.DataFrame:
    """
    Thực thi toàn bộ quy trình tải, chuẩn hóa RGB, lấy mẫu cân bằng và chia tập
    Train (80%) / Val (10%) / Test (10%) vào data/merged_dataset/.
    """
    if not force_rebuild and MERGED_DIR.exists() and METADATA_CSV.exists():
        df_existing = pd.read_csv(METADATA_CSV)
        if len(df_existing) > 1000 and set(df_existing["label"].unique()) == {"Closed", "Open", "no_yawn", "yawn"}:
            print(f"[*] Đã phát hiện tập dữ liệu hợp nhất sẵn sàng tại {MERGED_DIR} ({len(df_existing)} ảnh).")
            return df_existing

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)

    ds1_path, ds2_path, ds2_slug = download_kaggle_datasets()

    print("\n" + "=" * 75)
    print("BƯỚC 2: ÁNH XẠ NHÃN & CÂN BẰNG DỮ LIỆU PHÂN TẦNG (STRATIFIED SAMPLING)")
    print("=" * 75)
    ds1_records = scan_dataset_directory(
        ds1_path,
        source_name="Dataset1_YawnEye",
        allowed_labels={"Open", "Closed", "yawn", "no_yawn"},
    )
    ds2_raw_records = scan_dataset_directory(
        ds2_path,
        source_name="Dataset2_MRLEye",
        allowed_labels={"Open", "Closed"},
    )
    ds2_records = stratified_sample_mrl(ds2_raw_records, max_per_class=MRL_MAX_PER_CLASS, seed=RANDOM_SEED)

    all_records = ds1_records + ds2_records
    if len(all_records) == 0:
        raise RuntimeError("Không tìm thấy ảnh hợp lệ từ 2 bộ dữ liệu!")

    df_raw = pd.DataFrame(all_records)
    df_raw["stratify_key"] = df_raw["source_dataset"] + "__" + df_raw["label"]

    print("\n" + "=" * 75)
    print("BƯỚC 3: CHIA TẬP TRAIN (80%) / VAL (10%) / TEST (10%) & CHUẨN HÓA KÊNH MÀU RGB")
    print("=" * 75)

    train_df, temp_df = train_test_split(
        df_raw,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=df_raw["stratify_key"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_df["stratify_key"],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Tạo cấu trúc thư mục ImageFolder chuẩn
    for split_name in ["train", "val", "test"]:
        for cls_name in ["Closed", "Open", "no_yawn", "yawn"]:
            (MERGED_DIR / split_name / cls_name).mkdir(parents=True, exist_ok=True)

    metadata_rows: List[Dict] = []
    skipped = 0

    for idx, row in tqdm(combined_df.iterrows(), total=len(combined_df), desc="Chuẩn hóa RGB & Lưu ảnh"):
        src_path = Path(row["original_path"])
        split_name = row["split"]
        label = row["label"]
        source_ds = row["source_dataset"]
        prefix = "ds1_rgb" if source_ds == "Dataset1_YawnEye" else "ds2_nir"
        out_filename = f"{prefix}_{label.lower()}_{idx:05d}.jpg"
        out_path = MERGED_DIR / split_name / label / out_filename

        try:
            with Image.open(src_path) as img:
                rgb_img = img.convert("RGB")
                w, h = rgb_img.size
                if w < 12 or h < 12:
                    skipped += 1
                    continue
                if max(w, h) > 384:
                    rgb_img.thumbnail((384, 384), Image.Resampling.LANCZOS)
                    w, h = rgb_img.size
                rgb_img.save(out_path, format="JPEG", quality=94)

            metadata_rows.append({
                "filepath": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "filename": out_filename,
                "split": split_name,
                "label": label,
                "source_dataset": source_ds,
                "sub_stratum": row["sub_stratum"],
                "width": w,
                "height": h,
                "original_path": row["original_path"],
            })
        except Exception:
            skipped += 1
            continue

    meta_df = pd.DataFrame(metadata_rows)
    meta_df.to_csv(METADATA_CSV, index=False, encoding="utf-8-sig")

    # Lưu thống kê tổng hợp phục vụ báo cáo & giao diện Dashboard
    pivot_counts = (
        meta_df.groupby(["source_dataset", "label"])
        .size()
        .unstack(fill_value=0)
        .to_dict(orient="index")
    )
    split_counts = meta_df["split"].value_counts().to_dict()
    class_counts = meta_df["label"].value_counts().to_dict()

    summary = {
        "total_images": int(len(meta_df)),
        "skipped_corrupted": int(skipped),
        "dataset1_slug": "serenaraju/yawn-eye-dataset-new",
        "dataset2_slug": ds2_slug,
        "mrl_max_per_class": MRL_MAX_PER_CLASS,
        "split_counts": {k: int(v) for k, v in split_counts.items()},
        "class_counts": {k: int(v) for k, v in class_counts.items()},
        "source_class_breakdown": pivot_counts,
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 75)
    print("HOÀN TẤT HỢP NHẤT ĐA BỘ DỮ LIỆU (MULTI-DATASET PIPELINE)")
    print("=" * 75)
    print(f"- Tổng số ảnh hợp nhất : {len(meta_df)} ảnh (Bỏ qua lỗi: {skipped})")
    print(f"- Phân bố theo tập     : {split_counts}")
    print(f"- Phân bố theo lớp     : {class_counts}")
    print(f"- Đóng góp từng nguồn  : {pivot_counts}")
    print(f"- File metadata        : {METADATA_CSV}")
    print(f"- Thư mục dữ liệu gộp  : {MERGED_DIR}")
    return meta_df


if __name__ == "__main__":
    force = "--force" in sys.argv
    prepare_merged_dataset(force_rebuild=force)
