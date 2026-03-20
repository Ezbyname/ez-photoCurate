"""
Select best-quality images from D:\ריף disk, ~50-75 per age bracket.
Skips duplicates already in the sorted folder (via manifest hash).
Quality score = resolution * file_size (fast, no sharpness computation).
"""

import os
import sys
import re
import json
import shutil
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image
from collections import defaultdict

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# ── Configuration ──────────────────────────────────────────────────────────────
DISK_DIR = r"D:\ריף"
OUTPUT_DIR = r"C:\Codes\Reef images for bar mitza\sorted"
MANIFEST_FILE = r"C:\Codes\Reef images for bar mitza\manifest.json"
REEF_BIRTHDAY = datetime(2013, 7, 16)
TARGET_PER_AGE = 65  # target ~65, allow 50-75
MAX_PER_AGE = 75
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".heic", ".webp"}
MIN_WIDTH = 600
MIN_HEIGHT = 600
MIN_FILE_SIZE = 80 * 1024  # 80KB

# ── Age brackets ───────────────────────────────────────────────────────────────
AGE_BRACKETS = [
    ("00_birth",         0,    14),
    ("01_month1",       14,    60),
    ("02_months2-3",    60,   120),
    ("03_months4-6",   120,   210),
    ("04_months7-9",   210,   300),
    ("05_months10-12", 300,   365),
    ("06_year1",       365,   730),
    ("07_year2",       730,  1095),
    ("08_year3-4",    1095,  1825),
    ("09_year5-7",    1825,  2920),
    ("10_year8-10",   2920,  3650),
    ("11_year11-12",  3650,  4745),
    ("12_barmitzva",  4745,  5110),
]

BRACKET_DISPLAY = {
    "00_birth":        "Birth (0-2 weeks)",
    "01_month1":       "Month 1",
    "02_months2-3":    "Months 2-3",
    "03_months4-6":    "Months 4-6",
    "04_months7-9":    "Months 7-9",
    "05_months10-12":  "Months 10-12",
    "06_year1":        "Age 1 (1-2 years)",
    "07_year2":        "Age 2 (2-3 years)",
    "08_year3-4":      "Age 3-4",
    "09_year5-7":      "Age 5-7",
    "10_year8-10":     "Age 8-10",
    "11_year11-12":    "Age 11-12",
    "12_barmitzva":    "Bar Mitzva prep (12.5+)",
}

# ── Directory name -> age in days ──────────────────────────────────────────────
DIR_AGE_MAP = {
    "בית חולים":           0,
    "כשריף נולד":          0,
    "ריף מגיע הביתה":      3,
    "ברית":                44,
    "ריף בן חודש":         30,
    "28.9":                74,
    "8.9- 9.9":            55,
    "ריף בן חודשיים":      60,
    "ריף בן 3 חודשים":     90,
    "ריף בן 4 חודשים":    120,
    "ריף בן 5 חודשים":    150,
    "ריף בן חצי שנה":     180,
    "ריף בן 7 חודשים":    210,
    "ריף בן 8 חודשים":    240,
    "ריף בן 9 חודשים":    270,
    "ריף בן 10 חודשים":   300,
    "ריף בן 11 חודשים":   330,
    "ריף בן שנה":         365,
    "שנה וחודש":          395,
    "שנה וחודשיים":       425,
    "שנה ושלוש":          455,
    "שנה וארבע":          485,
    "שנה וחמש":           515,
    "שנה וחצי":           545,
    "שנה ושבע":           575,
    "שנה ושמונה":         605,
    "שנה ותשע":           635,
    "שנה ועשר":           665,
    "שנה ו5":             515,
    "כיפור 2013":          90,
    "בוק":                365,
    "משפחה ארז":          365,
    "ipad":               730,
    "הדפסה":              365,
    "תיקיה חדשה":         60,   # subfolder inside חודשיים
}


def file_hash(filepath):
    h = hashlib.md5(usedforsecurity=False)
    h.update(str(os.path.getsize(filepath)).encode())
    with open(filepath, "rb") as f:
        h.update(f.read(65536))
    return h.hexdigest()


def load_manifest():
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(manifest):
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def get_exif_date(filepath):
    try:
        img = Image.open(filepath)
        exif = img._getexif()
        if exif:
            for tag_id in (36867, 306, 36868):
                val = exif.get(tag_id)
                if val:
                    return datetime.strptime(val, "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return None


def dir_to_age_days(rel_path):
    parts = rel_path.replace("\\", "/").split("/")
    for part in parts:
        for pattern, age_days in DIR_AGE_MAP.items():
            if pattern in part:
                return age_days
    return None


def age_days_to_bracket(age_days):
    if age_days is None or age_days < 0:
        return None
    for label, start, end in AGE_BRACKETS:
        if start <= age_days < end:
            return label
    return None


def quality_score(width, height, file_size):
    """Fast quality score: resolution + file size."""
    resolution = width * height
    res_score = min(resolution / (3000 * 2000), 1.0)
    size_score = min(file_size / (3 * 1024 * 1024), 1.0)
    return (res_score * 0.6) + (size_score * 0.4)


def burst_key(filename):
    """Detect burst sequences like IMG_1234, IMG_1235."""
    base = os.path.splitext(filename)[0]
    m = re.match(r'(.+?)(\d{2,5})$', base)
    if m:
        return m.group(1), int(m.group(2))
    return base, 0


def select_best(images, max_count):
    """Select top images, limiting burst sequences to 3."""
    images.sort(key=lambda x: x["score"], reverse=True)
    chosen = []
    burst_counts = defaultdict(int)

    for img in images:
        prefix, num = burst_key(img["filename"])
        # Count nearby images from same burst
        nearby = sum(1 for c in chosen
                     if burst_key(c["filename"])[0] == prefix
                     and abs(burst_key(c["filename"])[1] - num) < 5)
        if nearby >= 3:
            continue
        chosen.append(img)
        if len(chosen) >= max_count:
            break

    return chosen


def age_bracket_to_folder(bracket):
    label_map = {
        "00_birth":        "2013_birth",
        "01_month1":       "2013_month1",
        "02_months2-3":    "2013_months2-3",
        "03_months4-6":    "2013-2014_months4-6",
        "04_months7-9":    "2014_months7-9",
        "05_months10-12":  "2014_months10-12",
        "06_year1":        "2014-2015_age1",
        "07_year2":        "2015-2016_age2",
        "08_year3-4":      "2016-2018_age3-4",
        "09_year5-7":      "2018-2020_age5-7",
        "10_year8-10":     "2021-2023_age8-10",
        "11_year11-12":    "2024-2025_age11-12",
        "12_barmitzva":    "2026_barmitzva",
    }
    return label_map.get(bracket, bracket)


def unique_dest_path(dest_dir, filename):
    dest = os.path.join(dest_dir, filename)
    if not os.path.exists(dest):
        return dest
    base, ext = os.path.splitext(filename)
    i = 1
    while True:
        dest = os.path.join(dest_dir, f"{base}_{i}{ext}")
        if not os.path.exists(dest):
            return dest
        i += 1


def main():
    print("=" * 60)
    print("Reef Image Selector - Disk D:\\")
    print("=" * 60)

    manifest = load_manifest()
    existing_hashes = set(manifest.keys())
    print(f"Existing in manifest: {len(existing_hashes)}")

    # Also hash images already in output dir
    for dirpath, _, filenames in os.walk(OUTPUT_DIR):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                fpath = os.path.join(dirpath, fname)
                try:
                    existing_hashes.add(file_hash(fpath))
                except Exception:
                    pass
    print(f"Total known (manifest + output): {len(existing_hashes)}")

    # ── Scan disk ──────────────────────────────────────────────────────────────
    print("\nScanning D:\\ disk...")
    images_by_bracket = defaultdict(list)
    skipped_dup = 0
    skipped_quality = 0
    skipped_no_age = 0
    scanned = 0

    for dirpath, dirnames, filenames in os.walk(DISK_DIR):
        rel_dir = os.path.relpath(dirpath, DISK_DIR)

        # Skip video directories
        if any(v in rel_dir for v in ["וידאו", "video"]):
            continue

        imgs = [f for f in filenames if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
        if not imgs:
            continue

        # Pre-compute directory age hint once per directory
        dir_age = dir_to_age_days(rel_dir)

        for fname in imgs:
            fpath = os.path.join(dirpath, fname)
            scanned += 1

            if scanned % 500 == 0:
                print(f"  Scanned {scanned} images...")

            # Quick duplicate check
            try:
                fhash = file_hash(fpath)
            except Exception:
                continue
            if fhash in existing_hashes:
                skipped_dup += 1
                continue

            # Basic quality gate (file size only - fast)
            try:
                file_size = os.path.getsize(fpath)
                if file_size < MIN_FILE_SIZE:
                    skipped_quality += 1
                    continue
            except Exception:
                continue

            # Get dimensions (opens file but doesn't decode pixels)
            try:
                img = Image.open(fpath)
                w, h = img.size
                img.close()
                if w < MIN_WIDTH and h < MIN_HEIGHT:
                    skipped_quality += 1
                    continue
            except Exception:
                skipped_quality += 1
                continue

            # Determine age
            age_days = None
            exif_date = get_exif_date(fpath)
            if exif_date:
                age_days = (exif_date - REEF_BIRTHDAY).days
            if age_days is None:
                age_days = dir_age

            bracket = age_days_to_bracket(age_days)
            if bracket is None:
                skipped_no_age += 1
                continue

            score = quality_score(w, h, file_size)
            images_by_bracket[bracket].append({
                "path": fpath,
                "hash": fhash,
                "width": w,
                "height": h,
                "file_size": file_size,
                "score": score,
                "age_days": age_days,
                "filename": fname,
            })

    print(f"\nScan complete: {scanned} images scanned")
    print(f"  Duplicates skipped: {skipped_dup}")
    print(f"  Low quality skipped: {skipped_quality}")
    print(f"  Unknown age skipped: {skipped_no_age}")
    print(f"  Candidates: {sum(len(v) for v in images_by_bracket.values())}")

    # ── Select best per bracket ────────────────────────────────────────────────
    print("\nSelecting best images per age bracket...")
    total_copied = 0
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Track totals for summary
    summary = {}

    for bracket_label, _, _ in AGE_BRACKETS:
        display = BRACKET_DISPLAY[bracket_label]
        candidates = images_by_bracket.get(bracket_label, [])
        folder_name = age_bracket_to_folder(bracket_label)
        dest_dir = os.path.join(OUTPUT_DIR, folder_name)

        # Count existing images in this folder
        existing_in_folder = 0
        if os.path.exists(dest_dir):
            existing_in_folder = sum(1 for f in os.listdir(dest_dir)
                                     if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS)

        if not candidates:
            summary[bracket_label] = {
                "display": display,
                "folder": folder_name,
                "existing": existing_in_folder,
                "added": 0,
                "total": existing_in_folder,
                "candidates": 0,
            }
            continue

        # How many more do we need?
        needed = max(0, MAX_PER_AGE - existing_in_folder)
        if needed == 0:
            print(f"  {display}: already has {existing_in_folder} images, skipping")
            summary[bracket_label] = {
                "display": display,
                "folder": folder_name,
                "existing": existing_in_folder,
                "added": 0,
                "total": existing_in_folder,
                "candidates": len(candidates),
            }
            continue

        selected = select_best(candidates, needed)
        os.makedirs(dest_dir, exist_ok=True)

        added = 0
        for img in selected:
            dest = unique_dest_path(dest_dir, img["filename"])
            try:
                shutil.copy2(img["path"], dest)
            except OSError as e:
                print(f"    WARN: could not copy {img['filename']}: {e}")
                continue
            manifest[img["hash"]] = {
                "source": img["path"],
                "dest": dest,
                "date": (REEF_BIRTHDAY + timedelta(days=img["age_days"])).isoformat(),
                "date_source": "disk_select",
                "folder": folder_name,
                "quality_score": round(img["score"], 3),
            }
            added += 1
            total_copied += 1

        summary[bracket_label] = {
            "display": display,
            "folder": folder_name,
            "existing": existing_in_folder,
            "added": added,
            "total": existing_in_folder + added,
            "candidates": len(candidates),
        }
        print(f"  {display}: +{added} images (was {existing_in_folder}, now {existing_in_folder + added})")

    save_manifest(manifest)

    # ── Final Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"{'Age Bracket':<25} {'Existing':>8} {'Added':>8} {'Total':>8}")
    print("-" * 60)

    total_all = 0
    missing = []
    low = []

    for bracket_label, _, _ in AGE_BRACKETS:
        s = summary.get(bracket_label, {"display": bracket_label, "existing": 0, "added": 0, "total": 0})
        display = s["display"]
        total = s["total"]
        total_all += total
        marker = ""
        if total == 0:
            missing.append(display)
            marker = "  << MISSING"
        elif total < 50:
            low.append((display, total))
            marker = "  << LOW"
        print(f"{display:<25} {s['existing']:>8} {s['added']:>8} {total:>8}{marker}")

    print("-" * 60)
    print(f"{'TOTAL':<25} {'':>8} {total_copied:>8} {total_all:>8}")

    if missing:
        print(f"\n!! MISSING AGES (0 images): {', '.join(missing)}")
    if low:
        print(f"\n!! LOW COVERAGE (<50 images):")
        for display, count in low:
            print(f"   {display}: only {count} images")

    print(f"\nTotal images copied this run: {total_copied}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
