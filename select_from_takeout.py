"""
Select best images from Google Takeout export of Reef's photos.
Uses JSON sidecar metadata for accurate dates, fills missing age brackets.
"""

import os
import sys
import re
import json
import shutil
import hashlib
from datetime import datetime, timedelta
from collections import defaultdict
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)

# ── Configuration ──────────────────────────────────────────────────────────────
TAKEOUT_DIR = r"C:\Users\erezg\OneDrive - Pen-Link Ltd\Desktop\אישי\ריף\בר מצווה\download\extracted\Takeout\Google Photos\reef"
OUTPUT_DIR = r"C:\Codes\Reef images for bar mitza\sorted"
ONEDRIVE_DIR = r"C:\Users\erezg\OneDrive - Pen-Link Ltd\Desktop\אישי\ריף\בר מצווה"
MANIFEST_FILE = r"C:\Codes\Reef images for bar mitza\manifest.json"
REEF_BIRTHDAY = datetime(2013, 7, 16)
MAX_PER_AGE = 75
TARGET_PER_AGE = 65
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".heic", ".webp"}
MIN_FILE_SIZE = 80 * 1024

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

BRACKET_FOLDER = {
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


def get_json_date(image_path):
    """Get date from Google Takeout JSON sidecar."""
    # Google Takeout creates .json files with same name + .json
    # e.g. IMG_1234.jpg -> IMG_1234.jpg.json
    json_path = image_path + ".json"
    if not os.path.exists(json_path):
        # Sometimes it's filename.json without the image extension
        base = os.path.splitext(image_path)[0]
        json_path = base + ".json"
    if not os.path.exists(json_path):
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # Google Takeout format: photoTakenTime.timestamp (epoch seconds)
        taken_time = meta.get("photoTakenTime", {}).get("timestamp")
        if taken_time:
            return datetime.fromtimestamp(int(taken_time))
        # Also try creationTime
        creation_time = meta.get("creationTime", {}).get("timestamp")
        if creation_time:
            return datetime.fromtimestamp(int(creation_time))
    except Exception:
        pass
    return None


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


def age_days_to_bracket(age_days):
    if age_days is None or age_days < 0:
        return None
    for label, start, end in AGE_BRACKETS:
        if start <= age_days < end:
            return label
    return None


def quality_score(width, height, file_size):
    resolution = width * height
    res_score = min(resolution / (3000 * 2000), 1.0)
    size_score = min(file_size / (3 * 1024 * 1024), 1.0)
    return (res_score * 0.6) + (size_score * 0.4)


def burst_key(filename):
    base = os.path.splitext(filename)[0]
    m = re.match(r'(.+?)(\d{2,5})$', base)
    if m:
        return m.group(1), int(m.group(2))
    return base, 0


def select_best(images, max_count):
    images.sort(key=lambda x: x["score"], reverse=True)
    chosen = []
    for img in images:
        prefix, num = burst_key(img["filename"])
        nearby = sum(1 for c in chosen
                     if burst_key(c["filename"])[0] == prefix
                     and abs(burst_key(c["filename"])[1] - num) < 5)
        if nearby >= 3:
            continue
        chosen.append(img)
        if len(chosen) >= max_count:
            break
    return chosen


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
    print("Reef Google Takeout Image Selector")
    print("=" * 60)

    manifest = load_manifest()
    existing_hashes = set(manifest.keys())

    # Also hash existing output images
    for dirpath, _, filenames in os.walk(OUTPUT_DIR):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                try:
                    existing_hashes.add(file_hash(os.path.join(dirpath, fname)))
                except Exception:
                    pass
    print(f"Known images (manifest + output): {len(existing_hashes)}")

    # Current counts per bracket
    existing_counts = {}
    for bracket_label, folder_name in BRACKET_FOLDER.items():
        folder_path = os.path.join(OUTPUT_DIR, folder_name)
        if os.path.exists(folder_path):
            existing_counts[bracket_label] = sum(1 for f in os.listdir(folder_path)
                                                  if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS)
        else:
            existing_counts[bracket_label] = 0

    print("\nCurrent counts:")
    for bracket_label, _, _ in AGE_BRACKETS:
        count = existing_counts[bracket_label]
        display = BRACKET_DISPLAY[bracket_label]
        status = "FULL" if count >= TARGET_PER_AGE else f"need ~{max(0, TARGET_PER_AGE - count)} more"
        print(f"  {display:<25} {count:>4}  ({status})")

    # ── Scan takeout images ────────────────────────────────────────────────────
    print(f"\nScanning takeout: {TAKEOUT_DIR}")
    by_bracket = defaultdict(list)
    skipped_dup = 0
    skipped_quality = 0
    skipped_no_date = 0
    scanned = 0

    files = [f for f in os.listdir(TAKEOUT_DIR)
             if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
    print(f"Image files found: {len(files)}")

    for fname in files:
        fpath = os.path.join(TAKEOUT_DIR, fname)
        scanned += 1
        if scanned % 200 == 0:
            print(f"  Scanned {scanned}/{len(files)}...")

        # Duplicate check
        try:
            fhash = file_hash(fpath)
        except Exception:
            continue
        if fhash in existing_hashes:
            skipped_dup += 1
            continue

        # File size check
        file_size = os.path.getsize(fpath)
        if file_size < MIN_FILE_SIZE:
            skipped_quality += 1
            continue

        # Get dimensions
        try:
            img = Image.open(fpath)
            w, h = img.size
            img.close()
            if w < 600 and h < 600:
                skipped_quality += 1
                continue
        except Exception:
            skipped_quality += 1
            continue

        # Get date: JSON sidecar first, then EXIF
        dt = get_json_date(fpath)
        if not dt:
            dt = get_exif_date(fpath)
        if not dt:
            skipped_no_date += 1
            continue

        age_days = (dt - REEF_BIRTHDAY).days
        bracket = age_days_to_bracket(age_days)
        if not bracket:
            skipped_no_date += 1
            continue

        score = quality_score(w, h, file_size)
        by_bracket[bracket].append({
            "path": fpath,
            "hash": fhash,
            "filename": fname,
            "width": w,
            "height": h,
            "file_size": file_size,
            "score": score,
            "age_days": age_days,
            "date": dt,
        })

    print(f"\nScan complete: {scanned} images")
    print(f"  Duplicates: {skipped_dup}")
    print(f"  Low quality: {skipped_quality}")
    print(f"  No date/out of range: {skipped_no_date}")
    print(f"  Candidates: {sum(len(v) for v in by_bracket.values())}")

    # ── Select and copy ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Selecting and copying best images...")
    print("=" * 60)

    total_copied = 0
    summary = {}

    for bracket_label, _, _ in AGE_BRACKETS:
        display = BRACKET_DISPLAY[bracket_label]
        folder_name = BRACKET_FOLDER[bracket_label]
        existing = existing_counts[bracket_label]
        needed = max(0, MAX_PER_AGE - existing)
        candidates = by_bracket.get(bracket_label, [])

        if needed == 0:
            print(f"\n{display}: full ({existing}), skipping")
            summary[bracket_label] = {"existing": existing, "added": 0, "total": existing}
            continue

        if not candidates:
            summary[bracket_label] = {"existing": existing, "added": 0, "total": existing}
            continue

        selected = select_best(candidates, needed)

        dest_dir = os.path.join(OUTPUT_DIR, folder_name)
        onedrive_dest = os.path.join(ONEDRIVE_DIR, folder_name)
        os.makedirs(dest_dir, exist_ok=True)
        os.makedirs(onedrive_dest, exist_ok=True)

        added = 0
        for img in selected:
            dest = unique_dest_path(dest_dir, img["filename"])
            try:
                shutil.copy2(img["path"], dest)
                # Also copy to OneDrive
                od_dest = unique_dest_path(onedrive_dest, img["filename"])
                shutil.copy2(img["path"], od_dest)
            except OSError as e:
                print(f"  WARN: {img['filename']}: {e}")
                continue

            manifest[img["hash"]] = {
                "source": img["path"],
                "dest": dest,
                "date": img["date"].isoformat(),
                "date_source": "google_takeout",
                "folder": folder_name,
                "quality_score": round(img["score"], 3),
            }
            added += 1
            total_copied += 1

        summary[bracket_label] = {"existing": existing, "added": added, "total": existing + added}
        if added > 0:
            print(f"\n{display}: +{added} (was {existing}, now {existing + added})")

    save_manifest(manifest)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"{'Age Bracket':<25} {'Before':>8} {'Added':>8} {'Total':>8}")
    print("-" * 60)

    total_all = 0
    for bracket_label, _, _ in AGE_BRACKETS:
        display = BRACKET_DISPLAY[bracket_label]
        s = summary.get(bracket_label, {"existing": 0, "added": 0, "total": 0})
        total = s["total"]
        total_all += total
        marker = ""
        if total == 0:
            marker = "  << MISSING"
        elif total < 50:
            marker = "  << LOW"
        print(f"{display:<25} {s['existing']:>8} {s['added']:>8} {total:>8}{marker}")

    print("-" * 60)
    print(f"{'TOTAL':<25} {'':>8} {total_copied:>8} {total_all:>8}")
    print(f"\nCopied {total_copied} images to sorted/ and OneDrive")


if __name__ == "__main__":
    main()
