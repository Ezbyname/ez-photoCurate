"""
Add new Reef images from Desktop/reef folder to scan_db.
Scans metadata, detects faces, grades quality, skips duplicates.
Then re-runs fill_best_75 selection.
"""
import os
import sys
import json
import time
import hashlib
import numpy as np
from datetime import datetime, date
from collections import Counter
from PIL import Image as PILImage, ExifTags

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SCAN_DB_PATH = os.path.join(PROJECT_DIR, "scan_db.json")
SOURCE_DIR = r"C:\Users\erezg\OneDrive - Pen-Link Ltd\Desktop\reef"
BIRTHDAY = date(2013, 7, 16)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".bmp", ".tiff", ".tif", ".webp"}

# Category mapping from folder names to category IDs
FOLDER_TO_CAT = {
    "8.9- 9.9": "00_birth",
    "Birth (0-2 weeks)": "00_birth",
    "28.9": "01_month1",
    "ברית ארז ויהלום 30.8.13": "02_months2-3",
}

# Category brackets by age_days
CATEGORY_BRACKETS = [
    (14, "00_birth"),
    (45, "01_month1"),
    (105, "02_months2-3"),
    (210, "03_months4-6"),
    (300, "04_months7-9"),
    (395, "05_months10-12"),
    (760, "06_age1-2"),
    (1125, "07_age2-3"),
    (1856, "08_age3-5"),
    (2921, "09_age5-8"),
    (3652, "10_age8-10"),
    (4748, "11_age10-13"),
    (99999, "12_bar_mitzva"),
]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_scan_db(db):
    tmp_path = SCAN_DB_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, cls=NumpyEncoder)
    os.replace(tmp_path, SCAN_DB_PATH)


def file_hash(fpath):
    h = hashlib.md5()
    with open(fpath, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_date_from_exif(fpath):
    """Try to extract date from EXIF data."""
    try:
        img = PILImage.open(fpath)
        exif = img._getexif()
        if exif:
            for tag_id, val in exif.items():
                tag = ExifTags.TAGS.get(tag_id, "")
                if tag in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
                    if isinstance(val, str) and len(val) >= 10:
                        dt = val.replace(":", "-", 2)[:10]
                        return dt
        img.close()
    except:
        pass
    return None


def get_date_from_filename(fname):
    """Try to extract date from filename patterns like IMG_20160101 or 2016-01-01."""
    import re
    # Pattern: 20YYMMDD
    m = re.search(r"(20[12]\d)(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])", fname)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    # Pattern: 2016-01-01
    m = re.search(r"(20[12]\d)-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])", fname)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def category_from_age_days(age_days):
    if age_days is None or age_days < 0:
        return None
    for max_days, cat_id in CATEGORY_BRACKETS:
        if age_days <= max_days:
            return cat_id
    return "11_age10-13"


def category_from_folder(folder_name):
    """Assign category based on folder name."""
    if folder_name in FOLDER_TO_CAT:
        return FOLDER_TO_CAT[folder_name]

    fl = folder_name.lower()
    if "birth" in fl or "0-2 week" in fl:
        return "00_birth"
    if "month 1" in fl or "month1" in fl:
        return "01_month1"
    if "months 2-3" in fl or "month2" in fl or "month3" in fl:
        return "02_months2-3"
    if "months 4-6" in fl or "month4" in fl:
        return "03_months4-6"
    if "months 7-9" in fl or "month7" in fl:
        return "04_months7-9"
    if "months 10-12" in fl or "month10" in fl:
        return "05_months10-12"
    if "age 1" in fl:
        return "06_age1-2"
    if "age 7" in fl or "7 year" in fl:
        return "09_age5-8"
    if "age 12" in fl or "12 year" in fl:
        return "11_age10-13"
    return None


def make_thumbnail(fpath, size=160):
    """Create base64 thumbnail."""
    import base64
    from io import BytesIO
    try:
        img = PILImage.open(fpath)
        img.thumbnail((size, size), PILImage.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=60)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except:
        return None


def scan_new_images():
    print("=" * 70)
    print("  ADD REEF IMAGES FROM DESKTOP")
    print("=" * 70)

    # Load existing DB
    print("\n[1/5] Loading scan database...")
    db = json.load(open(SCAN_DB_PATH, "r", encoding="utf-8"))
    images = db.get("images", [])
    config = db.get("config", {})

    # Build existing hash set for dedup
    existing_hashes = set(img.get("hash", "") for img in images)
    existing_paths = set(img["path"].replace("\\", "/").lower() for img in images)
    print(f"  {len(images)} existing images, {len(existing_hashes)} unique hashes")

    # Find new image files
    print("\n[2/5] Scanning for new images...")
    new_files = []
    for root, dirs, files in os.walk(SOURCE_DIR):
        # Skip the E-z Photo Collection subfolder (likely already processed)
        if "E-z Photo Collection" in root:
            continue
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext not in IMAGE_EXTS:
                continue
            fp = os.path.join(root, f)
            fp_norm = fp.replace("\\", "/").lower()
            if fp_norm in existing_paths:
                continue
            new_files.append(fp)

    print(f"  Found {len(new_files)} new images to process")
    if not new_files:
        print("  Nothing to add!")
        return

    # Process each new image
    print("\n[3/5] Extracting metadata and checking duplicates...")
    added = 0
    skipped_dup = 0
    skipped_err = 0
    t0 = time.time()

    for idx, fpath in enumerate(new_files):
        try:
            # Hash check
            fhash = file_hash(fpath)
            if fhash in existing_hashes:
                skipped_dup += 1
                continue

            # Get image dimensions
            pil_img = PILImage.open(fpath)
            w, h = pil_img.size
            pil_img.close()

            if w < 100 or h < 100:
                skipped_err += 1
                continue

            fname = os.path.basename(fpath)
            size_kb = os.path.getsize(fpath) / 1024

            # Get date
            img_date = get_date_from_exif(fpath)
            if not img_date:
                img_date = get_date_from_filename(fname)

            # Calculate age
            age_days = None
            if img_date:
                try:
                    dt = datetime.strptime(img_date[:10], "%Y-%m-%d").date()
                    age_days = (dt - BIRTHDAY).days
                except:
                    pass

            # Determine category
            rel_path = os.path.relpath(fpath, SOURCE_DIR)
            folder_name = rel_path.split(os.sep)[0]

            category = None
            # First try folder name
            category = category_from_folder(folder_name)
            # If no match, try age_days
            if not category and age_days is not None:
                category = category_from_age_days(age_days)

            # Screenshot check
            is_screenshot = (
                "screenshot" in fname.lower() or
                (w == 1080 and h == 1920) or
                (w == 1920 and h == 1080)
            )

            # Thumbnail
            thumb = make_thumbnail(fpath)

            # Normalize path
            norm_path = fpath.replace("\\", "/")

            entry = {
                "hash": fhash,
                "path": norm_path,
                "filename": fname,
                "source_label": "Desktop reef folder",
                "device": "other",
                "date": img_date,
                "age_days": age_days,
                "category": category,
                "face_count": None,
                "faces_found": [],
                "has_target_face": False,
                "width": w,
                "height": h,
                "size_kb": round(size_kb, 1),
                "is_screenshot": is_screenshot,
                "thumb": thumb,
                "status": "qualified",
                "reject_reason": None,
                "_face_checked": False,
            }

            images.append(entry)
            existing_hashes.add(fhash)
            added += 1

        except Exception as e:
            skipped_err += 1

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {idx+1}/{len(new_files)} processed, {added} added, {skipped_dup} dups, {skipped_err} errors")

    db["images"] = images
    save_scan_db(db)
    print(f"  Done! {added} new images added, {skipped_dup} duplicates skipped, {skipped_err} errors")

    # Face detection on new images
    print("\n[4/5] Face detection on new images...")
    sys.path.insert(0, PROJECT_DIR)
    from fill_best_75 import load_face_references, detect_face, compute_photo_grade, FACE_TOLERANCE

    ref_encodings = load_face_references()
    need_face = [img for img in images if not img.get("_face_checked")]
    print(f"  {len(need_face)} images need face checking")

    face_found = 0
    for idx, img in enumerate(need_face):
        fpath = img["path"].replace("/", os.sep)
        if not os.path.exists(fpath):
            img["_face_checked"] = True
            continue

        age_days = img.get("age_days")
        if age_days is not None and age_days <= 365:
            tol = 0.45
        elif age_days is not None and age_days <= 1095:
            tol = 0.50
        else:
            tol = FACE_TOLERANCE

        fc, ff, has_target, best_d = detect_face(fpath, ref_encodings, tol)
        img["face_count"] = fc
        img["faces_found"] = ff
        img["has_target_face"] = has_target
        img["face_distance"] = best_d
        img["_face_checked"] = True

        if has_target:
            face_found += 1

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / (time.time() - t0)
            eta = (len(need_face) - idx - 1) / rate if rate > 0 else 0
            print(f"  {idx+1}/{len(need_face)} checked, {face_found} Reef found ({rate:.1f}/s, ETA {eta/60:.0f}m)")

        if (idx + 1) % 200 == 0:
            save_scan_db(db)

    save_scan_db(db)
    print(f"  Done! {face_found} images with Reef's face out of {len(need_face)} new")

    # Photo grading on new images
    print("\n[5/5] Photo grading on new images...")
    need_grade = [img for img in images if not img.get("photo_grade") and img.get("media_type") != "video"]
    print(f"  {len(need_grade)} images need grading")

    graded = 0
    for idx, img in enumerate(need_grade):
        fpath = img["path"].replace("/", os.sep)
        if not os.path.exists(fpath):
            continue
        w = img.get("width", 0)
        h = img.get("height", 0)
        if w == 0 or h == 0:
            continue

        grade = compute_photo_grade(fpath, w, h)
        if grade:
            img["photo_grade"] = grade
            img["blur_score"] = grade["blur_score"]
            graded += 1

        if (idx + 1) % 200 == 0:
            print(f"  {idx+1}/{len(need_grade)} graded ({graded} successful)")
            save_scan_db(db)

    save_scan_db(db)
    print(f"  Done! {graded} images graded")

    # Summary
    print("\n" + "=" * 70)
    cat_counts = Counter(img.get("category") for img in images if img.get("category"))
    print(f"  SCAN COMPLETE: {len(images)} total images in DB")
    print(f"  New images added: {added}")
    print(f"  Category distribution:")
    for cat_id in sorted(cat_counts.keys()):
        reef_count = sum(1 for img in images if img.get("category") == cat_id and img.get("has_target_face"))
        face_count = sum(1 for img in images if img.get("category") == cat_id and img.get("face_count", 0) > 0)
        print(f"    {cat_id}: {cat_counts[cat_id]} total, {reef_count} Reef, {face_count} with faces")
    print("=" * 70)
    print("\n  Now run: python fill_best_75.py  to re-select best 75 per category")


if __name__ == "__main__":
    scan_new_images()
