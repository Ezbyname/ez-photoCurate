"""
Comprehensive scanner: scans all image sources, detects Reef via face recognition,
grades quality, maps to age brackets, and selects top ~75 per bracket for presentation.

Sources:
  1. USB disk D:\ריף
  2. OneDrive bar mitzva (excluding presentation/backup)
  3. Google Takeout

Pipeline:
  1. Collect all image paths (dedup by filename+size)
  2. Fast face detection (OpenCV Haar) - skip no-face images
  3. Face recognition matching against Reef references
  4. Quality grading (technical + face + composition)
  5. Age bracket assignment (EXIF date or folder name)
  6. Select top ~75 per bracket, copy to presentation
"""

import os
import sys
import json
import re
import shutil
import hashlib
import numpy as np
import cv2
from PIL import Image, ImageStat, ExifTags
from datetime import datetime, date
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

# ── Config ─────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".heic", ".webp"}
REEF_BIRTHDAY = date(2013, 7, 16)
TARGET_PER_BRACKET = 75
FACE_MATCH_THRESHOLD = 0.55  # distance threshold (lower = stricter)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_DIR, "scan_all_db.json")
REF_FACES_DIR = os.path.join(PROJECT_DIR, "ref_faces", "reef")

ONEDRIVE_BASE = r"C:\Users\erezg\OneDrive - Pen-Link Ltd\Desktop\אישי\ריף\בר מצווה"
PRESENTATION_DIR = os.path.join(ONEDRIVE_BASE, "the presentation")

SOURCES = [
    ("USB", r"D:\ריף"),
    ("OneDrive", ONEDRIVE_BASE),
]

# Folders to skip when scanning OneDrive (avoid scanning presentation itself)
SKIP_FOLDERS = {"the presentation", "backup pool", "removed_from_presentation"}

# Age brackets: (label, min_age_days, max_age_days)
AGE_BRACKETS = [
    ("2013_birth",        0,    30),
    ("2013_month1-3",    31,    120),
    ("2013_month3-6",   121,    210),
    ("2013-2014_month6-12", 211, 395),
    ("2014_age1",       396,    760),
    ("2014-2015_age1-2", 761,  1126),
    ("2015-2016_age2-3", 1127, 1491),
    ("2016-2018_age3-4", 1492, 2222),
    ("2018-2020_age5-7", 2223, 3000),
    ("2020-2021_age7-8", 3001, 3287),
    ("2021-2023_age8-10", 3288, 3835),
    ("2023-2024_age10-11", 3836, 4200),
    ("2024-2025_age11-12", 4201, 5000),
]


# ── Database ───────────────────────────────────────────────────────────────

def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"images": {}, "stats": {}}

def save_db(db):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=1)

def file_key(filepath):
    try:
        sz = os.path.getsize(filepath)
        name = os.path.basename(filepath)
        return f"{name}|{sz}"
    except:
        return filepath


# ── Image collection ───────────────────────────────────────────────────────

def collect_images():
    """Walk all sources and collect unique image paths."""
    seen_keys = {}
    all_images = []

    for source_name, source_path in SOURCES:
        if not os.path.isdir(source_path):
            print(f"  SKIP: {source_name} not found")
            continue

        count = 0
        for dirpath, dirnames, filenames in os.walk(source_path):
            # Skip certain folders in OneDrive
            if source_name == "OneDrive":
                rel = os.path.relpath(dirpath, source_path)
                parts = rel.split(os.sep)
                if any(skip in parts for skip in SKIP_FOLDERS):
                    continue

            for fname in filenames:
                if os.path.splitext(fname)[1].lower() not in IMAGE_EXTS:
                    continue
                fpath = os.path.join(dirpath, fname)
                key = file_key(fpath)
                if key not in seen_keys:
                    seen_keys[key] = True
                    all_images.append({
                        "path": fpath,
                        "source": source_name,
                        "key": key,
                        "folder": os.path.relpath(dirpath, source_path),
                    })
                    count += 1

        print(f"  {source_name}: {count} unique images")

    print(f"  Total unique: {len(all_images)}")
    return all_images


# ── EXIF date extraction ──────────────────────────────────────────────────

def get_image_date(filepath, folder_name=""):
    """Try to extract date from EXIF, then filename, then folder name."""
    # Try EXIF
    try:
        img = Image.open(filepath)
        exif = img._getexif()
        if exif:
            for tag_id, value in exif.items():
                tag = ExifTags.TAGS.get(tag_id, "")
                if tag in ("DateTimeOriginal", "DateTime", "DateTimeDigitized"):
                    if isinstance(value, str) and len(value) >= 10:
                        dt = datetime.strptime(value[:19], "%Y:%m:%d %H:%M:%S")
                        return dt.date()
        img.close()
    except:
        pass

    # Try filename patterns: IMG_20200313, 20170828_133753, etc.
    fname = os.path.basename(filepath)
    patterns = [
        r'(\d{4})(\d{2})(\d{2})',  # 20200313
        r'(\d{4})-(\d{2})-(\d{2})',  # 2020-03-13
    ]
    for pat in patterns:
        m = re.search(pat, fname)
        if m:
            try:
                y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if 2013 <= y <= 2026 and 1 <= mo <= 12 and 1 <= d <= 31:
                    return date(y, mo, d)
            except:
                pass

    # Try folder name for date patterns
    for pat in patterns:
        m = re.search(pat, folder_name)
        if m:
            try:
                y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if 2013 <= y <= 2026 and 1 <= mo <= 12 and 1 <= d <= 31:
                    return date(y, mo, d)
            except:
                pass

    return None


def get_age_bracket(img_date, folder_name=""):
    """Determine age bracket from date or folder name."""
    if img_date:
        age_days = (img_date - REEF_BIRTHDAY).days
        if age_days < 0:
            age_days = 0  # Pre-birth, assign to birth
        for label, min_d, max_d in AGE_BRACKETS:
            if min_d <= age_days <= max_d:
                return label
        # If beyond our brackets, use last one
        if age_days > AGE_BRACKETS[-1][2]:
            return AGE_BRACKETS[-1][0]
        return AGE_BRACKETS[0][0]

    # Fallback: try to match folder name to a bracket
    folder_lower = folder_name.lower()

    # Try to match age bracket from folder names like "2018-2020_age5-7"
    for label, _, _ in AGE_BRACKETS:
        if label.lower() in folder_lower:
            return label

    # Hebrew folder name mapping (from USB disk)
    hebrew_mappings = {
        "נולד": "2013_birth",
        "מגיע הביתה": "2013_birth",
        "ברית": "2013_birth",
        "בית חולים": "2013_birth",
        "בן חודש": "2013_month1-3",
        "בן 3 חודשים": "2013_month1-3",
        "בן 4 חודשים": "2013_month3-6",
        "בן 5 חודשים": "2013_month3-6",
        "בן חצי שנה": "2013_month3-6",
        "בן 7 חודשים": "2013-2014_month6-12",
        "בן 8 חודשים": "2013-2014_month6-12",
        "בן 9 חודשים": "2013-2014_month6-12",
        "בן 10 חודשים": "2013-2014_month6-12",
        "בן 11 חודשים": "2013-2014_month6-12",
        "בן שנה": "2014_age1",
        "שנה וחודש": "2014_age1",
        "שנה וחודשיים": "2014_age1",
        "שנה וארבע": "2014_age1",
        "שנה וחמש": "2014_age1",
        "שנה ו5": "2014-2015_age1-2",
        "8.9- 9.9": "2014_age1",
        "28.9": "2013_month1-3",
        "כיפור 2013": "2013_month1-3",
    }
    for hebrew, bracket in hebrew_mappings.items():
        if hebrew in folder_name:
            return bracket

    # Year-based fallback from folder
    year_match = re.search(r'(201[3-9]|202[0-6])', folder_name)
    if year_match:
        y = int(year_match.group(1))
        approx_date = date(y, 6, 1)
        age_days = (approx_date - REEF_BIRTHDAY).days
        if age_days < 0:
            age_days = 0
        for label, min_d, max_d in AGE_BRACKETS:
            if min_d <= age_days <= max_d:
                return label

    return "unknown"


# ── Face Detection (fast, OpenCV) ─────────────────────────────────────────

_face_cascade = None
def get_cascade():
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt2.xml"))
    return _face_cascade

def has_face_fast(filepath):
    """Quick face detection using OpenCV Haar cascade. Returns True/False."""
    try:
        pil_img = Image.open(filepath).convert("RGB")
        w, h = pil_img.size
        scale = min(600 / max(w, h), 1.0)
        if scale < 1:
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        arr = np.array(pil_img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        cascade = get_cascade()
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
        pil_img.close()
        return len(faces) > 0
    except:
        return False


# ── Face Recognition ──────────────────────────────────────────────────────

def load_reef_encodings():
    """Load all usable Reef face encodings from ref_faces."""
    import face_recognition
    encodings = []

    if not os.path.isdir(REF_FACES_DIR):
        print("  WARNING: ref_faces/reef/ not found")
        return encodings

    for fname in sorted(os.listdir(REF_FACES_DIR)):
        if os.path.splitext(fname)[1].lower() not in IMAGE_EXTS:
            continue
        fpath = os.path.join(REF_FACES_DIR, fname)
        try:
            img = Image.open(fpath).convert("RGB")
            w, h = img.size
            scale = min(600 / max(w, h), 1.0)
            if scale < 1:
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            arr = np.array(img)
            locs = face_recognition.face_locations(arr, number_of_times_to_upsample=2)
            if locs:
                encs = face_recognition.face_encodings(arr, locs)
                if encs:
                    encodings.append(encs[0])
                    print(f"    Loaded: {fname}")
                else:
                    print(f"    No encoding: {fname}")
            else:
                print(f"    No face: {fname}")
        except Exception as e:
            print(f"    Error {fname}: {e}")

    return encodings


def match_reef(filepath, reef_encodings):
    """Check if image contains Reef. Returns (matched, best_distance)."""
    import face_recognition

    if not reef_encodings:
        return False, 1.0

    try:
        img = Image.open(filepath).convert("RGB")
        w, h = img.size
        scale = min(800 / max(w, h), 1.0)
        if scale < 1:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        arr = np.array(img)

        face_locs = face_recognition.face_locations(arr, number_of_times_to_upsample=1)
        if not face_locs:
            return False, 1.0

        face_encs = face_recognition.face_encodings(arr, face_locs)
        if not face_encs:
            return False, 1.0

        best_dist = 1.0
        for ref_enc in reef_encodings:
            distances = face_recognition.face_distance(face_encs, ref_enc)
            min_d = float(np.min(distances))
            best_dist = min(best_dist, min_d)

        return best_dist <= FACE_MATCH_THRESHOLD, best_dist

    except:
        return False, 1.0


# ── Quality Grading ───────────────────────────────────────────────────────

def grade_image(filepath):
    """Compute quality score (0-100). Combines sharpness, exposure, resolution, face quality."""
    scores = {}
    try:
        pil_img = Image.open(filepath).convert("RGB")
        w, h = pil_img.size
        file_size_kb = os.path.getsize(filepath) / 1024

        # Resolution
        mp = (w * h) / 1_000_000
        scores["resolution"] = min(100, max(0, 20 + mp * 10))

        # Sharpness
        gray = pil_img.convert("L")
        scale = min(800 / max(w, h), 1.0)
        if scale < 1:
            gray = gray.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        gray_arr = np.array(gray)
        lap_var = cv2.Laplacian(gray_arr, cv2.CV_64F).var()
        scores["sharpness"] = min(100, max(0, lap_var / 8))

        # Exposure
        stat = ImageStat.Stat(pil_img)
        brightness = sum(stat.mean[:3]) / 3
        if 80 <= brightness <= 180:
            scores["exposure"] = 100
        elif brightness < 40 or brightness > 230:
            scores["exposure"] = 20
        elif brightness < 80:
            scores["exposure"] = 20 + (brightness - 40) * 2
        else:
            scores["exposure"] = 20 + (230 - brightness) * 1.6

        # Face quality via Haar
        pil_det = pil_img.copy()
        det_scale = min(600 / max(w, h), 1.0)
        if det_scale < 1:
            pil_det = pil_det.resize((int(w * det_scale), int(h * det_scale)), Image.LANCZOS)
        det_arr = np.array(pil_det)
        det_gray = cv2.cvtColor(det_arr, cv2.COLOR_RGB2GRAY)
        det_gray = cv2.equalizeHist(det_gray)
        cascade = get_cascade()
        faces = cascade.detectMultiScale(det_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))

        if len(faces) > 0:
            det_h, det_w = det_gray.shape[:2]
            areas = [fw * fh for (fx, fy, fw, fh) in faces]
            max_area = max(areas)
            size_ratio = min(100, (max_area / (det_w * det_h)) * 500)

            best_idx = areas.index(max_area)
            fx, fy, fw, fh = faces[best_idx]
            face_roi = det_gray[fy:fy+fh, fx:fx+fw]
            face_sharp = cv2.Laplacian(face_roi, cv2.CV_64F).var() if face_roi.size > 0 else 0
            scores["face_size"] = round(size_ratio, 1)
            scores["face_sharpness"] = min(100, face_sharp / 5)
            scores["face_count"] = int(len(faces))
        else:
            scores["face_size"] = 0
            scores["face_sharpness"] = 0
            scores["face_count"] = 0

        pil_img.close()

        # Overall weighted score
        overall = (
            scores["resolution"] * 0.10 +
            scores["sharpness"] * 0.25 +
            scores["exposure"] * 0.15 +
            scores["face_size"] * 0.20 +
            scores["face_sharpness"] * 0.30
        )
        scores["overall"] = round(overall, 1)

    except Exception as e:
        scores = {"overall": 0, "error": str(e)}

    return scores


# ── Main Pipeline ─────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  REEF BAR MITZVA - FULL IMAGE SCANNER")
    print("=" * 70)

    db = load_db()
    images_db = db.get("images", {})

    # Step 1: Collect all images
    print("\n[1/5] Collecting images from all sources...")
    all_images = collect_images()

    # Step 2: Load Reef face encodings
    print("\n[2/5] Loading Reef face encodings...")
    import face_recognition
    reef_encodings = load_reef_encodings()
    print(f"  Loaded {len(reef_encodings)} Reef encodings")

    if not reef_encodings:
        print("  ERROR: No Reef encodings. Add clear face photos to ref_faces/reef/")
        return

    # Step 3: Process each image
    print(f"\n[3/5] Processing {len(all_images)} images (face detect + recognize + grade)...")
    print("       This will take a while...\n")

    new_count = 0
    cached_count = 0
    face_found = 0
    reef_matched = 0
    no_face = 0
    errors = 0

    for i, img_info in enumerate(all_images):
        fpath = img_info["path"]
        key = img_info["key"]

        # Check cache
        if key in images_db and not images_db[key].get("error"):
            cached_count += 1
            if images_db[key].get("reef_match"):
                reef_matched += 1
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(all_images)} | new={new_count} cached={cached_count} "
                      f"reef={reef_matched} noface={no_face}")
            continue

        # Fast face detection first
        has_face = has_face_fast(fpath)
        if not has_face:
            # Store as no-face
            images_db[key] = {
                "path": fpath,
                "source": img_info["source"],
                "folder": img_info["folder"],
                "has_face": False,
                "reef_match": False,
                "reef_distance": 1.0,
                "grade": 0,
            }
            no_face += 1
            new_count += 1

            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(all_images)} | new={new_count} cached={cached_count} "
                      f"reef={reef_matched} noface={no_face}")
                save_db({"images": images_db, "stats": {}})
            continue

        face_found += 1

        # Face recognition: is this Reef?
        matched, distance = match_reef(fpath, reef_encodings)

        # Grade quality
        grade = grade_image(fpath) if matched else {"overall": 0}

        # Get date/bracket
        img_date = get_image_date(fpath, img_info["folder"])
        bracket = get_age_bracket(img_date, img_info["folder"])

        images_db[key] = {
            "path": fpath,
            "source": img_info["source"],
            "folder": img_info["folder"],
            "has_face": True,
            "reef_match": matched,
            "reef_distance": round(distance, 4),
            "grade": grade.get("overall", 0) if matched else 0,
            "grade_detail": grade if matched else {},
            "image_date": str(img_date) if img_date else None,
            "bracket": bracket,
            "filename": os.path.basename(fpath),
        }

        if matched:
            reef_matched += 1

        new_count += 1

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(all_images)} | new={new_count} cached={cached_count} "
                  f"reef={reef_matched} noface={no_face} faces={face_found}")
            save_db({"images": images_db, "stats": {}})

    # Final save
    save_db({"images": images_db, "stats": {
        "total_scanned": len(all_images),
        "faces_detected": face_found + sum(1 for v in images_db.values() if v.get("has_face")),
        "reef_matched": reef_matched + sum(1 for v in images_db.values() if v.get("reef_match")),
        "scan_date": datetime.now().isoformat(),
    }})

    print(f"\n  SCAN COMPLETE:")
    print(f"    Total: {len(all_images)}")
    print(f"    New: {new_count}, Cached: {cached_count}")
    print(f"    Faces detected: {face_found}")
    print(f"    Reef matched: {reef_matched}")
    print(f"    No face: {no_face}")

    # Step 4: Select top images per bracket
    print(f"\n[4/5] Selecting top {TARGET_PER_BRACKET} per bracket...")

    reef_images = [v for v in images_db.values() if v.get("reef_match")]
    print(f"  Total Reef images: {len(reef_images)}")

    by_bracket = defaultdict(list)
    for img in reef_images:
        bracket = img.get("bracket", "unknown")
        by_bracket[bracket].append(img)

    selections = {}
    for bracket in sorted(by_bracket.keys()):
        imgs = by_bracket[bracket]
        # Sort by grade descending
        imgs.sort(key=lambda x: x.get("grade", 0), reverse=True)
        selected = imgs[:TARGET_PER_BRACKET]
        selections[bracket] = selected
        print(f"  {bracket}: {len(imgs)} total, selected {len(selected)} "
              f"(grade range: {selected[-1].get('grade',0):.1f} - {selected[0].get('grade',0):.1f})")

    # Step 5: Copy to presentation
    print(f"\n[5/5] Copying selected images to presentation...")

    total_copied = 0
    for bracket, selected in sorted(selections.items()):
        bracket_dir = os.path.join(PRESENTATION_DIR, bracket)
        os.makedirs(bracket_dir, exist_ok=True)

        for img in selected:
            src_path = img["path"]
            fname = os.path.basename(src_path)
            dest_name = f"{bracket}__{fname}"
            dest_path = os.path.join(bracket_dir, dest_name)

            if os.path.exists(dest_path):
                continue

            try:
                shutil.copy2(src_path, dest_path)
                total_copied += 1
            except Exception as e:
                print(f"    ERROR copying {fname}: {e}")

    print(f"\n  Copied {total_copied} new images to presentation")
    print(f"  Presentation: {PRESENTATION_DIR}")

    # Summary
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    for bracket in sorted(selections.keys()):
        bracket_dir = os.path.join(PRESENTATION_DIR, bracket)
        if os.path.isdir(bracket_dir):
            n = len([f for f in os.listdir(bracket_dir)
                     if os.path.splitext(f)[1].lower() in IMAGE_EXTS])
            print(f"  {bracket}: {n} images")
    print("=" * 70)


if __name__ == "__main__":
    main()
