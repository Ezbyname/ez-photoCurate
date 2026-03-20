# Filter presentation images that contain no faces/people
# Uses OpenCV face detection (frontal + profile) + upper body detection
# Flags images with no detections for review, then removes + refills

import os
import sys
import shutil
import json
import hashlib
import re
import numpy as np
import cv2
from collections import defaultdict
from datetime import datetime, timedelta
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)

ONEDRIVE_DIR = r"C:\Users\erezg\OneDrive - Pen-Link Ltd\Desktop\אישי\ריף\בר מצווה"
PRESENTATION_DIR = os.path.join(ONEDRIVE_DIR, "the presentation")
BACKUP_DIR = os.path.join(ONEDRIVE_DIR, "backup pool")
TAKEOUT_DIR = os.path.join(ONEDRIVE_DIR, r"download\extracted\Takeout\Google Photos\reef")
DISK_DIR = r"D:\ריף"
PROJECT_DIR = r"C:\Codes\Reef images for bar mitza"
REMOVED_DIR = os.path.join(PROJECT_DIR, "removed_from_presentation")
REEF_BIRTHDAY = datetime(2013, 7, 16)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".heic", ".webp"}
VECTOR_SIZE = 64
SIM_THRESHOLD = 0.85
TARGET_PER_AGE = 75

AGE_BRACKETS = [
    ("00_birth",         0,    14,  "2013_birth"),
    ("01_month1",       14,    60,  "2013_month1"),
    ("02_months2-3",    60,   120,  "2013_months2-3"),
    ("03_months4-6",   120,   210,  "2013-2014_months4-6"),
    ("04_months7-9",   210,   300,  "2014_months7-9"),
    ("05_months10-12", 300,   365,  "2014_months10-12"),
    ("06_year1",       365,   730,  "2014-2015_age1"),
    ("07_year2",       730,  1095,  "2015-2016_age2"),
    ("08_year3-4",    1095,  1825,  "2016-2018_age3-4"),
    ("09_year5-7",    1825,  2920,  "2018-2020_age5-7"),
    ("10_year8-10",   2920,  3650,  "2021-2023_age8-10"),
    ("11_year11-12",  3650,  4745,  "2024-2025_age11-12"),
    ("12_barmitzva",  4745,  5110,  "2026_barmitzva"),
]

PREFIX_TO_BRACKET = {}
for _label, _start, _end, _folder in AGE_BRACKETS:
    PREFIX_TO_BRACKET[_folder] = _label
OLD_PREFIX_MAP = {"2013": "02_months2-3", "2014": "04_months7-9",
                  "2015": "07_year2", "2026": "12_barmitzva"}

DIR_AGE_MAP = {
    "בית חולים": 0, "כשריף נולד": 0, "ריף מגיע הביתה": 3,
    "ברית ארז": 44, "ברית": 8, "לידה": 3,
    "ריף בן חודש": 30, "חודש 1": 30,
    "8.9- 9.9": 55, "28.9": 74, "כיפור 2013": 90,
    "חודש 2-12": 180,
    "ריף בן חודשיים": 60, "ריף בן 3 חודשים": 90,
    "ריף בן 4 חודשים": 120, "ריף בן 5 חודשים": 150,
    "ריף בן חצי שנה": 180,
    "ריף בן 7 חודשים": 210, "ריף בן 8 חודשים": 240,
    "ריף בן 9 חודשים": 270, "ריף בן 10 חודשים": 300,
    "ריף בן 11 חודשים": 330,
    "ריף בן שנה": 365, "בוק": 365, "משפחה ארז": 365,
    "שנה - שנתיים": 545,
    "שנה וחודש": 395, "שנה וחודשיים": 425,
    "שנה ושלוש": 455, "שנה וארבע": 485,
    "שנה וחמש": 515, "שנה ו5": 515,
    "שנה וחצי": 545, "שנה ושבע": 575,
    "שנה ושמונה": 605, "שנה ותשע": 635,
    "שנה ועשר": 665,
    "ipad": 730, "תיקיה חדשה": 60, "mix": None,
}


# ── Face/person detection ─────────────────────────────────────────────────────

def load_detectors():
    """Load OpenCV cascade classifiers."""
    cv2_data = cv2.data.haarcascades
    detectors = []
    for cascade_name in [
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_profileface.xml",
        "haarcascade_upperbody.xml",
    ]:
        path = os.path.join(cv2_data, cascade_name)
        if os.path.exists(path):
            detectors.append((cascade_name.replace("haarcascade_", "").replace(".xml", ""),
                              cv2.CascadeClassifier(path)))
    return detectors


def detect_faces(image_path, detectors):
    """
    Returns (has_person, detail_string).
    Tries multiple detectors and scales.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            # Try loading with PIL for non-ASCII paths
            pil_img = Image.open(image_path).convert("RGB")
            img = np.array(pil_img)[:, :, ::-1]  # RGB -> BGR

        if img is None:
            return False, "unreadable"

        # Resize for speed (keep aspect ratio, max 800px)
        h, w = img.shape[:2]
        scale = min(800 / max(h, w), 1.0)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        for name, detector in detectors:
            # Try multiple scale factors for better detection
            for scale_factor in [1.1, 1.15, 1.2]:
                min_neighbors = 3 if "face" in name else 2
                min_size = (20, 20) if "face" in name else (40, 40)

                faces = detector.detectMultiScale(
                    gray,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=min_size,
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                if len(faces) > 0:
                    return True, f"{name}({len(faces)})"

        # Also try on rotated versions (some baby photos are rotated)
        for angle in [90, 270]:
            if angle == 90:
                rotated = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
            else:
                rotated = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)

            for name, detector in detectors:
                faces = detector.detectMultiScale(
                    rotated, scaleFactor=1.15, minNeighbors=3,
                    minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE
                )
                if len(faces) > 0:
                    return True, f"{name}_rot{angle}({len(faces)})"

        return False, "no_detection"

    except Exception as e:
        return False, f"error: {e}"


# ── Utility functions ─────────────────────────────────────────────────────────

def compute_vector(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        gray = img.convert("L").resize((VECTOR_SIZE, VECTOR_SIZE), Image.LANCZOS)
        pixels = np.array(gray, dtype=np.float32).flatten()
        img_small = img.resize((128, 128), Image.LANCZOS)
        arr = np.array(img_small)
        hist_r = np.histogram(arr[:, :, 0], bins=16, range=(0, 256))[0].astype(np.float32)
        hist_g = np.histogram(arr[:, :, 1], bins=16, range=(0, 256))[0].astype(np.float32)
        hist_b = np.histogram(arr[:, :, 2], bins=16, range=(0, 256))[0].astype(np.float32)
        vector = np.concatenate([pixels, hist_r * 10, hist_g * 10, hist_b * 10])
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector
    except Exception:
        return None


def quality_score(image_path):
    try:
        sz = os.path.getsize(image_path)
        img = Image.open(image_path)
        w, h = img.size
        img.close()
        return w * h * (sz / 1024)
    except Exception:
        return 0


def file_hash(filepath):
    h = hashlib.md5(usedforsecurity=False)
    h.update(str(os.path.getsize(filepath)).encode())
    with open(filepath, "rb") as f:
        h.update(f.read(65536))
    return h.hexdigest()


def presentation_name_to_bracket(filename):
    parts = filename.split("__", 1)
    if len(parts) != 2:
        return None
    prefix = parts[0]
    if prefix in PREFIX_TO_BRACKET:
        return PREFIX_TO_BRACKET[prefix]
    return OLD_PREFIX_MAP.get(prefix)


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


def get_json_date(image_path):
    for jpath in [image_path + ".json", os.path.splitext(image_path)[0] + ".json"]:
        if os.path.exists(jpath):
            try:
                with open(jpath, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                ts = meta.get("photoTakenTime", {}).get("timestamp")
                if ts:
                    return datetime.fromtimestamp(int(ts))
            except Exception:
                pass
    return None


def get_filename_date(filename):
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})', filename)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    m = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
    if m:
        try:
            d = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if 2013 <= d.year <= 2026:
                return d
        except ValueError:
            pass
    return None


def dir_to_age_days(rel_path):
    parts = rel_path.replace("\\", "/").split("/")
    for part in parts:
        for pattern, age_days in DIR_AGE_MAP.items():
            if age_days is not None and pattern in part:
                return age_days
    return None


def age_days_to_bracket(age_days):
    if age_days is None or age_days < 0:
        return None
    for label, start, end, folder in AGE_BRACKETS:
        if start <= age_days < end:
            return label
    return None


def determine_bracket(filepath, rel_dir=None):
    dt = get_exif_date(filepath)
    if dt:
        b = age_days_to_bracket((dt - REEF_BIRTHDAY).days)
        if b:
            return b
    dt = get_json_date(filepath)
    if dt:
        b = age_days_to_bracket((dt - REEF_BIRTHDAY).days)
        if b:
            return b
    dt = get_filename_date(os.path.basename(filepath))
    if dt:
        b = age_days_to_bracket((dt - REEF_BIRTHDAY).days)
        if b:
            return b
    if rel_dir:
        age_days = dir_to_age_days(rel_dir)
        if age_days is not None:
            b = age_days_to_bracket(age_days)
            if b:
                return b
    return None


def is_screenshot(image_path):
    try:
        img = Image.open(image_path).convert("RGB").resize((256, 256))
        arr = np.array(img, dtype=np.float64)
        pixels = arr.reshape(-1, 3)
        unique = len(set(map(tuple, pixels.astype(int).tolist())))
        return unique / len(pixels) < 0.08
    except Exception:
        return False


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


def bracket_to_folder(bracket_label):
    for label, start, end, folder in AGE_BRACKETS:
        if label == bracket_label:
            return folder
    return None


# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  FACE DETECTION FILTER")
    print("  Remove images with no people, refill from sources")
    print("=" * 70)

    os.makedirs(REMOVED_DIR, exist_ok=True)

    # ── Step 1: Load detectors ────────────────────────────────────────────────
    print("\n[Step 1] Loading face/body detectors...")
    detectors = load_detectors()
    print(f"  Loaded {len(detectors)} detectors: {[d[0] for d in detectors]}")

    # ── Step 2: Scan all presentation images ──────────────────────────────────
    print("\n[Step 2] Scanning presentation images for faces...")
    pres_files = sorted([f for f in os.listdir(PRESENTATION_DIR)
                         if os.path.splitext(f)[1].lower() in IMAGE_EXTS])
    print(f"  {len(pres_files)} images to scan")

    has_face = []
    no_face = []

    for i, fname in enumerate(pres_files):
        fpath = os.path.join(PRESENTATION_DIR, fname)
        found, detail = detect_faces(fpath, detectors)
        bracket = presentation_name_to_bracket(fname)

        if found:
            has_face.append(fname)
        else:
            no_face.append((fname, bracket, detail))

        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(pres_files)} scanned, {len(no_face)} no-face so far...")

    print(f"\n  With faces: {len(has_face)}")
    print(f"  No faces detected: {len(no_face)}")

    # ── Step 3: Report no-face images by bracket ─────────────────────────────
    print("\n[Step 3] No-face images by bracket:")
    no_face_by_bracket = defaultdict(list)
    for fname, bracket, detail in no_face:
        no_face_by_bracket[bracket or "unknown"].append((fname, detail))

    for label, start, end, folder in AGE_BRACKETS:
        items = no_face_by_bracket.get(label, [])
        if items:
            print(f"\n  {folder} ({len(items)} flagged):")
            for fname, detail in items:
                print(f"    {fname} [{detail}]")

    # ── Step 4: Remove no-face images ─────────────────────────────────────────
    print(f"\n[Step 4] Removing {len(no_face)} no-face images...")
    removed_by_bracket = defaultdict(int)

    for fname, bracket, detail in no_face:
        src = os.path.join(PRESENTATION_DIR, fname)
        if os.path.exists(src):
            dst = os.path.join(REMOVED_DIR, "noface__" + fname)
            shutil.move(src, dst)
            removed_by_bracket[bracket or "unknown"] += 1

    # ── Step 5: Count what needs refilling ────────────────────────────────────
    print("\n[Step 5] Post-removal counts:")
    remaining_files = sorted([f for f in os.listdir(PRESENTATION_DIR)
                              if os.path.splitext(f)[1].lower() in IMAGE_EXTS])
    bracket_remaining = defaultdict(list)
    for f in remaining_files:
        b = presentation_name_to_bracket(f)
        if b:
            bracket_remaining[b].append(f)

    needs_fill = {}
    for label, start, end, folder in AGE_BRACKETS:
        count = len(bracket_remaining.get(label, []))
        removed = removed_by_bracket.get(label, 0)
        needed = max(0, TARGET_PER_AGE - count)
        if needed > 0:
            needs_fill[label] = needed
        status = f"need {needed}" if needed > 0 else "OK"
        print(f"  {folder:<30} {count:>4} (-{removed}) {status}")

    total_needed = sum(needs_fill.values())
    print(f"\n  Total to refill: {total_needed}")

    if total_needed == 0:
        _final_summary(pres_files, no_face, [], remaining_files)
        return

    # ── Step 6: Refill from backup pool first, then sources ───────────────────
    print("\n[Step 6] Refilling gaps...")

    # Vectorize remaining presentation images
    bracket_vectors = defaultdict(list)
    existing_hashes = set()
    for f in remaining_files:
        fpath = os.path.join(PRESENTATION_DIR, f)
        b = presentation_name_to_bracket(f)
        if b:
            vec = compute_vector(fpath)
            if vec is not None:
                bracket_vectors[b].append(vec)
        try:
            existing_hashes.add(file_hash(fpath))
        except Exception:
            pass

    # 6a: Try backup pool first
    print("  Checking backup pool...")
    backup_added = 0
    if os.path.exists(BACKUP_DIR):
        backup_files = [f for f in os.listdir(BACKUP_DIR)
                        if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
        for bfname in backup_files:
            bpath = os.path.join(BACKUP_DIR, bfname)
            bracket = presentation_name_to_bracket(bfname)
            if not bracket or bracket not in needs_fill:
                continue

            # Face check on backup image
            found, _ = detect_faces(bpath, detectors)
            if not found:
                continue

            # Diversity check
            vec = compute_vector(bpath)
            if vec is None:
                continue
            existing_vecs = bracket_vectors.get(bracket, [])
            if existing_vecs:
                mat = np.array(existing_vecs, dtype=np.float32)
                sims = mat @ vec
                if np.max(sims) >= SIM_THRESHOLD:
                    continue

            # Move from backup to presentation
            dest = unique_dest_path(PRESENTATION_DIR, bfname)
            shutil.move(bpath, dest)
            bracket_vectors[bracket].append(vec)
            try:
                existing_hashes.add(file_hash(dest))
            except Exception:
                pass
            needs_fill[bracket] -= 1
            if needs_fill[bracket] <= 0:
                del needs_fill[bracket]
            backup_added += 1

    print(f"  From backup pool: {backup_added}")

    # 6b: Scan sources for remaining needs
    total_still_needed = sum(needs_fill.values())
    all_new = []

    if total_still_needed > 0:
        print(f"  Still need {total_still_needed}, scanning sources...")

        source_dirs = []
        # Disk
        if os.path.exists(DISK_DIR):
            source_dirs.append((DISK_DIR, "disk"))
        # OneDrive
        for name in ["extra images", "mix"]:
            d = os.path.join(ONEDRIVE_DIR, name)
            if os.path.exists(d):
                source_dirs.append((d, name))
        for label, start, end, folder in AGE_BRACKETS:
            d = os.path.join(ONEDRIVE_DIR, folder)
            if os.path.exists(d):
                source_dirs.append((d, folder))
        for heb in ["לידה", "חודש 1", "חודש 2-12", "שנה - שנתיים"]:
            d = os.path.join(ONEDRIVE_DIR, heb)
            if os.path.exists(d):
                source_dirs.append((d, heb))
        if os.path.exists(TAKEOUT_DIR):
            source_dirs.append((TAKEOUT_DIR, "takeout"))

        seen_hashes = set(existing_hashes)
        scanned = 0

        for source_path, source_name in source_dirs:
            if not os.path.exists(source_path):
                continue
            if not needs_fill:
                break

            for dirpath, dirnames, filenames in os.walk(source_path):
                if any(v in dirpath for v in ["וידאו", "video"]):
                    continue
                rel_dir = os.path.relpath(dirpath, os.path.dirname(source_path))

                for fname in filenames:
                    if not needs_fill:
                        break
                    if os.path.splitext(fname)[1].lower() not in IMAGE_EXTS:
                        continue

                    fpath = os.path.join(dirpath, fname)
                    scanned += 1
                    if scanned % 1000 == 0:
                        still = sum(needs_fill.values())
                        print(f"    Scanned {scanned}, still need {still}...")

                    try:
                        fhash = file_hash(fpath)
                    except Exception:
                        continue
                    if fhash in seen_hashes:
                        continue
                    seen_hashes.add(fhash)

                    # Quality gate
                    try:
                        if os.path.getsize(fpath) < 80 * 1024:
                            continue
                        img = Image.open(fpath)
                        w, h = img.size
                        img.close()
                        if w < 600 and h < 600:
                            continue
                    except Exception:
                        continue

                    if is_screenshot(fpath):
                        continue

                    bracket = determine_bracket(fpath, rel_dir)
                    if not bracket or bracket not in needs_fill:
                        continue

                    # Face check
                    found, _ = detect_faces(fpath, detectors)
                    if not found:
                        continue

                    # Diversity check
                    vec = compute_vector(fpath)
                    if vec is None:
                        continue
                    existing_vecs = bracket_vectors.get(bracket, [])
                    if existing_vecs:
                        mat = np.array(existing_vecs, dtype=np.float32)
                        sims = mat @ vec
                        if np.max(sims) >= SIM_THRESHOLD:
                            continue

                    # Copy to presentation
                    folder_name = bracket_to_folder(bracket)
                    pres_name = f"{folder_name}__{fname}"
                    dest = unique_dest_path(PRESENTATION_DIR, pres_name)
                    try:
                        shutil.copy2(fpath, dest)
                        bracket_vectors[bracket].append(vec)
                        existing_hashes.add(fhash)
                        all_new.append({"bracket": bracket, "filename": os.path.basename(dest)})
                        needs_fill[bracket] -= 1
                        if needs_fill[bracket] <= 0:
                            del needs_fill[bracket]
                    except OSError:
                        pass

        print(f"  From sources: {len(all_new)} (scanned {scanned})")

    _final_summary(pres_files, no_face, all_new, None, backup_added)


def _final_summary(original_files, no_face, new_from_sources, _, backup_added=0):
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)

    final_files = [f for f in os.listdir(PRESENTATION_DIR)
                   if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
    final_by_bracket = defaultdict(int)
    for f in final_files:
        b = presentation_name_to_bracket(f)
        if b:
            final_by_bracket[b] += 1

    print(f"\n  {'Bracket':<30} {'Final':>6} {'Status':>10}")
    print("  " + "-" * 50)
    total = 0
    for label, start, end, folder in AGE_BRACKETS:
        count = final_by_bracket.get(label, 0)
        total += count
        marker = "OK" if count >= TARGET_PER_AGE else ("LOW" if count < 50 else "close")
        print(f"  {folder:<30} {count:>6} {marker:>10}")

    print("  " + "-" * 50)
    print(f"  {'TOTAL':<30} {total:>6}")
    print(f"\n  No-face images removed: {len(no_face)}")
    print(f"  Refilled from backup: {backup_added}")
    print(f"  Refilled from sources: {len(new_from_sources)}")
    print(f"  Removed images saved to: {REMOVED_DIR}")


if __name__ == "__main__":
    main()
