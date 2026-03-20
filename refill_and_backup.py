# Refill presentation from disk + create backup pool
# 1. Remove manually flagged images (no Reef)
# 2. Fill each bracket to 75 from disk D:\
# 3. Create backup pool (~25 extra diverse candidates per bracket)

import os
import sys
import re
import shutil
import hashlib
import json
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)

DISK_DIR = r"D:\ריף"
ONEDRIVE_DIR = r"C:\Users\erezg\OneDrive - Pen-Link Ltd\Desktop\אישי\ריף\בר מצווה"
TAKEOUT_DIR = os.path.join(ONEDRIVE_DIR, r"download\extracted\Takeout\Google Photos\reef")
PRESENTATION_DIR = os.path.join(ONEDRIVE_DIR, "the presentation")
PROJECT_DIR = r"C:\Codes\Reef images for bar mitza"
REMOVED_DIR = os.path.join(PROJECT_DIR, "removed_from_presentation")
BACKUP_DIR = os.path.join(ONEDRIVE_DIR, "backup pool")
REEF_BIRTHDAY = datetime(2013, 7, 16)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".heic", ".webp"}
VECTOR_SIZE = 64
SIM_THRESHOLD = 0.85
TARGET_PER_AGE = 75
BACKUP_PER_AGE = 25

# Images to remove manually (no Reef in them)
MANUAL_REMOVE = [
    "2013_month1__Nitzan-Zohar-9708.jpg",
    "2013_month1__Nitzan-Zohar-9713.jpg",
    "2013_month1__Nitzan-Zohar-9714.jpg",
    "2013_month1__Nitzan-Zohar-9677.jpg",
    "2013_month1__Nitzan-Zohar-9682.jpg",
    "2013_month1__Nitzan-Zohar-9704.jpg",
    "2013_month1__Nitzan-Zohar-9706.jpg",
    "2013_month1__Nitzan-Zohar-9707.jpg",
]

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


def is_screenshot(image_path):
    try:
        img = Image.open(image_path).convert("RGB").resize((256, 256))
        arr = np.array(img, dtype=np.float64)
        pixels = arr.reshape(-1, 3)
        unique = len(set(map(tuple, pixels.astype(int).tolist())))
        return unique / len(pixels) < 0.08
    except Exception:
        return False


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


def bracket_to_folder(bracket_label):
    for label, start, end, folder in AGE_BRACKETS:
        if label == bracket_label:
            return folder
    return None


def determine_bracket_from_sources(filepath, rel_dir=None):
    dt = get_exif_date(filepath)
    if dt:
        age_days = (dt - REEF_BIRTHDAY).days
        b = age_days_to_bracket(age_days)
        if b:
            return b
    dt = get_json_date(filepath)
    if dt:
        age_days = (dt - REEF_BIRTHDAY).days
        b = age_days_to_bracket(age_days)
        if b:
            return b
    dt = get_filename_date(os.path.basename(filepath))
    if dt:
        age_days = (dt - REEF_BIRTHDAY).days
        b = age_days_to_bracket(age_days)
        if b:
            return b
    if rel_dir:
        age_days = dir_to_age_days(rel_dir)
        if age_days is not None:
            b = age_days_to_bracket(age_days)
            if b:
                return b
    return None


def presentation_name_to_bracket(filename):
    parts = filename.split("__", 1)
    if len(parts) != 2:
        return None
    prefix = parts[0]
    if prefix in PREFIX_TO_BRACKET:
        return PREFIX_TO_BRACKET[prefix]
    return OLD_PREFIX_MAP.get(prefix)


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
    print("=" * 70)
    print("  REFILL FROM DISK + BACKUP POOL")
    print("=" * 70)

    os.makedirs(REMOVED_DIR, exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)

    # ── Step 1: Remove manually flagged images ────────────────────────────────
    print("\n[Step 1] Removing manually flagged images (no Reef)...")
    manual_removed = 0
    for fname in MANUAL_REMOVE:
        fpath = os.path.join(PRESENTATION_DIR, fname)
        if os.path.exists(fpath):
            dst = os.path.join(REMOVED_DIR, "manual__" + fname)
            shutil.move(fpath, dst)
            manual_removed += 1
            print(f"  Removed: {fname}")
        else:
            print(f"  Already gone: {fname}")
    print(f"  Manual removals: {manual_removed}")

    # ── Step 2: Count current state per bracket ───────────────────────────────
    print("\n[Step 2] Current presentation state...")
    pres_files = sorted([f for f in os.listdir(PRESENTATION_DIR)
                         if os.path.splitext(f)[1].lower() in IMAGE_EXTS])

    bracket_files = defaultdict(list)
    for f in pres_files:
        b = presentation_name_to_bracket(f)
        if b:
            bracket_files[b].append(f)

    needs_fill = {}
    print(f"\n  {'Bracket':<30} {'Count':>6} {'Need':>6}")
    print("  " + "-" * 44)
    for label, start, end, folder in AGE_BRACKETS:
        count = len(bracket_files.get(label, []))
        needed = max(0, TARGET_PER_AGE - count)
        if needed > 0:
            needs_fill[label] = needed
        print(f"  {folder:<30} {count:>6} {needed if needed else 'OK':>6}")

    total_needed = sum(needs_fill.values())
    print(f"\n  Total to fill: {total_needed}")

    # ── Step 3: Vectorize existing presentation images ────────────────────────
    print("\n[Step 3] Vectorizing existing presentation images...")
    existing_hashes = set()
    bracket_vectors = defaultdict(list)

    for f in pres_files:
        fpath = os.path.join(PRESENTATION_DIR, f)
        try:
            existing_hashes.add(file_hash(fpath))
        except Exception:
            pass
        b = presentation_name_to_bracket(f)
        if b:
            vec = compute_vector(fpath)
            if vec is not None:
                bracket_vectors[b].append(vec)

    print(f"  {len(existing_hashes)} hashes, {sum(len(v) for v in bracket_vectors.values())} vectors")

    # ── Step 4: Scan ALL sources (disk + OneDrive + takeout) ──────────────────
    print("\n[Step 4] Scanning all sources...")

    source_dirs = []
    # Disk (primary for early ages)
    if os.path.exists(DISK_DIR):
        source_dirs.append((DISK_DIR, "disk"))
    # OneDrive folders
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
    # Takeout
    if os.path.exists(TAKEOUT_DIR):
        source_dirs.append((TAKEOUT_DIR, "takeout"))

    # We need candidates for BOTH fill and backup, so collect for all brackets
    all_brackets_needed = set()
    for label, _, _, _ in AGE_BRACKETS:
        all_brackets_needed.add(label)

    candidates_by_bracket = defaultdict(list)
    scanned = 0
    seen_hashes = set(existing_hashes)

    for source_path, source_name in source_dirs:
        if not os.path.exists(source_path):
            continue
        for dirpath, dirnames, filenames in os.walk(source_path):
            # Skip video dirs
            rel_dir = os.path.relpath(dirpath, os.path.dirname(source_path))
            if any(v in rel_dir for v in ["וידאו", "video"]):
                continue

            for fname in filenames:
                if os.path.splitext(fname)[1].lower() not in IMAGE_EXTS:
                    continue
                fpath = os.path.join(dirpath, fname)
                scanned += 1
                if scanned % 1000 == 0:
                    print(f"    Scanned {scanned}...")

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

                # Determine bracket
                bracket = determine_bracket_from_sources(fpath, rel_dir)
                if not bracket:
                    continue

                vec = compute_vector(fpath)
                if vec is None:
                    continue

                score = quality_score(fpath)
                candidates_by_bracket[bracket].append({
                    "filename": fname,
                    "path": fpath,
                    "vector": vec,
                    "bracket": bracket,
                    "score": score,
                    "hash": fhash,
                })

    print(f"\n  Scanned {scanned} images total")
    print(f"  Candidates per bracket:")
    for label, start, end, folder in AGE_BRACKETS:
        c = len(candidates_by_bracket.get(label, []))
        if c > 0:
            print(f"    {folder}: {c}")

    # ── Step 5: Fill presentation to 75 per bracket ───────────────────────────
    print("\n[Step 5] Filling presentation gaps...")

    # Verify disk is still accessible before copying
    if not os.path.exists(DISK_DIR):
        print("  ERROR: Disk not accessible! Please reconnect and re-run.")
        return
    total_added = 0

    for label, start, end, folder in AGE_BRACKETS:
        needed = needs_fill.get(label, 0)
        if needed == 0:
            continue

        candidates = candidates_by_bracket.get(label, [])
        if not candidates:
            print(f"  {folder}: need {needed}, 0 candidates")
            continue

        candidates.sort(key=lambda x: x["score"], reverse=True)

        existing_vecs = list(bracket_vectors.get(label, []))
        selected = []

        for cand in candidates:
            if existing_vecs:
                mat = np.array(existing_vecs, dtype=np.float32)
                sims = mat @ cand["vector"]
                if np.max(sims) >= SIM_THRESHOLD:
                    continue

            selected.append(cand)
            existing_vecs.append(cand["vector"])
            if len(selected) >= needed:
                break

        added = 0
        for cand in selected:
            pres_name = f"{folder}__{cand['filename']}"
            dest = unique_dest_path(PRESENTATION_DIR, pres_name)
            try:
                shutil.copy2(cand["path"], dest)
                existing_hashes.add(cand["hash"])
                # Update bracket_vectors for backup selection
                bracket_vectors[label].append(cand["vector"])
                added += 1
                total_added += 1
            except OSError as e:
                print(f"    WARN: {e}")

        print(f"  {folder}: +{added}/{needed} needed ({len(candidates)} candidates)")

    # ── Step 6: Create backup pool ────────────────────────────────────────────
    print(f"\n[Step 6] Creating backup pool (~{BACKUP_PER_AGE} per bracket)...")

    # Clear old backup pool
    if os.path.exists(BACKUP_DIR):
        for f in os.listdir(BACKUP_DIR):
            fp = os.path.join(BACKUP_DIR, f)
            if os.path.isfile(fp):
                os.remove(fp)

    # Re-read presentation hashes after fills
    pres_hashes = set()
    for f in os.listdir(PRESENTATION_DIR):
        fp = os.path.join(PRESENTATION_DIR, f)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
            try:
                pres_hashes.add(file_hash(fp))
            except Exception:
                pass

    total_backup = 0
    for label, start, end, folder in AGE_BRACKETS:
        candidates = candidates_by_bracket.get(label, [])
        if not candidates:
            continue

        candidates.sort(key=lambda x: x["score"], reverse=True)

        # Current vectors in presentation for this bracket
        cur_vecs = list(bracket_vectors.get(label, []))
        backup_selected = []

        for cand in candidates:
            if cand["hash"] in pres_hashes:
                continue

            # Must be diverse from presentation AND other backups
            all_vecs = cur_vecs + [b["vector"] for b in backup_selected]
            if all_vecs:
                mat = np.array(all_vecs, dtype=np.float32)
                sims = mat @ cand["vector"]
                if np.max(sims) >= SIM_THRESHOLD:
                    continue

            backup_selected.append(cand)
            if len(backup_selected) >= BACKUP_PER_AGE:
                break

        # Copy to backup dir
        backed = 0
        for cand in backup_selected:
            backup_name = f"{folder}__{cand['filename']}"
            dest = unique_dest_path(BACKUP_DIR, backup_name)
            try:
                shutil.copy2(cand["path"], dest)
                backed += 1
                total_backup += 1
            except OSError:
                pass

        if backed > 0:
            print(f"  {folder}: {backed} backup images")

    # ── Final Summary ─────────────────────────────────────────────────────────
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
        else:
            final_by_bracket["unknown"] += 1

    backup_files = [f for f in os.listdir(BACKUP_DIR)
                    if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
    backup_by_bracket = defaultdict(int)
    for f in backup_files:
        b = presentation_name_to_bracket(f)
        if b:
            backup_by_bracket[b] += 1

    print(f"\n  {'Bracket':<30} {'Presentation':>13} {'Backup':>8}")
    print("  " + "-" * 54)
    total_pres = 0
    for label, start, end, folder in AGE_BRACKETS:
        pcount = final_by_bracket.get(label, 0)
        bcount = backup_by_bracket.get(label, 0)
        total_pres += pcount
        marker = " OK" if pcount >= TARGET_PER_AGE else (" << LOW" if pcount < 50 else "")
        print(f"  {folder:<30} {pcount:>13} {bcount:>8}{marker}")

    print("  " + "-" * 54)
    print(f"  {'TOTAL':<30} {total_pres:>13} {total_backup:>8}")

    print(f"\n  Manual removals: {manual_removed}")
    print(f"  New images added: {total_added}")
    print(f"  Backup pool: {total_backup} images in backup pool folder")
    print(f"  To swap: move image from backup pool to presentation, delete unwanted from presentation")


if __name__ == "__main__":
    main()
