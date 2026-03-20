# Fill presentation gaps from D:\ריף disk-on-key
# Scans disk images, determines age bracket, selects diverse high-quality
# images to reach ~75 per age bracket in the presentation folder.

import os
import sys
import json
import re
import shutil
import hashlib
import pickle
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)

# ── Configuration ─────────────────────────────────────────────────────────────
DISK_DIR = r"D:\ריף"
ONEDRIVE_DIR = r"C:\Users\erezg\OneDrive - Pen-Link Ltd\Desktop\אישי\ריף\בר מצווה"
PRESENTATION_DIR = os.path.join(ONEDRIVE_DIR, "the presentation")
PROJECT_DIR = r"C:\Codes\Reef images for bar mitza"
VECTORS_FILE = os.path.join(PROJECT_DIR, "presentation_vectors.pkl")
REMOVED_DIR = os.path.join(PROJECT_DIR, "removed_from_presentation")
REEF_BIRTHDAY = datetime(2013, 7, 16)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".heic", ".webp"}
SIMILARITY_THRESHOLD = 0.94
TARGET_PER_AGE = 75
VECTOR_SIZE = 64
MIN_FILE_SIZE = 80 * 1024
MIN_DIM = 600

# ── Age brackets ──────────────────────────────────────────────────────────────
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

# Hebrew directory -> age in days from birthday
DIR_AGE_MAP = {
    "בית חולים": 0, "כשריף נולד": 0, "ריף מגיע הביתה": 3,
    "ברית": 8, "ברית ארז": 44,
    "ריף בן חודש": 30, "8.9- 9.9": 55, "28.9": 74,
    "כיפור 2013": 90,
    "ריף בן 3 חודשים": 90, "ריף בן חודשיים": 60,
    "ריף בן 4 חודשים": 120, "ריף בן 5 חודשים": 150,
    "ריף בן חצי שנה": 180,
    "ריף בן 7 חודשים": 210, "ריף בן 8 חודשים": 240,
    "ריף בן 9 חודשים": 270, "ריף בן 10 חודשים": 300,
    "ריף בן 11 חודשים": 330,
    "ריף בן שנה": 365, "בוק": 365, "משפחה ארז": 365,
    "שנה וחודש": 395, "שנה וחודשיים": 425,
    "שנה ושלוש": 455, "שנה וארבע": 485,
    "שנה וחמש": 515, "שנה ו5": 515,
    "שנה וחצי": 545, "שנה ושבע": 575,
    "שנה ושמונה": 605, "שנה ותשע": 635,
    "שנה ועשר": 665,
    "ipad": 730,
    "תיקיה חדשה": 60,
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


def is_screenshot(image_path):
    try:
        img = Image.open(image_path).convert("RGB").resize((256, 256))
        arr = np.array(img, dtype=np.float64)
        pixels = arr.reshape(-1, 3)
        unique = len(set(map(tuple, pixels.astype(int).tolist())))
        ratio = unique / len(pixels)
        return ratio < 0.08
    except Exception:
        return False


def file_hash(filepath):
    h = hashlib.md5(usedforsecurity=False)
    h.update(str(os.path.getsize(filepath)).encode())
    with open(filepath, "rb") as f:
        h.update(f.read(65536))
    return h.hexdigest()


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
    for label, start, end, folder in AGE_BRACKETS:
        if start <= age_days < end:
            return label
    return None


def bracket_to_folder(bracket_label):
    for label, start, end, folder in AGE_BRACKETS:
        if label == bracket_label:
            return folder
    return None


def presentation_name_to_bracket(filename):
    PREFIX_TO_BRACKET = {}
    for label, start, end, folder in AGE_BRACKETS:
        PREFIX_TO_BRACKET[folder] = label
    OLD_MAP = {"2013": "02_months2-3", "2014": "04_months7-9",
               "2015": "07_year2", "2026": "12_barmitzva"}
    parts = filename.split("__", 1)
    if len(parts) != 2:
        return None
    prefix = parts[0]
    if prefix in PREFIX_TO_BRACKET:
        return PREFIX_TO_BRACKET[prefix]
    return OLD_MAP.get(prefix)


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
    print("  FILL PRESENTATION FROM DISK D:\\")
    print("=" * 70)

    # ── Step 1: Count current presentation per bracket ────────────────────────
    print("\n[Step 1] Current presentation state...")
    pres_files = [f for f in os.listdir(PRESENTATION_DIR)
                  if os.path.splitext(f)[1].lower() in IMAGE_EXTS]

    bracket_files = defaultdict(list)
    for f in pres_files:
        b = presentation_name_to_bracket(f)
        if b:
            bracket_files[b].append(f)

    needs_fill = {}
    print(f"\n  {'Bracket':<30} {'Current':>8} {'Need':>8}")
    print("  " + "-" * 50)
    for label, start, end, folder in AGE_BRACKETS:
        count = len(bracket_files.get(label, []))
        needed = max(0, TARGET_PER_AGE - count)
        if needed > 0:
            needs_fill[label] = needed
        status = "OK" if needed == 0 else f"{needed}"
        print(f"  {folder:<30} {count:>8} {status:>8}")

    total_needed = sum(needs_fill.values())
    print(f"\n  Total needed: {total_needed}")
    if total_needed == 0:
        print("  All brackets full!")
        return

    # ── Step 2: Hash existing presentation images ─────────────────────────────
    print("\n[Step 2] Hashing existing presentation images...")
    existing_hashes = set()
    for f in pres_files:
        try:
            existing_hashes.add(file_hash(os.path.join(PRESENTATION_DIR, f)))
        except Exception:
            pass
    print(f"  {len(existing_hashes)} existing hashes")

    # ── Step 3: Compute vectors for current presentation images per bracket ───
    print("\n[Step 3] Vectorizing current presentation images for diversity check...")
    bracket_vectors = defaultdict(list)  # bracket -> list of vectors
    for label in needs_fill:
        for fname in bracket_files.get(label, []):
            fpath = os.path.join(PRESENTATION_DIR, fname)
            vec = compute_vector(fpath)
            if vec is not None:
                bracket_vectors[label].append(vec)
    print(f"  Vectorized {sum(len(v) for v in bracket_vectors.values())} existing images in target brackets")

    # ── Step 4: Scan disk ─────────────────────────────────────────────────────
    print(f"\n[Step 4] Scanning disk...")
    candidates_by_bracket = defaultdict(list)
    scanned = 0
    skipped_dup = 0
    skipped_quality = 0
    skipped_no_bracket = 0
    skipped_not_needed = 0
    seen_hashes = set(existing_hashes)

    for dirpath, dirnames, filenames in os.walk(DISK_DIR):
        # Skip video directories
        rel_dir = os.path.relpath(dirpath, DISK_DIR)
        if any(v in rel_dir for v in ["וידאו", "video"]):
            continue

        dir_age = dir_to_age_days(rel_dir)

        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in IMAGE_EXTS:
                continue

            fpath = os.path.join(dirpath, fname)
            scanned += 1
            if scanned % 500 == 0:
                print(f"    Scanned {scanned} images, {sum(len(v) for v in candidates_by_bracket.values())} candidates...")

            # Hash dedup
            try:
                fhash = file_hash(fpath)
            except Exception:
                continue
            if fhash in seen_hashes:
                skipped_dup += 1
                continue
            seen_hashes.add(fhash)

            # Quality gate
            try:
                file_size = os.path.getsize(fpath)
                if file_size < MIN_FILE_SIZE:
                    skipped_quality += 1
                    continue
                img = Image.open(fpath)
                w, h = img.size
                img.close()
                if w < MIN_DIM and h < MIN_DIM:
                    skipped_quality += 1
                    continue
            except Exception:
                skipped_quality += 1
                continue

            # Determine age bracket
            age_days = None
            exif_date = get_exif_date(fpath)
            if exif_date:
                age_days = (exif_date - REEF_BIRTHDAY).days
            if age_days is None:
                age_days = dir_age

            bracket = age_days_to_bracket(age_days)
            if not bracket:
                skipped_no_bracket += 1
                continue

            if bracket not in needs_fill:
                skipped_not_needed += 1
                continue

            # Screenshot check
            if is_screenshot(fpath):
                skipped_quality += 1
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
                "dir": rel_dir,
            })

    print(f"\n  Scan complete: {scanned} images")
    print(f"    Duplicates: {skipped_dup}")
    print(f"    Low quality: {skipped_quality}")
    print(f"    No bracket: {skipped_no_bracket}")
    print(f"    Not needed: {skipped_not_needed}")
    print(f"    Candidates: {sum(len(v) for v in candidates_by_bracket.values())}")

    # ── Step 5: Select diverse high-quality images ────────────────────────────
    print("\n[Step 5] Selecting diverse images per bracket...")
    total_added = 0
    all_new = []

    for label, start, end, folder in AGE_BRACKETS:
        needed = needs_fill.get(label, 0)
        if needed == 0:
            continue

        candidates = candidates_by_bracket.get(label, [])
        if not candidates:
            print(f"  {folder}: need {needed}, 0 candidates")
            continue

        # Sort by quality (best first)
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # Greedy diverse selection
        existing_vecs = list(bracket_vectors.get(label, []))
        selected = []

        for cand in candidates:
            is_too_similar = False
            if existing_vecs:
                existing_mat = np.array(existing_vecs, dtype=np.float32)
                sims = existing_mat @ cand["vector"]
                if np.max(sims) >= SIMILARITY_THRESHOLD:
                    is_too_similar = True

            if is_too_similar:
                continue

            selected.append(cand)
            existing_vecs.append(cand["vector"])
            if len(selected) >= needed:
                break

        # Copy to presentation
        added = 0
        for cand in selected:
            pres_name = f"{folder}__{cand['filename']}"
            dest = unique_dest_path(PRESENTATION_DIR, pres_name)
            try:
                shutil.copy2(cand["path"], dest)
                cand["pres_filename"] = os.path.basename(dest)
                all_new.append(cand)
                added += 1
                total_added += 1
            except OSError as e:
                print(f"    WARN: {cand['filename']}: {e}")

        print(f"  {folder}: +{added}/{needed} needed ({len(candidates)} candidates)")

    # ── Step 6: Update vector map ─────────────────────────────────────────────
    print("\n[Step 6] Updating vector map...")
    # Load existing vectors
    existing_data = []
    if os.path.exists(VECTORS_FILE):
        with open(VECTORS_FILE, "rb") as f:
            existing_data = pickle.load(f)

    for cand in all_new:
        existing_data.append({
            "filename": cand["pres_filename"],
            "bracket": cand["bracket"],
            "score": cand["score"],
            "hash": cand["hash"],
            "vector": cand["vector"].tolist(),
        })

    with open(VECTORS_FILE, "wb") as f:
        pickle.dump(existing_data, f)
    print(f"  Saved {len(existing_data)} total vectors")

    # Update JSON map
    summary_file = os.path.join(PROJECT_DIR, "presentation_map.json")
    summary = [{"filename": d["filename"], "bracket": d["bracket"],
                "score": round(d["score"], 1)} for d in existing_data]
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ── Final Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY (after disk fill)")
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

    print(f"\n  {'Bracket':<30} {'Before':>8} {'Added':>8} {'Final':>8}")
    print("  " + "-" * 58)
    total_before = 0
    total_final = 0
    for label, start, end, folder in AGE_BRACKETS:
        before = len(bracket_files.get(label, []))
        added_now = sum(1 for d in all_new if d["bracket"] == label)
        final = final_by_bracket.get(label, 0)
        total_before += before
        total_final += final
        marker = ""
        if final < 50:
            marker = " << LOW"
        elif final >= TARGET_PER_AGE:
            marker = " OK"
        print(f"  {folder:<30} {before:>8} {added_now:>8} {final:>8}{marker}")

    print("  " + "-" * 58)
    print(f"  {'TOTAL':<30} {total_before:>8} {total_added:>8} {total_final:>8}")
    print(f"\n  New images from disk: {total_added}")
    print(f"  Total presentation images: {len(final_files)}")


if __name__ == "__main__":
    main()
