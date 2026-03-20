"""
Visual dedup v2 - stricter approach:
1. Only group images within the SAME age folder (no cross-age merging)
2. Use 0.95 threshold for cosine similarity
3. Verify each pair directly (no transitive chaining)
4. Rebuild from source: re-select best diverse images to fill each bracket
"""

import os
import sys
import json
import re
import shutil
import hashlib
import numpy as np
from datetime import datetime, timedelta
from PIL import Image
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)

PROJECT_DIR = r"C:\Codes\Reef images for bar mitza"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "sorted")
ONEDRIVE_DIR = r"C:\Users\erezg\OneDrive - Pen-Link Ltd\Desktop\אישי\ריף\בר מצווה"
TAKEOUT_DIR = os.path.join(ONEDRIVE_DIR, r"download\extracted\Takeout\Google Photos\reef")
MANIFEST_FILE = os.path.join(PROJECT_DIR, "manifest.json")
REEF_BIRTHDAY = datetime(2013, 7, 16)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".heic", ".webp"}
SIMILARITY_THRESHOLD = 0.95
TARGET_PER_FOLDER = 50  # Target after dedup
VECTOR_SIZE = 64

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

# Also handle original folders from first sort
EXTRA_FOLDERS = ["2013", "2014", "2015", "2026"]


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
        if ratio < 0.08:
            return True
    except Exception:
        pass
    return False


def file_hash(filepath):
    h = hashlib.md5(usedforsecurity=False)
    h.update(str(os.path.getsize(filepath)).encode())
    with open(filepath, "rb") as f:
        h.update(f.read(65536))
    return h.hexdigest()


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


def dedup_within_folder(folder_path):
    """
    Dedup images within a single folder.
    Returns list of filenames to remove.
    """
    files = sorted([f for f in os.listdir(folder_path)
                   if os.path.splitext(f)[1].lower() in IMAGE_EXTS])
    if len(files) < 2:
        return []

    # Compute vectors
    vectors = []
    valid_files = []
    for fname in files:
        fpath = os.path.join(folder_path, fname)
        vec = compute_vector(fpath)
        if vec is not None:
            vectors.append(vec)
            valid_files.append(fname)

    if len(vectors) < 2:
        return []

    vectors = np.array(vectors, dtype=np.float32)

    # Compute pairwise similarity
    sim = vectors @ vectors.T

    # Find pairs above threshold - NO union-find, just direct pairs
    # Mark images to remove (keep the better quality one)
    to_remove = set()

    for i in range(len(valid_files)):
        if valid_files[i] in to_remove:
            continue
        for j in range(i + 1, len(valid_files)):
            if valid_files[j] in to_remove:
                continue
            if sim[i, j] >= SIMILARITY_THRESHOLD:
                # Keep better quality
                qi = quality_score(os.path.join(folder_path, valid_files[i]))
                qj = quality_score(os.path.join(folder_path, valid_files[j]))
                if qi >= qj:
                    to_remove.add(valid_files[j])
                else:
                    to_remove.add(valid_files[i])
                    break  # i got removed, skip remaining j's

    # Also remove screenshots
    for fname in valid_files:
        if fname in to_remove:
            continue
        if is_screenshot(os.path.join(folder_path, fname)):
            to_remove.add(fname)

    return list(to_remove)


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
    print("Visual Dedup v2 + Refill")
    print("=" * 60)

    # ── Step 1: First restore all images from sources ──────────────────────────
    # Since v1 over-deleted, let's rebuild from scratch by re-running the
    # takeout selector. But first, let's just work with what we have and refill.

    # ── Step 2: Dedup within each folder ───────────────────────────────────────
    print("\nStep 1: Dedup within each age folder (threshold=0.95)...")
    all_removals = {}  # folder -> [filenames]
    total_removed = 0

    all_folders = [f[3] for f in AGE_BRACKETS] + EXTRA_FOLDERS
    for folder_name in all_folders:
        folder_path = os.path.join(OUTPUT_DIR, folder_name)
        if not os.path.exists(folder_path):
            continue

        current_count = sum(1 for f in os.listdir(folder_path)
                          if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
        if current_count == 0:
            continue

        removals = dedup_within_folder(folder_path)
        if removals:
            all_removals[folder_name] = removals
            total_removed += len(removals)
            print(f"  {folder_name}: remove {len(removals)}/{current_count}")

    print(f"\nTotal duplicates found: {total_removed}")

    # ── Step 3: Remove duplicates ──────────────────────────────────────────────
    print("\nStep 2: Removing duplicates...")
    pres_sorted = os.path.join(OUTPUT_DIR, "the presentation")
    pres_onedrive = os.path.join(ONEDRIVE_DIR, "the presentation")

    for folder_name, removals in all_removals.items():
        for fname in removals:
            # Remove from sorted
            p = os.path.join(OUTPUT_DIR, folder_name, fname)
            if os.path.exists(p):
                os.remove(p)
            # Remove from OneDrive
            p = os.path.join(ONEDRIVE_DIR, folder_name, fname)
            if os.path.exists(p):
                os.remove(p)
            # Remove from presentation
            pres_name = f"{folder_name}__{fname}"
            for d in [pres_sorted, pres_onedrive]:
                if d and os.path.exists(d):
                    p = os.path.join(d, pres_name)
                    if os.path.exists(p):
                        os.remove(p)

    # ── Step 4: Refill from takeout ────────────────────────────────────────────
    print("\nStep 3: Refilling from takeout to reach target per folder...")

    # Collect hashes of remaining images
    existing_hashes = set()
    for dirpath, _, filenames in os.walk(OUTPUT_DIR):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTS:
                try:
                    existing_hashes.add(file_hash(os.path.join(dirpath, fname)))
                except:
                    pass

    print(f"  Remaining images: {len(existing_hashes)}")

    # Pre-compute vectors for remaining images per folder (for diversity check)
    total_refilled = 0

    for bracket_label, start_days, end_days, folder_name in AGE_BRACKETS:
        folder_path = os.path.join(OUTPUT_DIR, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        current = sum(1 for f in os.listdir(folder_path)
                     if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
        needed = max(0, TARGET_PER_FOLDER - current)

        if needed == 0:
            continue

        # Compute vectors for existing images in this folder
        existing_vectors = []
        existing_files = [f for f in os.listdir(folder_path)
                         if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
        for fname in existing_files:
            vec = compute_vector(os.path.join(folder_path, fname))
            if vec is not None:
                existing_vectors.append(vec)

        # Find candidates from takeout
        candidates = []
        if os.path.exists(TAKEOUT_DIR):
            for fname in os.listdir(TAKEOUT_DIR):
                if os.path.splitext(fname)[1].lower() not in IMAGE_EXTS:
                    continue
                fpath = os.path.join(TAKEOUT_DIR, fname)

                try:
                    fhash = file_hash(fpath)
                except:
                    continue
                if fhash in existing_hashes:
                    continue

                # Check date
                dt = get_json_date(fpath) or get_exif_date(fpath)
                if not dt:
                    continue
                age_days = (dt - REEF_BIRTHDAY).days
                if not (start_days <= age_days < end_days):
                    continue

                # Skip screenshots
                if is_screenshot(fpath):
                    continue

                # Check file quality
                try:
                    sz = os.path.getsize(fpath)
                    if sz < 80 * 1024:
                        continue
                    img = Image.open(fpath)
                    w, h = img.size
                    img.close()
                    if w < 600 and h < 600:
                        continue
                except:
                    continue

                vec = compute_vector(fpath)
                if vec is None:
                    continue

                # Check similarity against existing images in folder
                is_dup = False
                if existing_vectors:
                    existing_mat = np.array(existing_vectors, dtype=np.float32)
                    sims = existing_mat @ vec
                    if np.max(sims) >= SIMILARITY_THRESHOLD:
                        is_dup = True

                if is_dup:
                    continue

                score = quality_score(fpath)
                candidates.append({
                    "path": fpath,
                    "hash": fhash,
                    "filename": fname,
                    "score": score,
                    "vector": vec,
                    "date": dt,
                })

        # Select diverse candidates: sort by date, pick evenly spread
        candidates.sort(key=lambda x: x["date"])
        selected = []
        if len(candidates) > needed:
            step = len(candidates) / needed
            selected = [candidates[int(i * step)] for i in range(needed)]
        else:
            selected = candidates

        # Final diversity check: don't add candidates too similar to each other
        final_selected = []
        for cand in selected:
            is_dup = False
            for prev in final_selected:
                sim = float(np.dot(cand["vector"], prev["vector"]))
                if sim >= SIMILARITY_THRESHOLD:
                    is_dup = True
                    break
            if not is_dup:
                final_selected.append(cand)
                existing_vectors.append(cand["vector"])

        # Copy selected
        added = 0
        for cand in final_selected:
            dest = unique_dest_path(folder_path, cand["filename"])
            try:
                shutil.copy2(cand["path"], dest)
                # OneDrive
                od_dir = os.path.join(ONEDRIVE_DIR, folder_name)
                os.makedirs(od_dir, exist_ok=True)
                od_dest = unique_dest_path(od_dir, cand["filename"])
                shutil.copy2(cand["path"], od_dest)
                # Presentation
                pres_name = f"{folder_name}__{cand['filename']}"
                for d in [pres_sorted, pres_onedrive]:
                    if d:
                        os.makedirs(d, exist_ok=True)
                        shutil.copy2(cand["path"], os.path.join(d, pres_name))
                existing_hashes.add(cand["hash"])
                added += 1
            except OSError as e:
                print(f"    WARN: {cand['filename']}: {e}")
                continue

        total_refilled += added
        final_count = sum(1 for f in os.listdir(folder_path)
                         if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
        if added > 0 or needed > 0:
            print(f"  {folder_name}: was {current}, needed {needed}, added {added}, now {final_count}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Duplicates removed: {total_removed}")
    print(f"New diverse images added: {total_refilled}")
    print(f"\nFolder counts:")
    total_all = 0
    for bracket_label, _, _, folder_name in AGE_BRACKETS:
        folder_path = os.path.join(OUTPUT_DIR, folder_name)
        if os.path.exists(folder_path):
            count = sum(1 for f in os.listdir(folder_path)
                       if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
            total_all += count
            print(f"  {folder_name}: {count}")

    for folder_name in EXTRA_FOLDERS:
        folder_path = os.path.join(OUTPUT_DIR, folder_name)
        if os.path.exists(folder_path):
            count = sum(1 for f in os.listdir(folder_path)
                       if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
            if count > 0:
                total_all += count
                print(f"  {folder_name}: {count}")

    pres_count = 0
    if os.path.exists(pres_sorted):
        pres_count = sum(1 for f in os.listdir(pres_sorted)
                        if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
    print(f"\n  the presentation: {pres_count}")
    print(f"  total sorted: {total_all}")


if __name__ == "__main__":
    main()
