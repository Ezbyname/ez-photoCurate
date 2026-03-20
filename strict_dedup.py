# Strict dedup of presentation folder:
# 1. Group by visual similarity (0.85 threshold) AND sequential filenames
# 2. Keep only the best quality from each group
# 3. Refill from sources (OneDrive + takeout + disk) to maintain ~75 per bracket

import os
import sys
import re
import shutil
import hashlib
import pickle
import json
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)

ONEDRIVE_DIR = r"C:\Users\erezg\OneDrive - Pen-Link Ltd\Desktop\אישי\ריף\בר מצווה"
PRESENTATION_DIR = os.path.join(ONEDRIVE_DIR, "the presentation")
TAKEOUT_DIR = os.path.join(ONEDRIVE_DIR, r"download\extracted\Takeout\Google Photos\reef")
DISK_DIR = r"D:\ריף"
PROJECT_DIR = r"C:\Codes\Reef images for bar mitza"
VECTORS_FILE = os.path.join(PROJECT_DIR, "presentation_vectors.pkl")
REMOVED_DIR = os.path.join(PROJECT_DIR, "removed_from_presentation")
REEF_BIRTHDAY = datetime(2013, 7, 16)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".heic", ".webp"}
VECTOR_SIZE = 64
TARGET_PER_AGE = 75

# Thresholds
SIM_THRESHOLD = 0.85       # cosine similarity for visual dedup
BURST_SIM_THRESHOLD = 0.55  # lower threshold if filenames are sequential
BURST_NUM_GAP = 30          # max gap in sequential numbering

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
    "ipad": 730, "תיקיה חדשה": 60,
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


def extract_number(filename):
    """Extract the trailing number from a filename like IMG_4421.JPG -> 4421, Nitzan-Zohar-9716.jpg -> 9716."""
    # Get just the original filename (after __)
    parts = filename.split("__", 1)
    name = parts[1] if len(parts) == 2 else parts[0]
    base = os.path.splitext(name)[0]
    # Find last sequence of digits
    nums = re.findall(r'(\d+)', base)
    if nums:
        return int(nums[-1])
    return None


def extract_name_prefix(filename):
    """Extract prefix before number: IMG_4421.JPG -> IMG_, Nitzan-Zohar-9716.jpg -> Nitzan-Zohar-"""
    parts = filename.split("__", 1)
    name = parts[1] if len(parts) == 2 else parts[0]
    base = os.path.splitext(name)[0]
    m = re.match(r'^(.*?)(\d+)$', base)
    if m:
        return m.group(1)
    return base


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


def determine_bracket(filepath, rel_dir=None):
    dt = get_exif_date(filepath)
    if dt:
        age_days = (dt - REEF_BIRTHDAY).days
        bracket = age_days_to_bracket(age_days)
        if bracket:
            return bracket, dt
    dt = get_json_date(filepath)
    if dt:
        age_days = (dt - REEF_BIRTHDAY).days
        bracket = age_days_to_bracket(age_days)
        if bracket:
            return bracket, dt
    dt = get_filename_date(os.path.basename(filepath))
    if dt:
        age_days = (dt - REEF_BIRTHDAY).days
        bracket = age_days_to_bracket(age_days)
        if bracket:
            return bracket, dt
    if rel_dir:
        age_days = dir_to_age_days(rel_dir)
        if age_days is not None:
            bracket = age_days_to_bracket(age_days)
            if bracket:
                return bracket, REEF_BIRTHDAY + timedelta(days=age_days)
    return None, None


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


def main():
    print("=" * 70)
    print("  STRICT DEDUP + REFILL")
    print("  Visual similarity (0.85) + sequential filename grouping")
    print("=" * 70)

    os.makedirs(REMOVED_DIR, exist_ok=True)

    # ── Step 1: Load & vectorize all presentation images ──────────────────────
    print("\n[Step 1] Vectorizing presentation images...")
    pres_files = sorted([f for f in os.listdir(PRESENTATION_DIR)
                         if os.path.splitext(f)[1].lower() in IMAGE_EXTS])
    print(f"  {len(pres_files)} images in presentation")

    images = []
    for i, fname in enumerate(pres_files):
        fpath = os.path.join(PRESENTATION_DIR, fname)
        vec = compute_vector(fpath)
        bracket = presentation_name_to_bracket(fname)
        score = quality_score(fpath)
        num = extract_number(fname)
        prefix = extract_name_prefix(fname)

        images.append({
            "idx": i,
            "filename": fname,
            "path": fpath,
            "vector": vec,
            "bracket": bracket,
            "score": score,
            "num": num,
            "prefix": prefix,
            "hash": file_hash(fpath),
        })
        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{len(pres_files)}...")

    print(f"  Vectorized {sum(1 for im in images if im['vector'] is not None)}/{len(images)}")

    # ── Step 2: Build similarity + burst groups per bracket ───────────────────
    print("\n[Step 2] Finding similar pairs (visual + sequential)...")

    # Group images by bracket
    by_bracket = defaultdict(list)
    for im in images:
        if im["bracket"]:
            by_bracket[im["bracket"]].append(im)

    # Union-find
    parent = {im["idx"]: im["idx"] for im in images}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    total_pairs = 0
    burst_pairs = 0
    visual_pairs = 0

    for bracket, bimages in by_bracket.items():
        n = len(bimages)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = bimages[i], bimages[j]

                # Check if sequential filenames (same prefix, close numbers)
                is_sequential = False
                if (a["prefix"] and b["prefix"] and a["prefix"] == b["prefix"]
                        and a["num"] is not None and b["num"] is not None):
                    gap = abs(a["num"] - b["num"])
                    if gap <= BURST_NUM_GAP:
                        is_sequential = True

                # Compute visual similarity
                if a["vector"] is not None and b["vector"] is not None:
                    sim = float(a["vector"] @ b["vector"])
                else:
                    sim = 0.0

                # Decide if they should be grouped
                should_group = False
                if sim >= SIM_THRESHOLD:
                    should_group = True
                    visual_pairs += 1
                elif is_sequential and sim >= BURST_SIM_THRESHOLD:
                    should_group = True
                    burst_pairs += 1

                if should_group:
                    union(a["idx"], b["idx"])
                    total_pairs += 1

    print(f"  Visual pairs (sim >= {SIM_THRESHOLD}): {visual_pairs}")
    print(f"  Burst pairs (sequential + sim >= {BURST_SIM_THRESHOLD}): {burst_pairs}")
    print(f"  Total pairs: {total_pairs}")

    # Build groups
    groups = defaultdict(list)
    for im in images:
        groups[find(im["idx"])].append(im)
    dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"  Groups with duplicates: {len(dup_groups)}")

    # ── Step 3: Keep best from each group, remove rest ────────────────────────
    print("\n[Step 3] Removing duplicates (keeping best quality)...")
    to_remove = []

    for group_id, members in dup_groups.items():
        members.sort(key=lambda x: x["score"], reverse=True)
        keep = members[0]
        for m in members[1:]:
            to_remove.append(m)

        if len(members) <= 6:
            print(f"  Group ({members[0]['bracket']}):")
            print(f"    KEEP: {keep['filename']} (score={keep['score']:.0f})")
            for m in members[1:]:
                sim = float(keep["vector"] @ m["vector"]) if keep["vector"] is not None and m["vector"] is not None else 0
                print(f"    REMOVE: {m['filename']} (sim={sim:.2f})")

    print(f"\n  Total to remove: {len(to_remove)}")

    # Move removed files
    removed_set = set()
    for m in to_remove:
        src = m["path"]
        if os.path.exists(src):
            dst = os.path.join(REMOVED_DIR, "dedup2__" + m["filename"])
            shutil.move(src, dst)
            removed_set.add(m["filename"])

    # ── Step 4: Count what's left per bracket ─────────────────────────────────
    remaining = [im for im in images if im["filename"] not in removed_set]
    bracket_remaining = defaultdict(list)
    for im in remaining:
        if im["bracket"]:
            bracket_remaining[im["bracket"]].append(im)

    print("\n[Step 4] Post-dedup counts:")
    needs_fill = {}
    for label, start, end, folder in AGE_BRACKETS:
        count = len(bracket_remaining.get(label, []))
        needed = max(0, TARGET_PER_AGE - count)
        if needed > 0:
            needs_fill[label] = needed
        status = f"need {needed}" if needed > 0 else "OK"
        print(f"  {folder:<30} {count:>4} ({status})")

    total_needed = sum(needs_fill.values())
    print(f"\n  Total to refill: {total_needed}")

    if total_needed == 0:
        _print_summary(images, removed_set, [], bracket_remaining)
        return

    # ── Step 5: Scan sources for refill candidates ────────────────────────────
    print("\n[Step 5] Scanning sources for refill candidates...")

    existing_hashes = set(im["hash"] for im in remaining)

    # Build source list
    source_dirs = []
    # OneDrive folders
    for name in ["extra images", "mix"]:
        d = os.path.join(ONEDRIVE_DIR, name)
        if os.path.exists(d):
            source_dirs.append((d, name))
    for label, start, end, folder in AGE_BRACKETS:
        d = os.path.join(ONEDRIVE_DIR, folder)
        if os.path.exists(d):
            source_dirs.append((d, folder))
    # Hebrew folders
    for heb in ["לידה", "חודש 1", "חודש 2-12", "שנה - שנתיים"]:
        d = os.path.join(ONEDRIVE_DIR, heb)
        if os.path.exists(d):
            source_dirs.append((d, heb))
    # Takeout
    if os.path.exists(TAKEOUT_DIR):
        source_dirs.append((TAKEOUT_DIR, "takeout"))
    # Disk
    if os.path.exists(DISK_DIR):
        source_dirs.append((DISK_DIR, "disk"))

    candidates_by_bracket = defaultdict(list)
    scanned = 0
    seen_hashes = set(existing_hashes)

    for source_path, source_name in source_dirs:
        if not os.path.exists(source_path):
            continue
        for dirpath, _, filenames in os.walk(source_path):
            rel_dir = os.path.relpath(dirpath, os.path.dirname(source_path))
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
                bracket, dt = determine_bracket(fpath, rel_dir)
                if not bracket or bracket not in needs_fill:
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

    print(f"  Scanned {scanned} images, {sum(len(v) for v in candidates_by_bracket.values())} candidates")

    # ── Step 6: Select diverse refills ────────────────────────────────────────
    print("\n[Step 6] Selecting diverse refills...")
    all_new = []

    for label, start, end, folder in AGE_BRACKETS:
        needed = needs_fill.get(label, 0)
        if needed == 0:
            continue

        candidates = candidates_by_bracket.get(label, [])
        if not candidates:
            print(f"  {folder}: need {needed}, 0 candidates")
            continue

        candidates.sort(key=lambda x: x["score"], reverse=True)

        # Existing vectors for this bracket (for diversity)
        existing_vecs = [im["vector"] for im in bracket_remaining.get(label, [])
                         if im["vector"] is not None]
        selected_vecs = list(existing_vecs)
        selected = []

        for cand in candidates:
            if selected_vecs:
                mat = np.array(selected_vecs, dtype=np.float32)
                sims = mat @ cand["vector"]
                if np.max(sims) >= SIM_THRESHOLD:
                    continue

            selected.append(cand)
            selected_vecs.append(cand["vector"])
            if len(selected) >= needed:
                break

        added = 0
        for cand in selected:
            pres_name = f"{folder}__{cand['filename']}"
            dest = unique_dest_path(PRESENTATION_DIR, pres_name)
            try:
                shutil.copy2(cand["path"], dest)
                cand["pres_filename"] = os.path.basename(dest)
                all_new.append(cand)
                added += 1
            except OSError as e:
                print(f"    WARN: {e}")

        print(f"  {folder}: +{added}/{needed} needed ({len(candidates)} candidates)")

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_summary(images, removed_set, all_new, bracket_remaining)


def _print_summary(images, removed_set, all_new, bracket_remaining):
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

    print(f"\n  {'Bracket':<30} {'Before':>8} {'Removed':>8} {'Added':>8} {'Final':>8}")
    print("  " + "-" * 62)
    total_final = 0
    for label, start, end, folder in AGE_BRACKETS:
        before_count = sum(1 for im in images if im["bracket"] == label)
        removed_count = sum(1 for im in images if im["bracket"] == label and im["filename"] in removed_set)
        added_count = sum(1 for c in all_new if c["bracket"] == label)
        final = final_by_bracket.get(label, 0)
        total_final += final
        marker = " OK" if final >= TARGET_PER_AGE else (" << LOW" if final < 50 else "")
        print(f"  {folder:<30} {before_count:>8} {removed_count:>8} {added_count:>8} {final:>8}{marker}")

    unknown = final_by_bracket.get("unknown", 0)
    if unknown:
        total_final += unknown
        print(f"  {'(unknown)':<30} {'':>8} {'':>8} {'':>8} {unknown:>8}")

    print("  " + "-" * 62)
    print(f"  {'TOTAL':<30} {len(images):>8} {len(removed_set):>8} {len(all_new):>8} {total_final:>8}")
    print(f"\n  Removed images saved to: {REMOVED_DIR}")


if __name__ == "__main__":
    main()
