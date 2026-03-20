# Curate the bar mitzva presentation folder:
# 1. Vectorize all images (presentation + sources)
# 2. Remove similar/duplicate images from presentation (keep best quality)
# 3. Fill each age bracket to ~75 images from local sources
# 4. Save vector map for future analysis

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
from PIL import Image, ImageFilter

sys.stdout.reconfigure(line_buffering=True)

# ── Configuration ─────────────────────────────────────────────────────────────
ONEDRIVE_DIR = r"C:\Users\erezg\OneDrive - Pen-Link Ltd\Desktop\אישי\ריף\בר מצווה"
PRESENTATION_DIR = os.path.join(ONEDRIVE_DIR, "the presentation")
TAKEOUT_DIR = os.path.join(ONEDRIVE_DIR, r"download\extracted\Takeout\Google Photos\reef")
PROJECT_DIR = r"C:\Codes\Reef images for bar mitza"
VECTORS_FILE = os.path.join(PROJECT_DIR, "presentation_vectors.pkl")
REMOVED_DIR = os.path.join(PROJECT_DIR, "removed_from_presentation")
REEF_BIRTHDAY = datetime(2013, 7, 16)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".heic", ".webp"}
SIMILARITY_THRESHOLD = 0.94  # cosine similarity to consider "same scene"
TARGET_PER_AGE = 75
VECTOR_SIZE = 64

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

BRACKET_DISPLAY = {b[0]: b[0] for b in AGE_BRACKETS}

# Map folder prefixes in presentation filenames to bracket
PREFIX_TO_BRACKET = {}
for label, start, end, folder in AGE_BRACKETS:
    PREFIX_TO_BRACKET[folder] = label

# Also map old-style folder prefixes (from sort_images.py)
OLD_PREFIX_MAP = {
    "2013": "02_months2-3",   # generic 2013 -> months 2-3 range
    "2014": "04_months7-9",   # generic 2014
    "2015": "07_year2",       # generic 2015
    "2026": "12_barmitzva",
}

# Hebrew directory -> age days
DIR_AGE_MAP = {
    "בית חולים": 0, "כשריף נולד": 0, "ריף מגיע הביתה": 3,
    "ברית": 44, "ריף בן חודש": 30, "לידה": 3,
    "חודש 1": 30, "חודש 2-12": 180,
    "שנה - שנתיים": 545,
    "ריף בן חודשיים": 60, "ריף בן 3 חודשים": 90,
    "ריף בן 4 חודשים": 120, "ריף בן 5 חודשים": 150,
    "ריף בן חצי שנה": 180, "ריף בן 7 חודשים": 210,
    "ריף בן 8 חודשים": 240, "ריף בן 9 חודשים": 270,
    "ריף בן 10 חודשים": 300, "ריף בן 11 חודשים": 330,
    "ריף בן שנה": 365, "שנה וחודש": 395,
    "שנה וחודשיים": 425, "שנה ושלוש": 455,
    "שנה וארבע": 485, "שנה וחמש": 515,
    "שנה וחצי": 545, "שנה ושבע": 575,
    "שנה ושמונה": 605, "שנה ותשע": 635,
    "שנה ועשר": 665, "mix": None,
}


def compute_vector(image_path):
    """Feature vector: grayscale pixels + color histogram."""
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
    """Higher = better quality (resolution * file_size)."""
    try:
        sz = os.path.getsize(image_path)
        img = Image.open(image_path)
        w, h = img.size
        img.close()
        return w * h * (sz / 1024)
    except Exception:
        return 0


def is_screenshot(image_path):
    """Detect non-photo images (screenshots, game captures, UI)."""
    try:
        img = Image.open(image_path).convert("RGB").resize((256, 256))
        arr = np.array(img, dtype=np.float64)
        pixels = arr.reshape(-1, 3)
        unique = len(set(map(tuple, pixels.astype(int).tolist())))
        ratio = unique / len(pixels)
        if ratio < 0.08:
            return True, f"too few colors ({ratio:.2f})"
        # PNG with low variance
        ext = os.path.splitext(image_path)[1].lower()
        if ext == ".png":
            from PIL import ImageStat
            stat = ImageStat.Stat(img)
            avg_std = sum(stat.stddev) / 3
            if avg_std < 40 and ratio < 0.12:
                return True, f"PNG low variance ({avg_std:.0f})"
        return False, ""
    except Exception:
        return False, ""


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
    """Parse presentation filename (folder__original.jpg) to determine age bracket."""
    parts = filename.split("__", 1)
    if len(parts) != 2:
        return None
    prefix = parts[0]
    if prefix in PREFIX_TO_BRACKET:
        return PREFIX_TO_BRACKET[prefix]
    if prefix in OLD_PREFIX_MAP:
        return OLD_PREFIX_MAP[prefix]
    return None


def dir_path_to_age_days(rel_path):
    """Estimate age from directory path using Hebrew folder name hints."""
    parts = rel_path.replace("\\", "/").split("/")
    for part in parts:
        for pattern, age_days in DIR_AGE_MAP.items():
            if pattern in part:
                return age_days
    return None


def determine_bracket(filepath, rel_dir=None):
    """Determine age bracket from all available signals."""
    # 1. EXIF date
    dt = get_exif_date(filepath)
    if dt:
        age_days = (dt - REEF_BIRTHDAY).days
        bracket = age_days_to_bracket(age_days)
        if bracket:
            return bracket, dt

    # 2. JSON sidecar (takeout)
    dt = get_json_date(filepath)
    if dt:
        age_days = (dt - REEF_BIRTHDAY).days
        bracket = age_days_to_bracket(age_days)
        if bracket:
            return bracket, dt

    # 3. Filename date
    dt = get_filename_date(os.path.basename(filepath))
    if dt:
        age_days = (dt - REEF_BIRTHDAY).days
        bracket = age_days_to_bracket(age_days)
        if bracket:
            return bracket, dt

    # 4. Directory hint
    if rel_dir:
        age_days = dir_path_to_age_days(rel_dir)
        if age_days is not None:
            bracket = age_days_to_bracket(age_days)
            if bracket:
                return bracket, REEF_BIRTHDAY + timedelta(days=age_days)

    return None, None


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


# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  PRESENTATION CURATOR")
    print("  Target: ~75 diverse, high-quality images per age bracket")
    print("=" * 70)

    os.makedirs(REMOVED_DIR, exist_ok=True)

    # ── Step 1: Vectorize current presentation images ─────────────────────────
    print("\n[Step 1] Vectorizing presentation images...")
    pres_files = sorted([f for f in os.listdir(PRESENTATION_DIR)
                         if os.path.splitext(f)[1].lower() in IMAGE_EXTS])
    print(f"  Found {len(pres_files)} images in presentation")

    pres_data = []  # list of dicts with all image metadata
    for i, fname in enumerate(pres_files):
        fpath = os.path.join(PRESENTATION_DIR, fname)
        vec = compute_vector(fpath)
        bracket = presentation_name_to_bracket(fname)
        score = quality_score(fpath)
        is_ss, ss_reason = is_screenshot(fpath)
        fhash = file_hash(fpath)

        pres_data.append({
            "filename": fname,
            "path": fpath,
            "vector": vec,
            "bracket": bracket,
            "score": score,
            "hash": fhash,
            "is_screenshot": is_ss,
            "ss_reason": ss_reason,
        })
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(pres_files)} processed...")

    # ── Step 2: Remove screenshots / non-photos ──────────────────────────────
    print("\n[Step 2] Identifying non-photo images (screenshots, games, etc.)...")
    screenshots = [d for d in pres_data if d["is_screenshot"]]
    print(f"  Found {len(screenshots)} non-photo images:")
    for d in screenshots:
        print(f"    REMOVE: {d['filename']} ({d['ss_reason']})")

    # Move screenshots to removed folder
    for d in screenshots:
        src = d["path"]
        dst = os.path.join(REMOVED_DIR, "screenshot__" + d["filename"])
        if os.path.exists(src):
            shutil.move(src, dst)
    remaining = [d for d in pres_data if not d["is_screenshot"]]
    print(f"  Remaining after screenshot removal: {len(remaining)}")

    # ── Step 3: Find and remove similar images within presentation ────────────
    print("\n[Step 3] Finding similar image pairs (threshold={:.2f})...".format(SIMILARITY_THRESHOLD))
    valid = [d for d in remaining if d["vector"] is not None]
    if len(valid) < 2:
        print("  Not enough valid images for similarity check")
        similar_removed = []
    else:
        vectors = np.array([d["vector"] for d in valid], dtype=np.float32)
        sim_matrix = vectors @ vectors.T

        # Group by bracket first, then find similarities
        to_remove_indices = set()
        pairs_found = 0

        for i in range(len(valid)):
            if i in to_remove_indices:
                continue
            for j in range(i + 1, len(valid)):
                if j in to_remove_indices:
                    continue
                if sim_matrix[i, j] >= SIMILARITY_THRESHOLD:
                    pairs_found += 1
                    # Keep the one with higher quality score
                    if valid[i]["score"] >= valid[j]["score"]:
                        to_remove_indices.add(j)
                    else:
                        to_remove_indices.add(i)
                        break  # i is removed, stop comparing

        similar_removed = [valid[i] for i in to_remove_indices]
        print(f"  Found {pairs_found} similar pairs, removing {len(similar_removed)} lower-quality duplicates:")
        for d in similar_removed:
            print(f"    REMOVE: {d['filename']} (score={d['score']:.0f})")

        # Move to removed folder
        for d in similar_removed:
            src = d["path"]
            dst = os.path.join(REMOVED_DIR, "similar__" + d["filename"])
            if os.path.exists(src):
                shutil.move(src, dst)

    removed_filenames = set(d["filename"] for d in screenshots + similar_removed)
    remaining = [d for d in remaining if d["filename"] not in removed_filenames]
    print(f"  Remaining after dedup: {len(remaining)}")

    # ── Step 4: Count per bracket ─────────────────────────────────────────────
    print("\n[Step 4] Current images per age bracket:")
    bracket_counts = defaultdict(list)
    no_bracket = []
    for d in remaining:
        if d["bracket"]:
            bracket_counts[d["bracket"]].append(d)
        else:
            no_bracket.append(d)

    needs_fill = {}
    for label, start, end, folder in AGE_BRACKETS:
        count = len(bracket_counts.get(label, []))
        needed = max(0, TARGET_PER_AGE - count)
        status = "OK" if count >= TARGET_PER_AGE else f"need {needed} more"
        print(f"  {folder:<30} {count:>4} images  ({status})")
        if needed > 0:
            needs_fill[label] = needed

    if no_bracket:
        print(f"  Unknown bracket: {len(no_bracket)} images")

    total_needed = sum(needs_fill.values())
    print(f"\n  Total images needed: {total_needed}")

    if total_needed == 0:
        print("\n  All brackets are full! Nothing to fill.")
        _save_vectors(remaining)
        return

    # ── Step 5: Scan local source folders for candidates ──────────────────────
    print("\n[Step 5] Scanning local source folders for candidates...")

    # Collect hashes of images already in presentation
    existing_hashes = set(d["hash"] for d in remaining)
    existing_vectors = [d["vector"] for d in remaining if d["vector"] is not None]

    # Source directories to scan
    source_dirs = [
        (os.path.join(ONEDRIVE_DIR, "לידה"), "לידה"),
        (os.path.join(ONEDRIVE_DIR, "חודש 1"), "חודש 1"),
        (os.path.join(ONEDRIVE_DIR, "חודש 2-12"), "חודש 2-12"),
        (os.path.join(ONEDRIVE_DIR, "שנה - שנתיים"), "שנה - שנתיים"),
        (os.path.join(ONEDRIVE_DIR, "mix"), "mix"),
        (os.path.join(ONEDRIVE_DIR, "extra images"), "extra images"),
    ]
    # Add age-based folders from OneDrive
    for label, start, end, folder in AGE_BRACKETS:
        d = os.path.join(ONEDRIVE_DIR, folder)
        if os.path.exists(d):
            source_dirs.append((d, folder))

    # Add takeout (download/extracted)
    if os.path.exists(TAKEOUT_DIR):
        source_dirs.append((TAKEOUT_DIR, "takeout"))

    candidates_by_bracket = defaultdict(list)
    scanned = 0
    skipped_dup = 0
    skipped_quality = 0
    skipped_no_bracket = 0

    seen_hashes = set(existing_hashes)  # track across sources to avoid re-scanning

    for source_path, source_name in source_dirs:
        if not os.path.exists(source_path):
            continue

        # Walk recursively to catch nested subfolders
        for dirpath, _, filenames in os.walk(source_path):
            rel_dir = os.path.relpath(dirpath, ONEDRIVE_DIR)
            for fname in filenames:
                if os.path.splitext(fname)[1].lower() not in IMAGE_EXTS:
                    continue

                fpath = os.path.join(dirpath, fname)
                scanned += 1
                if scanned % 500 == 0:
                    print(f"    Scanned {scanned} images...")

                # Hash-based dedup
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
                    if file_size < 50 * 1024:
                        skipped_quality += 1
                        continue
                    img = Image.open(fpath)
                    w, h = img.size
                    img.close()
                    if w < 500 and h < 500:
                        skipped_quality += 1
                        continue
                except Exception:
                    skipped_quality += 1
                    continue

                # Screenshot check
                is_ss, _ = is_screenshot(fpath)
                if is_ss:
                    skipped_quality += 1
                    continue

                # Determine bracket (use rel_dir for directory hints)
                bracket, dt = determine_bracket(fpath, rel_dir)
                if not bracket:
                    skipped_no_bracket += 1
                    continue

                # Only collect for brackets that need filling
                if bracket not in needs_fill:
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
                    "date": dt,
                    "source": source_name,
                })

    print(f"\n  Scan complete: {scanned} images scanned")
    print(f"    Duplicates: {skipped_dup}")
    print(f"    Low quality/screenshots: {skipped_quality}")
    print(f"    No bracket/not needed: {skipped_no_bracket}")
    print(f"    Candidates: {sum(len(v) for v in candidates_by_bracket.values())}")

    # ── Step 6: Select diverse, high-quality images to fill each bracket ──────
    print("\n[Step 6] Selecting diverse images to fill each bracket...")

    total_added = 0
    all_new = []

    for label, start, end, folder in AGE_BRACKETS:
        needed = needs_fill.get(label, 0)
        if needed == 0:
            continue

        candidates = candidates_by_bracket.get(label, [])
        if not candidates:
            print(f"  {folder}: need {needed}, but 0 candidates available")
            continue

        # Get existing vectors for this bracket (for diversity check)
        bracket_existing_vecs = [d["vector"] for d in bracket_counts.get(label, [])
                                 if d["vector"] is not None]

        # Sort candidates by quality
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # Select greedily: pick highest quality that is diverse from already selected
        selected = []
        selected_vecs = list(bracket_existing_vecs)

        for cand in candidates:
            # Check similarity against existing + already selected
            is_too_similar = False
            if selected_vecs:
                existing_mat = np.array(selected_vecs, dtype=np.float32)
                sims = existing_mat @ cand["vector"]
                if np.max(sims) >= SIMILARITY_THRESHOLD:
                    is_too_similar = True

            if is_too_similar:
                continue

            selected.append(cand)
            selected_vecs.append(cand["vector"])
            if len(selected) >= needed:
                break

        # Copy selected images to presentation
        added = 0
        for cand in selected:
            pres_name = f"{folder}__{cand['filename']}"
            dest = unique_dest_path(PRESENTATION_DIR, pres_name)
            try:
                shutil.copy2(cand["path"], dest)
                existing_hashes.add(cand["hash"])
                cand["pres_filename"] = os.path.basename(dest)
                cand["pres_path"] = dest
                all_new.append(cand)
                added += 1
                total_added += 1
            except OSError as e:
                print(f"    WARN: could not copy {cand['filename']}: {e}")

        print(f"  {folder}: +{added}/{needed} needed ({len(candidates)} candidates)")

    # ── Step 7: Save vectors for mapping ──────────────────────────────────────
    print("\n[Step 7] Saving vector map...")
    all_images = remaining + all_new
    _save_vectors(all_images)

    # ── Final Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)

    # Recount
    final_files = [f for f in os.listdir(PRESENTATION_DIR)
                   if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
    final_by_bracket = defaultdict(int)
    for f in final_files:
        b = presentation_name_to_bracket(f)
        if b:
            final_by_bracket[b] += 1
        else:
            final_by_bracket["unknown"] += 1

    print(f"\n{'Age Bracket':<30} {'Before':>8} {'Removed':>8} {'Added':>8} {'Final':>8}")
    print("-" * 70)
    total_final = 0
    for label, start, end, folder in AGE_BRACKETS:
        before = len(bracket_counts.get(label, []))
        removed = sum(1 for d in screenshots + similar_removed if d.get("bracket") == label)
        added = sum(1 for d in all_new if d["bracket"] == label)
        final = final_by_bracket.get(label, 0)
        total_final += final
        marker = ""
        if final < 50:
            marker = " << LOW"
        elif final >= TARGET_PER_AGE:
            marker = " OK"
        print(f"  {folder:<28} {before:>8} {removed:>8} {added:>8} {final:>8}{marker}")

    unknown = final_by_bracket.get("unknown", 0)
    if unknown:
        print(f"  {'(unknown bracket)':<28} {'':>8} {'':>8} {'':>8} {unknown:>8}")
        total_final += unknown

    print("-" * 70)
    print(f"  {'TOTAL':<28} {len(pres_data):>8} {len(screenshots) + len(similar_removed):>8} {total_added:>8} {total_final:>8}")

    print(f"\n  Screenshots removed: {len(screenshots)}")
    print(f"  Similar duplicates removed: {len(similar_removed)}")
    print(f"  New diverse images added: {total_added}")
    print(f"  Removed images saved to: {REMOVED_DIR}")
    print(f"  Vector map saved to: {VECTORS_FILE}")


def _save_vectors(all_images):
    """Save image vectors + metadata for future mapping."""
    save_data = []
    for d in all_images:
        entry = {
            "filename": d.get("pres_filename", d.get("filename")),
            "bracket": d.get("bracket"),
            "score": d.get("score", 0),
            "hash": d.get("hash"),
        }
        if d.get("vector") is not None:
            entry["vector"] = d["vector"].tolist()
        save_data.append(entry)

    with open(VECTORS_FILE, "wb") as f:
        pickle.dump(save_data, f)
    print(f"  Saved {len(save_data)} image vectors to {VECTORS_FILE}")

    # Also save a human-readable JSON summary (without vectors)
    summary_file = os.path.join(PROJECT_DIR, "presentation_map.json")
    summary = []
    for d in save_data:
        summary.append({
            "filename": d["filename"],
            "bracket": d["bracket"],
            "score": round(d["score"], 1),
        })
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  Saved human-readable map to {summary_file}")


if __name__ == "__main__":
    main()
