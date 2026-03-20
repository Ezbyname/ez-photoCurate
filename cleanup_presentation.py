"""
Clean up presentation folder:
1. Remove non-photo images (screenshots, game images, etc.) using image analysis
2. Remove near-duplicate burst shots, keep only best per burst
3. Replace removed images with diverse alternatives from takeout
"""

import os
import sys
import json
import shutil
import hashlib
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from PIL import Image, ImageStat, ImageFilter

sys.stdout.reconfigure(line_buffering=True)

OUTPUT_DIR = r"C:\Codes\Reef images for bar mitza\sorted"
ONEDRIVE_DIR = r"C:\Users\erezg\OneDrive - Pen-Link Ltd\Desktop\אישי\ריף\בר מצווה"
TAKEOUT_DIR = r"C:\Users\erezg\OneDrive - Pen-Link Ltd\Desktop\אישי\ריף\בר מצווה\download\extracted\Takeout\Google Photos\reef"
MANIFEST_FILE = r"C:\Codes\Reef images for bar mitza\manifest.json"
THUMB_DIR = r"C:\Codes\Reef images for bar mitza\thumbnails"
REEF_BIRTHDAY = datetime(2013, 7, 16)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".heic", ".webp"}


def image_histogram_similarity(img1_path, img2_path):
    """Compare two images using histogram correlation. Returns 0-1 (1=identical)."""
    try:
        img1 = Image.open(img1_path).convert("RGB").resize((128, 128))
        img2 = Image.open(img2_path).convert("RGB").resize((128, 128))
        h1 = np.array(img1.histogram(), dtype=np.float64)
        h2 = np.array(img2.histogram(), dtype=np.float64)
        # Normalize
        h1 /= (h1.sum() + 1e-10)
        h2 /= (h2.sum() + 1e-10)
        # Correlation
        return float(np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-10))
    except Exception:
        return 0


def is_screenshot_or_non_photo(image_path):
    """
    Detect screenshots, game images, UI captures - not real photos.
    Returns (is_screenshot, reason).
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img_small = img.resize((256, 256))
        arr = np.array(img_small, dtype=np.float64)

        # Check for very uniform colors (solid backgrounds typical of UI/games)
        # Real photos have more natural color variance
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

        # Check if image has lots of perfectly identical pixels (UI elements)
        pixels = arr.reshape(-1, 3)
        unique_colors = len(set(map(tuple, pixels.astype(int).tolist())))
        color_ratio = unique_colors / len(pixels)

        # Screenshots/game images often have very few unique colors (<15% of pixels)
        if color_ratio < 0.08:
            return True, f"too few colors ({color_ratio:.2f})"

        # Check for sharp edges typical of UI (lots of single-pixel transitions)
        gray = img_small.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_arr = np.array(edges, dtype=np.float64)
        edge_ratio = np.mean(edge_arr > 128)

        # UI/game screenshots have very sharp edges
        if edge_ratio > 0.15 and color_ratio < 0.15:
            return True, f"UI-like edges ({edge_ratio:.2f}) + low colors ({color_ratio:.2f})"

        # Check for text-heavy images (lots of high contrast small regions)
        # PNG screenshots from games/apps
        ext = os.path.splitext(image_path)[1].lower()
        if ext == ".png":
            # PNGs that are screenshots tend to have very uniform areas
            stat = ImageStat.Stat(img_small)
            # Low stddev in channels = uniform/synthetic
            avg_std = sum(stat.stddev) / 3
            if avg_std < 40 and color_ratio < 0.12:
                return True, f"PNG with low variance ({avg_std:.0f}) + low colors"

        return False, ""

    except Exception as e:
        return False, str(e)


def detect_burst_groups(files, folder_path):
    """
    Group files that are burst shots (same timestamp within seconds,
    or sequential filenames).
    Returns list of groups, each group is a list of filenames.
    """
    groups = []
    current_group = []

    for fname in sorted(files):
        if not current_group:
            current_group = [fname]
            continue

        prev = current_group[-1]
        # Check if same date prefix in filename (e.g. IMG_20220428_132345 vs IMG_20220428_132348)
        prev_base = os.path.splitext(prev)[0]
        curr_base = os.path.splitext(fname)[0]

        # Extract timestamp parts
        import re
        prev_match = re.search(r'(\d{8})[_-](\d{6})', prev_base)
        curr_match = re.search(r'(\d{8})[_-](\d{6})', curr_base)

        is_burst = False
        if prev_match and curr_match:
            if prev_match.group(1) == curr_match.group(1):  # Same date
                prev_time = int(prev_match.group(2))
                curr_time = int(curr_match.group(2))
                if abs(curr_time - prev_time) <= 10:  # Within 10 seconds
                    is_burst = True

        # Also check sequential numbering (IMG_1234, IMG_1235)
        if not is_burst:
            prev_num = re.search(r'(\d{3,5})(?:\.\w+)?$', prev_base)
            curr_num = re.search(r'(\d{3,5})(?:\.\w+)?$', curr_base)
            if prev_num and curr_num:
                prev_prefix = prev_base[:prev_num.start()]
                curr_prefix = curr_base[:curr_num.start()]
                if prev_prefix == curr_prefix:
                    diff = abs(int(curr_num.group(1)) - int(prev_num.group(1)))
                    if diff <= 3:
                        # Also verify visual similarity
                        prev_path = os.path.join(folder_path, prev)
                        curr_path = os.path.join(folder_path, fname)
                        sim = image_histogram_similarity(prev_path, curr_path)
                        if sim > 0.95:
                            is_burst = True

        if is_burst:
            current_group.append(fname)
        else:
            if len(current_group) > 1:
                groups.append(current_group)
            current_group = [fname]

    if len(current_group) > 1:
        groups.append(current_group)

    return groups


def get_json_date(image_path):
    json_path = image_path + ".json"
    if not os.path.exists(json_path):
        base = os.path.splitext(image_path)[0]
        json_path = base + ".json"
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
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


def file_hash(filepath):
    h = hashlib.md5(usedforsecurity=False)
    h.update(str(os.path.getsize(filepath)).encode())
    with open(filepath, "rb") as f:
        h.update(f.read(65536))
    return h.hexdigest()


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


def main():
    print("=" * 60)
    print("Presentation Cleanup: Remove non-Reef + Dedup bursts")
    print("=" * 60)

    to_remove = []  # (folder, filename, reason)
    burst_removals = []  # (folder, filename, reason)

    # ── Step 1: Scan each age folder for non-photos ────────────────────────────
    print("\nStep 1: Finding non-photo images (screenshots, games, etc.)...")
    for bracket_label, folder_name in sorted(BRACKET_FOLDER.items(), key=lambda x: x[0]):
        folder_path = os.path.join(OUTPUT_DIR, folder_name)
        if not os.path.exists(folder_path):
            continue

        files = sorted([f for f in os.listdir(folder_path)
                       if os.path.splitext(f)[1].lower() in IMAGE_EXTS])

        for fname in files:
            fpath = os.path.join(folder_path, fname)
            is_ss, reason = is_screenshot_or_non_photo(fpath)
            if is_ss:
                to_remove.append((folder_name, fname, f"non-photo: {reason}"))

    print(f"  Found {len(to_remove)} non-photo images to remove")
    for folder, fname, reason in to_remove:
        print(f"    {folder}/{fname} - {reason}")

    # ── Step 2: Find burst duplicates ──────────────────────────────────────────
    print("\nStep 2: Finding burst duplicate groups...")
    total_burst_removals = 0

    for bracket_label, folder_name in sorted(BRACKET_FOLDER.items(), key=lambda x: x[0]):
        folder_path = os.path.join(OUTPUT_DIR, folder_name)
        if not os.path.exists(folder_path):
            continue

        files = sorted([f for f in os.listdir(folder_path)
                       if os.path.splitext(f)[1].lower() in IMAGE_EXTS])

        # Skip files already marked for removal
        remove_set = set(fname for fn, fname, _ in to_remove if fn == folder_name)
        files = [f for f in files if f not in remove_set]

        burst_groups = detect_burst_groups(files, folder_path)

        for group in burst_groups:
            # Keep the best one (highest resolution * filesize)
            scores = []
            for fname in group:
                fpath = os.path.join(folder_path, fname)
                try:
                    img = Image.open(fpath)
                    w, h = img.size
                    img.close()
                    sz = os.path.getsize(fpath)
                    scores.append((w * h * sz, fname))
                except:
                    scores.append((0, fname))

            scores.sort(reverse=True)
            keep = scores[0][1]
            for _, fname in scores[1:]:
                burst_removals.append((folder_name, fname, f"burst dup of {keep}"))
                total_burst_removals += 1

            if len(group) > 1:
                print(f"    {folder_name}: burst of {len(group)} -> keep {keep}, remove {len(group)-1}")

    print(f"  Found {total_burst_removals} burst duplicates to remove")

    all_removals = to_remove + burst_removals
    print(f"\nTotal to remove: {len(all_removals)}")

    if not all_removals:
        print("Nothing to clean up!")
        return

    # ── Step 3: Remove from sorted folders and presentation ────────────────────
    print("\nStep 3: Removing images...")

    removed_per_folder = defaultdict(int)
    pres_sorted = os.path.join(OUTPUT_DIR, "the presentation")
    pres_onedrive = os.path.join(ONEDRIVE_DIR, "the presentation")

    for folder_name, fname, reason in all_removals:
        # Remove from sorted folder
        src_path = os.path.join(OUTPUT_DIR, folder_name, fname)
        if os.path.exists(src_path):
            os.remove(src_path)

        # Remove from OneDrive age folder
        od_path = os.path.join(ONEDRIVE_DIR, folder_name, fname)
        if os.path.exists(od_path):
            os.remove(od_path)

        # Remove from presentation (prefixed name)
        pres_name = f"{folder_name}__{fname}"
        for pres_dir in [pres_sorted, pres_onedrive]:
            pres_path = os.path.join(pres_dir, pres_name)
            if os.path.exists(pres_path):
                os.remove(pres_path)
            # Also try .jpg version for PNG originals
            pres_jpg = os.path.splitext(pres_path)[0] + ".jpg"
            if os.path.exists(pres_jpg):
                os.remove(pres_jpg)

        # Remove thumbnail
        thumb_name = f"{folder_name}__{os.path.splitext(fname)[0]}.jpg"
        thumb_path = os.path.join(THUMB_DIR, thumb_name)
        if os.path.exists(thumb_path):
            os.remove(thumb_path)

        removed_per_folder[folder_name] += 1

    print("  Removed per folder:")
    for folder, count in sorted(removed_per_folder.items()):
        print(f"    {folder}: {count} removed")

    # ── Step 4: Find replacements from takeout ─────────────────────────────────
    print("\nStep 4: Finding replacement images from takeout...")

    manifest = json.load(open(MANIFEST_FILE, "r", encoding="utf-8"))
    existing_hashes = set(manifest.keys())
    # Add hashes of remaining sorted images
    for dirpath, _, filenames in os.walk(OUTPUT_DIR):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTS:
                try:
                    existing_hashes.add(file_hash(os.path.join(dirpath, fname)))
                except:
                    pass

    total_replaced = 0

    for bracket_label, start_days, end_days in AGE_BRACKETS:
        folder_name = BRACKET_FOLDER[bracket_label]
        folder_path = os.path.join(OUTPUT_DIR, folder_name)
        needed = removed_per_folder.get(folder_name, 0)
        if needed == 0:
            continue

        # Find candidates from takeout in this age range
        candidates = []
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

            # Check not a screenshot
            is_ss, _ = is_screenshot_or_non_photo(fpath)
            if is_ss:
                continue

            try:
                img = Image.open(fpath)
                w, h = img.size
                img.close()
                sz = os.path.getsize(fpath)
            except:
                continue

            candidates.append({
                "path": fpath,
                "hash": fhash,
                "filename": fname,
                "width": w,
                "height": h,
                "score": w * h * sz,
                "date": dt,
            })

        # Sort by quality, pick diverse ones (spread across dates)
        candidates.sort(key=lambda x: x["date"])
        # Take evenly spaced candidates for variety
        if len(candidates) > needed:
            step = len(candidates) / needed
            selected = [candidates[int(i * step)] for i in range(needed)]
        else:
            selected = candidates[:needed]

        for img in selected:
            dest_path = os.path.join(folder_path, img["filename"])
            if os.path.exists(dest_path):
                continue

            shutil.copy2(img["path"], dest_path)

            # Copy to OneDrive
            od_dest = os.path.join(ONEDRIVE_DIR, folder_name, img["filename"])
            os.makedirs(os.path.dirname(od_dest), exist_ok=True)
            shutil.copy2(img["path"], od_dest)

            # Copy to presentation
            pres_name = f"{folder_name}__{img['filename']}"
            for pres_dir in [pres_sorted, pres_onedrive]:
                shutil.copy2(img["path"], os.path.join(pres_dir, pres_name))

            existing_hashes.add(img["hash"])
            manifest[img["hash"]] = {
                "source": img["path"],
                "dest": dest_path,
                "date": img["date"].isoformat(),
                "date_source": "takeout_replacement",
                "folder": folder_name,
            }
            total_replaced += 1

        if selected:
            print(f"  {folder_name}: replaced {len(selected)}/{needed} with diverse alternatives")

    json.dump(manifest, open(MANIFEST_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CLEANUP SUMMARY")
    print("=" * 60)
    print(f"Non-photo images removed: {len(to_remove)}")
    print(f"Burst duplicates removed: {len(burst_removals)}")
    print(f"Total removed: {len(all_removals)}")
    print(f"Replacements added: {total_replaced}")

    # Final counts
    print("\nFinal folder counts:")
    for bracket_label, folder_name in sorted(BRACKET_FOLDER.items(), key=lambda x: x[0]):
        folder_path = os.path.join(OUTPUT_DIR, folder_name)
        if os.path.exists(folder_path):
            count = sum(1 for f in os.listdir(folder_path)
                       if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
            print(f"  {folder_name}: {count}")

    pres_count = sum(1 for f in os.listdir(pres_sorted)
                    if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
    print(f"\n  the presentation: {pres_count}")


if __name__ == "__main__":
    main()
