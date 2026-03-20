"""
Reef Bar Mitzva Image Sorter
-----------------------------
Scans source directories for images, determines the date each photo was taken,
and copies them into date-based subdirectories under an output folder.

Date detection priority:
1. EXIF DateTimeOriginal / DateTime
2. Date embedded in filename (e.g. WhatsApp "2026-01-06")
3. Hint from source subdirectory name (e.g. "חודש 1", "לידה", "שנה - שנתיים")
4. File modification time as last resort

Incremental: keeps a manifest of already-sorted files so re-runs only process new images.
Adaptive splitting: max 100 images per output folder (year -> 6mo -> 3mo -> 1mo -> day).
"""

import os
import sys
import re
import json
import shutil
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS

# ── Configuration ──────────────────────────────────────────────────────────────
SOURCE_DIR = r"C:\Users\erezg\OneDrive - Pen-Link Ltd\Desktop\אישי\ריף\בר מצווה"
OUTPUT_DIR = r"C:\Codes\Reef images for bar mitza\sorted"
MANIFEST_FILE = r"C:\Codes\Reef images for bar mitza\manifest.json"
REEF_BIRTHDAY = datetime(2013, 7, 16)
MAX_PER_FOLDER = 100
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".heic", ".webp"}

# ── Directory-name date hints ──────────────────────────────────────────────────
# Maps Hebrew subdirectory name patterns to approximate date ranges relative to birthday.
DIRECTORY_HINTS = {
    "לידה": (timedelta(days=0), timedelta(days=14)),          # birth ~ 2 weeks
    "חודש 1": (timedelta(days=14), timedelta(days=60)),       # month 1 (2wk-2mo)
    "חודש 2-12": (timedelta(days=60), timedelta(days=365)),   # months 2-12
    "שנה - שנתיים": (timedelta(days=365), timedelta(days=730)),  # age 1-2
}


def file_hash(filepath):
    """Quick hash: first 64KB + file size for fast dedup."""
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


def get_exif_date(filepath):
    """Try to extract date from EXIF data."""
    try:
        img = Image.open(filepath)
        exif = img._getexif()
        if exif:
            for tag_id in (36867, 306, 36868):  # DateTimeOriginal, DateTime, DateTimeDigitized
                val = exif.get(tag_id)
                if val:
                    return datetime.strptime(val, "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return None


def get_filename_date(filename):
    """Extract date from filename patterns like 'WhatsApp Image 2026-01-06 at 19.28.22'."""
    # Pattern: YYYY-MM-DD
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})', filename)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    # Pattern: YYYYMMDD
    m = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
    if m:
        try:
            d = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if 2000 <= d.year <= 2030:
                return d
        except ValueError:
            pass
    return None


def get_directory_hint_date(rel_dir):
    """Estimate date from source subdirectory name using known age hints."""
    for pattern, (start_delta, end_delta) in DIRECTORY_HINTS.items():
        if pattern in rel_dir:
            midpoint = REEF_BIRTHDAY + start_delta + (end_delta - start_delta) / 2
            return midpoint
    # Try to extract age in years from directory name (e.g. "גיל 5", "5 שנים")
    m = re.search(r'(\d+)\s*(?:שנ|year|age)', rel_dir, re.IGNORECASE)
    if m:
        years = int(m.group(1))
        return REEF_BIRTHDAY + timedelta(days=years * 365)
    m = re.search(r'(?:גיל|age)\s*(\d+)', rel_dir, re.IGNORECASE)
    if m:
        years = int(m.group(1))
        return REEF_BIRTHDAY + timedelta(days=years * 365)
    return None


def get_file_mod_date(filepath):
    """Last resort: file modification time."""
    return datetime.fromtimestamp(os.path.getmtime(filepath))


def determine_date(filepath, rel_dir):
    """Determine the date a photo was taken, using all available signals."""
    # 1. EXIF
    d = get_exif_date(filepath)
    if d:
        return d, "exif"
    # 2. Filename
    d = get_filename_date(os.path.basename(filepath))
    if d:
        return d, "filename"
    # 3. Directory hint
    d = get_directory_hint_date(rel_dir)
    if d:
        return d, "dir_hint"
    # 4. File modification date
    return get_file_mod_date(filepath), "file_mod"


def collect_images(source_dir, manifest):
    """Scan source for new images not yet in manifest."""
    new_images = []
    for root, dirs, files in os.walk(source_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue
            fpath = os.path.join(root, fname)
            fhash = file_hash(fpath)
            if fhash in manifest:
                continue
            rel_dir = os.path.relpath(root, source_dir)
            date, source = determine_date(fpath, rel_dir)
            new_images.append({
                "path": fpath,
                "hash": fhash,
                "date": date,
                "date_source": source,
                "filename": fname,
            })
    return new_images


def compute_period_folders(all_dates, existing_counts):
    """
    Determine the folder structure. Adaptively split time ranges so no folder
    exceeds MAX_PER_FOLDER images.

    Returns a function: date -> folder_name
    """
    if not all_dates:
        return lambda d: "unknown"

    min_year = min(d.year for d in all_dates)
    max_year = max(d.year for d in all_dates)

    # Count images per year (existing + new)
    year_counts = {}
    for d in all_dates:
        year_counts[d.year] = year_counts.get(d.year, 0) + 1

    # Add existing counts
    for folder_name, count in existing_counts.items():
        # Parse year from folder name
        m = re.match(r"(\d{4})", folder_name)
        if m:
            y = int(m.group(1))
            year_counts[y] = year_counts.get(y, 0) + count

    # Build split rules per year
    split_rules = {}  # year -> list of (month_start, month_end) tuples

    for year in range(min_year, max_year + 1):
        total = year_counts.get(year, 0)
        if total == 0:
            continue
        if total <= MAX_PER_FOLDER:
            split_rules[year] = [(1, 12)]  # whole year
        else:
            # Try 6-month splits
            splits_6 = [(1, 6), (7, 12)]
            max_half = _max_count_in_splits(all_dates, existing_counts, year, splits_6)
            if max_half <= MAX_PER_FOLDER:
                split_rules[year] = splits_6
            else:
                # Try quarterly
                splits_q = [(1, 3), (4, 6), (7, 9), (10, 12)]
                max_q = _max_count_in_splits(all_dates, existing_counts, year, splits_q)
                if max_q <= MAX_PER_FOLDER:
                    split_rules[year] = splits_q
                else:
                    # Monthly
                    splits_m = [(m, m) for m in range(1, 13)]
                    max_m = _max_count_in_splits(all_dates, existing_counts, year, splits_m)
                    if max_m <= MAX_PER_FOLDER:
                        split_rules[year] = splits_m
                    else:
                        # Daily will be handled via overflow
                        split_rules[year] = splits_m

    def date_to_folder(d):
        year = d.year
        if year not in split_rules:
            return str(year)
        for m_start, m_end in split_rules[year]:
            if m_start <= d.month <= m_end:
                if m_start == 1 and m_end == 12:
                    return f"{year}"
                elif m_end - m_start >= 5:  # half year
                    return f"{year}_{'H1' if m_start <= 6 else 'H2'}"
                elif m_end - m_start >= 2:  # quarter
                    q = (m_start - 1) // 3 + 1
                    return f"{year}_Q{q}"
                else:  # month
                    return f"{year}_{m_start:02d}"
        return str(year)

    return date_to_folder


def _max_count_in_splits(all_dates, existing_counts, year, splits):
    """Count max images in any split bucket for a given year."""
    counts = {(s, e): 0 for s, e in splits}
    for d in all_dates:
        if d.year != year:
            continue
        for s, e in splits:
            if s <= d.month <= e:
                counts[(s, e)] += 1
                break
    # Add existing
    for folder_name, count in existing_counts.items():
        m = re.match(r"(\d{4})(?:_(\w+))?", folder_name)
        if not m or int(m.group(1)) != year:
            continue
        suffix = m.group(2)
        if suffix is None:
            # Whole year folder - distribute proportionally (rough)
            for key in counts:
                counts[key] += count // len(counts)
        # For finer splits, try to map
    return max(counts.values()) if counts else 0


def get_existing_folder_counts(output_dir):
    """Count images already in output folders."""
    counts = {}
    if not os.path.exists(output_dir):
        return counts
    for folder_name in os.listdir(output_dir):
        folder_path = os.path.join(output_dir, folder_name)
        if os.path.isdir(folder_path):
            count = sum(1 for f in os.listdir(folder_path)
                       if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS)
            if count > 0:
                counts[folder_name] = count
    return counts


def unique_dest_path(dest_dir, filename):
    """Ensure no filename collision in destination."""
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


def age_label(date):
    """Human-readable age of Reef at given date."""
    delta = date - REEF_BIRTHDAY
    days = delta.days
    if days < 0:
        return "prenatal"
    if days < 30:
        return "newborn"
    months = days // 30
    if months < 12:
        return f"{months}mo"
    years = days // 365
    rem_months = (days % 365) // 30
    if rem_months > 0:
        return f"{years}y{rem_months}m"
    return f"{years}y"


def main():
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    manifest = load_manifest()
    print(f"Already sorted: {len(manifest)} images")

    # Collect new images
    print("Scanning for new images...")
    new_images = collect_images(SOURCE_DIR, manifest)
    print(f"Found {len(new_images)} new images")

    if not new_images:
        print("Nothing to do.")
        return

    # Show date source stats
    sources = {}
    for img in new_images:
        sources[img["date_source"]] = sources.get(img["date_source"], 0) + 1
    print(f"Date sources: {sources}")

    # Get existing folder counts
    existing_counts = get_existing_folder_counts(OUTPUT_DIR)

    # Compute folder mapping considering existing + new
    all_new_dates = [img["date"] for img in new_images]
    # Also include dates implied by existing folders
    all_dates_for_planning = list(all_new_dates)

    date_to_folder = compute_period_folders(all_dates_for_planning, existing_counts)

    # Sort and copy
    new_images.sort(key=lambda x: x["date"])
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    copied = 0
    for img in new_images:
        folder_name = date_to_folder(img["date"])
        dest_dir = os.path.join(OUTPUT_DIR, folder_name)
        os.makedirs(dest_dir, exist_ok=True)

        dest_path = unique_dest_path(dest_dir, img["filename"])
        shutil.copy2(img["path"], dest_path)

        manifest[img["hash"]] = {
            "source": img["path"],
            "dest": dest_path,
            "date": img["date"].isoformat(),
            "date_source": img["date_source"],
            "folder": folder_name,
        }
        copied += 1

    save_manifest(manifest)
    print(f"\nCopied {copied} images into sorted folders.")

    # Print summary
    print("\nFolder summary:")
    for folder_name in sorted(os.listdir(OUTPUT_DIR)):
        folder_path = os.path.join(OUTPUT_DIR, folder_name)
        if os.path.isdir(folder_path):
            count = sum(1 for f in os.listdir(folder_path)
                       if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS)
            # Try to show age range
            m = re.match(r"(\d{4})", folder_name)
            age_info = ""
            if m:
                y = int(m.group(1))
                age_info = f"  (Reef ~{age_label(datetime(y, 7, 1))})"
            print(f"  {folder_name}: {count} images{age_info}")


if __name__ == "__main__":
    main()
