"""
Select 500 best quality images from all scanned sources that aren't already sorted.
Places them in 'extra images' subdirectory.
"""

import os
import sys
import json
import hashlib
import shutil
from datetime import datetime, timedelta
from PIL import Image
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)

OUTPUT_DIR = r"C:\Codes\Reef images for bar mitza\sorted"
ONEDRIVE_DIR = r"C:\Users\erezg\OneDrive - Pen-Link Ltd\Desktop\אישי\ריף\בר מצווה"
EXTRAS_FOLDER = "extra images"
MANIFEST_FILE = r"C:\Codes\Reef images for bar mitza\manifest.json"
REEF_BIRTHDAY = datetime(2013, 7, 16)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".heic", ".webp"}
TARGET = 500

# All source directories to scan
SOURCES = [
    r"C:\Users\erezg\OneDrive - Pen-Link Ltd\Desktop\אישי\ריף\בר מצווה\download\extracted\Takeout\Google Photos\reef",
    r"C:\Users\erezg\OneDrive - Pen-Link Ltd\Desktop\אישי\ריף\בר מצווה\לידה",
    r"C:\Users\erezg\OneDrive - Pen-Link Ltd\Desktop\אישי\ריף\בר מצווה\חודש 1",
    r"C:\Users\erezg\OneDrive - Pen-Link Ltd\Desktop\אישי\ריף\בר מצווה\חודש 2-12",
    r"C:\Users\erezg\OneDrive - Pen-Link Ltd\Desktop\אישי\ריף\בר מצווה\שנה - שנתיים",
    r"C:\Users\erezg\OneDrive - Pen-Link Ltd\Desktop\אישי\ריף\בר מצווה\mix",
]

# Also add disk if available
DISK_DIR = r"D:\ריף"


def file_hash(filepath):
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
    json_path = image_path + ".json"
    if not os.path.exists(json_path):
        base = os.path.splitext(image_path)[0]
        json_path = base + ".json"
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        taken_time = meta.get("photoTakenTime", {}).get("timestamp")
        if taken_time:
            return datetime.fromtimestamp(int(taken_time))
    except Exception:
        pass
    return None


def age_label(dt):
    if dt is None:
        return "unknown"
    days = (dt - REEF_BIRTHDAY).days
    if days < 0:
        return "pre-birth"
    if days < 30:
        return "newborn"
    months = days // 30
    if months < 12:
        return f"{months}mo"
    years = days // 365
    rem = (days % 365) // 30
    if rem > 0:
        return f"{years}y{rem}m"
    return f"{years}y"


def quality_score(width, height, file_size):
    resolution = width * height
    res_score = min(resolution / (4000 * 3000), 1.0)  # 12MP cap
    size_score = min(file_size / (4 * 1024 * 1024), 1.0)  # 4MB cap
    return (res_score * 0.6) + (size_score * 0.4)


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
    print("Select 500 Best Extra Images")
    print("=" * 60)

    manifest = load_manifest()
    existing_hashes = set(manifest.keys())

    # Also hash all images in sorted output
    for dirpath, _, filenames in os.walk(OUTPUT_DIR):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                try:
                    existing_hashes.add(file_hash(os.path.join(dirpath, fname)))
                except Exception:
                    pass
    print(f"Already sorted: {len(existing_hashes)} images")

    # ── Collect all candidate images ───────────────────────────────────────────
    all_sources = list(SOURCES)
    if os.path.exists(DISK_DIR):
        print(f"Disk D: available, including it")
        all_sources.append(DISK_DIR)

    candidates = []
    seen_hashes = set()
    scanned = 0

    for source in all_sources:
        if not os.path.exists(source):
            continue
        source_name = os.path.basename(source) or source
        count_before = len(candidates)

        for dirpath, _, filenames in os.walk(source):
            for fname in filenames:
                if os.path.splitext(fname)[1].lower() not in IMAGE_EXTENSIONS:
                    continue

                fpath = os.path.join(dirpath, fname)
                scanned += 1

                if scanned % 1000 == 0:
                    print(f"  Scanned {scanned} images, {len(candidates)} candidates...")

                try:
                    fhash = file_hash(fpath)
                except Exception:
                    continue

                # Skip already sorted or already seen
                if fhash in existing_hashes or fhash in seen_hashes:
                    continue
                seen_hashes.add(fhash)

                try:
                    file_size = os.path.getsize(fpath)
                    if file_size < 50 * 1024:
                        continue
                    img = Image.open(fpath)
                    w, h = img.size
                    img.close()
                    if w < 400 and h < 400:
                        continue
                except Exception:
                    continue

                # Get date
                dt = get_json_date(fpath) or get_exif_date(fpath)

                score = quality_score(w, h, file_size)
                candidates.append({
                    "path": fpath,
                    "hash": fhash,
                    "filename": fname,
                    "width": w,
                    "height": h,
                    "file_size": file_size,
                    "score": score,
                    "date": dt,
                })

        added = len(candidates) - count_before
        print(f"  {source_name}: +{added} candidates")

    print(f"\nTotal scanned: {scanned}")
    print(f"Total candidates: {len(candidates)}")

    if not candidates:
        print("No extra images found.")
        return

    # ── Sort by quality and pick top 500 ───────────────────────────────────────
    candidates.sort(key=lambda x: x["score"], reverse=True)
    selected = candidates[:TARGET]

    print(f"\nSelected top {len(selected)} images")
    print(f"  Quality range: {selected[-1]['score']:.3f} - {selected[0]['score']:.3f}")
    print(f"  Resolution range: {selected[-1]['width']}x{selected[-1]['height']} - {selected[0]['width']}x{selected[0]['height']}")

    # Show date distribution
    date_dist = defaultdict(int)
    for img in selected:
        if img["date"]:
            label = age_label(img["date"])
            date_dist[label] += 1
        else:
            date_dist["unknown"] += 1
    print(f"\n  Age distribution of selected extras:")
    for label, count in sorted(date_dist.items()):
        print(f"    {label}: {count}")

    # ── Copy to extra images folders ───────────────────────────────────────────
    dest_dir = os.path.join(OUTPUT_DIR, EXTRAS_FOLDER)
    onedrive_dest = os.path.join(ONEDRIVE_DIR, EXTRAS_FOLDER)
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(onedrive_dest, exist_ok=True)

    copied = 0
    for img in selected:
        dest = unique_dest_path(dest_dir, img["filename"])
        try:
            shutil.copy2(img["path"], dest)
            od_dest = unique_dest_path(onedrive_dest, img["filename"])
            shutil.copy2(img["path"], od_dest)
        except OSError as e:
            print(f"  WARN: {img['filename']}: {e}")
            continue

        manifest[img["hash"]] = {
            "source": img["path"],
            "dest": dest,
            "date": img["date"].isoformat() if img["date"] else "unknown",
            "date_source": "extra_quality_pick",
            "folder": EXTRAS_FOLDER,
            "quality_score": round(img["score"], 3),
        }
        copied += 1

    save_manifest(manifest)
    print(f"\nDone! Copied {copied} extra images")
    print(f"  -> {dest_dir}")
    print(f"  -> {onedrive_dest}")


if __name__ == "__main__":
    main()
