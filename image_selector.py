# Image Selector - Build folders of curated images with flexible filters
# Supports: date ranges, age filters, face recognition (1+ people),
# boolean face logic (all/any), quality gates, diversity filtering
#
# Usage examples:
#   # Select images of Reef aged 1-2 years
#   python image_selector.py --source "D:\reef" --output ./out --age-from 365 --age-to 730
#
#   # Select images where Reef appears (face recognition)
#   python image_selector.py --source "D:\reef" --output ./out --faces ./ref_faces
#
#   # Select images where BOTH Reef and Erez appear
#   python image_selector.py --source "D:\reef" --output ./out --faces ./ref_faces --face-mode all --face-names reef,erez
#
#   # Select images from 2015, at least one of reef/mom, max 50, diverse
#   python image_selector.py --source "D:\reef" --output ./out --faces ./ref_faces \
#       --face-mode any --face-names reef,mom --date-from 2015-01-01 --date-to 2015-12-31 \
#       --max-images 50 --diverse
#
# Reference faces directory structure:
#   ref_faces/
#     reef/          <- person name = folder name
#       photo1.jpg
#       photo2.jpg   <- multiple reference photos improve accuracy
#     erez/
#       photo1.jpg

import os
import sys
import argparse
import shutil
import hashlib
import json
import re
import numpy as np
from datetime import datetime
from collections import defaultdict
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".heic", ".webp"}
REEF_BIRTHDAY = datetime(2013, 7, 16)
VECTOR_SIZE = 64

# ── Face recognition ─────────────────────────────────────────────────────────

_face_recognition = None

def _get_face_recognition():
    global _face_recognition
    if _face_recognition is None:
        import face_recognition as fr
        _face_recognition = fr
    return _face_recognition


def load_reference_faces(faces_dir):
    """
    Load reference face encodings from a directory structure:
      faces_dir/person_name/photo.jpg
    Returns dict: {person_name: [encoding1, encoding2, ...]}
    """
    fr = _get_face_recognition()
    ref_encodings = {}

    if not os.path.isdir(faces_dir):
        print(f"  ERROR: faces directory not found: {faces_dir}")
        return ref_encodings

    for person_name in os.listdir(faces_dir):
        person_dir = os.path.join(faces_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        encodings = []
        for fname in os.listdir(person_dir):
            if os.path.splitext(fname)[1].lower() not in IMAGE_EXTS:
                continue
            fpath = os.path.join(person_dir, fname)
            try:
                img = fr.load_image_file(fpath)
                face_encs = fr.face_encodings(img)
                if face_encs:
                    encodings.append(face_encs[0])
                    print(f"    {person_name}/{fname}: face found")
                else:
                    print(f"    {person_name}/{fname}: NO face found, skipping")
            except Exception as e:
                print(f"    {person_name}/{fname}: error: {e}")

        if encodings:
            ref_encodings[person_name] = encodings
            print(f"  {person_name}: {len(encodings)} reference encodings loaded")
        else:
            print(f"  WARNING: {person_name}: no valid face encodings found")

    return ref_encodings


def check_faces_in_image(image_path, ref_encodings, face_mode="any",
                         required_names=None, tolerance=0.6):
    """
    Check if required faces appear in the image.

    Args:
        image_path: path to image file
        ref_encodings: dict from load_reference_faces()
        face_mode: "any" = at least one required person appears
                   "all" = all required persons must appear
        required_names: list of person names to look for (None = all loaded)
        tolerance: face match distance threshold (lower = stricter, default 0.6)

    Returns: (matches: bool, found_persons: list[str])
    """
    fr = _get_face_recognition()

    if required_names is None:
        required_names = list(ref_encodings.keys())

    if not required_names:
        return True, []

    try:
        # Use PIL to load (handles non-ASCII paths)
        pil_img = Image.open(image_path).convert("RGB")
        img_array = np.array(pil_img)

        face_locations = fr.face_locations(img_array, model="hog")
        if not face_locations:
            return False, []

        face_encs = fr.face_encodings(img_array, face_locations)
        if not face_encs:
            return False, []

        found_persons = set()
        for person_name in required_names:
            if person_name not in ref_encodings:
                continue
            for ref_enc in ref_encodings[person_name]:
                distances = fr.face_distance(face_encs, ref_enc)
                if np.any(distances <= tolerance):
                    found_persons.add(person_name)
                    break

        found_list = list(found_persons)

        if face_mode == "all":
            return len(found_persons) == len(required_names), found_list
        else:  # "any"
            return len(found_persons) > 0, found_list

    except Exception:
        return False, []


# ── Date / age detection ─────────────────────────────────────────────────────

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


def get_image_date(filepath, rel_dir=None):
    """Get best-effort date for an image from all sources."""
    dt = get_exif_date(filepath)
    if dt:
        return dt
    dt = get_json_date(filepath)
    if dt:
        return dt
    dt = get_filename_date(os.path.basename(filepath))
    if dt:
        return dt
    if rel_dir:
        age_days = dir_to_age_days(rel_dir)
        if age_days is not None:
            return REEF_BIRTHDAY + __import__('datetime').timedelta(days=age_days)
    return None


# ── Quality / diversity / utility ─────────────────────────────────────────────

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

def parse_args():
    p = argparse.ArgumentParser(
        description="Select and filter images into a curated output folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All images of Reef aged 1-2 years from disk
  python image_selector.py --source "D:\\reef" --output ./out --age-from 365 --age-to 730

  # Images where Reef's face appears
  python image_selector.py --source "D:\\reef" --output ./out --faces ./ref_faces --face-names reef

  # Images where BOTH Reef and Erez appear, from 2015
  python image_selector.py --source "D:\\reef" --output ./out --faces ./ref_faces \\
      --face-mode all --face-names reef,erez --date-from 2015-01-01 --date-to 2015-12-31

  # Max 50 diverse images, any of reef/mom
  python image_selector.py --source "D:\\reef" --output ./out --faces ./ref_faces \\
      --face-mode any --face-names reef,mom --max-images 50 --diverse

Reference faces directory:
  ref_faces/
    reef/photo1.jpg, photo2.jpg   (multiple refs improve accuracy)
    erez/photo1.jpg
""")
    p.add_argument("--source", required=True, nargs="+",
                   help="Source directories to scan for images")
    p.add_argument("--output", required=True,
                   help="Output directory for selected images")

    # Date filters
    p.add_argument("--date-from", type=str, default=None,
                   help="Start date (YYYY-MM-DD)")
    p.add_argument("--date-to", type=str, default=None,
                   help="End date (YYYY-MM-DD)")

    # Age filters (days from REEF_BIRTHDAY)
    p.add_argument("--age-from", type=int, default=None,
                   help="Min age in days from birth (July 16, 2013)")
    p.add_argument("--age-to", type=int, default=None,
                   help="Max age in days from birth")

    # Face recognition
    p.add_argument("--faces", type=str, default=None,
                   help="Directory with reference face subdirs (person_name/photos)")
    p.add_argument("--face-names", type=str, default=None,
                   help="Comma-separated person names to filter by (default: all loaded)")
    p.add_argument("--face-mode", choices=["all", "any"], default="any",
                   help="'all' = every named person must appear; 'any' = at least one (default: any)")
    p.add_argument("--face-tolerance", type=float, default=0.6,
                   help="Face match distance (lower=stricter, default 0.6)")

    # Quality / diversity
    p.add_argument("--min-size", type=int, default=80,
                   help="Min file size in KB (default: 80)")
    p.add_argument("--min-dim", type=int, default=600,
                   help="Min image dimension in pixels (default: 600)")
    p.add_argument("--no-screenshots", action="store_true", default=True,
                   help="Skip screenshot-like images (default: on)")
    p.add_argument("--diverse", action="store_true",
                   help="Apply diversity filter (skip similar images)")
    p.add_argument("--sim-threshold", type=float, default=0.85,
                   help="Similarity threshold for diversity (default: 0.85)")
    p.add_argument("--max-images", type=int, default=None,
                   help="Max total images to select")
    p.add_argument("--sort-by", choices=["quality", "date", "name"], default="quality",
                   help="Sort candidates by (default: quality)")

    # Misc
    p.add_argument("--copy", action="store_true", default=True,
                   help="Copy files (default)")
    p.add_argument("--move", action="store_true",
                   help="Move files instead of copying")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be selected without copying")

    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("  IMAGE SELECTOR")
    print("=" * 70)

    # Parse date filters
    date_from = datetime.strptime(args.date_from, "%Y-%m-%d") if args.date_from else None
    date_to = datetime.strptime(args.date_to, "%Y-%m-%d") if args.date_to else None

    # Convert age filters to dates
    from datetime import timedelta
    if args.age_from is not None:
        age_date_from = REEF_BIRTHDAY + timedelta(days=args.age_from)
        if date_from is None or age_date_from > date_from:
            date_from = age_date_from
    if args.age_to is not None:
        age_date_to = REEF_BIRTHDAY + timedelta(days=args.age_to)
        if date_to is None or age_date_to < date_to:
            date_to = age_date_to

    # Print filter summary
    print("\n  Filters:")
    if date_from:
        print(f"    Date from:    {date_from.strftime('%Y-%m-%d')}")
    if date_to:
        print(f"    Date to:      {date_to.strftime('%Y-%m-%d')}")
    if args.faces:
        print(f"    Face dir:     {args.faces}")
        print(f"    Face mode:    {args.face_mode}")
        if args.face_names:
            print(f"    Face names:   {args.face_names}")
        print(f"    Tolerance:    {args.face_tolerance}")
    if args.diverse:
        print(f"    Diversity:    threshold={args.sim_threshold}")
    if args.max_images:
        print(f"    Max images:   {args.max_images}")
    print(f"    Min size:     {args.min_size}KB, min dim: {args.min_dim}px")
    print(f"    Sources:      {', '.join(args.source)}")
    print(f"    Output:       {args.output}")

    # ── Load face references ────────────────────────────────────────────────
    ref_encodings = {}
    required_face_names = None
    use_face_filter = False

    if args.faces:
        print("\n[Face Recognition] Loading reference faces...")
        ref_encodings = load_reference_faces(args.faces)
        if not ref_encodings:
            print("  WARNING: No face encodings loaded. Face filter disabled.")
        else:
            use_face_filter = True
            if args.face_names:
                required_face_names = [n.strip() for n in args.face_names.split(",")]
                missing = [n for n in required_face_names if n not in ref_encodings]
                if missing:
                    print(f"  WARNING: No references for: {', '.join(missing)}")
                    required_face_names = [n for n in required_face_names if n in ref_encodings]
            else:
                required_face_names = list(ref_encodings.keys())
            print(f"  Will filter for: {', '.join(required_face_names)} (mode={args.face_mode})")

    # ── Scan sources ────────────────────────────────────────────────────────
    print("\n[Scanning] Collecting candidate images...")
    candidates = []
    seen_hashes = set()
    scanned = 0
    skipped = defaultdict(int)

    for source_dir in args.source:
        if not os.path.exists(source_dir):
            print(f"  WARNING: source not found: {source_dir}")
            continue

        for dirpath, dirnames, filenames in os.walk(source_dir):
            # Skip video dirs
            if any(v in dirpath for v in ["וידאו", "video"]):
                continue

            rel_dir = os.path.relpath(dirpath, source_dir)

            for fname in filenames:
                if os.path.splitext(fname)[1].lower() not in IMAGE_EXTS:
                    continue

                fpath = os.path.join(dirpath, fname)
                scanned += 1
                if scanned % 500 == 0:
                    print(f"    Scanned {scanned}, {len(candidates)} candidates...")

                # Hash dedup
                try:
                    fhash = file_hash(fpath)
                except Exception:
                    skipped["unreadable"] += 1
                    continue
                if fhash in seen_hashes:
                    skipped["duplicate"] += 1
                    continue
                seen_hashes.add(fhash)

                # Quality gate
                try:
                    if os.path.getsize(fpath) < args.min_size * 1024:
                        skipped["too_small"] += 1
                        continue
                    img = Image.open(fpath)
                    w, h = img.size
                    img.close()
                    if w < args.min_dim and h < args.min_dim:
                        skipped["low_res"] += 1
                        continue
                except Exception:
                    skipped["unreadable"] += 1
                    continue

                if args.no_screenshots and is_screenshot(fpath):
                    skipped["screenshot"] += 1
                    continue

                # Date filter
                img_date = get_image_date(fpath, rel_dir)
                if date_from or date_to:
                    if img_date is None:
                        skipped["no_date"] += 1
                        continue
                    if date_from and img_date < date_from:
                        skipped["before_date"] += 1
                        continue
                    if date_to and img_date > date_to:
                        skipped["after_date"] += 1
                        continue

                candidates.append({
                    "filename": fname,
                    "path": fpath,
                    "hash": fhash,
                    "date": img_date,
                    "score": quality_score(fpath),
                })

    print(f"\n  Scanned: {scanned}")
    print(f"  Pre-filter candidates: {len(candidates)}")
    if skipped:
        for reason, count in sorted(skipped.items()):
            print(f"    Skipped ({reason}): {count}")

    if not candidates:
        print("\n  No candidates found. Exiting.")
        return

    # ── Sort candidates ─────────────────────────────────────────────────────
    if args.sort_by == "quality":
        candidates.sort(key=lambda x: x["score"], reverse=True)
    elif args.sort_by == "date":
        candidates.sort(key=lambda x: (x["date"] or datetime.min))
    else:
        candidates.sort(key=lambda x: x["filename"])

    # ── Face recognition filter ─────────────────────────────────────────────
    if use_face_filter:
        print(f"\n[Face Recognition] Filtering {len(candidates)} candidates...")
        face_passed = []
        for i, cand in enumerate(candidates):
            if (i + 1) % 100 == 0:
                print(f"    {i + 1}/{len(candidates)} checked, {len(face_passed)} passed...")

            matches, found = check_faces_in_image(
                cand["path"], ref_encodings,
                face_mode=args.face_mode,
                required_names=required_face_names,
                tolerance=args.face_tolerance,
            )
            if matches:
                cand["found_faces"] = found
                face_passed.append(cand)

        print(f"  Face filter: {len(face_passed)}/{len(candidates)} passed")
        candidates = face_passed

    if not candidates:
        print("\n  No images passed face filter. Exiting.")
        return

    # ── Diversity filter ────────────────────────────────────────────────────
    if args.diverse:
        print(f"\n[Diversity] Filtering {len(candidates)} candidates (threshold={args.sim_threshold})...")
        selected = []
        selected_vecs = []

        for cand in candidates:
            vec = compute_vector(cand["path"])
            if vec is None:
                continue

            if selected_vecs:
                mat = np.array(selected_vecs, dtype=np.float32)
                sims = mat @ vec
                if np.max(sims) >= args.sim_threshold:
                    continue

            selected.append(cand)
            selected_vecs.append(vec)

            if args.max_images and len(selected) >= args.max_images:
                break

        print(f"  Diversity: {len(selected)}/{len(candidates)} selected")
        candidates = selected
    elif args.max_images:
        candidates = candidates[:args.max_images]

    # ── Output ──────────────────────────────────────────────────────────────
    print(f"\n[Output] {'DRY RUN - ' if args.dry_run else ''}{len(candidates)} images -> {args.output}")

    if not args.dry_run:
        os.makedirs(args.output, exist_ok=True)

    copied = 0
    manifest = []

    for cand in candidates:
        dest = unique_dest_path(args.output, cand["filename"])
        dest_name = os.path.basename(dest)

        entry = {
            "filename": dest_name,
            "source": cand["path"],
            "score": round(cand["score"], 1),
        }
        if cand.get("date"):
            entry["date"] = cand["date"].strftime("%Y-%m-%d")
        if cand.get("found_faces"):
            entry["faces"] = cand["found_faces"]
        manifest.append(entry)

        if not args.dry_run:
            try:
                if args.move:
                    shutil.move(cand["path"], dest)
                else:
                    shutil.copy2(cand["path"], dest)
                copied += 1
            except OSError as e:
                print(f"    WARN: {cand['filename']}: {e}")

    # Save manifest
    if not args.dry_run:
        manifest_path = os.path.join(args.output, "_selection_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({
                "created": datetime.now().isoformat(),
                "filters": {
                    "date_from": args.date_from,
                    "date_to": args.date_to,
                    "age_from": args.age_from,
                    "age_to": args.age_to,
                    "face_mode": args.face_mode if use_face_filter else None,
                    "face_names": args.face_names if use_face_filter else None,
                    "diverse": args.diverse,
                    "max_images": args.max_images,
                },
                "total_selected": copied,
                "images": manifest,
            }, f, ensure_ascii=False, indent=2)

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Selected: {len(candidates)} images")
    if not args.dry_run:
        print(f"  {'Moved' if args.move else 'Copied'}: {copied}")
        print(f"  Output: {args.output}")
        print(f"  Manifest: {os.path.join(args.output, '_selection_manifest.json')}")
    else:
        print("  (dry run - no files copied)")

    # Date range of selected
    dated = [c for c in candidates if c.get("date")]
    if dated:
        dates = [c["date"] for c in dated]
        print(f"  Date range: {min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}")

    # Face stats
    if use_face_filter:
        face_counts = defaultdict(int)
        for c in candidates:
            for f in c.get("found_faces", []):
                face_counts[f] += 1
        if face_counts:
            print("  Faces found:")
            for name, count in sorted(face_counts.items()):
                print(f"    {name}: {count} images")


if __name__ == "__main__":
    main()
