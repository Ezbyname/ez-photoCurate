# Image Grader - Comprehensive quality scoring for images
# Computes and stores per-image grades in a local JSON database
#
# Scores (each 0-100):
#   technical  - resolution, file size, sharpness, exposure
#   face       - face detected, face size, face count, face sharpness
#   recognition - similarity to reference faces (requires ref_faces/)
#   composition - face centering, aspect ratio
#   overall    - weighted composite
#
# Usage:
#   python image_grader.py --input "folder" [--faces ref_faces/] [--db grades.json]
#   python image_grader.py --input "folder" --show-worst 20
#   python image_grader.py --input "folder" --show-best 20
#   python image_grader.py --input "folder" --below 30 --list

import os
import sys
import json
import argparse
import hashlib
import numpy as np
import cv2
from datetime import datetime
from collections import defaultdict
from PIL import Image, ImageStat, ImageFilter

sys.stdout.reconfigure(line_buffering=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".heic", ".webp"}
DEFAULT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image_grades.json")


# ── Database ──────────────────────────────────────────────────────────────────

def load_db(db_path):
    if os.path.exists(db_path):
        with open(db_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"version": 2, "images": {}}


def save_db(db, db_path):
    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def file_key(filepath):
    """Stable key: filename + size."""
    try:
        sz = os.path.getsize(filepath)
        return f"{os.path.basename(filepath)}|{sz}"
    except Exception:
        return os.path.basename(filepath)


# ── Technical Quality ─────────────────────────────────────────────────────────

def grade_technical(filepath):
    """
    Technical quality score (0-100).
    Components: resolution, file_size, sharpness, exposure.
    """
    scores = {}
    try:
        pil_img = Image.open(filepath)
        w, h = pil_img.size
        file_size_kb = os.path.getsize(filepath) / 1024

        # Resolution: megapixels, scaled 0-100
        # 0.5MP=20, 2MP=50, 5MP=70, 12MP=90, 20MP+=100
        megapixels = (w * h) / 1_000_000
        scores["resolution"] = min(100, max(0, 20 + megapixels * 10))

        # File size relative to resolution (compression quality)
        # Higher KB per megapixel = less compressed = better quality
        kb_per_mp = file_size_kb / max(megapixels, 0.01)
        # 200 KB/MP = low quality, 500 = ok, 1000+ = great
        scores["file_quality"] = min(100, max(0, kb_per_mp / 15))

        # Sharpness via Laplacian variance
        pil_gray = pil_img.convert("L")
        # Resize for consistent measurement
        measure_size = min(800, max(w, h))
        scale = measure_size / max(w, h)
        if scale < 1:
            pil_gray = pil_gray.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        gray_arr = np.array(pil_gray)
        laplacian = cv2.Laplacian(gray_arr, cv2.CV_64F)
        sharpness_var = laplacian.var()
        # Typical range: <50 = blurry, 100-300 = ok, 500+ = sharp
        scores["sharpness"] = min(100, max(0, sharpness_var / 8))

        # Exposure: check if image is too dark or too bright
        pil_rgb = pil_img.convert("RGB")
        stat = ImageStat.Stat(pil_rgb)
        mean_brightness = sum(stat.mean[:3]) / 3  # 0-255
        # Ideal: 80-180, penalize extremes
        if 80 <= mean_brightness <= 180:
            scores["exposure"] = 100
        elif mean_brightness < 40 or mean_brightness > 230:
            scores["exposure"] = 20
        elif mean_brightness < 80:
            scores["exposure"] = 20 + (mean_brightness - 40) * 2
        else:
            scores["exposure"] = 20 + (230 - mean_brightness) * 1.6

        pil_img.close()

        # Weighted average
        overall = (
            scores["resolution"] * 0.25 +
            scores["file_quality"] * 0.15 +
            scores["sharpness"] * 0.40 +
            scores["exposure"] * 0.20
        )
        scores["overall"] = round(overall, 1)

    except Exception as e:
        scores = {"resolution": 0, "file_quality": 0, "sharpness": 0,
                  "exposure": 0, "overall": 0, "error": str(e)}

    return {k: round(v, 1) if isinstance(v, float) else v for k, v in scores.items()}


# ── Face Detection & Quality ──────────────────────────────────────────────────

_face_cascade = None
_profile_cascade = None

def _get_cascades():
    global _face_cascade, _profile_cascade
    if _face_cascade is None:
        cv2_data = cv2.data.haarcascades
        _face_cascade = cv2.CascadeClassifier(
            os.path.join(cv2_data, "haarcascade_frontalface_alt2.xml"))
        _profile_cascade = cv2.CascadeClassifier(
            os.path.join(cv2_data, "haarcascade_profileface.xml"))
    return _face_cascade, _profile_cascade


def grade_face(filepath):
    """
    Face detection score (0-100).
    Components: detected, count, size_ratio, face_sharpness.
    """
    scores = {}
    face_regions = []
    try:
        pil_img = Image.open(filepath).convert("RGB")
        w, h = pil_img.size

        # Resize for detection
        scale = min(800 / max(w, h), 1.0)
        if scale < 1:
            det_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        else:
            det_img = pil_img

        img_arr = np.array(det_img)
        gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        frontal, profile = _get_cascades()

        # Try frontal
        faces = frontal.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4,
                                         minSize=(20, 20))
        if len(faces) == 0:
            faces = frontal.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=3,
                                             minSize=(20, 20))
        if len(faces) == 0:
            # Try profile
            faces = profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3,
                                             minSize=(20, 20))
        if len(faces) == 0:
            # Try rotated
            for rot in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                rotated = cv2.rotate(gray, rot)
                faces = frontal.detectMultiScale(rotated, scaleFactor=1.15,
                                                 minNeighbors=3, minSize=(20, 20))
                if len(faces) > 0:
                    break

        det_h, det_w = gray.shape[:2]

        if len(faces) == 0:
            scores["detected"] = False
            scores["count"] = 0
            scores["size_ratio"] = 0
            scores["face_sharpness"] = 0
            scores["overall"] = 0
        else:
            scores["detected"] = True
            scores["count"] = int(len(faces))

            # Largest face size relative to image
            areas = [fw * fh for (fx, fy, fw, fh) in faces]
            max_area = max(areas)
            image_area = det_w * det_h
            ratio = max_area / image_area
            # 0.01 = tiny face in background, 0.05 = medium, 0.15+ = close-up
            scores["size_ratio"] = round(min(100, ratio * 500), 1)

            # Face sharpness (Laplacian on largest face region)
            best_idx = areas.index(max_area)
            fx, fy, fw, fh = faces[best_idx]
            face_roi = gray[fy:fy+fh, fx:fx+fw]
            if face_roi.size > 0:
                lap = cv2.Laplacian(face_roi, cv2.CV_64F)
                face_sharp = lap.var()
                scores["face_sharpness"] = round(min(100, face_sharp / 5), 1)
            else:
                scores["face_sharpness"] = 0

            # Store face regions (scaled back to original coords)
            for (fx, fy, fw, fh) in faces:
                face_regions.append({
                    "x": int(fx / scale), "y": int(fy / scale),
                    "w": int(fw / scale), "h": int(fh / scale)
                })

            # Weighted score
            count_score = min(100, scores["count"] * 30 + 40)  # 1 face=70, 2=100
            overall = (
                count_score * 0.20 +
                scores["size_ratio"] * 0.35 +
                scores["face_sharpness"] * 0.45
            )
            scores["overall"] = round(overall, 1)

        pil_img.close()

    except Exception as e:
        scores = {"detected": False, "count": 0, "size_ratio": 0,
                  "face_sharpness": 0, "overall": 0, "error": str(e)}

    return scores, face_regions


# ── Face Recognition Score ────────────────────────────────────────────────────

_fr = None

def _get_fr():
    global _fr
    if _fr is None:
        import face_recognition
        _fr = face_recognition
    return _fr


def load_reference_faces(faces_dir):
    """Load reference face encodings: {person: [encoding, ...]}"""
    fr = _get_fr()
    refs = {}
    if not os.path.isdir(faces_dir):
        return refs

    for person in os.listdir(faces_dir):
        pdir = os.path.join(faces_dir, person)
        if not os.path.isdir(pdir):
            continue
        encs = []
        for fname in os.listdir(pdir):
            if os.path.splitext(fname)[1].lower() not in IMAGE_EXTS:
                continue
            fpath = os.path.join(pdir, fname)
            try:
                img = Image.open(fpath).convert("RGB")
                arr = np.array(img)
                found = fr.face_encodings(arr)
                if found:
                    encs.append(found[0])
            except Exception:
                pass
        if encs:
            refs[person] = encs
    return refs


def grade_recognition(filepath, ref_encodings):
    """
    Face recognition score (0-100 per person).
    Returns {person_name: score} where score = 100 - distance*100.
    """
    if not ref_encodings:
        return {}

    fr = _get_fr()
    scores = {}
    try:
        pil_img = Image.open(filepath).convert("RGB")
        w, h = pil_img.size
        scale = min(1000 / max(w, h), 1.0)
        if scale < 1:
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        arr = np.array(pil_img)

        face_locs = fr.face_locations(arr, model="hog")
        if not face_locs:
            return {person: 0.0 for person in ref_encodings}

        face_encs = fr.face_encodings(arr, face_locs)
        if not face_encs:
            return {person: 0.0 for person in ref_encodings}

        for person, ref_list in ref_encodings.items():
            best_score = 0.0
            for ref_enc in ref_list:
                distances = fr.face_distance(face_encs, ref_enc)
                min_dist = float(np.min(distances))
                # distance 0 = perfect match, 0.6 = threshold, 1.0+ = no match
                # Convert to 0-100: score = max(0, (1 - dist/0.8) * 100)
                score = max(0, (1 - min_dist / 0.8) * 100)
                best_score = max(best_score, score)
            scores[person] = round(best_score, 1)

        pil_img.close()
    except Exception:
        scores = {person: 0.0 for person in ref_encodings}

    return scores


# ── Composition Score ─────────────────────────────────────────────────────────

def grade_composition(filepath, face_regions):
    """
    Composition score (0-100).
    Based on face centering and image aspect ratio.
    """
    scores = {}
    try:
        pil_img = Image.open(filepath)
        w, h = pil_img.size
        pil_img.close()

        # Aspect ratio: penalize very narrow/tall images
        ratio = w / h if h > 0 else 1
        if 0.6 <= ratio <= 1.8:
            scores["aspect"] = 100
        elif ratio < 0.4 or ratio > 3:
            scores["aspect"] = 30
        else:
            scores["aspect"] = 65

        # Face centering: how close is the main face to center
        if face_regions:
            # Find largest face
            largest = max(face_regions, key=lambda f: f["w"] * f["h"])
            face_cx = (largest["x"] + largest["w"] / 2) / w
            face_cy = (largest["y"] + largest["h"] / 2) / h
            # Distance from center (0 = perfect center, 0.5 = edge)
            dx = abs(face_cx - 0.5)
            dy = abs(face_cy - 0.45)  # Slightly above center is ideal
            dist = (dx ** 2 + dy ** 2) ** 0.5
            # 0.0 = 100, 0.35+ = 30
            scores["centering"] = round(max(30, 100 - dist * 200), 1)
        else:
            scores["centering"] = 50  # neutral if no face

        overall = scores["aspect"] * 0.3 + scores["centering"] * 0.7
        scores["overall"] = round(overall, 1)

    except Exception as e:
        scores = {"aspect": 50, "centering": 50, "overall": 50, "error": str(e)}

    return scores


# ── Overall Composite ─────────────────────────────────────────────────────────

def compute_overall(technical, face, recognition, composition):
    """
    Weighted overall score (0-100).
    Weights: technical=25%, face=30%, recognition=20%, composition=25%
    If no face recognition refs, redistribute that weight.
    """
    t = technical.get("overall", 0)
    f = face.get("overall", 0)
    c = composition.get("overall", 50)

    if recognition:
        r = max(recognition.values()) if recognition else 0
        overall = t * 0.20 + f * 0.30 + r * 0.25 + c * 0.25
    else:
        overall = t * 0.25 + f * 0.40 + c * 0.35

    return round(overall, 1)


# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Grade images by quality, faces, recognition, composition.")
    p.add_argument("--input", required=True, help="Directory of images to grade")
    p.add_argument("--faces", default=None, help="Reference faces directory for recognition scoring")
    p.add_argument("--db", default=DEFAULT_DB, help=f"Database file (default: {DEFAULT_DB})")
    p.add_argument("--force", action="store_true", help="Re-grade all images (ignore cache)")
    p.add_argument("--show-best", type=int, default=None, help="Show N best-graded images")
    p.add_argument("--show-worst", type=int, default=None, help="Show N worst-graded images")
    p.add_argument("--below", type=float, default=None, help="List images below this overall score")
    p.add_argument("--above", type=float, default=None, help="List images above this overall score")
    p.add_argument("--list", action="store_true", help="Just list from DB, don't re-scan")
    p.add_argument("--summary", action="store_true", help="Show score distribution summary")
    return p.parse_args()


def main():
    args = parse_args()

    db = load_db(args.db)
    images_db = db.get("images", {})

    input_dir = args.input
    if not os.path.isdir(input_dir):
        print(f"ERROR: {input_dir} not found")
        return

    files = sorted([f for f in os.listdir(input_dir)
                    if os.path.splitext(f)[1].lower() in IMAGE_EXTS])

    # ── List/query mode ───────────────────────────────────────────────────
    if args.list or args.show_best or args.show_worst or args.summary:
        entries = []
        for fname in files:
            key = file_key(os.path.join(input_dir, fname))
            if key in images_db:
                entries.append((fname, images_db[key]))

        if not entries:
            print("No graded images in DB. Run without --list first.")
            return

        if args.summary:
            overalls = [e[1]["overall"] for e in entries]
            print(f"\n  Graded images: {len(entries)}")
            print(f"  Overall scores: min={min(overalls):.1f}, max={max(overalls):.1f}, "
                  f"avg={sum(overalls)/len(overalls):.1f}, median={sorted(overalls)[len(overalls)//2]:.1f}")
            # Distribution
            brackets = [(0, 20, "Very Low"), (20, 40, "Low"), (40, 60, "Medium"),
                        (60, 80, "Good"), (80, 100.1, "Excellent")]
            print(f"\n  {'Range':<15} {'Count':>6} {'Bar'}")
            for lo, hi, label in brackets:
                count = sum(1 for s in overalls if lo <= s < hi)
                bar = "#" * (count * 40 // len(entries)) if entries else ""
                print(f"  {label:<15} {count:>6}  {bar}")

            # Per-bracket breakdown
            by_bracket = defaultdict(list)
            for fname, data in entries:
                prefix = fname.split("__")[0] if "__" in fname else "unknown"
                by_bracket[prefix].append(data["overall"])
            print(f"\n  {'Bracket':<35} {'Count':>5} {'Avg':>6} {'Min':>6}")
            print("  " + "-" * 55)
            for prefix in sorted(by_bracket):
                vals = by_bracket[prefix]
                print(f"  {prefix:<35} {len(vals):>5} {sum(vals)/len(vals):>6.1f} {min(vals):>6.1f}")
            return

        if args.below is not None:
            entries = [(f, d) for f, d in entries if d["overall"] < args.below]
        if args.above is not None:
            entries = [(f, d) for f, d in entries if d["overall"] >= args.above]

        entries.sort(key=lambda x: x[1]["overall"])

        if args.show_worst:
            entries = entries[:args.show_worst]
        elif args.show_best:
            entries = entries[-args.show_best:]
            entries.reverse()

        print(f"\n  {'Filename':<55} {'Overall':>7} {'Tech':>5} {'Face':>5} {'Comp':>5} {'Recog':>5}")
        print("  " + "-" * 85)
        for fname, data in entries:
            recog = max(data.get("recognition", {}).values()) if data.get("recognition") else "-"
            recog_str = f"{recog:>5.1f}" if isinstance(recog, float) else f"{recog:>5}"
            print(f"  {fname:<55} {data['overall']:>7.1f} "
                  f"{data['technical']['overall']:>5.1f} "
                  f"{data['face']['overall']:>5.1f} "
                  f"{data['composition']['overall']:>5.1f} "
                  f"{recog_str}")
        print(f"\n  Total: {len(entries)} images")
        return

    # ── Grading mode ──────────────────────────────────────────────────────
    print("=" * 70)
    print("  IMAGE GRADER")
    print("=" * 70)
    print(f"  Input: {input_dir.encode('ascii', 'replace').decode()}")
    print(f"  Images: {len(files)}")
    print(f"  Database: {args.db.encode('ascii', 'replace').decode()}")

    # Load face recognition references
    ref_encodings = {}
    if args.faces:
        print(f"\n  Loading reference faces...")
        ref_encodings = load_reference_faces(args.faces)
        for person, encs in ref_encodings.items():
            print(f"    {person}: {len(encs)} encodings")
        if not ref_encodings:
            print("    No reference encodings loaded.")

    # Grade each image
    new_count = 0
    skip_count = 0

    print(f"\n  Grading images...")
    for i, fname in enumerate(files):
        fpath = os.path.join(input_dir, fname)
        key = file_key(fpath)

        # Skip if already graded (unless --force)
        if key in images_db and not args.force:
            # But re-do recognition if we now have refs and didn't before
            existing = images_db[key]
            if ref_encodings and not existing.get("recognition"):
                pass  # fall through to re-grade recognition
            else:
                skip_count += 1
                if (i + 1) % 100 == 0:
                    print(f"    {i+1}/{len(files)} ({new_count} new, {skip_count} cached)...")
                continue

        # Grade components
        technical = grade_technical(fpath)
        face, face_regions = grade_face(fpath)
        recognition = grade_recognition(fpath, ref_encodings) if ref_encodings else {}
        composition = grade_composition(fpath, face_regions)
        overall = compute_overall(technical, face, recognition, composition)

        images_db[key] = {
            "filename": fname,
            "path": fpath,
            "graded_at": datetime.now().isoformat(),
            "overall": overall,
            "technical": technical,
            "face": face,
            "recognition": recognition,
            "composition": composition,
            "face_regions": face_regions,
        }
        new_count += 1

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(files)} ({new_count} new, {skip_count} cached)...")
            # Save periodically
            db["images"] = images_db
            save_db(db, args.db)

    # Final save
    db["images"] = images_db
    db["last_graded"] = datetime.now().isoformat()
    save_db(db, args.db)

    print(f"\n  Done: {new_count} graded, {skip_count} cached")

    # Quick summary
    graded = [images_db[file_key(os.path.join(input_dir, f))]
              for f in files if file_key(os.path.join(input_dir, f)) in images_db]
    if graded:
        overalls = [g["overall"] for g in graded]
        face_scores = [g["face"]["overall"] for g in graded]
        no_face = sum(1 for g in graded if not g["face"].get("detected", False))

        print(f"\n  Summary:")
        print(f"    Overall: avg={sum(overalls)/len(overalls):.1f}, "
              f"min={min(overalls):.1f}, max={max(overalls):.1f}")
        print(f"    Face detection: {len(graded) - no_face}/{len(graded)} detected "
              f"({no_face} no-face)")
        print(f"    Face score: avg={sum(face_scores)/len(face_scores):.1f}")

        if ref_encodings:
            for person in ref_encodings:
                recog_scores = [g["recognition"].get(person, 0) for g in graded]
                matched = sum(1 for s in recog_scores if s > 25)
                print(f"    Recognition '{person}': {matched}/{len(graded)} matched, "
                      f"avg={sum(recog_scores)/len(recog_scores):.1f}")

        # Worst 10
        graded.sort(key=lambda g: g["overall"])
        print(f"\n  Bottom 10:")
        for g in graded[:10]:
            print(f"    {g['overall']:5.1f}  {g['filename']}")


if __name__ == "__main__":
    main()
