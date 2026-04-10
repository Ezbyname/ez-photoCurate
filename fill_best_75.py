"""
Fill Best 75 Pipeline — Find the best 75+ images per category for Reef's bar mitzva.

Pipeline:
1. Load scan_db (already scanned 12K+ images)
2. Run face detection on all unchecked images (find Reef)
3. Compute photo grades on all ungraded images
4. Filter: must have Reef's face, not NSFW, not screenshot, not blurry
5. Remove near-duplicates (keep highest grade)
6. Select top 75 per category by composite grade
7. Save results to Downloads

Usage:
    python fill_best_75.py
"""

import os
import sys
import json
import time
import shutil
import hashlib
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter
from PIL import Image as PILImage

sys.stdout.reconfigure(line_buffering=True)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SCAN_DB_PATH = os.path.join(PROJECT_DIR, "scan_db.json")
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Downloads", "Reef Bar Mitzva - Best 75")

MIN_PER_CAT = 75
BLUR_THRESHOLD = 50
FACE_TOLERANCE = 0.55  # Default; infant/toddler gets tighter
DEDUP_THRESHOLD = 0.92  # Cosine similarity threshold for near-duplicates


def load_scan_db():
    with open(SCAN_DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_scan_db(db):
    tmp_path = SCAN_DB_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, cls=NumpyEncoder)
    os.replace(tmp_path, SCAN_DB_PATH)


# ── Face Detection ──────────────────────────────────────────────────────────

def load_face_references():
    """Load reference face encodings for Reef."""
    import face_recognition as fr
    ref_dir = os.path.join(PROJECT_DIR, "ref_faces")
    refs = {}

    # Try cache first
    cache_path = os.path.join(ref_dir, "_encodings_cache.json")
    if os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path, "r"))
            for person, encs in cache.items():
                refs[person] = [np.array(e) for e in encs]
            print(f"  Loaded {sum(len(v) for v in refs.values())} cached encodings for {list(refs.keys())}")
            return refs
        except Exception:
            pass

    for person in os.listdir(ref_dir):
        pdir = os.path.join(ref_dir, person)
        if not os.path.isdir(pdir):
            continue
        encs = []
        for fname in os.listdir(pdir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            fpath = os.path.join(pdir, fname)
            try:
                img = PILImage.open(fpath).convert("RGB")
                arr = np.array(img)
                found = fr.face_encodings(arr)
                if found:
                    encs.append(found[0])
            except Exception:
                pass
        if encs:
            refs[person] = encs
            print(f"  {person}: {len(encs)} encodings")

    # Save cache
    if refs:
        cache = {p: [e.tolist() for e in encs] for p, encs in refs.items()}
        with open(cache_path, "w") as f:
            json.dump(cache, f)
        print(f"  Saved encodings cache")

    return refs


def detect_face(fpath, ref_encodings, tolerance=0.55):
    """Detect faces and check for Reef. Returns (face_count, faces_found, has_target, best_distance)."""
    import face_recognition as fr

    try:
        pil_img = PILImage.open(fpath).convert("RGB")
        w, h = pil_img.size

        # Resize for speed
        max_dim = 800
        scale = min(max_dim / max(w, h), 1.0)
        if scale < 1:
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)

        arr = np.array(pil_img)

        # Quick check: any faces?
        face_locs = fr.face_locations(arr, model="hog")
        if not face_locs:
            return 0, [], False, None

        face_encs = fr.face_encodings(arr, face_locs)
        if not face_encs:
            return len(face_locs), [], False, None

        # Match against references
        faces_found = []
        best_distance = None
        for person, ref_list in ref_encodings.items():
            for ref_enc in ref_list:
                distances = fr.face_distance(face_encs, ref_enc)
                min_dist = float(np.min(distances))
                if min_dist <= tolerance:
                    if person not in faces_found:
                        faces_found.append(person)
                    if best_distance is None or min_dist < best_distance:
                        best_distance = min_dist

        has_target = len(faces_found) > 0
        return len(face_locs), faces_found, has_target, best_distance

    except Exception as e:
        return 0, [], False, None


# ── Photo Grading ───────────────────────────────────────────────────────────

def compute_photo_grade(fpath, w, h):
    """Compute comprehensive photo quality grade."""
    import cv2
    from PIL import ImageStat

    try:
        pil_img = PILImage.open(fpath)
        pil_rgb = pil_img.convert("RGB")
        file_size_kb = os.path.getsize(fpath) / 1024
        megapixels = (w * h) / 1_000_000

        measure_size = min(800, max(w, h))
        scale = measure_size / max(w, h)
        if scale < 1:
            pil_small = pil_rgb.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
        else:
            pil_small = pil_rgb
        rgb_arr = np.array(pil_small)
        gray_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)
        sh, sw = gray_arr.shape

        # Resolution
        resolution = min(100, max(0, 20 + megapixels * 10))

        # Sharpness
        laplacian = cv2.Laplacian(gray_arr, cv2.CV_64F)
        sharpness_var = laplacian.var()
        sharpness = min(100, max(0, sharpness_var / 8))

        # Noise
        median_filtered = cv2.medianBlur(gray_arr, 5)
        noise_diff = gray_arr.astype(np.float32) - median_filtered.astype(np.float32)
        noise_std = noise_diff.std()
        noise = min(100, max(0, 100 - noise_std * 5))

        # Compression
        kb_per_mp = file_size_kb / max(megapixels, 0.01)
        compression = min(100, max(0, kb_per_mp / 15))

        # Color
        stat = ImageStat.Stat(pil_small)
        hsv_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2HSV)
        mean_sat = hsv_arr[:, :, 1].mean()
        sat_score = min(100, mean_sat / 1.5)
        p5 = np.percentile(gray_arr, 5)
        p95 = np.percentile(gray_arr, 95)
        dyn_range = p95 - p5
        range_score = min(100, max(0, dyn_range / 2.0))
        color = sat_score * 0.4 + range_score * 0.6

        # Exposure
        mean_brightness = sum(stat.mean[:3]) / 3
        if 80 <= mean_brightness <= 180:
            exposure = 100
        elif mean_brightness < 40 or mean_brightness > 230:
            exposure = 20
        elif mean_brightness < 80:
            exposure = 20 + (mean_brightness - 40) * 2
        else:
            exposure = 20 + (230 - mean_brightness) * 1.6
        brightness_std = np.std(gray_arr)
        if 30 <= brightness_std <= 80:
            contrast_bonus = 0
        elif brightness_std < 15:
            contrast_bonus = -15
        elif brightness_std > 100:
            contrast_bonus = -10
        else:
            contrast_bonus = -5
        exposure = max(0, min(100, exposure + contrast_bonus))

        # Focus
        ch, cw = sh // 4, sw // 4
        center = gray_arr[ch:sh-ch, cw:sw-cw]
        center_lap = cv2.Laplacian(center, cv2.CV_64F).var()
        if center_lap > 20:
            focus = min(100, center_lap / 8 + 5)
        else:
            focus = max(0, center_lap / 8 * 100)

        # Distortion
        ratio = w / h if h > 0 else 1
        if 0.6 <= ratio <= 1.8:
            distortion = 100
        elif ratio < 0.4 or ratio > 3:
            distortion = 30
        else:
            distortion = 65

        pil_img.close()

        composite = (
            resolution * 0.10 + sharpness * 0.20 + noise * 0.10 +
            compression * 0.05 + color * 0.10 + exposure * 0.15 +
            focus * 0.20 + distortion * 0.10
        )

        return {
            "resolution": round(resolution, 1),
            "sharpness": round(sharpness, 1),
            "noise": round(noise, 1),
            "compression": round(compression, 1),
            "color": round(color, 1),
            "exposure": round(exposure, 1),
            "focus": round(focus, 1),
            "distortion": round(distortion, 1),
            "composite": round(composite, 1),
            "blur_score": round(sharpness_var, 1),
        }
    except Exception:
        return None


# ── Deduplication ───────────────────────────────────────────────────────────

VECTOR_SIZE = 64

def compute_image_vector(fpath):
    """Compute visual feature vector for dedup comparison."""
    try:
        img = PILImage.open(fpath).convert("RGB")
        gray = img.convert("L").resize((VECTOR_SIZE, VECTOR_SIZE), PILImage.LANCZOS)
        gray_arr = np.array(gray, dtype=np.float32).flatten() / 255.0

        small = img.resize((VECTOR_SIZE, VECTOR_SIZE), PILImage.LANCZOS)
        arr = np.array(small)
        hist_features = []
        for ch in range(3):
            hist, _ = np.histogram(arr[:, :, ch], bins=16, range=(0, 256))
            hist_features.extend(hist / hist.sum())

        vec = np.concatenate([gray_arr, hist_features])
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        img.close()
        return vec
    except Exception:
        return None


def remove_near_duplicates(images, threshold=DEDUP_THRESHOLD):
    """Remove near-duplicate images, keeping the highest graded one."""
    if len(images) <= 1:
        return images

    # Compute vectors
    vectors = {}
    for img in images:
        fpath = img["path"].replace("/", os.sep)
        if os.path.exists(fpath):
            vec = compute_image_vector(fpath)
            if vec is not None:
                vectors[img["hash"]] = vec

    # Find duplicates
    hashes = list(vectors.keys())
    to_remove = set()

    for i in range(len(hashes)):
        if hashes[i] in to_remove:
            continue
        for j in range(i + 1, len(hashes)):
            if hashes[j] in to_remove:
                continue
            sim = np.dot(vectors[hashes[i]], vectors[hashes[j]])
            if sim > threshold:
                # Keep the one with higher grade
                img_i = next(im for im in images if im["hash"] == hashes[i])
                img_j = next(im for im in images if im["hash"] == hashes[j])
                grade_i = (img_i.get("photo_grade") or {}).get("composite", 0)
                grade_j = (img_j.get("photo_grade") or {}).get("composite", 0)
                if grade_i >= grade_j:
                    to_remove.add(hashes[j])
                else:
                    to_remove.add(hashes[i])

    return [img for img in images if img["hash"] not in to_remove]


# ── Main Pipeline ───────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  FILL BEST 75 — Reef Bar Mitzva Pipeline")
    print("=" * 70)

    t0 = time.time()

    # Load database
    print("\n[1/6] Loading scan database...")
    db = load_scan_db()
    images = db["images"]
    config = db.get("config", {})
    print(f"  {len(images)} images loaded")

    # ── Step 1: Face Detection ──────────────────────────────────────────
    print("\n[2/6] Face detection (finding Reef in all images)...")

    import face_recognition as fr
    ref_encodings = load_face_references()
    if not ref_encodings:
        print("  ERROR: No reference faces found in ref_faces/")
        return

    need_face = [img for img in images if not img.get("_face_checked") and img.get("media_type") != "video"]
    print(f"  {len(need_face)} images need face checking")

    face_found_count = 0
    for idx, img in enumerate(need_face):
        fpath = img["path"].replace("/", os.sep)
        if not os.path.exists(fpath):
            img["_face_checked"] = True
            continue

        # Adjust tolerance by age
        age_days = img.get("age_days")
        if age_days is not None and age_days <= 365:
            tol = 0.45
        elif age_days is not None and age_days <= 1095:
            tol = 0.50
        else:
            tol = FACE_TOLERANCE

        fc, ff, has_target, best_d = detect_face(fpath, ref_encodings, tol)
        img["face_count"] = fc
        img["faces_found"] = ff
        img["has_target_face"] = has_target
        img["face_distance"] = best_d
        img["_face_checked"] = True

        if has_target:
            face_found_count += 1

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(need_face) - idx - 1) / rate if rate > 0 else 0
            print(f"  {idx+1}/{len(need_face)} checked, {face_found_count} Reef found "
                  f"({rate:.1f}/s, ETA {eta/60:.0f}m)")

        # Save periodically
        if (idx + 1) % 500 == 0:
            save_scan_db(db)

    total_reef = sum(1 for img in images if img.get("has_target_face"))
    print(f"  Done! {total_reef} images with Reef's face")
    save_scan_db(db)

    # ── Step 2: Photo Grading ───────────────────────────────────────────
    print("\n[3/6] Photo quality grading...")

    need_grade = [img for img in images if not img.get("photo_grade") and img.get("media_type") != "video"]
    print(f"  {len(need_grade)} images need grading")

    graded = 0
    for idx, img in enumerate(need_grade):
        fpath = img["path"].replace("/", os.sep)
        if not os.path.exists(fpath):
            continue

        w = img.get("width", 0)
        h = img.get("height", 0)
        if w == 0 or h == 0:
            continue

        grade = compute_photo_grade(fpath, w, h)
        if grade:
            img["photo_grade"] = grade
            img["blur_score"] = grade["blur_score"]
            graded += 1

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  {idx+1}/{len(need_grade)} graded ({graded} successful)")

        if (idx + 1) % 1000 == 0:
            save_scan_db(db)

    print(f"  Done! {graded} images graded")
    save_scan_db(db)

    # ── Step 3: Filter candidates ───────────────────────────────────────
    print("\n[4/6] Filtering candidates...")

    candidates_reef = []   # Images with Reef's face (preferred)
    candidates_face = []   # Images with any face but not Reef (fallback for baby categories)
    reject_reasons = Counter()

    # Baby/infant categories where face_recognition can't match Reef
    BABY_CATEGORIES = {"00_birth", "01_month1", "02_months2-3", "03_months4-6",
                       "04_months7-9", "05_months10-12", "06_age1-2"}

    for img in images:
        if img.get("media_type") == "video":
            reject_reasons["video"] += 1
            continue
        # Only include images whose files are accessible
        if not os.path.exists(img["path"].replace("/", os.sep)):
            reject_reasons["file_missing"] += 1
            continue
        if img.get("is_screenshot"):
            reject_reasons["screenshot"] += 1
            continue

        # Blur check
        blur = img.get("blur_score")
        if blur is not None and blur < BLUR_THRESHOLD:
            reject_reasons["blurry"] += 1
            continue

        # NSFW check
        if img.get("nsfw"):
            reject_reasons["nsfw"] += 1
            continue

        if not img.get("category"):
            reject_reasons["no_category"] += 1
            continue

        if img.get("has_target_face"):
            candidates_reef.append(img)
        elif img.get("category") in BABY_CATEGORIES and img.get("face_count", 0) > 0:
            candidates_face.append(img)
        elif img.get("category") in BABY_CATEGORIES:
            # For baby categories, accept even no-face images as last resort
            candidates_face.append(img)
        else:
            reject_reasons["no_reef_face"] += 1
            continue

    candidates = candidates_reef + candidates_face
    print(f"  {len(candidates_reef)} Reef-face candidates + {len(candidates_face)} baby/fallback candidates = {len(candidates)} total")
    print(f"  Rejected: {dict(reject_reasons)}")

    # ── Step 4: Group by category and dedup ─────────────────────────────
    print("\n[5/6] Deduplicating and selecting best 75 per category...")

    by_cat = defaultdict(list)
    for img in candidates:
        by_cat[img["category"]].append(img)

    # Get category display names and ensure ALL categories are in by_cat
    config_cats = config.get("categories", [])
    cat_display = {}
    for cat in config_cats:
        cat_display[cat["id"]] = cat.get("display", cat["id"])
        if cat["id"] not in by_cat:
            by_cat[cat["id"]] = []  # Ensure empty categories are included

    selected = {}
    total_selected = 0

    for cat_id in sorted(by_cat.keys()):
        cat_imgs = by_cat[cat_id]
        display = cat_display.get(cat_id, cat_id)

        # Dedup within category
        before = len(cat_imgs)
        cat_imgs = remove_near_duplicates(cat_imgs)
        dupes = before - len(cat_imgs)

        # Sort: Reef face > any face > no face, then by grade, face distance, preference
        cat_imgs.sort(key=lambda x: (
            (20 if x.get("has_target_face") else 10 if x.get("face_count", 0) > 0 else 0) +
            (x.get("photo_grade") or {}).get("composite", 0) * 10 +
            max(0, (0.6 - (x.get("face_distance") or 0.6)) * 20) +
            (8 if x.get("preference") == "like" else -6 if x.get("preference") == "dislike" else 0)
        ), reverse=True)

        pick = cat_imgs[:max(MIN_PER_CAT, MIN_PER_CAT)]
        selected[cat_id] = pick
        total_selected += len(pick)

        avg_grade = sum((p.get("photo_grade") or {}).get("composite", 0) for p in pick) / max(len(pick), 1)
        print(f"  {display}: {len(pick)}/{MIN_PER_CAT} selected "
              f"(from {before} candidates, {dupes} dupes removed, avg grade {avg_grade:.0f})")

    print(f"\n  Total: {total_selected} images selected across {len(selected)} categories")

    # Check for shortfalls
    shortfalls = []
    for cat_id, picks in selected.items():
        if len(picks) < MIN_PER_CAT:
            shortfalls.append((cat_display.get(cat_id, cat_id), len(picks), MIN_PER_CAT))

    if shortfalls:
        print(f"\n  WARNING: {len(shortfalls)} categories below {MIN_PER_CAT}:")
        for name, got, need in shortfalls:
            print(f"    {name}: {got}/{need}")

        # Try to fill shortfalls by relaxing face distance or including non-Reef faces
        print("\n  Attempting to fill shortfalls with relaxed criteria...")
        for cat_id, picks in selected.items():
            if len(picks) >= MIN_PER_CAT:
                continue

            display = cat_display.get(cat_id, cat_id)
            need = MIN_PER_CAT - len(picks)
            picked_hashes = set(p["hash"] for p in picks)

            # Get ALL images in this category (even without Reef face), sorted by grade
            all_cat = [img for img in images
                       if img.get("category") == cat_id
                       and img.get("media_type") != "video"
                       and not img.get("is_screenshot")
                       and not img.get("nsfw")
                       and img["hash"] not in picked_hashes
                       and (img.get("blur_score") is None or img.get("blur_score", 999) >= BLUR_THRESHOLD)]

            # Prefer images with ANY face > no face
            all_cat.sort(key=lambda x: (
                2 if x.get("has_target_face") else 1 if x.get("face_count", 0) > 0 else 0,
                (x.get("photo_grade") or {}).get("composite", 0)
            ), reverse=True)

            extras = all_cat[:need]
            picks.extend(extras)
            if extras:
                reef_count = sum(1 for e in extras if e.get("has_target_face"))
                face_count = sum(1 for e in extras if e.get("face_count", 0) > 0)
                print(f"    {display}: added {len(extras)} extras ({reef_count} with Reef, {face_count} with any face)")

    # ── Step 5: Copy to output ──────────────────────────────────────────
    print(f"\n[6/6] Copying to {OUTPUT_DIR}...")

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    total_copied = 0
    total_errors = 0

    for cat_id in sorted(selected.keys()):
        picks = selected[cat_id]
        display = cat_display.get(cat_id, cat_id)
        cat_dir = os.path.join(OUTPUT_DIR, display)
        os.makedirs(cat_dir, exist_ok=True)

        for i, img in enumerate(picks):
            src = img["path"].replace("/", os.sep)
            if not os.path.exists(src):
                total_errors += 1
                continue

            # Name: rank__grade__original_filename
            grade = (img.get("photo_grade") or {}).get("composite", 0)
            fname = img.get("filename", f"img_{i}.jpg")
            dst_name = f"{i+1:03d}_g{grade:.0f}__{fname}"
            dst = os.path.join(cat_dir, dst_name)

            try:
                shutil.copy2(src, dst)
                total_copied += 1
            except Exception:
                total_errors += 1

        actual_count = len([f for f in os.listdir(cat_dir) if os.path.isfile(os.path.join(cat_dir, f))])
        print(f"  {display}: {actual_count} images copied")

    # Update scan_db with selections
    selected_hashes = set()
    for picks in selected.values():
        for img in picks:
            selected_hashes.add(img["hash"])

    for img in images:
        if img["hash"] in selected_hashes:
            img["status"] = "selected"

    save_scan_db(db)

    # Update the "Reef Bar Mitzva" project with the new scan_db
    proj_dir = os.path.join(PROJECT_DIR, "projects", "Reef Bar Mitzva")
    if os.path.exists(proj_dir):
        shutil.copy2(SCAN_DB_PATH, os.path.join(proj_dir, "scan_db.json"))
        # Also copy config
        cfg_path = os.path.join(PROJECT_DIR, "curate_config.json")
        if os.path.exists(cfg_path):
            shutil.copy2(cfg_path, os.path.join(proj_dir, "curate_config.json"))
        print(f"  Updated project 'Reef Bar Mitzva' with new selections")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  DONE! {total_copied} images saved to:")
    print(f"  {OUTPUT_DIR}")
    print(f"  ({total_errors} errors, {elapsed/60:.1f} minutes)")
    print(f"  Project 'Reef Bar Mitzva' updated.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
