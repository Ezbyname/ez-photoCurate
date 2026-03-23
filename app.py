"""
E-z Photo Organizer Web UI — Step-by-step wizard for building photo collections.
Runs locally, opens in browser.

Usage:
    python app.py
    python app.py --port 5050
"""

import os
import sys
import json
import threading
import webbrowser
import argparse
from datetime import datetime
from collections import defaultdict

import base64
import shutil

from flask import Flask, request, jsonify, send_file, session, redirect

sys.stdout.reconfigure(line_buffering=True)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECTS_DIR = os.path.join(PROJECT_DIR, "projects")
SCAN_DB_PATH = os.path.join(PROJECT_DIR, "scan_db.json")
CONFIG_PATH = os.path.join(PROJECT_DIR, "curate_config.json")

app = Flask(__name__, static_folder=None)
app.secret_key = os.environ.get("SECRET_KEY", "ez-photo-organizer-dev-key-change-in-prod")
app.config["PERMANENT_SESSION_LIFETIME"] = 60 * 60 * 24 * 30  # 30 days

# ── Auth ─────────────────────────────────────────────────────────────────────
from auth import init_db, register_auth_routes, login_required
init_db()
register_auth_routes(app)

@app.before_request
def require_auth():
    """Protect all routes except login/auth endpoints."""
    open_paths = ("/login", "/api/auth/")
    if any(request.path == p or request.path.startswith(p) for p in open_paths):
        return None
    if not session.get("user"):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Not authenticated"}), 401
        return redirect("/login")
    return None

# Lock for scan_db.json reads/writes
_db_lock = threading.Lock()

# ── Background task state ─────────────────────────────────────────────────────

_task = {"running": False, "type": None, "progress": "", "lines": [], "done": False, "error": None, "cancelled": False}
_task_lock = threading.Lock()


def _reset_task(task_type):
    with _task_lock:
        _task["running"] = True
        _task["type"] = task_type
        _task["progress"] = "Starting..."
        _task["lines"] = []
        _task["done"] = False
        _task["error"] = None
        _task["cancelled"] = False


def _is_cancelled():
    with _task_lock:
        return _task["cancelled"]


def _update_task(line):
    with _task_lock:
        _task["progress"] = line
        _task["lines"].append(line)
        if len(_task["lines"]) > 200:
            _task["lines"] = _task["lines"][-100:]


def _finish_task(error=None):
    with _task_lock:
        _task["running"] = False
        _task["done"] = True
        _task["error"] = error


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config():
    if os.path.isfile(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_config(config):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def load_scan_db():
    if os.path.isfile(SCAN_DB_PATH):
        with _db_lock:
            try:
                with open(SCAN_DB_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, ValueError):
                return None
    return None


def save_scan_db(db):
    with _db_lock:
        with open(SCAN_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False)


def _check_nsfw(fpath, classifier):
    """Check if image contains NSFW content using NudeDetector.
    Returns (is_nsfw: bool, labels: list of detected NSFW class names)."""
    # NudeDetector labels considered inappropriate
    NSFW_LABELS = {
        "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED",
        "MALE_GENITALIA_EXPOSED", "BUTTOCKS_EXPOSED",
        "ANUS_EXPOSED",
    }
    try:
        detections = classifier.detect(fpath)
        found = [d["class"] for d in detections
                 if d["class"] in NSFW_LABELS and d["score"] >= 0.6]
        return bool(found), found
    except Exception:
        return False, []


def _estimate_age(fpath):
    """Estimate the age of the primary face in an image using DeepFace.
    Returns estimated age (int) or None on failure."""
    try:
        from deepface import DeepFace
        results = DeepFace.analyze(
            img_path=fpath, actions=["age"],
            enforce_detection=False, silent=True,
            detector_backend="opencv",
        )
        if isinstance(results, list) and results:
            return results[0].get("age")
        elif isinstance(results, dict):
            return results.get("age")
    except Exception:
        pass
    return None


def _fast_face_detect(fpath, ref_encodings, face_names, tolerance=0.6):
    """Fast face detection: auto-rotate + resize to 800px max.
    Returns (face_count, faces_found, ok, best_distance).
    best_distance is the minimum distance across all ref encodings (lower = better match)."""
    import face_recognition as fr
    import numpy as np
    from PIL import Image, ImageOps
    try:
        pil_img = ImageOps.exif_transpose(Image.open(fpath)).convert("RGB")
        w, h = pil_img.size
        max_dim = 800
        if w > max_dim or h > max_dim:
            pil_img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        arr = np.array(pil_img)
        locations = fr.face_locations(arr, model="hog")
        if not locations:
            return 0, [], True, None
        encodings = fr.face_encodings(arr, locations)
        if not encodings:
            return len(locations), [], True, None
        found = set()
        best_dist = 999.0
        for person, ref_encs in ref_encodings.items():
            for ref_enc in ref_encs:
                dists = fr.face_distance(encodings, ref_enc)
                min_d = float(np.min(dists))
                if min_d < best_dist:
                    best_dist = min_d
                if min_d <= tolerance:
                    found.add(person)
        return len(locations), sorted(found), True, round(best_dist, 3) if best_dist < 999 else None
    except Exception:
        return 0, [], False, None


def _verify_single_photo(fpath):
    """Verify a single photo for face detection. Returns dict with status/message."""
    try:
        import face_recognition as fr
        import numpy as np
        from PIL import Image
        pil_img = Image.open(fpath).convert("RGB")
        w, h = pil_img.size
        max_dim = 1600
        if w > max_dim or h > max_dim:
            pil_img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        arr = np.array(pil_img)
        locations = fr.face_locations(arr, model="hog")
        if not locations:
            return {"status": "no_face", "message": "No face detected", "dimensions": f"{w}x{h}"}
        elif len(locations) > 1:
            areas = [(b-t)*(r-l) for t, r, b, l in locations]
            best = areas.index(max(areas))
            enc = fr.face_encodings(arr, [locations[best]])
            if enc:
                return {"status": "ok_multi", "message": f"{len(locations)} faces, using largest", "dimensions": f"{w}x{h}"}
            return {"status": "encode_fail", "message": "Face found but encoding failed", "dimensions": f"{w}x{h}"}
        else:
            enc = fr.face_encodings(arr, locations)
            if enc:
                return {"status": "ok", "message": "Face detected and encoded", "dimensions": f"{w}x{h}"}
            return {"status": "encode_fail", "message": "Encoding failed", "dimensions": f"{w}x{h}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def auto_rotate_image(filepath):
    """Apply EXIF orientation and save back. Returns True if rotated."""
    from PIL import Image, ImageOps
    try:
        img = Image.open(filepath)
        rotated = ImageOps.exif_transpose(img)
        if rotated is not img:
            rotated.save(filepath, quality=95)
            return True
    except Exception:
        pass
    return False


def list_templates():
    templates_dir = os.path.join(PROJECT_DIR, "templates")
    result = []
    if not os.path.isdir(templates_dir):
        return result
    for fname in sorted(os.listdir(templates_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(templates_dir, fname), "r", encoding="utf-8") as f:
            t = json.load(f)
        result.append({
            "event_type": t["event_type"],
            "display_name": t["display_name"],
            "description": t["description"],
            "categorization": t["categorization"],
            "num_categories": len(t.get("categories", [])),
            "required_fields": t.get("required_fields", {}),
            "tips": t.get("tips", []),
            "extra": t.get("extra", False),
            "categories": [{"id": c["id"], "display": c["display"], "target": c.get("target")} for c in t.get("categories", [])],
        })
    return result


def load_template(event_type):
    tpath = os.path.join(PROJECT_DIR, "templates", f"{event_type}.json")
    if os.path.isfile(tpath):
        with open(tpath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ── API Routes ────────────────────────────────────────────────────────────────

@app.route("/api/templates")
def api_templates():
    return jsonify(list_templates())


@app.route("/api/config", methods=["GET"])
def api_get_config():
    config = load_config()
    if config:
        return jsonify(config)
    return jsonify(None)


@app.route("/api/config", methods=["POST"])
def api_save_config():
    config = request.json
    save_config(config)
    return jsonify({"ok": True})


@app.route("/api/init", methods=["POST"])
def api_init():
    data = request.json
    event_type = data.get("event_type")
    template = load_template(event_type)
    if not template:
        return jsonify({"error": f"Unknown event type: {event_type}"}), 400

    defaults = template.get("defaults", {})
    config = {
        "event_type": event_type,
        "template": template["display_name"],
        "categorization": template["categorization"],
        "ref_faces_dir": "./ref_faces",
        "face_names": data.get("face_names", []),
        "face_tolerance": defaults.get("face_tolerance", 0.6),
        "sources": data.get("sources", []),
        "target_per_category": defaults.get("target_per_category", 75),
        "min_size_kb": defaults.get("min_size_kb", 80),
        "min_dim": defaults.get("min_dim", 600),
        "thumb_size": defaults.get("thumb_size", 120),
        "categories": template["categories"],
    }

    # Fill required fields
    if data.get("subject_birthday"):
        config["subject_birthday"] = data["subject_birthday"]
    if data.get("event_date"):
        config["event_date"] = data["event_date"]
    if data.get("end_date"):
        config["end_date"] = data["end_date"]
    if data.get("year"):
        config["year"] = data["year"]

    save_config(config)
    return jsonify({"ok": True, "config": config})


@app.route("/api/scan/start", methods=["POST"])
def api_scan_start():
    if _task["running"]:
        return jsonify({"error": "A task is already running"}), 409

    full = request.json.get("full", False) if request.json else False
    nsfw_filter = request.json.get("nsfw_filter", False) if request.json else False
    age_estimation = request.json.get("age_estimation", None) if request.json else None

    def run_scan():
        try:
            _reset_task("scan")
            _update_task("Loading config...")

            config = load_config()
            if not config:
                _finish_task("No config found. Complete setup first.")
                return

            # Import scan logic
            from curate import (
                load_reference_faces, detect_faces_in_image,
                get_image_date, is_screenshot, file_hash,
                guess_device_source, make_thumbnail_b64,
                categorize_by_template, load_template as load_tpl,
                age_days_to_bracket, IMAGE_EXTS, VIDEO_EXTS, MEDIA_EXTS, REEF_BIRTHDAY,
                get_video_date, get_video_info, make_video_thumbnail_b64,
            )
            from PIL import Image as PILImage

            sources = config.get("sources", [])
            face_names = config.get("face_names", [])
            face_dir = config.get("ref_faces_dir", "")
            tolerance = config.get("face_tolerance", 0.6)
            min_size_kb = config.get("min_size_kb", 80)
            min_dim = config.get("min_dim", 600)
            thumb_size = config.get("thumb_size", 120)

            # Template
            event_type = config.get("event_type")
            template = None
            if event_type:
                template = load_tpl(event_type)
                if not template and "categories" in config and "categorization" in config:
                    template = {"categorization": config["categorization"], "categories": config["categories"]}
            use_template = template is not None

            # Faces
            ref_encodings = {}
            use_faces = False
            if face_names and face_dir and os.path.isdir(face_dir):
                _update_task("Loading face references...")
                if _is_cancelled():
                    _update_task("Stopped by user.")
                    _finish_task("Cancelled")
                    return
                ref_encodings = load_reference_faces(face_dir)
                use_faces = bool(ref_encodings)

            if _is_cancelled():
                _update_task("Stopped by user.")
                _finish_task("Cancelled")
                return

            # NSFW detection
            nsfw_classifier = None
            if nsfw_filter:
                try:
                    from nudenet import NudeDetector
                    _update_task("Loading NSFW detection model (first time may download ~25 MB)...")
                    nsfw_classifier = NudeDetector()
                    _update_task("NSFW filter ready.")
                except ImportError:
                    _update_task("nudenet not installed — installing...")
                    import subprocess
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "nudenet"], stdout=subprocess.DEVNULL)
                    from nudenet import NudeDetector
                    nsfw_classifier = NudeDetector()
                    _update_task("NSFW filter installed and ready.")
                except Exception as e:
                    _update_task(f"NSFW filter failed to load: {e}. Continuing without it.")

            if _is_cancelled():
                _update_task("Stopped by user.")
                _finish_task("Cancelled")
                return

            # Age estimation setup
            age_est_enabled = False
            age_est_scope = "all"
            age_est_folders = []
            if age_estimation and age_estimation.get("enabled"):
                try:
                    from deepface import DeepFace
                    _update_task("Loading age estimation model (first run may download ~500 MB)...")
                    # Warm up the model with a dummy analysis
                    import numpy as np
                    _dummy = np.zeros((100, 100, 3), dtype=np.uint8)
                    try:
                        DeepFace.analyze(_dummy, actions=["age"], enforce_detection=False, silent=True)
                    except Exception:
                        pass
                    age_est_enabled = True
                    age_est_scope = age_estimation.get("scope", "all")
                    age_est_folders = [f.replace("\\", "/") for f in age_estimation.get("folders", [])]
                    _update_task("Age estimation model ready.")
                except ImportError:
                    _update_task("deepface not installed — installing...")
                    import subprocess
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "deepface", "tf-keras"], stdout=subprocess.DEVNULL)
                    from deepface import DeepFace
                    age_est_enabled = True
                    age_est_scope = age_estimation.get("scope", "all")
                    age_est_folders = [f.replace("\\", "/") for f in age_estimation.get("folders", [])]
                    _update_task("Age estimation installed and ready.")
                except Exception as e:
                    _update_task(f"Age estimation failed to load: {e}. Continuing without it.")

            if _is_cancelled():
                _update_task("Stopped by user.")
                _finish_task("Cancelled")
                return

            # Existing DB for incremental
            existing_db = {}
            if os.path.isfile(SCAN_DB_PATH) and not full:
                old = load_scan_db()
                if old:
                    for img in old.get("images", []):
                        existing_db[img["hash"]] = img
                _update_task(f"Incremental mode: {len(existing_db)} cached images")

            if _is_cancelled():
                _update_task("Stopped by user.")
                _finish_task("Cancelled")
                return

            all_images = []
            seen_hashes = set()
            scanned = 0
            face_rechecked = 0
            skipped = defaultdict(int)

            for source in sources:
                if _is_cancelled():
                    _update_task("Stopped by user.")
                    _finish_task("Cancelled")
                    return
                if isinstance(source, str):
                    src_path = source
                    src_label = os.path.basename(source) or source
                else:
                    src_path = source.get("path", "")
                    src_label = source.get("label", "Unknown")

                if not os.path.isdir(src_path):
                    _update_task(f"Source not found: {src_label}")
                    continue

                _update_task(f"Scanning: {src_label}...")
                src_count = 0

                for dirpath, dirnames, filenames in os.walk(src_path):
                    rel_dir = os.path.relpath(dirpath, src_path)

                    for fname in filenames:
                        if _is_cancelled():
                            _update_task("Stopped by user.")
                            _finish_task("Cancelled")
                            return

                        ext = os.path.splitext(fname)[1].lower()
                        if ext not in MEDIA_EXTS:
                            continue
                        is_video = ext in VIDEO_EXTS

                        fpath = os.path.join(dirpath, fname)
                        scanned += 1

                        if scanned % 100 == 0:
                            _update_task(f"Scanning {src_label}: {scanned} files, {len(all_images)} kept...")

                        try:
                            fhash = file_hash(fpath)
                        except Exception:
                            skipped["unreadable"] += 1
                            continue
                        if fhash in seen_hashes:
                            skipped["duplicate"] += 1
                            continue
                        seen_hashes.add(fhash)

                        if fhash in existing_db:
                            entry = existing_db[fhash]
                            entry["path"] = fpath.replace("\\", "/")
                            entry["source_label"] = src_label

                            # Re-run face detection on cached entries that weren't face-checked
                            # Also backfill face_distance for entries that have faces but no distance
                            needs_face_check = (use_faces
                                and not entry.get("_face_checked")
                                and entry.get("status") != "rejected")
                            needs_distance_backfill = (use_faces
                                and entry.get("has_target_face")
                                and entry.get("face_distance") is None
                                and entry.get("status") != "rejected")
                            if needs_face_check or needs_distance_backfill:
                                try:
                                    fc, ff, ok, best_d = _fast_face_detect(
                                        fpath, ref_encodings, face_names, tolerance)
                                    entry["face_count"] = fc
                                    entry["faces_found"] = ff
                                    entry["has_target_face"] = any(n in ff for n in face_names) if face_names else (fc > 0)
                                    if best_d is not None:
                                        entry["face_distance"] = best_d
                                    entry["_face_checked"] = True
                                    face_rechecked += 1
                                    if face_rechecked % 100 == 0:
                                        found_so_far = sum(1 for e in all_images if e.get("has_target_face"))
                                        _update_task(f"Face-checking: {face_rechecked} done, {found_so_far} with target face...")
                                    # Periodic save every 500 face-checks
                                    if face_rechecked % 500 == 0:
                                        _interim_db = {"scan_date": datetime.now().isoformat(), "config": config,
                                            "stats": {"total_scanned": scanned, "total_kept": len(all_images), "skipped": dict(skipped), "face_rechecked": face_rechecked},
                                            "images": all_images + [existing_db[h] for h in existing_db if h not in seen_hashes]}
                                        save_scan_db(_interim_db)
                                        _update_task(f"Progress saved ({face_rechecked} face-checked)...")
                                    if _is_cancelled():
                                        _update_task("Stopped by user.")
                                        _finish_task("Cancelled")
                                        return
                                    # Update status based on face results
                                    if face_names and not entry["has_target_face"]:
                                        if entry.get("status") in ("qualified", "selected"):
                                            entry["status"] = "pool"
                                            entry["reject_reason"] = "no_faces" if fc == 0 else "wrong_person"
                                    elif entry.get("status") == "pool" and entry.get("reject_reason") in ("no_faces", "wrong_person"):
                                        if entry.get("category"):
                                            entry["status"] = "qualified"
                                            entry["reject_reason"] = None
                                except Exception:
                                    entry["_face_checked"] = True

                            # NSFW check on cached entries if filter enabled and not yet checked
                            if nsfw_classifier and "nsfw" not in entry and entry.get("status") != "rejected":
                                is_nsfw, nsfw_labels = _check_nsfw(fpath, nsfw_classifier)
                                entry["nsfw"] = is_nsfw
                                if is_nsfw:
                                    entry["nsfw_labels"] = nsfw_labels
                                    entry["status"] = "rejected"
                                    entry["reject_reason"] = "nsfw"

                            # Age estimation on cached entries if enabled and not yet done
                            if (age_est_enabled
                                and "estimated_age" not in entry
                                and entry.get("has_target_face")
                                and entry.get("media_type") != "video"
                                and entry.get("status") != "rejected"):
                                run_age = False
                                if age_est_scope == "all":
                                    run_age = True
                                elif age_est_scope == "folders":
                                    src_norm = src_path.replace("\\", "/")
                                    run_age = src_norm in age_est_folders
                                if run_age:
                                    est = _estimate_age(fpath)
                                    if est is not None:
                                        entry["estimated_age"] = est

                            all_images.append(entry)
                            src_count += 1
                            continue

                        try:
                            file_size = os.path.getsize(fpath)
                            if not is_video and file_size < min_size_kb * 1024:
                                skipped["too_small"] += 1
                                continue
                            if is_video:
                                w, h, duration = get_video_info(fpath)
                            else:
                                img = PILImage.open(fpath)
                                w, h = img.size
                                img.close()
                                duration = 0
                            if not is_video and w < min_dim and h < min_dim:
                                skipped["low_res"] += 1
                                continue
                        except Exception:
                            skipped["unreadable"] += 1
                            continue

                        if is_video:
                            img_date = get_video_date(fpath)
                        else:
                            img_date = get_image_date(fpath, rel_dir)
                        age_days = None
                        bracket = None
                        if img_date:
                            if use_template:
                                bracket = categorize_by_template(template, config, img_date)
                                bday_str = config.get("subject_birthday")
                                if bday_str:
                                    from datetime import datetime as dt
                                    bday = dt.strptime(bday_str, "%Y-%m-%d")
                                    age_days = (img_date - bday).days
                            else:
                                age_days = (img_date - REEF_BIRTHDAY).days
                                bracket = age_days_to_bracket(age_days)

                        screenshot = False
                        if not is_video and file_size < 500 * 1024:
                            screenshot = is_screenshot(fpath)

                        face_count = 0
                        faces_found = []
                        face_dist = None
                        if use_faces and not is_video:
                            face_count, faces_found, ok, face_dist = _fast_face_detect(
                                fpath, ref_encodings, face_names, tolerance)

                        if is_video:
                            thumb = make_video_thumbnail_b64(fpath, thumb_size)
                        else:
                            thumb = make_thumbnail_b64(fpath, thumb_size)
                        device = guess_device_source(fname)

                        entry = {
                            "hash": fhash,
                            "path": fpath.replace("\\", "/"),
                            "filename": fname,
                            "source_label": src_label,
                            "device": device,
                            "media_type": "video" if is_video else "image",
                            "date": img_date.strftime("%Y-%m-%d") if img_date else None,
                            "age_days": age_days,
                            "category": bracket,
                            "face_count": face_count,
                            "faces_found": faces_found,
                            "has_target_face": any(n in faces_found for n in face_names) if face_names else (face_count > 0),
                            "face_distance": face_dist,
                            "width": w, "height": h,
                            "duration": round(duration, 1) if is_video else 0,
                            "size_kb": round(file_size / 1024),
                            "is_screenshot": screenshot,
                            "thumb": thumb,
                        }

                        # For manual categorization, assign all to first category
                        if not bracket and config.get("categorization") == "manual" and config.get("categories"):
                            bracket = config["categories"][0]["id"]
                            entry["category"] = bracket

                        # NSFW check
                        if nsfw_classifier:
                            is_nsfw, nsfw_labels = _check_nsfw(fpath, nsfw_classifier)
                            entry["nsfw"] = is_nsfw
                            if is_nsfw:
                                entry["nsfw_labels"] = nsfw_labels

                        # Age estimation
                        if age_est_enabled and not is_video and entry["has_target_face"]:
                            run_age = False
                            if age_est_scope == "all":
                                run_age = True
                            elif age_est_scope == "folders":
                                src_norm = src_path.replace("\\", "/")
                                run_age = src_norm in age_est_folders
                            if run_age:
                                est = _estimate_age(fpath)
                                if est is not None:
                                    entry["estimated_age"] = est

                        if nsfw_classifier and entry.get("nsfw"):
                            entry["status"] = "rejected"
                            entry["reject_reason"] = "nsfw"
                        elif screenshot:
                            entry["status"] = "rejected"
                            entry["reject_reason"] = "screenshot"
                        elif face_names and not entry["has_target_face"]:
                            entry["status"] = "pool"
                            entry["reject_reason"] = "no_faces" if face_count == 0 else "wrong_person"
                        elif not bracket:
                            entry["status"] = "pool"
                            entry["reject_reason"] = "no_date"
                        else:
                            entry["status"] = "qualified"
                            entry["reject_reason"] = None

                        all_images.append(entry)
                        src_count += 1

                _update_task(f"{src_label}: {src_count} images kept")

            # Save
            db = {
                "scan_date": datetime.now().isoformat(),
                "config": config,
                "stats": {"total_scanned": scanned, "total_kept": len(all_images), "skipped": dict(skipped)},
                "images": all_images,
            }
            save_scan_db(db)

            _update_task(f"Done! {len(all_images)} images from {scanned} scanned.")
            _finish_task()

        except Exception as e:
            _finish_task(str(e))

    threading.Thread(target=run_scan, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/scan/status")
def api_scan_status():
    with _task_lock:
        return jsonify({
            "running": _task["running"],
            "type": _task["type"],
            "progress": _task["progress"],
            "lines": _task["lines"][-20:],
            "done": _task["done"],
            "error": _task["error"],
            "cancelled": _task["cancelled"],
        })


@app.route("/api/task/stop", methods=["POST"])
def api_task_stop():
    with _task_lock:
        if _task["running"]:
            _task["cancelled"] = True
            return jsonify({"ok": True})
        return jsonify({"ok": False, "msg": "No task running"})


@app.route("/api/analyze")
def api_analyze():
    db = load_scan_db()
    if not db:
        return jsonify({"error": "No scan data. Run scan first."}), 404
    config = load_config()
    # Sync scan_db config with current config categories
    if config and config.get("categories"):
        db["config"] = config
    from event_agent import analyze_collection
    analysis = analyze_collection(db)
    # Filter to only categories in current config
    if config and isinstance(config.get("categories"), list):
        valid_ids = {c["id"] for c in config["categories"]}
        analysis["categories"] = [c for c in analysis.get("categories", []) if c["id"] in valid_ids]
        analysis["total_target"] = sum(c.get("target", 0) for c in analysis["categories"])
        analysis["total_qualified"] = sum(c.get("count", 0) for c in analysis["categories"])
    return jsonify(analysis)


@app.route("/api/auto-select", methods=["POST"])
def api_auto_select():
    if _task["running"]:
        return jsonify({"error": "A task is already running"}), 409

    data = request.json or {}
    strategy = data.get("strategy", "balanced")
    threshold = data.get("sim_threshold", 0.85)

    def run_select():
        try:
            _reset_task("auto-select")
            db = load_scan_db()
            if not db:
                _finish_task("No scan data.")
                return

            # Inject current config so auto_select sees up-to-date categories
            current_config = load_config()
            if current_config:
                db["config"] = current_config

            # Monkey-patch print to capture progress
            import builtins
            _orig_print = builtins.print
            def _capture_print(*args, **kwargs):
                if _is_cancelled():
                    raise InterruptedError("Cancelled by user")
                line = " ".join(str(a) for a in args)
                # Skip internal/traceback lines
                skip = line.lstrip().startswith(("File \"", "Traceback (", "json.decoder"))
                if not skip:
                    _update_task(line.strip())
                _orig_print(*args, **kwargs)
            builtins.print = _capture_print

            from event_agent import auto_select
            db, report = auto_select(db, strategy=strategy, sim_threshold=threshold)

            builtins.print = _orig_print

            save_scan_db(db)

            _update_task(f"Done! Selected {report['total_selected']} images.")
            _finish_task()
        except InterruptedError:
            builtins.print = _orig_print
            _update_task("Stopped by user.")
            _finish_task("Cancelled")
        except Exception as e:
            try:
                builtins.print = _orig_print
            except Exception:
                pass
            _finish_task(str(e))

    threading.Thread(target=run_select, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/quick-fill", methods=["POST"])
def api_quick_fill():
    """Quick fill: select top images by quality score without vector diversity (fast)."""
    if _task["running"]:
        return jsonify({"error": "A task is already running"}), 409

    def run_quick():
        try:
            _reset_task("auto-select")
            db = load_scan_db()
            if not db:
                _finish_task("No scan data.")
                return

            current_config = load_config()
            if current_config:
                db["config"] = current_config

            config = db.get("config", {})
            images = db.get("images", [])
            target_per_cat = config.get("target_per_category", 75)

            # Get categories
            from event_agent import get_categories_from_config, compute_quality_score, get_event_type, EVENT_KNOWLEDGE
            categories = get_categories_from_config(config, images)
            event_type = get_event_type(config)
            knowledge = EVENT_KNOWLEDGE.get(event_type, EVENT_KNOWLEDGE.get("photo_book"))
            weights = knowledge.get("quality_weights", {})

            # Group by category (only face-matched when face names are configured)
            face_names = config.get("face_names", [])
            unlimited = config.get("unlimited_mode", False)
            by_cat = {}
            for img in images:
                if img.get("status") == "rejected":
                    continue
                # For videos, skip face check (can't detect faces in video thumbs)
                if face_names and img.get("media_type") != "video" and not img.get("has_target_face"):
                    continue
                cat = img.get("category")
                if cat:
                    by_cat.setdefault(cat, []).append(img)

            total_selected = 0
            total_videos = 0
            for cat in categories:
                if _is_cancelled():
                    _update_task("Stopped by user.")
                    _finish_task("Cancelled")
                    return

                cid = cat["id"]
                img_target = cat.get("target", target_per_cat)
                vid_target = cat.get("video_target", 0)
                pool = by_cat.get(cid, [])

                # Split pool into images and videos
                img_pool = [i for i in pool if i.get("media_type") != "video"]
                vid_pool = [i for i in pool if i.get("media_type") == "video"]

                def _score_and_select(candidates, target_count, media_label):
                    # Face distance filtering for images
                    if face_names and media_label == "images":
                        age_days_to = cat.get("age_days_to", 99999)
                        if age_days_to <= 365:
                            max_dist = 0.45
                        elif age_days_to <= 1095:
                            max_dist = 0.50
                        else:
                            max_dist = 0.55
                        candidates = [i for i in candidates
                                      if i.get("face_distance") is not None and i.get("face_distance") <= max_dist]

                    already = [i for i in candidates if i.get("status") == "selected"]
                    unselected = [i for i in candidates if i.get("status") != "selected"]

                    for img in unselected:
                        base_score = compute_quality_score(img, weights)
                        fd = img.get("face_distance")
                        if fd is not None and face_names:
                            base_score += max(0, (0.6 - fd)) * 5
                        img["_score"] = base_score
                    unselected.sort(key=lambda x: x["_score"], reverse=True)

                    if unlimited:
                        remaining = len(unselected)
                    else:
                        remaining = max(0, target_count - len(already))

                    picked = 0
                    for img in unselected[:remaining]:
                        img["status"] = "selected"
                        picked += 1
                    return len(already) + picked, picked

                img_total, img_picked = _score_and_select(img_pool, img_target, "images")
                vid_total, vid_picked = _score_and_select(vid_pool, vid_target, "videos")

                total_selected += img_total
                total_videos += vid_total

                parts = []
                if unlimited:
                    parts.append(f"{img_picked} images, {vid_picked} videos")
                else:
                    parts.append(f"{img_total}/{img_target} images")
                    if vid_target > 0 or vid_picked > 0:
                        parts.append(f"{vid_total}/{vid_target} videos")
                _update_task(f"{cat.get('display', cid)}: {', '.join(parts)}")

            # Clean temp scores
            for img in images:
                img.pop("_score", None)

            save_scan_db(db)
            msg = f"Done! {total_selected} images"
            if total_videos > 0:
                msg += f" + {total_videos} videos"
            msg += " selected."
            _update_task(msg)
            _finish_task()
        except Exception as e:
            _finish_task(str(e))

    threading.Thread(target=run_quick, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/images")
def api_images():
    """Get images for gallery. Supports pagination and filters."""
    db = load_scan_db()
    if not db:
        return jsonify({"images": [], "total": 0})

    images = db["images"]
    category = request.args.get("category")
    status = request.args.get("status")
    source = request.args.get("source")
    offset = int(request.args.get("offset", 0))
    limit = int(request.args.get("limit", 200))

    filtered = images
    if category:
        filtered = [i for i in filtered if i.get("category") == category]
    if status:
        filtered = [i for i in filtered if i.get("status") == status]
    if source:
        filtered = [i for i in filtered if i.get("source_label") == source]

    total = len(filtered)
    page = filtered[offset:offset + limit]

    # Strip thumbnails for lighter response if requested
    compact = request.args.get("compact") == "1"
    if compact:
        page = [{k: v for k, v in img.items() if k != "thumb"} for img in page]

    return jsonify({"images": page, "total": total, "offset": offset})


@app.route("/api/images/move", methods=["POST"])
def api_images_move():
    """Move images between categories or to pool."""
    data = request.json
    hashes = set(data.get("hashes", []))
    to_category = data.get("to_category")
    to_status = data.get("to_status", "qualified")

    db = load_scan_db()
    if not db:
        return jsonify({"error": "No scan data"}), 404

    moved = 0
    for img in db["images"]:
        if img["hash"] in hashes:
            if to_category:
                img["category"] = to_category
            img["status"] = to_status
            if to_status == "pool":
                img["reject_reason"] = "manual_reject"
            else:
                img["reject_reason"] = None
            moved += 1

    save_scan_db(db)
    return jsonify({"ok": True, "moved": moved})


@app.route("/api/images/serve/<img_hash>")
def api_images_serve(img_hash):
    """Serve a full-size image by hash."""
    db = load_scan_db()
    if not db:
        return jsonify({"error": "No scan data"}), 404
    for img in db["images"]:
        if img["hash"] == img_hash:
            fpath = img["path"].replace("/", os.sep)
            if os.path.isfile(fpath):
                return send_file(fpath)
            break
    return jsonify({"error": "Not found"}), 404


@app.route("/api/images/select", methods=["POST"])
def api_images_select():
    """Select or deselect individual images."""
    data = request.json
    hashes = set(data.get("hashes", []))
    action = data.get("action", "select")  # "select" or "deselect"

    db = load_scan_db()
    if not db:
        return jsonify({"error": "No scan data"}), 404

    changed = 0
    for img in db["images"]:
        if img["hash"] in hashes:
            if action == "select":
                img["status"] = "selected"
            else:
                img["status"] = "qualified"
            changed += 1

    save_scan_db(db)
    return jsonify({"ok": True, "changed": changed})


@app.route("/api/selections/reset", methods=["POST"])
def api_selections_reset():
    """Reset all selected images back to qualified."""
    db = load_scan_db()
    if not db:
        return jsonify({"error": "No scan data"}), 404
    count = 0
    for img in db["images"]:
        if img.get("status") == "selected":
            img["status"] = "qualified"
            count += 1
    save_scan_db(db)
    return jsonify({"ok": True, "reset": count})


@app.route("/api/categories/summary")
def api_categories_summary():
    """Get per-category counts for selection UI."""
    db = load_scan_db()
    config = load_config()
    if not db:
        return jsonify([])

    images = db["images"]
    config_cats = config.get("categories", []) if config else []
    valid_ids = {c["id"] for c in config_cats} if config_cats else set()

    cat_data = {}
    for c in config_cats:
        cat_data[c["id"]] = {
            "id": c["id"],
            "display": c.get("display", c["id"]),
            "target": c.get("target", config.get("target_per_category", 75)),
            "qualified": 0,
            "selected": 0,
            "total": 0,
        }

    for img in images:
        cat = img.get("category")
        if cat and cat in cat_data:
            cat_data[cat]["total"] += 1
            if img.get("status") == "selected":
                cat_data[cat]["selected"] += 1
            elif img.get("status") == "qualified":
                cat_data[cat]["qualified"] += 1

    return jsonify(list(cat_data.values()))


@app.route("/api/categories/update-target", methods=["POST"])
def api_categories_update_target():
    """Update target for a specific category."""
    data = request.json
    cat_id = data.get("id")
    new_target = data.get("target")

    config = load_config()
    if not config or not config.get("categories"):
        return jsonify({"error": "No config"}), 400

    for c in config["categories"]:
        if c["id"] == cat_id:
            c["target"] = new_target
            break

    save_config(config)
    return jsonify({"ok": True})


@app.route("/api/export", methods=["POST"])
def api_export():
    """Export selected/qualified images to output folder."""
    if _task["running"]:
        return jsonify({"error": "A task is already running"}), 409

    data = request.json or {}
    default_downloads = os.path.join(os.path.expanduser("~"), "Downloads", "E-z Photo Collection")
    output_dir = data.get("output_dir", default_downloads)
    status_filter = data.get("status", "selected")

    def run_export():
        try:
            import shutil
            _reset_task("export")
            db = load_scan_db()
            if not db:
                _finish_task("No scan data.")
                return

            os.makedirs(output_dir, exist_ok=True)
            exported = 0

            images = [i for i in db["images"] if i.get("status") == status_filter]
            if not images:
                _update_task("No selected media found. Use the Select step first.")
                _finish_task("No media to export")
                return

            total = len(images)
            vid_count = sum(1 for i in images if i.get("media_type") == "video")
            img_count = total - vid_count
            label = f"{img_count} images"
            if vid_count:
                label += f" + {vid_count} videos"
            _update_task(f"Exporting {label}...")

            skipped = 0
            for i, img in enumerate(images):
                if _is_cancelled():
                    _update_task("Stopped by user.")
                    _finish_task("Cancelled")
                    return

                src = img["path"].replace("/", os.sep)
                if not os.path.isfile(src):
                    skipped += 1
                    continue

                cat = img.get("category", "uncategorized")
                dest_dir = os.path.join(output_dir, cat)
                os.makedirs(dest_dir, exist_ok=True)

                dest = os.path.join(dest_dir, img["filename"])
                if os.path.exists(dest):
                    base, ext = os.path.splitext(img["filename"])
                    j = 1
                    while os.path.exists(dest):
                        dest = os.path.join(dest_dir, f"{base}_{j}{ext}")
                        j += 1

                shutil.copy2(src, dest)
                exported += 1

                if (i + 1) % 50 == 0:
                    _update_task(f"Exported {exported}/{total}...")

            msg = f"Done! {exported} files exported to {output_dir}"
            if skipped:
                msg += f" ({skipped} skipped — source files not found, check if all drives are connected)"
            _update_task(msg)
            _finish_task()

        except Exception as e:
            _finish_task(str(e))

    threading.Thread(target=run_export, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/ref-faces")
def api_ref_faces():
    """List reference face folders."""
    ref_dir = os.path.join(PROJECT_DIR, "ref_faces")
    if not os.path.isdir(ref_dir):
        return jsonify([])
    result = []
    for person in sorted(os.listdir(ref_dir)):
        pdir = os.path.join(ref_dir, person)
        if os.path.isdir(pdir):
            photos = [f for f in os.listdir(pdir) if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"}]
            result.append({"name": person, "photo_count": len(photos)})
    return jsonify(result)


@app.route("/api/ref-faces/upload", methods=["POST"])
def api_ref_faces_upload():
    """Upload reference face photos for a person.
    Accepts multipart form with 'person' field and 'photos' files,
    or JSON with 'person' and base64-encoded 'photos' array.
    """
    ref_dir = os.path.join(PROJECT_DIR, "ref_faces")
    os.makedirs(ref_dir, exist_ok=True)

    if request.content_type and "multipart" in request.content_type:
        person = request.form.get("person", "").strip().lower()
        if not person:
            return jsonify({"error": "Person name is required"}), 400
        pdir = os.path.join(ref_dir, person)
        os.makedirs(pdir, exist_ok=True)
        saved = []
        for f in request.files.getlist("photos"):
            if f.filename:
                safe_name = os.path.basename(f.filename)
                dest = os.path.join(pdir, safe_name)
                f.save(dest)
                auto_rotate_image(dest)
                saved.append(safe_name)
        return jsonify({"ok": True, "person": person, "saved": saved})
    else:
        data = request.json or {}
        person = data.get("person", "").strip().lower()
        if not person:
            return jsonify({"error": "Person name is required"}), 400
        pdir = os.path.join(ref_dir, person)
        os.makedirs(pdir, exist_ok=True)
        saved = []
        for i, photo_b64 in enumerate(data.get("photos", [])):
            img_data = base64.b64decode(photo_b64)
            fname = f"ref_{i+1:03d}.jpg"
            with open(os.path.join(pdir, fname), "wb") as f:
                f.write(img_data)
            saved.append(fname)
        return jsonify({"ok": True, "person": person, "saved": saved})


@app.route("/api/ref-faces/<person>/replace", methods=["POST"])
def api_ref_faces_replace(person):
    """Replace a single reference photo. Expects multipart form with 'filename' and 'photo'."""
    ref_dir = os.path.join(PROJECT_DIR, "ref_faces")
    pdir = os.path.join(ref_dir, person)
    if not os.path.isdir(pdir):
        return jsonify({"error": "Person not found"}), 404

    old_name = request.form.get("filename", "").strip()
    photo = request.files.get("photo")
    if not old_name or not photo:
        return jsonify({"error": "filename and photo are required"}), 400

    # Remove old file
    old_path = os.path.join(pdir, old_name)
    if os.path.isfile(old_path):
        os.remove(old_path)

    # Save new file with same name, auto-rotate
    new_path = os.path.join(pdir, old_name)
    photo.save(new_path)
    auto_rotate_image(new_path)
    return jsonify({"ok": True, "filename": old_name})


@app.route("/api/ref-faces/<person>/verify-photos", methods=["POST"])
def api_ref_faces_verify_photos(person):
    """Verify specific photos only. Expects JSON {filenames: [...]}."""
    data = request.json or {}
    filenames = data.get("filenames", [])
    pdir = os.path.join(PROJECT_DIR, "ref_faces", person)
    if not os.path.isdir(pdir):
        return jsonify({"error": "Person not found"}), 404
    results = {}
    for fn in filenames:
        fpath = os.path.join(pdir, fn)
        if os.path.isfile(fpath):
            results[fn] = _verify_single_photo(fpath)
        else:
            results[fn] = {"status": "error", "message": "File not found"}
    return jsonify({"ok": True, "results": results})


@app.route("/api/ref-faces/<person>/rotate", methods=["POST"])
def api_ref_faces_rotate(person):
    """Rotate a reference photo 90 degrees clockwise. Expects JSON {filename, direction}."""
    data = request.json or {}
    filename = data.get("filename", "")
    direction = data.get("direction", "cw")  # cw or ccw

    pdir = os.path.join(PROJECT_DIR, "ref_faces", person)
    fpath = os.path.join(pdir, filename)
    if not os.path.isfile(fpath):
        return jsonify({"error": "File not found"}), 404

    from PIL import Image
    try:
        img = Image.open(fpath)
        angle = -90 if direction == "cw" else 90
        rotated = img.rotate(angle, expand=True)
        rotated.save(fpath, quality=95)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ref-faces/<person>/photo/<filename>", methods=["DELETE"])
def api_ref_faces_delete_photo(person, filename):
    """Delete a single reference photo."""
    pdir = os.path.join(PROJECT_DIR, "ref_faces", person)
    fpath = os.path.join(pdir, filename)
    if os.path.isfile(fpath):
        os.remove(fpath)
    return jsonify({"ok": True})


@app.route("/api/ref-faces/<person>", methods=["DELETE"])
def api_ref_faces_delete(person):
    """Delete all reference photos for a person."""
    pdir = os.path.join(PROJECT_DIR, "ref_faces", person)
    if os.path.isdir(pdir):
        shutil.rmtree(pdir)
    return jsonify({"ok": True})


@app.route("/api/ref-faces/verify", methods=["POST"])
def api_ref_faces_verify():
    """Verify face reference quality. Checks:
    - Each photo has a detectable face
    - Enough photos for reliable multi-age recognition
    - Encoding diversity (different angles/ages)
    Returns per-photo results and overall readiness score.
    """
    data = request.json or {}
    person = data.get("person")
    ref_dir = os.path.join(PROJECT_DIR, "ref_faces")

    # If no person specified, verify all
    persons_to_check = []
    if person:
        persons_to_check = [person]
    else:
        if os.path.isdir(ref_dir):
            persons_to_check = [d for d in sorted(os.listdir(ref_dir))
                                if os.path.isdir(os.path.join(ref_dir, d))]

    if not persons_to_check:
        return jsonify({"persons": [], "ready": False,
                        "message": "No reference faces found. Add photos first."})

    try:
        import face_recognition as fr
        import numpy as np
    except ImportError:
        return jsonify({"error": "face_recognition library not installed. "
                        "Install with: pip install face_recognition"}), 500

    results = []
    for pname in persons_to_check:
        pdir = os.path.join(ref_dir, pname)
        if not os.path.isdir(pdir):
            results.append({"person": pname, "photos": [], "encodings": 0,
                            "ready": False, "issues": ["Folder not found"]})
            continue

        photos = []
        encodings = []
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        for fname in sorted(os.listdir(pdir)):
            if os.path.splitext(fname)[1].lower() not in image_exts:
                continue
            fpath = os.path.join(pdir, fname)
            photo_result = {"filename": fname, "status": "unknown", "message": ""}

            try:
                from PIL import Image
                pil_img = Image.open(fpath).convert("RGB")
                w, h = pil_img.size
                photo_result["dimensions"] = f"{w}x{h}"

                # Resize large images to prevent dlib segfaults
                max_dim = 1600
                if w > max_dim or h > max_dim:
                    pil_img.thumbnail((max_dim, max_dim), Image.LANCZOS)

                arr = np.array(pil_img)
                locations = fr.face_locations(arr, model="hog")
                if not locations:
                    photo_result["status"] = "no_face"
                    photo_result["message"] = "No face detected in this photo"
                elif len(locations) > 1:
                    # Multiple faces — use the largest
                    areas = [(b-t)*(r-l) for t, r, b, l in locations]
                    best = areas.index(max(areas))
                    enc = fr.face_encodings(arr, [locations[best]])
                    if enc:
                        encodings.append(enc[0])
                        photo_result["status"] = "ok_multi"
                        photo_result["message"] = f"{len(locations)} faces found, using largest"
                    else:
                        photo_result["status"] = "encode_fail"
                        photo_result["message"] = "Face found but encoding failed"
                else:
                    enc = fr.face_encodings(arr, locations)
                    if enc:
                        encodings.append(enc[0])
                        photo_result["status"] = "ok"
                        photo_result["message"] = "Face detected and encoded"
                    else:
                        photo_result["status"] = "encode_fail"
                        photo_result["message"] = "Face found but encoding failed"
            except Exception as e:
                photo_result["status"] = "error"
                photo_result["message"] = str(e)

            photos.append(photo_result)

        # Analyze encoding diversity
        n_enc = len(encodings)
        issues = []
        tips = []
        diversity_score = 0.0

        if n_enc == 0:
            issues.append("No faces could be encoded from any photo")
        elif n_enc < 3:
            issues.append(f"Only {n_enc} encoding(s). Need at least 3-5 for reliable recognition")
            tips.append("Add more photos with clear, front-facing views")
        else:
            # Compute pairwise distances to measure diversity
            dists = []
            for i in range(n_enc):
                for j in range(i+1, n_enc):
                    dists.append(np.linalg.norm(encodings[i] - encodings[j]))
            avg_dist = np.mean(dists) if dists else 0
            min_dist = np.min(dists) if dists else 0
            max_dist = np.max(dists) if dists else 0
            diversity_score = min(avg_dist / 0.6, 1.0)  # 0.6 is typical tolerance

            if avg_dist < 0.3:
                issues.append("Photos are very similar — all seem to be the same angle/age")
                tips.append("Add photos from different ages and angles for better multi-age recognition")
            elif avg_dist < 0.45:
                tips.append("Consider adding photos from more diverse ages/angles")

            if n_enc < 5:
                tips.append(f"You have {n_enc} good encodings. 5-10 is ideal for multi-age recognition")

        # Age coverage assessment
        if n_enc >= 3:
            tips.append("For best results across ages: include baby, toddler, child, and recent photos")

        ok_count = sum(1 for p in photos if p["status"] in ("ok", "ok_multi"))
        fail_count = len(photos) - ok_count
        ready = n_enc >= 3 and len(issues) == 0

        person_result = {
            "person": pname,
            "photos": photos,
            "total_photos": len(photos),
            "encodings": n_enc,
            "ok_count": ok_count,
            "fail_count": fail_count,
            "diversity_score": round(diversity_score, 2),
            "ready": ready,
            "issues": issues,
            "tips": tips,
        }
        results.append(person_result)

    all_ready = all(r["ready"] for r in results)
    return jsonify({
        "persons": results,
        "ready": all_ready,
        "message": "All faces verified and ready!" if all_ready else
                   "Some issues found — see details below"
    })


@app.route("/api/ref-faces/<person>/photos")
def api_ref_faces_photos(person):
    """Get thumbnails of reference photos for a person."""
    from PIL import Image, ImageOps
    import io

    pdir = os.path.join(PROJECT_DIR, "ref_faces", person)
    if not os.path.isdir(pdir):
        return jsonify([])

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    photos = []
    for fname in sorted(os.listdir(pdir)):
        if os.path.splitext(fname)[1].lower() not in image_exts:
            continue
        fpath = os.path.join(pdir, fname)
        try:
            img = Image.open(fpath)
            img = ImageOps.exif_transpose(img)
            img.thumbnail((120, 120))
            if img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            thumb = base64.b64encode(buf.getvalue()).decode()
            photos.append({"filename": fname, "thumb": thumb})
        except Exception:
            photos.append({"filename": fname, "thumb": ""})

    return jsonify(photos)


@app.route("/api/ref-faces/<person>/photo/<filename>")
def api_ref_faces_photo_full(person, filename):
    """Serve a full-size reference photo."""
    fpath = os.path.join(PROJECT_DIR, "ref_faces", person, filename)
    if not os.path.isfile(fpath):
        return jsonify({"error": "Not found"}), 404
    return send_file(fpath)


@app.route("/api/stats")
def api_stats():
    """Quick stats for the dashboard."""
    config = load_config()
    db = load_scan_db()

    has_config = config is not None
    has_scan = db is not None
    has_sources = bool(config.get("sources", [])) if config else False

    stats = {
        "has_config": has_config,
        "has_scan": has_scan,
        "has_sources": has_sources,
        "event_type": config.get("event_type") if config else None,
        "template_name": config.get("template") if config else None,
    }

    if db:
        images = db["images"]
        stats["total_images"] = sum(1 for i in images if i.get("media_type") != "video")
        stats["total_videos"] = sum(1 for i in images if i.get("media_type") == "video")
        stats["total_media"] = len(images)
        stats["qualified"] = sum(1 for i in images if i.get("status") == "qualified")
        stats["selected"] = sum(1 for i in images if i.get("status") == "selected")
        stats["selected_videos"] = sum(1 for i in images if i.get("status") == "selected" and i.get("media_type") == "video")
        stats["pool"] = sum(1 for i in images if i.get("status") == "pool")
        stats["sources"] = list(set(i.get("source_label", "") for i in images))

    stats["default_export_dir"] = os.path.join(os.path.expanduser("~"), "Downloads", "E-z Photo Collection")
    return jsonify(stats)


# ── Report generation ─────────────────────────────────────────────────────────

@app.route("/api/report")
def api_report():
    """Generate the interactive HTML gallery report and serve it."""
    if not os.path.isfile(SCAN_DB_PATH):
        return jsonify({"error": "No scan data. Run scan first."}), 400

    # Import report generation from curate.py
    sys.path.insert(0, PROJECT_DIR)
    import curate

    class FakeArgs:
        output = os.path.join(PROJECT_DIR, "curate_report.html")
        no_open = True

    try:
        curate.cmd_report(FakeArgs())
    except SystemExit:
        pass

    report_path = os.path.join(PROJECT_DIR, "curate_report.html")
    if os.path.isfile(report_path):
        return send_file(report_path, mimetype="text/html")
    return jsonify({"error": "Report generation failed"}), 500


# ── Project save/load ─────────────────────────────────────────────────────────

def _safe_project_name(name):
    """Sanitize project name for use as directory name."""
    import re
    safe = re.sub(r'[<>:"/\\|?*]', '_', name.strip())
    return safe[:100] or "untitled"


def _project_meta_path(pdir):
    return os.path.join(pdir, "project_meta.json")


def _current_project_state():
    """Gather current wizard step from config, for saving."""
    config = load_config()
    return {
        "has_config": config is not None,
        "event_type": config.get("event_type") if config else None,
        "template": config.get("template") if config else None,
    }


@app.route("/api/projects")
def api_projects_list():
    """List all saved projects."""
    os.makedirs(PROJECTS_DIR, exist_ok=True)
    projects = []
    for name in sorted(os.listdir(PROJECTS_DIR)):
        pdir = os.path.join(PROJECTS_DIR, name)
        if not os.path.isdir(pdir):
            continue
        meta_path = _project_meta_path(pdir)
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            meta = {"name": name}
        meta["dir_name"] = name
        projects.append(meta)
    return jsonify(projects)


@app.route("/api/projects/save", methods=["POST"])
def api_projects_save():
    """Save current state as a project. Expects {name, step, overwrite?}."""
    data = request.json or {}
    raw_name = data.get("name", "").strip()
    if not raw_name:
        return jsonify({"error": "Project name is required"}), 400

    step = data.get("step", 0)
    allow_overwrite = data.get("overwrite", False)
    dir_name = _safe_project_name(raw_name)
    pdir = os.path.join(PROJECTS_DIR, dir_name)

    # Check for duplicate name
    if os.path.isdir(pdir) and not allow_overwrite:
        return jsonify({"error": "A project with this name already exists. Choose a different name."}), 409

    os.makedirs(pdir, exist_ok=True)

    # Save meta
    meta_path = _project_meta_path(pdir)
    existing_meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            existing_meta = json.load(f)

    config = load_config()
    meta = {
        "name": raw_name,
        "dir_name": dir_name,
        "created": existing_meta.get("created", datetime.now().isoformat()),
        "modified": datetime.now().isoformat(),
        "step": step,
        "event_type": config.get("event_type") if config else None,
        "template": config.get("template") if config else None,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Copy config
    if os.path.isfile(CONFIG_PATH):
        shutil.copy2(CONFIG_PATH, os.path.join(pdir, "curate_config.json"))

    # Copy scan_db
    if os.path.isfile(SCAN_DB_PATH):
        with _db_lock:
            shutil.copy2(SCAN_DB_PATH, os.path.join(pdir, "scan_db.json"))

    # Copy ref_faces
    ref_src = os.path.join(PROJECT_DIR, "ref_faces")
    ref_dst = os.path.join(pdir, "ref_faces")
    if os.path.isdir(ref_src):
        if os.path.isdir(ref_dst):
            shutil.rmtree(ref_dst)
        shutil.copytree(ref_src, ref_dst)

    return jsonify({"ok": True, "project": meta})


@app.route("/api/projects/load", methods=["POST"])
def api_projects_load():
    """Load a saved project. Expects {dir_name}."""
    data = request.json or {}
    dir_name = data.get("dir_name", "")
    pdir = os.path.join(PROJECTS_DIR, dir_name)
    if not os.path.isdir(pdir):
        return jsonify({"error": "Project not found"}), 404

    meta_path = _project_meta_path(pdir)
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    # Restore config
    src_config = os.path.join(pdir, "curate_config.json")
    if os.path.isfile(src_config):
        shutil.copy2(src_config, CONFIG_PATH)
    elif os.path.isfile(CONFIG_PATH):
        os.remove(CONFIG_PATH)

    # Restore scan_db
    src_db = os.path.join(pdir, "scan_db.json")
    with _db_lock:
        if os.path.isfile(src_db):
            shutil.copy2(src_db, SCAN_DB_PATH)
        elif os.path.isfile(SCAN_DB_PATH):
            os.remove(SCAN_DB_PATH)

    # Restore ref_faces
    ref_dst = os.path.join(PROJECT_DIR, "ref_faces")
    ref_src = os.path.join(pdir, "ref_faces")
    if os.path.isdir(ref_dst):
        shutil.rmtree(ref_dst)
    if os.path.isdir(ref_src):
        shutil.copytree(ref_src, ref_dst)
    else:
        os.makedirs(ref_dst, exist_ok=True)

    return jsonify({"ok": True, "step": meta.get("step", 0), "project": meta})


@app.route("/api/projects/new", methods=["POST"])
def api_projects_new():
    """Start a fresh project. Clears current config, scan_db, and ref_faces."""
    if os.path.isfile(CONFIG_PATH):
        os.remove(CONFIG_PATH)
    with _db_lock:
        if os.path.isfile(SCAN_DB_PATH):
            os.remove(SCAN_DB_PATH)
    ref_dir = os.path.join(PROJECT_DIR, "ref_faces")
    if os.path.isdir(ref_dir):
        shutil.rmtree(ref_dir)
    os.makedirs(ref_dir, exist_ok=True)
    return jsonify({"ok": True})


@app.route("/api/projects/<dir_name>/rename", methods=["POST"])
def api_projects_rename(dir_name):
    """Rename a saved project. Expects {name}."""
    data = request.json or {}
    new_name = data.get("name", "").strip()
    if not new_name:
        return jsonify({"error": "Name is required"}), 400

    pdir = os.path.join(PROJECTS_DIR, _safe_project_name(dir_name))
    if not os.path.isdir(pdir):
        return jsonify({"error": "Project not found"}), 404

    meta_path = _project_meta_path(pdir)
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    meta["name"] = new_name
    meta["modified"] = datetime.now().isoformat()
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return jsonify({"ok": True, "name": new_name})


@app.route("/api/projects/<dir_name>", methods=["DELETE"])
def api_projects_delete(dir_name):
    """Delete a saved project."""
    pdir = os.path.join(PROJECTS_DIR, _safe_project_name(dir_name))
    if not os.path.isdir(pdir):
        return jsonify({"error": "Project not found"}), 404
    shutil.rmtree(pdir)
    return jsonify({"ok": True})


# ── Cleanup / Trash ──────────────────────────────────────────────────────────

@app.route("/api/cleanup/images")
def api_cleanup_images():
    """Get all images for cleanup review, grouped by category."""
    db = load_scan_db()
    if not db:
        return jsonify({"images": [], "total": 0})

    images = db["images"]
    status_filter = request.args.get("status")  # e.g. "candidate", "pool"
    category = request.args.get("category")
    media_type = request.args.get("media_type")
    trash_only = request.args.get("trash") == "1"

    filtered = images
    if trash_only:
        filtered = [i for i in filtered if i.get("trash")]
    if status_filter:
        filtered = [i for i in filtered if i.get("status") == status_filter]
    if category:
        filtered = [i for i in filtered if i.get("category") == category]
    if media_type:
        filtered = [i for i in filtered if i.get("media_type", "image") == media_type]

    offset = int(request.args.get("offset", 0))
    limit = int(request.args.get("limit", 200))
    total = len(filtered)
    page = filtered[offset:offset + limit]
    return jsonify({"images": page, "total": total, "offset": offset})


@app.route("/api/cleanup/mark-trash", methods=["POST"])
def api_cleanup_mark_trash():
    """Mark images for trash (soft mark, not deleted yet)."""
    data = request.json or {}
    hashes = set(data.get("hashes", []))
    if not hashes:
        return jsonify({"error": "No hashes provided"}), 400

    db = load_scan_db()
    if not db:
        return jsonify({"error": "No scan data"}), 404

    marked = 0
    for img in db["images"]:
        if img["hash"] in hashes:
            img["trash"] = True
            marked += 1

    save_scan_db(db)
    return jsonify({"ok": True, "marked": marked})


@app.route("/api/cleanup/unmark-trash", methods=["POST"])
def api_cleanup_unmark_trash():
    """Remove trash mark from images."""
    data = request.json or {}
    hashes = set(data.get("hashes", []))

    db = load_scan_db()
    if not db:
        return jsonify({"error": "No scan data"}), 404

    unmarked = 0
    for img in db["images"]:
        if img["hash"] in hashes and img.get("trash"):
            img["trash"] = False
            unmarked += 1

    save_scan_db(db)
    return jsonify({"ok": True, "unmarked": unmarked})


@app.route("/api/cleanup/trash-count")
def api_cleanup_trash_count():
    """Get count of images marked for trash."""
    db = load_scan_db()
    if not db:
        return jsonify({"count": 0, "size_mb": 0})
    count = 0
    size_kb = 0
    for img in db["images"]:
        if img.get("trash"):
            count += 1
            size_kb += img.get("size_kb", 0)
    return jsonify({"count": count, "size_mb": round(size_kb / 1024, 1)})


@app.route("/api/cleanup/confirm-trash", methods=["POST"])
def api_cleanup_confirm_trash():
    """Send all trash-marked images to the OS recycle bin."""
    try:
        from send2trash import send2trash
    except ImportError:
        return jsonify({"error": "send2trash not installed. Run: pip install send2trash"}), 500

    db = load_scan_db()
    if not db:
        return jsonify({"error": "No scan data"}), 404

    trashed = []
    failed = []
    for img in db["images"]:
        if img.get("trash"):
            fpath = img["path"].replace("/", os.sep)
            if os.path.isfile(fpath):
                try:
                    send2trash(fpath)
                    trashed.append(img["hash"])
                except Exception as e:
                    failed.append({"hash": img["hash"], "error": str(e)})
            else:
                trashed.append(img["hash"])  # Already gone

    # Remove trashed entries from scan_db
    db["images"] = [i for i in db["images"] if i["hash"] not in set(trashed)]
    save_scan_db(db)

    return jsonify({
        "ok": True,
        "recycled": len(trashed),
        "failed": len(failed),
        "failures": failed[:10],
    })


# ── Age Assessment (standalone) ──────────────────────────────────────────────

@app.route("/api/age-assess/start", methods=["POST"])
def api_age_assess_start():
    """Run age assessment as a standalone background task on selected folders."""
    if _task["running"]:
        return jsonify({"error": "A task is already running"}), 409

    data = request.json or {}
    folders = data.get("folders", [])
    if not folders:
        return jsonify({"error": "No folders selected"}), 400
    face_mode = data.get("face_mode", "all")  # "all" or "specific"
    person_name = data.get("person_name", "")

    def run_age_assess():
        try:
            _reset_task("age_assess")
            _update_task("Loading age estimation model (first run may download ~500 MB)...")

            from deepface import DeepFace
            import numpy as np

            # Warm up
            _dummy = np.zeros((100, 100, 3), dtype=np.uint8)
            try:
                DeepFace.analyze(_dummy, actions=["age"], enforce_detection=False, silent=True)
            except Exception:
                pass
            _update_task("Model ready.")

            if _is_cancelled():
                _finish_task("Cancelled")
                return

            # Load face references if specific person mode
            ref_encodings = {}
            use_face_filter = False
            if face_mode == "specific" and person_name:
                from curate import load_reference_faces
                face_dir = os.path.join(PROJECT_DIR, "ref_faces")
                if os.path.isdir(face_dir):
                    _update_task(f"Loading face references for {person_name}...")
                    ref_encodings = load_reference_faces(face_dir)
                    if person_name in ref_encodings:
                        use_face_filter = True
                        _update_task(f"Face references loaded for {person_name}.")
                    else:
                        _update_task(f"No face reference found for '{person_name}'. Running on all faces.")

            from curate import IMAGE_EXTS, file_hash, make_thumbnail_b64

            results = []
            total_files = 0
            processed = 0
            face_matched = 0

            # Count files first
            for folder in folders:
                if os.path.isdir(folder):
                    for dp, dn, fns in os.walk(folder):
                        for fn in fns:
                            if os.path.splitext(fn)[1].lower() in IMAGE_EXTS:
                                total_files += 1

            _update_task(f"Found {total_files} images to analyze...")

            for folder in folders:
                if not os.path.isdir(folder):
                    _update_task(f"Folder not found: {folder}")
                    continue

                folder_label = os.path.basename(folder) or folder

                for dp, dn, fns in os.walk(folder):
                    for fn in fns:
                        if _is_cancelled():
                            _finish_task("Cancelled")
                            return

                        ext = os.path.splitext(fn)[1].lower()
                        if ext not in IMAGE_EXTS:
                            continue

                        fpath = os.path.join(dp, fn)
                        processed += 1

                        if processed % 10 == 0:
                            _update_task(f"Analyzing {processed}/{total_files} — {len(results)} faces aged...")

                        # If specific person, check face first
                        if use_face_filter:
                            fc, ff, ok, dist = _fast_face_detect(
                                fpath, ref_encodings, [person_name], 0.6)
                            if person_name not in ff:
                                continue
                            face_matched += 1

                        est = _estimate_age(fpath)
                        if est is not None:
                            thumb = make_thumbnail_b64(fpath, 120)
                            fh = file_hash(fpath)
                            results.append({
                                "path": fpath.replace("\\", "/"),
                                "filename": fn,
                                "folder": folder_label,
                                "hash": fh,
                                "estimated_age": est,
                                "person": person_name if use_face_filter else None,
                                "thumb": thumb,
                            })

            # Also store results in scan_db entries if they exist
            db = load_scan_db()
            if db:
                result_map = {r["hash"]: r["estimated_age"] for r in results}
                updated = 0
                for img in db.get("images", []):
                    if img["hash"] in result_map and "estimated_age" not in img:
                        img["estimated_age"] = result_map[img["hash"]]
                        updated += 1
                if updated > 0:
                    save_scan_db(db)
                    _update_task(f"Updated {updated} entries in scan database.")

            # Save results to a temp file for the UI to fetch
            results_path = os.path.join(PROJECT_DIR, "age_results.json")
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False)

            msg = f"Done! Estimated age for {len(results)} images out of {processed} analyzed."
            if use_face_filter:
                msg += f" ({face_matched} photos matched {person_name})"
            _update_task(msg)
            _finish_task()

        except Exception as e:
            _finish_task(str(e))

    threading.Thread(target=run_age_assess, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/age-assess/results")
def api_age_assess_results():
    """Return saved age assessment results."""
    results_path = os.path.join(PROJECT_DIR, "age_results.json")
    if not os.path.isfile(results_path):
        return jsonify([])
    with open(results_path, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))


# ── Serve UI ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return WIZARD_HTML


# ── Wizard HTML (single page) ────────────────────────────────────────────────

WIZARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>E-z Photo Organizer</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background:#f5f7fa; color:#2d3748; min-height:100vh; }

/* ── Layout ── */
.app { max-width:1200px; margin:0 auto; padding:20px; padding-left:68px; }
.header { text-align:center; padding:30px 0 20px; }
.header h1 { font-size:2em; color:#2b6cb0; margin-bottom:5px; }
.header p { color:#718096; font-size:.9em; }

/* ── Steps nav ── */
.steps { display:flex; justify-content:center; gap:0; margin:25px 0; }
.step-dot {
    display:flex; align-items:center; gap:8px; padding:10px 18px;
    background:#fff; cursor:pointer; font-size:.85em; color:#a0aec0;
    border:1px solid #e2e8f0; transition:all .2s;
}
.step-dot:first-child { border-radius:8px 0 0 8px; }
.step-dot:last-child { border-radius:0 8px 8px 0; }
.step-dot.active { background:#ebf8ff; color:#2b6cb0; border-color:#90cdf4; }
.step-dot.done { background:#f0fff4; color:#276749; border-color:#9ae6b4; }
.step-dot .num {
    width:24px; height:24px; border-radius:50%; background:#e2e8f0; color:#a0aec0;
    display:flex; align-items:center; justify-content:center; font-size:.75em; font-weight:bold;
}
.step-dot.active .num { background:#2b6cb0; color:white; }
.step-dot.done .num { background:#38a169; color:white; }

/* ── Panels ── */
.panel { display:none; background:#fff; border-radius:12px; padding:30px; margin-top:10px; box-shadow:0 1px 3px rgba(0,0,0,.1); border:1px solid #e2e8f0; }
.panel.active { display:block; }
.panel h2 { color:#2b6cb0; margin-bottom:15px; font-size:1.3em; }
.panel p { color:#4a5568; line-height:1.6; margin-bottom:15px; }

/* ── Forms ── */
label { display:block; font-size:.85em; color:#718096; margin-bottom:4px; margin-top:12px; }
input, select { width:100%; padding:10px 12px; border-radius:6px; border:1px solid #cbd5e0; background:#fff; color:#2d3748; font-size:.9em; }
input:focus, select:focus { outline:none; border-color:#63b3ed; box-shadow:0 0 0 3px rgba(66,153,225,.15); }
.row { display:flex; gap:12px; }
.row > * { flex:1; }

/* ── Buttons ── */
.btn {
    padding:10px 24px; border:none; border-radius:6px; cursor:pointer;
    font-size:.9em; font-weight:600; transition:all .15s;
}
.btn-primary { background:#3182ce; color:white; }
.btn-primary:hover { background:#2b6cb0; }
.btn-primary:disabled { background:#cbd5e0; color:#a0aec0; cursor:not-allowed; }
.btn-secondary { background:#edf2f7; color:#2b6cb0; }
.btn-secondary:hover { background:#e2e8f0; }
.btn-danger { background:#fed7d7; color:#c53030; }
.btn-group { display:flex; gap:10px; margin-top:20px; }

/* ── Cards ── */
.template-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(280px, 1fr)); gap:12px; margin-top:15px; }
.template-card {
    background:#fff; border:2px solid #e2e8f0; border-radius:8px; padding:16px;
    cursor:pointer; transition:all .15s;
}
.template-card:hover { border-color:#63b3ed; box-shadow:0 2px 8px rgba(66,153,225,.15); }
.template-card.selected { border-color:#3182ce; background:#ebf8ff; }
.template-card h3 { color:#2b6cb0; margin-bottom:6px; }
.template-card .desc { color:#718096; font-size:.8em; line-height:1.4; }
.template-card .meta { color:#a0aec0; font-size:.75em; margin-top:8px; }

/* ── Sources list ── */
.source-list { margin-top:10px; }
.source-item {
    display:flex; align-items:center; gap:10px; padding:8px 12px;
    background:#f7fafc; border-radius:6px; margin-bottom:6px; border:1px solid #e2e8f0;
}
.source-item input { flex:1; }
.source-item .remove { color:#e53e3e; cursor:pointer; font-size:1.2em; padding:0 8px; }

/* ── Progress ── */
.progress-box {
    background:#1a202c; border-radius:8px; padding:15px; margin-top:15px;
    font-family:monospace; font-size:.8em; max-height:300px; overflow-y:auto;
}
.progress-box .line { padding:2px 0; color:#63b3ed; }
.progress-box .current { color:#fc8181; font-weight:bold; }

/* ── Analysis ── */
.analysis-table { width:100%; border-collapse:collapse; margin-top:10px; }
.analysis-table th, .analysis-table td { padding:8px 12px; text-align:left; border-bottom:1px solid #e2e8f0; font-size:.85em; }
.analysis-table th { color:#718096; }
.status-ok { color:#38a169; }
.status-close { color:#d69e2e; }
.status-low { color:#dd6b20; }
.status-critical { color:#e53e3e; }
.status-empty { color:#e53e3e; font-weight:bold; }
.status-overflow { color:#3182ce; }

.rec-card {
    background:#f7fafc; border-radius:6px; padding:12px; margin-bottom:8px;
    border-left:3px solid #cbd5e0;
}
.rec-card.critical { border-left-color:#e53e3e; }
.rec-card.warning { border-left-color:#dd6b20; }
.rec-card.info { border-left-color:#3182ce; }
.rec-card.tip { border-left-color:#38a169; }
.rec-card .title { font-weight:600; margin-bottom:4px; color:#2d3748; }
.rec-card .detail { color:#718096; font-size:.85em; }

/* ── Gallery (embedded) ── */
.gallery-frame { width:100%; height:80vh; border:none; border-radius:8px; margin-top:15px; background:#f7fafc; }

/* ── Export ── */
.export-summary { background:#ebf8ff; border-radius:8px; padding:20px; margin-top:15px; border:1px solid #bee3f8; }
.export-summary .big-num { font-size:2.5em; color:#2b6cb0; font-weight:bold; }
@keyframes spin { to { transform: translate(-50%,-50%) rotate(360deg); } }

/* ── Lightbox ── */
.lightbox { display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,.7); z-index:9999; align-items:center; justify-content:center; }
.lightbox.open { display:flex; }
.lightbox img { max-width:90vw; max-height:90vh; border-radius:8px; box-shadow:0 4px 30px rgba(0,0,0,.4); }
.lightbox .close-btn { position:absolute; top:20px; right:30px; font-size:2em; color:white; cursor:pointer; background:rgba(0,0,0,.5); border:none; border-radius:50%; width:44px; height:44px; display:flex; align-items:center; justify-content:center; }
.lightbox .close-btn:hover { background:rgba(0,0,0,.8); }

/* ── Inline spinner ── */
.inline-loader { display:inline-flex; align-items:center; gap:10px; padding:12px 16px; background:#ebf8ff; border:1px solid #bee3f8; border-radius:8px; color:#2b6cb0; font-size:.9em; }
.inline-loader .spin { width:18px; height:18px; border:3px solid #bee3f8; border-top:3px solid #3182ce; border-radius:50%; animation:spinA .7s linear infinite; }
@keyframes spinA { to { transform:rotate(360deg); } }

/* ── Task overlay ── */
#task-overlay { display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(255,255,255,.88); z-index:9990; align-items:center; justify-content:center; flex-direction:column; }
#task-overlay.active { display:flex; }
#task-overlay .task-box { background:#fff; border-radius:12px; box-shadow:0 4px 24px rgba(0,0,0,.12); padding:30px 40px; min-width:400px; max-width:600px; text-align:center; }
#task-overlay .task-title { font-size:1.2em; font-weight:600; color:#2d3748; margin-bottom:12px; }
#task-overlay .task-status { font-size:.9em; color:#718096; margin-bottom:16px; min-height:1.2em; }
#task-overlay .task-bar-bg { height:8px; background:#e2e8f0; border-radius:4px; overflow:hidden; margin-bottom:16px; }
#task-overlay .task-bar { height:100%; background:linear-gradient(90deg, #3182ce, #63b3ed); border-radius:4px; transition:width .5s ease; }
@keyframes taskPulse { 0%,100% { opacity:.6; } 50% { opacity:1; } }
#task-overlay .task-bar.indeterminate { width:100% !important; animation: taskPulse 1.5s ease-in-out infinite; }
#task-overlay .task-lines { text-align:left; max-height:120px; overflow-y:auto; font-size:.8em; color:#718096; background:#f7fafc; border-radius:6px; padding:8px 12px; margin-bottom:16px; }
#task-overlay .task-lines .line { margin:2px 0; }
#task-overlay .btn-stop { background:#e53e3e; color:#fff; border:none; padding:8px 24px; border-radius:6px; font-size:.9em; cursor:pointer; }
#task-overlay .btn-stop:hover { background:#c53030; }

/* ── Side menu (projects) ── */
/* ── Icon rail (always visible) ── */
#icon-rail { position:fixed; top:0; left:0; width:48px; height:100%; background:#fff; z-index:10000; display:flex; flex-direction:column; align-items:center; padding:10px 0; border-right:1px solid #e2e8f0; box-shadow:1px 0 6px rgba(0,0,0,.04); }
#icon-rail .rail-btn { width:36px; height:36px; border:none; background:none; border-radius:8px; cursor:pointer; display:flex; align-items:center; justify-content:center; color:#718096; transition:all .15s; position:relative; margin-bottom:4px; }
#icon-rail .rail-btn:hover { background:#ebf8ff; color:#2b6cb0; }
#icon-rail .rail-btn.active { background:#ebf8ff; color:#2b6cb0; }
#icon-rail .rail-btn svg { width:20px; height:20px; }
#icon-rail .rail-btn .rail-tip { display:none; position:absolute; left:46px; top:50%; transform:translateY(-50%); background:#2d3748; color:#fff; font-size:.72em; padding:4px 10px; border-radius:5px; white-space:nowrap; pointer-events:none; z-index:10; }
#icon-rail .rail-btn:hover .rail-tip { display:block; }
#icon-rail .rail-spacer { flex:1; }
#icon-rail .rail-divider { width:24px; height:1px; background:#e2e8f0; margin:6px 0; }
.menu-btn { display:none; }
/* ── Cleanup overlay ── */
#cleanup-overlay { display:none; position:fixed; top:0; left:48px; width:calc(100% - 48px); height:100%; background:#f5f7fa; z-index:1300; overflow-y:auto; }
#cleanup-overlay.active { display:block; }
#cleanup-overlay .cleanup-header { position:sticky; top:0; background:#fff; z-index:10; padding:16px 24px; border-bottom:1px solid #e2e8f0; display:flex; justify-content:space-between; align-items:center; box-shadow:0 2px 8px rgba(0,0,0,.05); }
#cleanup-overlay .cleanup-header h2 { margin:0; color:#2d3748; font-size:1.3em; }
#cleanup-overlay .cleanup-body { padding:20px 24px; max-width:1400px; margin:0 auto; }
#cleanup-overlay .cleanup-filters { display:flex; gap:12px; flex-wrap:wrap; align-items:center; margin-bottom:16px; }
#cleanup-overlay .cleanup-filters select, #cleanup-overlay .cleanup-filters input { padding:6px 10px; border:1px solid #e2e8f0; border-radius:6px; font-size:.85em; }
#age-overlay { display:none; position:fixed; top:0; left:48px; width:calc(100% - 48px); height:100%; background:#f5f7fa; z-index:1300; overflow-y:auto; }
#age-overlay.active { display:block; }
#age-overlay .age-header { position:sticky; top:0; background:#fff; z-index:10; padding:16px 24px; border-bottom:1px solid #e2e8f0; display:flex; justify-content:space-between; align-items:center; box-shadow:0 2px 8px rgba(0,0,0,.05); }
#age-overlay .age-header h2 { margin:0; color:#2d3748; font-size:1.3em; }
#age-overlay .age-body { padding:20px 24px; max-width:1200px; margin:0 auto; }
#age-overlay .age-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(140px, 1fr)); gap:12px; }
#age-overlay .age-card { background:#fff; border:1px solid #e2e8f0; border-radius:8px; overflow:hidden; text-align:center; transition:box-shadow .2s; }
#age-overlay .age-card:hover { box-shadow:0 4px 12px rgba(0,0,0,.1); }
#age-overlay .age-card img { width:100%; height:120px; object-fit:cover; }
#age-overlay .age-card .age-info { padding:6px 8px; font-size:.78em; color:#4a5568; }
#age-overlay .age-card .age-badge { display:inline-block; background:#ebf8ff; color:#2b6cb0; padding:2px 8px; border-radius:10px; font-weight:600; font-size:.85em; }
.cleanup-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(150px, 1fr)); gap:10px; }
.cleanup-card { position:relative; border-radius:8px; overflow:hidden; cursor:pointer; border:2px solid transparent; transition:all .15s; background:#fff; box-shadow:0 1px 3px rgba(0,0,0,.08); }
.cleanup-card:hover { box-shadow:0 2px 8px rgba(0,0,0,.15); }
.cleanup-card.trashed { border-color:#e53e3e; opacity:.7; }
.cleanup-card.trashed::after { content:'TRASH'; position:absolute; top:50%; left:50%; transform:translate(-50%,-50%) rotate(-20deg); font-size:1.4em; font-weight:900; color:#e53e3e; background:rgba(255,255,255,.85); padding:4px 16px; border-radius:6px; pointer-events:none; }
.cleanup-card img { width:100%; height:130px; object-fit:cover; display:block; }
.cleanup-card .card-info { padding:6px 8px; font-size:.72em; color:#718096; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.cleanup-trash-bar { position:sticky; bottom:0; background:#fff; border-top:1px solid #e2e8f0; padding:12px 24px; display:flex; justify-content:space-between; align-items:center; box-shadow:0 -2px 8px rgba(0,0,0,.05); z-index:10; }
.cleanup-trash-bar .trash-count { font-size:.95em; color:#4a5568; font-weight:600; }
.cleanup-trash-bar .trash-size { font-size:.8em; color:#a0aec0; margin-left:8px; }

#side-drawer { position:fixed; top:0; left:-320px; width:300px; height:100%; background:#fff; z-index:1150; box-shadow:4px 0 20px rgba(0,0,0,.1); transition:left .25s ease; display:flex; flex-direction:column; padding-left:48px; }
#side-drawer.open { left:0; }
#side-drawer .drawer-header { padding:16px 18px; border-bottom:1px solid #e2e8f0; display:flex; justify-content:space-between; align-items:center; }
#side-drawer .drawer-header h3 { color:#2d3748; margin:0; font-size:1em; }
#side-drawer .drawer-close { background:none; border:none; font-size:1.4em; cursor:pointer; color:#718096; padding:0 4px; }
#side-drawer .drawer-close:hover { color:#2d3748; }
#side-drawer .drawer-actions { padding:12px 18px; display:flex; gap:8px; border-bottom:1px solid #e2e8f0; }
#side-drawer .drawer-actions button { flex:1; padding:7px 0; border-radius:5px; font-size:.8em; cursor:pointer; border:1px solid #e2e8f0; background:#f7fafc; color:#4a5568; }
#side-drawer .drawer-actions button:hover { background:#ebf8ff; border-color:#90cdf4; color:#2b6cb0; }
#side-drawer .project-list { flex:1; overflow-y:auto; padding:8px 0; }
#side-drawer .project-item { padding:10px 18px; cursor:pointer; border-bottom:1px solid #f7fafc; transition:background .1s; }
#side-drawer .project-item:hover { background:#f0f9ff; }
#side-drawer .project-item .p-name { font-weight:500; color:#2d3748; font-size:.9em; }
#side-drawer .project-item .p-meta { color:#a0aec0; font-size:.72em; margin-top:2px; }
#side-drawer .project-item .p-actions { display:flex; gap:6px; margin-top:6px; }
#side-drawer .project-item .p-del { background:none; border:none; color:#e53e3e; cursor:pointer; font-size:.75em; padding:2px 6px; border-radius:3px; }
#side-drawer .project-item .p-del:hover { background:#fed7d7; }
#drawer-backdrop { display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,.2); z-index:1150; }
#drawer-backdrop.open { display:block; }

/* ── Tutorial overlay ── */
#tour-backdrop { display:none; position:fixed; top:0; left:0; width:100%; height:100%; z-index:2000; }
#tour-backdrop.active { display:block; }
#tour-highlight {
    position:absolute; z-index:2001; border-radius:8px;
    box-shadow: 0 0 0 4000px rgba(0,0,0,.55);
    transition: all .4s cubic-bezier(.4,0,.2,1);
    pointer-events:none;
}
#tour-tooltip {
    position:absolute; z-index:2002; background:white; border-radius:12px; padding:20px 24px;
    box-shadow: 0 8px 30px rgba(0,0,0,.2); max-width:360px; min-width:260px;
    opacity:0; transform:translateY(12px); transition: opacity .35s ease, transform .35s ease;
}
#tour-tooltip.show { opacity:1; transform:translateY(0); }
#tour-tooltip h3 { font-size:1em; color:#2b6cb0; margin-bottom:6px; }
#tour-tooltip p { font-size:.88em; color:#4a5568; line-height:1.5; margin:0 0 14px; }
#tour-tooltip .tour-actions { display:flex; justify-content:space-between; align-items:center; }
#tour-tooltip .tour-dots { display:flex; gap:5px; }
#tour-tooltip .tour-dot { width:7px; height:7px; border-radius:50%; background:#e2e8f0; }
#tour-tooltip .tour-dot.active { background:#667eea; }
#tour-tooltip .tour-skip { font-size:.8em; color:#a0aec0; cursor:pointer; border:none; background:none; }
#tour-tooltip .tour-skip:hover { color:#718096; }
#tour-tooltip .tour-next {
    padding:7px 18px; border:none; border-radius:6px; background:#667eea; color:white;
    font-size:.85em; font-weight:600; cursor:pointer; transition: background .2s;
}
#tour-tooltip .tour-next:hover { background:#5a67d8; }
@keyframes tour-pulse { 0%,100% { box-shadow: 0 0 0 4000px rgba(0,0,0,.55); } 50% { box-shadow: 0 0 0 4000px rgba(0,0,0,.5), 0 0 0 8px rgba(102,126,234,.4); } }
#tour-highlight.pulse { animation: tour-pulse 1.5s ease infinite; }
</style>
</head>
<body>

<!-- Icon rail (always visible) -->
<div id="icon-rail">
    <button class="rail-btn" onclick="toggleDrawer()" id="rail-toggle" title="Menu">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
        <span class="rail-tip">Menu</span>
    </button>
    <div class="rail-divider"></div>
    <button class="rail-btn" onclick="newProject()" title="New Project">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
        <span class="rail-tip">New Project</span>
    </button>
    <button class="rail-btn" onclick="openDrawerToProjects()" title="Saved Projects">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z"/></svg>
        <span class="rail-tip">Saved Projects</span>
    </button>
    <div class="rail-divider"></div>
    <div style="position:relative;" id="rail-cleanup-group">
        <button class="rail-btn" onclick="toggleRailCleanup()" title="Cleanup Assistant" id="rail-cleanup-btn">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg>
            <span class="rail-tip">Cleanup Assistant</span>
        </button>
        <div id="rail-cleanup-sub" style="display:none; position:absolute; left:46px; top:0; background:#fff; border:1px solid #e2e8f0; border-radius:8px; box-shadow:0 4px 16px rgba(0,0,0,.12); padding:6px; z-index:20; white-space:nowrap;">
            <button onclick="openCleanup(); closeRailCleanup();" style="display:flex; align-items:center; gap:8px; width:100%; padding:8px 14px; border:none; background:none; cursor:pointer; border-radius:6px; font-size:.82em; color:#4a5568; text-align:left;" onmouseover="this.style.background='#ebf8ff'" onmouseout="this.style.background='none'">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="2" y="3" width="20" height="14" rx="2" ry="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>
                Hard Drives
            </button>
            <button onclick="openPhoneImages(); closeRailCleanup();" style="display:flex; align-items:center; gap:8px; width:100%; padding:8px 14px; border:none; background:none; cursor:pointer; border-radius:6px; font-size:.82em; color:#4a5568; text-align:left;" onmouseover="this.style.background='#ebf8ff'" onmouseout="this.style.background='none'">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="5" y="2" width="14" height="20" rx="2" ry="2"/><line x1="12" y1="18" x2="12.01" y2="18"/></svg>
                Mobile Images
            </button>
        </div>
    </div>
    <button class="rail-btn" onclick="openAgeAssessment()" title="Age Assessment">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="8" r="4"/><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><path d="M16 3.13a4 4 0 010 7.75"/></svg>
        <span class="rail-tip">Age Assessment</span>
    </button>
    <div class="rail-spacer"></div>
    <div class="rail-divider"></div>
    <button class="rail-btn" onclick="startTutorial()" title="Tutorial">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 015.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
        <span class="rail-tip">Tutorial</span>
    </button>
    <button class="rail-btn" onclick="logout()" title="Sign Out">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>
        <span class="rail-tip">Sign Out</span>
    </button>
</div>

<!-- Side drawer (slides from behind icon rail) -->
<div id="drawer-backdrop" onclick="toggleDrawer()"></div>
<div id="side-drawer">
    <div class="drawer-header">
        <h3 id="drawer-title">Menu</h3>
        <button class="drawer-close" onclick="toggleDrawer()">&times;</button>
    </div>

    <!-- New Project button -->
    <div style="padding:12px 18px; border-bottom:1px solid #e2e8f0;">
        <button onclick="newProject()" style="width:100%; padding:10px 12px; border:1px solid #bee3f8; border-radius:8px; background:#ebf8ff; color:#2b6cb0; font-size:.85em; font-weight:600; cursor:pointer; display:flex; align-items:center; gap:8px;">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
            New Project
        </button>
    </div>

    <!-- Saved Projects section (collapsible) -->
    <div style="border-bottom:1px solid #e2e8f0;">
        <div onclick="toggleProjectsSection()" style="display:flex; align-items:center; justify-content:space-between; padding:12px 18px; cursor:pointer; user-select:none;" id="projects-toggle">
            <span style="font-weight:600; font-size:.9em; color:#2d3748; display:flex; align-items:center; gap:8px;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z"/></svg>
                Saved Projects
            </span>
            <svg id="projects-arrow" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" style="transition:transform .2s;"><polyline points="6 9 12 15 18 9"/></svg>
        </div>
        <div id="projects-section" style="display:none;">
            <div class="project-list" id="project-list">
                <div style="padding:18px; color:#a0aec0; font-size:.85em; text-align:center;">Loading...</div>
            </div>
        </div>
    </div>

    <!-- Cleanup section -->
    <div style="border-bottom:1px solid #e2e8f0;">
        <div onclick="toggleCleanupSection()" style="display:flex; align-items:center; justify-content:space-between; padding:12px 18px; cursor:pointer; user-select:none;">
            <span style="font-weight:600; font-size:.9em; color:#2d3748; display:flex; align-items:center; gap:8px;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg>
                Cleanup
            </span>
            <svg id="cleanup-arrow" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" style="transition:transform .2s;"><polyline points="6 9 12 15 18 9"/></svg>
        </div>
        <div id="cleanup-section" style="display:none; padding:0 18px 12px;">
            <button onclick="toggleDrawer(); openCleanup();" style="width:100%; padding:10px 12px; border:1px solid #e2e8f0; border-radius:8px; background:#f7fafc; color:#4a5568; font-size:.85em; cursor:pointer; margin-bottom:6px; display:flex; align-items:center; gap:8px; text-align:left;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="2" y="3" width="20" height="14" rx="2" ry="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>
                Desktop / Laptop / Portable Memory
            </button>
            <button onclick="toggleDrawer(); openPhoneImages();" style="width:100%; padding:10px 12px; border:1px solid #e2e8f0; border-radius:8px; background:#f7fafc; color:#4a5568; font-size:.85em; cursor:pointer; display:flex; align-items:center; gap:8px; text-align:left;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="5" y="2" width="14" height="20" rx="2" ry="2"/><line x1="12" y1="18" x2="12.01" y2="18"/></svg>
                Mobile
            </button>
        </div>
    </div>

    <!-- Age Assessment -->
    <div style="border-bottom:1px solid #e2e8f0;">
        <button onclick="toggleDrawer(); openAgeAssessment();" style="display:flex; align-items:center; gap:8px; padding:12px 18px; width:100%; border:none; background:none; cursor:pointer; font-weight:600; font-size:.9em; color:#2d3748; text-align:left;">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="8" r="4"/><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><path d="M16 3.13a4 4 0 010 7.75"/></svg>
            Age Assessment
        </button>
    </div>

    <!-- Bottom actions -->
    <div style="flex:1;"></div>
    <div style="padding:12px 18px; border-top:1px solid #e2e8f0;">
        <div id="user-info" style="font-size:.8em; color:#718096; margin-bottom:8px;"></div>
        <button onclick="toggleDrawer(); startTutorial();" style="width:100%; padding:8px; border:1px solid #e2e8f0; border-radius:6px; background:#f7fafc; color:#4a5568; font-size:.85em; cursor:pointer; margin-bottom:8px; display:flex; align-items:center; justify-content:center; gap:6px;">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 015.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
            Tutorial
        </button>
        <button onclick="logout()" style="width:100%; padding:8px; border:1px solid #fed7d7; border-radius:6px; background:#fff5f5; color:#e53e3e; font-size:.85em; cursor:pointer; display:flex; align-items:center; justify-content:center; gap:6px;">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>
            Sign Out
        </button>
    </div>
</div>

<!-- Cleanup overlay -->
<div id="cleanup-overlay">
    <div class="cleanup-header">
        <h2>Cleanup Tool</h2>
        <div style="display:flex; gap:10px; align-items:center;">
            <span id="cleanup-trash-badge" style="background:#fed7d7; color:#c53030; padding:4px 12px; border-radius:12px; font-size:.85em; font-weight:600; display:none;">0 in trash</span>
            <button class="btn btn-secondary" onclick="showCleanupTrash()" id="btn-review-trash" style="display:none;">Review Trash</button>
            <button class="btn btn-secondary" onclick="closeCleanup()" style="font-size:.85em;">Close</button>
        </div>
    </div>
    <div class="cleanup-body">
        <div class="cleanup-filters">
            <select id="cleanup-filter-cat" onchange="loadCleanupImages()">
                <option value="">All Categories</option>
            </select>
            <select id="cleanup-filter-status" onchange="loadCleanupImages()">
                <option value="">All Statuses</option>
                <option value="candidate">Candidate</option>
                <option value="qualified">Qualified</option>
                <option value="selected">Selected</option>
                <option value="pool">Rejected / Pool</option>
            </select>
            <select id="cleanup-filter-media" onchange="loadCleanupImages()">
                <option value="">All Media</option>
                <option value="image">Images Only</option>
                <option value="video">Videos Only</option>
            </select>
            <label style="display:flex; align-items:center; gap:4px; font-size:.85em; color:#4a5568; cursor:pointer;">
                <input type="checkbox" id="cleanup-show-trash" onchange="loadCleanupImages()"> Show trash only
            </label>
            <div style="flex:1;"></div>
            <button class="btn btn-secondary" onclick="cleanupSelectAll()" style="font-size:.8em; padding:6px 14px;">Select All Visible</button>
            <button class="btn btn-secondary" onclick="cleanupDeselectAll()" style="font-size:.8em; padding:6px 14px;">Deselect All</button>
        </div>
        <div id="cleanup-grid" class="cleanup-grid">
            <div style="grid-column:1/-1; text-align:center; color:#a0aec0; padding:40px;">Open the cleanup tool to review and trash unwanted images.</div>
        </div>
        <div id="cleanup-load-more" style="text-align:center; padding:16px; display:none;">
            <button class="btn btn-secondary" onclick="loadCleanupMore()">Load More</button>
        </div>
    </div>
    <div class="cleanup-trash-bar" id="cleanup-bottom-bar" style="display:none;">
        <div>
            <span class="trash-count" id="cleanup-bar-count">0 items in trash</span>
            <span class="trash-size" id="cleanup-bar-size"></span>
        </div>
        <div style="display:flex; gap:10px;">
            <button class="btn btn-secondary" onclick="cleanupClearTrash()">Clear All Trash Marks</button>
            <button class="btn" style="background:#e53e3e; color:#fff;" onclick="cleanupConfirmRecycle()">Move to Recycle Bin</button>
        </div>
    </div>
</div>

<!-- Age Assessment overlay -->
<div id="age-overlay">
    <div class="age-header">
        <h2>Age Assessment</h2>
        <div style="display:flex; gap:10px; align-items:center;">
            <span id="age-status" style="font-size:.85em; color:#718096;"></span>
            <button class="btn btn-secondary" onclick="closeAgeAssessment()" style="font-size:.85em;">Close</button>
        </div>
    </div>
    <div class="age-body">
        <p style="color:#4a5568; font-size:.9em; margin-bottom:16px;">Estimate the age of people in your photos using AI face analysis.</p>

        <div id="age-setup" style="margin-bottom:20px;">
            <!-- Step 1: Face mode -->
            <div style="font-weight:600; font-size:.9em; color:#2d3748; margin-bottom:6px;">1. Who to analyze</div>
            <div style="margin-bottom:12px;">
                <table style="border-collapse:collapse;">
                    <tr>
                        <td style="padding:2px 6px 2px 0; vertical-align:middle;"><input type="radio" name="age-face-mode" value="all" checked style="margin:0; cursor:pointer;" onchange="toggleAgeFaceMode()"></td>
                        <td style="padding:2px 0; font-size:.82em; cursor:pointer;" onclick="this.previousElementSibling.querySelector('input').checked=true; toggleAgeFaceMode()"><strong style="color:#4a5568;">All faces</strong> <span style="color:#a0aec0;">— Estimate age for every face found in each image</span></td>
                    </tr>
                    <tr>
                        <td style="padding:2px 6px 2px 0; vertical-align:middle;"><input type="radio" name="age-face-mode" value="specific" style="margin:0; cursor:pointer;" onchange="toggleAgeFaceMode()"></td>
                        <td style="padding:2px 0; font-size:.82em; cursor:pointer;" onclick="this.previousElementSibling.querySelector('input').checked=true; toggleAgeFaceMode()"><strong style="color:#4a5568;">Specific person</strong> <span style="color:#a0aec0;">— Only estimate age for a recognized person (requires reference photos)</span></td>
                    </tr>
                </table>
            </div>

            <!-- Face reference upload (shown only for specific person) -->
            <div id="age-face-ref" style="display:none; margin-bottom:14px; padding:12px 16px; background:#f7fafc; border:1px solid #e2e8f0; border-radius:8px;">
                <div style="font-weight:600; font-size:.85em; color:#2d3748; margin-bottom:6px;">Reference photos for recognition</div>
                <div style="display:flex; gap:8px; align-items:center; margin-bottom:10px;">
                    <input type="text" id="age-ref-person-name" placeholder="Person name (e.g. Reef)" style="padding:6px 10px; border:1px solid #e2e8f0; border-radius:6px; font-size:.82em; max-width:180px;">
                    <label class="btn btn-secondary" style="font-size:.8em; padding:6px 12px; margin:0; cursor:pointer;" for="age-ref-upload">+ Add Photos</label>
                    <input type="file" id="age-ref-upload" multiple accept="image/*" style="display:none" onchange="uploadAgeRefPhotos(this.files)">
                    <button class="btn btn-secondary" id="btn-age-verify" onclick="verifyAgeRefFaces()" style="font-size:.8em; padding:6px 12px; margin:0; display:none;">Verify Faces</button>
                </div>
                <div id="age-ref-status" style="font-size:.82em; color:#718096; margin-bottom:6px;"></div>
                <div id="age-ref-thumbs" style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:8px;"></div>
                <div id="age-verify-result" style="display:none; font-size:.82em; margin-bottom:8px; padding:8px 12px; border-radius:6px;"></div>
                <div style="font-size:.75em; color:#a0aec0;">Upload 3-5 clear face photos from different angles. Verify before running.</div>
                <div id="age-use-existing" style="margin-top:8px;"></div>
            </div>

            <!-- Step 2: Folders -->
            <div style="font-weight:600; font-size:.9em; color:#2d3748; margin-bottom:6px;">2. Select folders to analyze</div>
            <div id="age-folder-list" style="margin-bottom:12px;"></div>

            <!-- Run -->
            <div style="display:flex; gap:10px; align-items:center; margin-bottom:12px;">
                <button class="btn btn-primary" id="btn-run-age" onclick="runAgeAssessment()">Run Age Assessment</button>
                <button class="btn btn-secondary" id="btn-stop-age" onclick="stopAgeAssessment()" style="display:none;">Stop</button>
            </div>
            <div id="age-progress" style="display:none; padding:10px 14px; background:#f7fafc; border:1px solid #e2e8f0; border-radius:6px; font-size:.85em; color:#4a5568;"></div>
        </div>

        <div id="age-results" style="display:none;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <div style="font-weight:600; font-size:.95em; color:#2d3748;">Results <span id="age-result-count" style="font-weight:400; color:#718096;"></span></div>
                <div style="display:flex; gap:8px; align-items:center;">
                    <label style="font-size:.8em; color:#4a5568;">Sort:</label>
                    <select id="age-sort" onchange="renderAgeResults()" style="padding:4px 8px; border:1px solid #e2e8f0; border-radius:5px; font-size:.8em;">
                        <option value="age-asc">Age (youngest first)</option>
                        <option value="age-desc">Age (oldest first)</option>
                        <option value="name">Person name</option>
                        <option value="date">Photo date</option>
                    </select>
                    <label style="font-size:.8em; color:#4a5568;">Filter person:</label>
                    <select id="age-filter-person" onchange="renderAgeResults()" style="padding:4px 8px; border:1px solid #e2e8f0; border-radius:5px; font-size:.8em;">
                        <option value="">All</option>
                    </select>
                </div>
            </div>
            <div id="age-grid" class="age-grid"></div>
        </div>
    </div>
</div>

<!-- Task overlay -->
<div id="task-overlay">
    <div class="task-box">
        <div class="task-title" id="task-overlay-title">Working...</div>
        <div class="task-status" id="task-overlay-status"></div>
        <div class="task-bar-bg"><div class="task-bar indeterminate" id="task-overlay-bar"></div></div>
        <div class="task-lines" id="task-overlay-lines"></div>
        <button class="btn-stop" onclick="stopTask()">Stop</button>
    </div>
</div>

<div class="app">

<div class="header">
    <h1>E-z Photo Organizer</h1>
    <p id="header-greeting">Build a meaningful photo collection for your special event</p>
</div>

<div style="display:flex; align-items:center; gap:12px; margin:25px 0;">
    <div class="steps" id="steps-nav" style="margin:0;">
        <div class="step-dot active" onclick="goStep(0)"><span class="num">1</span> Event</div>
        <div class="step-dot" onclick="goStep(1)"><span class="num">2</span> Categories</div>
        <div class="step-dot" onclick="goStep(2)"><span class="num">3</span> Sources</div>
        <div class="step-dot" onclick="goStep(3)"><span class="num">4</span> Faces</div>
        <div class="step-dot" onclick="goStep(4)"><span class="num">5</span> Scan</div>
        <div class="step-dot" onclick="goStep(5)"><span class="num">6</span> Analyze</div>
        <div class="step-dot" onclick="goStep(6)"><span class="num">7</span> Select</div>
        <div class="step-dot" onclick="goStep(7)"><span class="num">8</span> Review</div>
        <div class="step-dot" onclick="goStep(8)"><span class="num">9</span> Export</div>
    </div>
    <button id="btn-save-project" onclick="saveCurrentProject()" title="Save Project" style="flex-shrink:0; width:38px; height:38px; border:1px solid #e2e8f0; background:#fff; border-radius:8px; cursor:pointer; display:flex; align-items:center; justify-content:center; color:#4a5568; transition:all .2s;" onmouseover="this.style.background='#ebf8ff';this.style.borderColor='#90cdf4';this.style.color='#2b6cb0'" onmouseout="this.style.background='#fff';this.style.borderColor='#e2e8f0';this.style.color='#4a5568'">
        <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M19 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v11a2 2 0 01-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg>
    </button>
</div>

<!-- ── STEP 0: Choose Event ── -->
<div class="panel active" id="panel-0">
    <h2>What are you creating?</h2>
    <p>Choose the type of event. This sets up the right categories and targets for your photo collection.</p>
    <div class="template-grid" id="template-grid"></div>
    <div id="more-events" style="margin-top:18px; display:none;">
        <div style="color:#718096; font-size:.85em; margin-bottom:6px; cursor:pointer;" onclick="document.getElementById('more-events-list').style.display = document.getElementById('more-events-list').style.display === 'none' ? 'block' : 'none'; this.querySelector('span').textContent = document.getElementById('more-events-list').style.display === 'none' ? '&#9654;' : '&#9660;'">
            <span>&#9654;</span> More events...
        </div>
        <div id="more-events-list" style="display:none;"></div>
    </div>

    <div id="event-fields" style="margin-top:20px; display:none;">
        <div class="row">
            <div id="field-birthday" style="display:none">
                <label>Child's Birthday</label>
                <div style="display:flex; align-items:center; gap:10px;">
                    <input type="date" id="inp-birthday" style="width:160px;" max="">
                    <label style="font-size:.8em; color:#718096; display:flex; align-items:center; gap:4px; white-space:nowrap;"><input type="checkbox" id="skip-birthday" onchange="toggleSkipDate('birthday')"> Not needed</label>
                </div>
            </div>
            <div id="field-event-date" style="display:none">
                <label>Event Date</label>
                <div style="display:flex; align-items:center; gap:10px;">
                    <input type="date" id="inp-event-date" style="width:160px;">
                    <label style="font-size:.8em; color:#718096; display:flex; align-items:center; gap:4px; white-space:nowrap;"><input type="checkbox" id="skip-event-date" onchange="toggleSkipDate('event-date')"> Not needed</label>
                </div>
            </div>
            <div id="field-end-date" style="display:none">
                <label>End Date</label>
                <div style="display:flex; align-items:center; gap:10px;">
                    <input type="date" id="inp-end-date" style="width:160px;">
                    <label style="font-size:.8em; color:#718096; display:flex; align-items:center; gap:4px; white-space:nowrap;"><input type="checkbox" id="skip-end-date" onchange="toggleSkipDate('end-date')"> Not needed</label>
                </div>
            </div>
            <div id="field-year" style="display:none">
                <label>Year</label>
                <div style="display:flex; align-items:center; gap:10px;">
                    <input type="number" id="inp-year" placeholder="2024" min="2000" max="2030" style="width:100px;">
                    <label style="font-size:.8em; color:#718096; display:flex; align-items:center; gap:4px; white-space:nowrap;"><input type="checkbox" id="skip-year" onchange="toggleSkipDate('year')"> Not needed</label>
                </div>
            </div>
        </div>
    </div>

    <div id="event-tips" style="margin-top:15px; display:none;"></div>

    <div class="btn-group">
        <button class="btn btn-primary" id="btn-next-0" disabled onclick="completeStep0()">Next: Categories</button>
    </div>
</div>

<!-- ── STEP 1: Categories ── -->
<div class="panel" id="panel-1">
    <h2>Customize Your Categories</h2>
    <p>These categories come from your event template. Adjust names, targets, or add/remove as needed.</p>

    <div style="margin-bottom:16px;">
        <label style="font-weight:600; font-size:.9em; color:#2d3748;">Project Name <span style="color:#e53e3e;">*</span></label>
        <input type="text" id="inp-project-name" placeholder="e.g. Reef's Bar Mitzva Photos" style="width:100%; max-width:400px; margin-top:4px; padding:8px 12px; border:1px solid #e2e8f0; border-radius:6px; font-size:.9em;" oninput="validateProjectName()">
        <div id="project-name-error" style="color:#e53e3e; font-size:.8em; margin-top:4px; display:none;">Project name is required</div>
    </div>

    <div style="display:flex; gap:20px; margin-bottom:15px; flex-wrap:wrap;">
        <div style="background:#ebf8ff; border:1px solid #bee3f8; border-radius:8px; padding:15px; flex:1; min-width:150px; text-align:center;">
            <div style="font-size:1.8em; font-weight:bold; color:#2b6cb0;" id="cat-total-count">0</div>
            <div style="color:#718096; font-size:.85em;">Categories</div>
        </div>
        <div style="background:#f0fff4; border:1px solid #c6f6d5; border-radius:8px; padding:15px; flex:1; min-width:150px; text-align:center;">
            <div style="font-size:1.8em; font-weight:bold; color:#38a169;" id="cat-total-target">0</div>
            <div style="color:#718096; font-size:.85em;">Total Target (images + videos)</div>
        </div>
    </div>

    <div style="margin-bottom:12px;">
        <label style="display:inline-flex; align-items:center; gap:6px; cursor:pointer; font-size:.85em; color:#4a5568; font-weight:600; white-space:nowrap;">
            <input type="checkbox" id="chk-unlimited" onchange="toggleUnlimited(this.checked)">
            Select all matching media (no count limit)
            <span style="font-weight:400; font-size:.88em; color:#a0aec0;">— Selects every image/video that matches your face reference</span>
        </label>
    </div>

    <table class="analysis-table" id="cat-table">
        <thead>
            <tr>
                <th style="width:40px;">#</th>
                <th>Category Name</th>
                <th style="width:100px;">Photos</th>
                <th style="width:100px;">Videos</th>
                <th style="width:60px;"></th>
            </tr>
        </thead>
        <tbody id="cat-tbody"></tbody>
    </table>

    <div class="btn-group">
        <button class="btn btn-secondary" onclick="addCategory()">+ Add Category</button>
    </div>

    <div class="btn-group">
        <button class="btn btn-secondary" onclick="goStep(0)">Back</button>
        <button class="btn btn-primary" onclick="completeCategoriesStep()">Next: Add Sources</button>
    </div>
</div>

<!-- ── STEP 2: Sources ── -->
<div class="panel" id="panel-2">
    <h2>Where are your photos?</h2>
    <p>Add the folders where your photos are stored. These can be USB drives, cloud exports, phone backups, etc. The more sources, the better your collection.</p>

    <div class="source-list" id="source-list"></div>
    <button class="btn btn-secondary" onclick="addSource()" style="margin-top:10px">+ Add Source</button>

    <div style="margin-top:14px; padding:10px 14px; background:#fffff0; border:1px solid #fefcbf; border-radius:6px; font-size:.82em; color:#744210;">
        <strong>Tip:</strong> If your images are not properly rotated (e.g. sideways photos from phones), face detection may miss faces and scanning will take longer as it retries with rotation correction. For best results, make sure your source images have correct EXIF orientation.
    </div>

    <div class="btn-group">
        <button class="btn btn-secondary" onclick="goStep(1)">Back</button>
        <button class="btn btn-primary" onclick="completeStep2()">Next: Face References</button>
    </div>
</div>

<!-- ── STEP 3: Face References ── -->
<div class="panel" id="panel-3">
    <h2>Face Recognition Setup</h2>
    <p>Add reference photos of the people you want to find in your collection. More photos from different ages and angles means better recognition.</p>

    <div style="margin-bottom:20px;">
        <label>Person Name</label>
        <div style="display:flex; gap:10px; align-items:center;">
            <input type="text" id="inp-face-name" placeholder="e.g. daniel" style="max-width:250px">
            <button class="btn btn-secondary" onclick="addPerson()" style="margin:0">Add Person</button>
        </div>
    </div>

    <div id="face-persons"></div>

    <div id="face-match-mode" style="display:none; margin:16px 0; padding:14px 18px; background:#ebf8ff; border:1px solid #bee3f8; border-radius:8px;">
        <div style="font-weight:600; font-size:.9em; color:#2d3748; margin-bottom:4px; display:flex; align-items:center; gap:6px;">Face matching mode <span style="color:#e53e3e;">*</span>
            <span style="position:relative; display:inline-flex; cursor:help;" id="face-mode-info">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#718096" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
                <span style="display:none; position:absolute; left:24px; top:-8px; width:320px; padding:10px 14px; background:#2d3748; color:#fff; font-size:.8em; font-weight:400; border-radius:6px; line-height:1.5; z-index:10; box-shadow:0 4px 12px rgba(0,0,0,.2); pointer-events:none;" id="face-mode-tooltip">
                    <strong>Any person:</strong> A photo is included if it contains at least one of the people you added. Good for collecting all photos of each person separately.<br><br>
                    <strong>All people together:</strong> A photo is only included if every person appears in it. Great for finding group shots where everyone is together.
                </span>
            </span>
        </div>
        <table style="border-collapse:collapse;">
            <tr>
                <td style="padding:2px 6px 2px 0; vertical-align:middle;"><input type="radio" name="face-match" value="any" checked style="margin:0; cursor:pointer;"></td>
                <td style="padding:2px 0; font-size:.8em; cursor:pointer;" onclick="this.previousElementSibling.querySelector('input').checked=true"><strong style="color:#4a5568;">Any person</strong> <span style="color:#a0aec0;">— At least one appears</span></td>
            </tr>
            <tr>
                <td style="padding:2px 6px 2px 0; vertical-align:middle;"><input type="radio" name="face-match" value="all" style="margin:0; cursor:pointer;"></td>
                <td style="padding:2px 0; font-size:.8em; cursor:pointer;" onclick="this.previousElementSibling.querySelector('input').checked=true"><strong style="color:#4a5568;">All together</strong> <span style="color:#a0aec0;">— Everyone must appear</span></td>
            </tr>
        </table>
        <div id="face-match-people" style="margin-top:8px; font-size:.8em; color:#718096;"></div>
    </div>

    <div class="btn-group" id="face-verify-btn-group" style="display:none;">
        <button class="btn btn-primary" onclick="runVerifyAll()">Verify All Faces</button>
    </div>

    <div id="face-verify-results" style="display:none; margin-top:20px;"></div>

    <div class="btn-group">
        <button class="btn btn-secondary" onclick="goStep(2)">Back</button>
        <button class="btn btn-secondary" id="btn-skip-faces" onclick="skipFaces()">Skip (no face recognition)</button>
        <button class="btn btn-primary" id="btn-next-3" disabled onclick="completeFacesStep()">Next: Start Scan</button>
    </div>
</div>

<!-- ── STEP 4: Scan ── -->
<div class="panel" id="panel-4">
    <h2>Scanning your photos</h2>
    <p>This scans all your sources, extracts dates, detects duplicates, and creates thumbnails. This runs once — future scans are incremental.</p>

    <div style="margin-bottom:10px;">
        <label style="display:inline-flex; align-items:center; gap:5px; cursor:pointer; font-size:.8em; color:#4a5568; font-weight:600; white-space:nowrap;">
            <input type="checkbox" id="chk-nsfw-filter" onchange="toggleNsfwFilter(this.checked)">
            Filter out nudity / inappropriate content <span style="font-weight:400; color:#a0aec0;">(AI detection, ~25 MB model)</span>
        </label>
    </div>

    <div style="margin-bottom:10px;">
        <label style="display:inline-flex; align-items:center; gap:5px; cursor:pointer; font-size:.8em; color:#4a5568; font-weight:600; white-space:nowrap;">
            <input type="checkbox" id="chk-age-estimation" onchange="toggleAgeEstimation(this.checked)">
            Estimate age from faces <span style="font-weight:400; color:#a0aec0;">(AI age detection — helps sort undated photos, first run downloads ~500 MB model)</span>
        </label>
        <div id="age-est-options" style="display:none; margin-top:8px; margin-left:22px;">
            <table style="border-collapse:collapse;">
                <tr>
                    <td style="padding:2px 6px 2px 0; vertical-align:middle;"><input type="radio" name="age-est-scope" value="all" checked style="margin:0; cursor:pointer;"></td>
                    <td style="padding:2px 0; font-size:.8em; cursor:pointer;" onclick="this.previousElementSibling.querySelector('input').checked=true"><strong style="color:#4a5568;">All scanned images</strong> <span style="color:#a0aec0;">— Estimate age on every photo with a recognized face</span></td>
                </tr>
                <tr>
                    <td style="padding:2px 6px 2px 0; vertical-align:middle;"><input type="radio" name="age-est-scope" value="folders" style="margin:0; cursor:pointer;"></td>
                    <td style="padding:2px 0; font-size:.8em; cursor:pointer;" onclick="this.previousElementSibling.querySelector('input').checked=true; showAgeEstFolders()"><strong style="color:#4a5568;">Specific folders only</strong> <span style="color:#a0aec0;">— Choose which source folders to run age estimation on</span></td>
                </tr>
            </table>
            <div id="age-est-folders" style="display:none; margin-top:6px; text-align:left;"></div>
        </div>
    </div>

    <div class="btn-group">
        <button class="btn btn-primary" id="btn-start-scan" onclick="startScan(false)">Start Scan</button>
        <button class="btn btn-secondary" onclick="startScan(true)">Full Rescan</button>
    </div>

    <div class="progress-box" id="scan-progress" style="display:none"></div>

    <div class="btn-group">
        <button class="btn btn-secondary" onclick="goStep(3)">Back</button>
        <button class="btn btn-primary" id="btn-next-4" disabled onclick="goStep(5)">Next: Analyze</button>
    </div>
</div>

<!-- ── STEP 5: Analyze ── -->
<div class="panel" id="panel-5">
    <h2>Collection Analysis</h2>
    <p>Understanding what you have and what's missing.</p>

    <!-- Quick stats (loads instantly) -->
    <div id="a-quick" style="margin-top:12px;"></div>

    <!-- Actions -->
    <div style="margin-top:16px; display:flex; gap:12px; flex-wrap:wrap;">
        <button class="btn btn-secondary" onclick="runFullAnalysis()" id="btn-full-analysis">Run Full Analysis (recommendations)</button>
        <button class="btn btn-secondary" style="color:#e53e3e; border-color:#fed7d7;" onclick="resetAllSelections()">Reset All Selections</button>
    </div>
    <div id="a-full-results" style="display:none; margin-top:16px;"></div>

    <div class="btn-group">
        <button class="btn btn-secondary" onclick="goStep(4)">Back</button>
        <button class="btn btn-primary" id="btn-next-5" onclick="goStep(6)">Next: Select</button>
    </div>
</div>

<!-- ── STEP 6: Select ── -->
<div class="panel" id="panel-6" style="max-width:1200px">
    <h2>Select Photos</h2>
    <p>Pick images for each category manually, or auto-fill remaining slots.</p>

    <div style="display:flex; gap:16px; margin-top:12px;">
        <!-- Category sidebar -->
        <div id="sel-cat-list" style="min-width:240px; max-width:280px; border-right:1px solid #e2e8f0; padding-right:12px; max-height:70vh; overflow-y:auto;">
            <div style="font-weight:600; margin-bottom:8px; color:#2d3748;">Categories</div>
        </div>

        <!-- Image grid area -->
        <div style="flex:1; min-width:0;">
            <div id="sel-cat-header" style="display:flex; align-items:center; gap:12px; margin-bottom:10px; flex-wrap:wrap;">
                <span id="sel-cat-title" style="font-size:1.1em; font-weight:600; color:#2d3748;">Select a category</span>
                <span id="sel-cat-counter" style="font-size:.9em; color:#718096;"></span>
                <div id="sel-target-edit" style="display:none; margin-left:auto; align-items:center; gap:6px; font-size:.85em;">
                    <label style="color:#4a5568; margin:0;">Target:</label>
                    <input type="number" id="sel-target-input" style="width:60px; padding:2px 6px; border:1px solid #cbd5e0; border-radius:4px;" min="0" max="999">
                    <button class="btn btn-secondary" style="padding:2px 10px; font-size:.85em;" onclick="saveTarget()">Set</button>
                </div>
            </div>
            <div id="sel-filter-bar" style="display:none; margin-bottom:8px; font-size:.85em; align-items:center; gap:10px;">
                <label style="display:inline-flex; align-items:center; gap:4px; margin:0; cursor:pointer; font-size:1em; color:#4a5568;">
                    <input type="checkbox" id="sel-show-selected" checked onchange="renderSelGrid()" style="margin:0;"> Show selected
                </label>
                <button class="btn btn-secondary" style="padding:2px 10px; font-size:.85em;" onclick="selectAllVisible()">Select All</button>
                <button class="btn btn-secondary" style="padding:2px 10px; font-size:.85em;" onclick="deselectAllVisible()">Deselect All</button>
            </div>
            <div id="sel-grid" style="display:flex; flex-wrap:wrap; gap:6px; max-height:55vh; overflow-y:auto; padding:4px;"></div>
            <div id="sel-grid-paging" style="margin-top:8px; display:none; font-size:.85em; color:#718096;">
                <button class="btn btn-secondary" style="padding:2px 10px; font-size:.85em;" id="sel-prev" onclick="selPage(-1)">Prev</button>
                <span id="sel-page-info"></span>
                <button class="btn btn-secondary" style="padding:2px 10px; font-size:.85em;" id="sel-next" onclick="selPage(1)">Next</button>
            </div>
        </div>
    </div>

    <!-- Auto-fill section -->
    <div style="margin-top:16px; padding-top:12px; border-top:1px solid #e2e8f0;">
        <div style="font-weight:600; color:#2d3748; margin-bottom:8px;">Auto-fill remaining slots</div>
        <div style="display:flex; gap:12px; flex-wrap:wrap;">
            <button class="btn btn-primary" onclick="runQuickFill()">Quick Fill (fast)</button>
            <button class="btn btn-secondary" onclick="runAutoSelect()">Smart Fill (slow, dedup)</button>
        </div>
        <div style="font-size:.8em; color:#718096; margin-top:6px;">
            Quick Fill picks top-quality images instantly. Smart Fill also removes visual duplicates (takes several minutes).
        </div>
    </div>

    <div class="btn-group">
        <button class="btn btn-secondary" onclick="goStep(5)">Back</button>
        <button class="btn btn-primary" id="btn-next-6" onclick="goStep(7)">Next: Review</button>
    </div>
</div>

<!-- ── STEP 7: Review ── -->
<div class="panel" id="panel-7">
    <h2>Review Your Selection</h2>
    <p>Open the interactive gallery to review, move, and reject images. When done, come back here to export.</p>
    <button class="btn btn-primary" onclick="openGallery()">Open Gallery in New Tab</button>
    <p style="margin-top:10px; font-size:.85em; color:#a0aec0;">Review your images in the gallery, then return here to export.</p>

    <div class="btn-group">
        <button class="btn btn-secondary" onclick="goStep(6)">Back</button>
        <button class="btn btn-primary" onclick="goStep(8)">Next: Export</button>
    </div>
</div>

<!-- ── STEP 8: Export ── -->
<div class="panel" id="panel-8">
    <h2>Export Your Collection</h2>
    <p>Copy your curated photos to a final folder, organized by category. Ready to use for your presentation, photo book, or slideshow.</p>

    <label>Output Folder</label>
    <input type="text" id="inp-export-dir">

    <div id="export-stats" style="margin-top:15px;"></div>

    <div class="btn-group">
        <button class="btn btn-primary" id="btn-export" onclick="runExport()">Export Photos</button>
    </div>

    <div class="progress-box" id="export-progress" style="display:none"></div>

    <div class="btn-group">
        <button class="btn btn-secondary" onclick="goStep(7)">Back</button>
    </div>
</div>

</div><!-- .app -->

<div class="lightbox" id="lightbox" onclick="closeLightbox()">
    <button class="close-btn" onclick="closeLightbox()">&times;</button>
    <img id="lightbox-img" src="">
    <video id="lightbox-video" controls style="display:none; max-width:90vw; max-height:80vh;" onclick="event.stopPropagation()"></video>
    <div id="lightbox-info" style="color:#ccc; font-size:.85em; text-align:center; margin-top:8px;"></div>
</div>

<script>
let currentStep = 0;
let selectedTemplate = null;
let templates = [];
let config = null;
let taskPoll = null;

function showLoader(el, msg) {
    if (typeof el === 'string') el = document.getElementById(el);
    el.innerHTML = '<div class=inline-loader><div class=spin></div><span>' + esc(msg || 'Loading...') + '</span></div>';
    el.style.display = 'block';
}

// ── Task overlay ──
const TASK_TITLES = {
    'scan': 'Scanning Images...',
    'auto-select': 'Auto-Selecting Photos...',
    'export': 'Exporting Collection...',
};

function showTaskOverlay(taskType) {
    const overlay = document.getElementById('task-overlay');
    document.getElementById('task-overlay-title').textContent = TASK_TITLES[taskType] || 'Working...';
    document.getElementById('task-overlay-status').textContent = 'Starting...';
    document.getElementById('task-overlay-lines').innerHTML = '';
    const bar = document.getElementById('task-overlay-bar');
    bar.className = 'task-bar indeterminate';
    bar.style.width = '100%';
    overlay.classList.add('active');

    if (taskPoll) clearInterval(taskPoll);
    taskPoll = setInterval(async () => {
        try {
            const res = await fetch('/api/scan/status');
            const st = await res.json();
            document.getElementById('task-overlay-status').textContent = st.progress || '';
            const linesEl = document.getElementById('task-overlay-lines');
            const clean = st.lines.filter(l => !l.trimStart().startsWith('File "') && !l.startsWith('Traceback') && !l.trimStart().startsWith('json.decoder'));
            linesEl.innerHTML = clean.map(l => '<div class="line">' + esc(l) + '</div>').join('');
            linesEl.scrollTop = linesEl.scrollHeight;

            if (st.done || st.error || st.cancelled) {
                clearInterval(taskPoll);
                taskPoll = null;
                overlay.classList.remove('active');
                if (st.cancelled || st.error === 'Cancelled') {
                    // User stopped — stay on current step
                } else if (st.error) {
                    alert('Error: ' + st.error);
                }
            }
        } catch(e) {}
    }, 1500);
}

async function stopTask() {
    document.getElementById('task-overlay-status').textContent = 'Stopping...';
    document.querySelector('#task-overlay .btn-stop').disabled = true;
    try {
        await fetch('/api/task/stop', { method: 'POST' });
    } catch(e) {}
    // Wait briefly for backend to acknowledge, then dismiss
    setTimeout(() => {
        if (taskPoll) { clearInterval(taskPoll); taskPoll = null; }
        document.getElementById('task-overlay').classList.remove('active');
        document.querySelector('#task-overlay .btn-stop').disabled = false;
    }, 2000);
}

// ── Step navigation ──
function goStep(n) {
    currentStep = n;
    document.querySelectorAll('.panel').forEach((p, i) => p.classList.toggle('active', i === n));
    document.querySelectorAll('.step-dot').forEach((d, i) => {
        d.classList.toggle('active', i === n);
    });

    if (n === 1) loadCategoriesStep();
    if (n === 2) renderSources();
    if (n === 3) loadFaceStep();
    if (n === 4) checkExistingScan();
    if (n === 5) runAnalysis();
    if (n === 6) loadSelCategories();
    if (n === 8) loadExportStats();
}

// ── Step 0: Choose event ──
async function loadTemplates() {
    const res = await fetch('/api/templates');
    templates = await res.json();
    var main = templates.filter(t => !t.extra);
    var extras = templates.filter(t => t.extra);
    main.sort((a, b) => (a.event_type === 'custom') - (b.event_type === 'custom'));
    extras.sort((a, b) => a.display_name.localeCompare(b.display_name));
    // Reorder: main first, then extras
    templates = main.concat(extras);
    var grid = document.getElementById('template-grid');
    grid.innerHTML = main.map(t => `
        <div class="template-card" onclick="selectTemplate('${t.event_type}')" id="tpl-${t.event_type}">
            <h3>${t.display_name}</h3>
            <div class="desc">${t.description}</div>
            <div class="meta">${t.num_categories} categories | ${t.categorization}</div>
        </div>
    `).join('');
    var moreEl = document.getElementById('more-events');
    var listEl = document.getElementById('more-events-list');
    if (extras.length) {
        moreEl.style.display = 'block';
        listEl.innerHTML = '';
        extras.forEach(function(t) {
            var row = document.createElement('div');
            row.id = 'tpl-' + t.event_type;
            row.style.cssText = 'display:flex; align-items:center; gap:12px; padding:8px 12px; cursor:pointer; border-radius:6px; border:1px solid #e2e8f0; margin-bottom:4px; transition:all .15s;';
            row.onmouseenter = function() { row.style.borderColor = '#63b3ed'; row.style.background = '#f0f9ff'; };
            row.onmouseleave = function() { if (!row.classList.contains('selected')) { row.style.borderColor = '#e2e8f0'; row.style.background = ''; } };
            row.onclick = function() { selectTemplate(t.event_type); };
            var name = document.createElement('span');
            name.style.cssText = 'font-weight:500; color:#2b6cb0; min-width:160px;';
            name.textContent = t.display_name;
            var desc = document.createElement('span');
            desc.style.cssText = 'color:#718096; font-size:.82em; flex:1;';
            desc.textContent = t.description;
            var meta = document.createElement('span');
            meta.style.cssText = 'color:#a0aec0; font-size:.75em; white-space:nowrap;';
            meta.textContent = t.num_categories + ' cats';
            row.appendChild(name);
            row.appendChild(desc);
            row.appendChild(meta);
            listEl.appendChild(row);
        });
    }

    // Check existing config
    const cfgRes = await fetch('/api/config');
    config = await cfgRes.json();
    if (config && config.event_type) {
        selectTemplate(config.event_type);
        // Restore fields
        if (config.subject_birthday) document.getElementById('inp-birthday').value = config.subject_birthday;
        if (config.event_date) document.getElementById('inp-event-date').value = config.event_date;
        if (config.end_date) document.getElementById('inp-end-date').value = config.end_date;
        if (config.year) document.getElementById('inp-year').value = config.year;
    }
}

function selectTemplate(type) {
    selectedTemplate = templates.find(t => t.event_type === type);
    // Clear all selections (cards + extra rows)
    document.querySelectorAll('.template-card').forEach(c => c.classList.remove('selected'));
    document.querySelectorAll('#more-events-list > div').forEach(function(r) {
        r.classList.remove('selected');
        r.style.borderColor = '#e2e8f0';
        r.style.background = '';
    });
    var el = document.getElementById('tpl-' + type);
    if (el) {
        el.classList.add('selected');
        if (el.parentElement && el.parentElement.id === 'more-events-list') {
            el.style.borderColor = '#3182ce';
            el.style.background = '#ebf8ff';
            // Auto-expand the more events list
            document.getElementById('more-events-list').style.display = 'block';
        }
    }
    document.getElementById('btn-next-0').disabled = false;

    // Show required fields
    const fields = selectedTemplate.required_fields || {};
    document.getElementById('event-fields').style.display = 'block';
    document.getElementById('field-birthday').style.display = 'subject_birthday' in fields ? 'block' : 'none';
    document.getElementById('field-event-date').style.display = 'event_date' in fields ? 'block' : 'none';
    document.getElementById('field-end-date').style.display = 'end_date' in fields ? 'block' : 'none';
    document.getElementById('field-year').style.display = 'year' in fields ? 'block' : 'none';

    // Tips
    const tips = selectedTemplate.tips || [];
    const tipsEl = document.getElementById('event-tips');
    if (tips.length) {
        tipsEl.style.display = 'block';
        tipsEl.innerHTML = '<h3 style="color:#4caf50; font-size:.9em; margin-bottom:8px;">Tips</h3>' +
            tips.map(t => '<div style="color:#718096; font-size:.85em; padding:3px 0;">• ' + t + '</div>').join('');
    }
}

function toggleSkipDate(field) {
    var inp = document.getElementById('inp-' + field);
    var skip = document.getElementById('skip-' + field).checked;
    inp.disabled = skip;
    inp.style.opacity = skip ? '0.4' : '1';
    if (skip) inp.value = '';
}

async function completeStep0() {
    if (!selectedTemplate) return;

    const data = {
        event_type: selectedTemplate.event_type,
        sources: config?.sources || [],
        face_names: config?.face_names || [],
    };
    const bday = document.getElementById('inp-birthday').value;
    const edate = document.getElementById('inp-event-date').value;
    const enddate = document.getElementById('inp-end-date').value;
    const year = document.getElementById('inp-year').value;
    if (bday) data.subject_birthday = bday;
    if (edate) data.event_date = edate;
    if (enddate) data.end_date = enddate;
    if (year) data.year = year;

    await fetch('/api/init', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data) });

    const cfgRes = await fetch('/api/config');
    config = await cfgRes.json();

    document.querySelectorAll('.step-dot')[0].classList.add('done');
    goStep(1);
}

// ── Step 1: Categories ──
function loadCategoriesStep() {
    // Populate project name from config
    var nameInp = document.getElementById('inp-project-name');
    if (config && config.project_name) {
        nameInp.value = config.project_name;
    }
    const cats = config?.categories || [];
    renderCategories(cats);
}

function renderCategories(cats) {
    const unlimited = config?.unlimited_mode || false;
    document.getElementById('chk-unlimited').checked = unlimited;
    const tbody = document.getElementById('cat-tbody');
    tbody.innerHTML = cats.map((c, i) => `
        <tr>
            <td style="color:#a0aec0;">${i + 1}</td>
            <td><input type="text" value="${esc(c.display || c.id || '')}" onchange="updateCatField(${i}, 'display', this.value)" style="border:1px solid #e2e8f0;"></td>
            <td><input type="number" value="${c.target || config?.target_per_category || 75}" min="0" max="500" onchange="updateCatField(${i}, 'target', parseInt(this.value))" style="width:80px; border:1px solid #e2e8f0;" ${unlimited ? 'disabled' : ''}></td>
            <td><input type="number" value="${c.video_target || 0}" min="0" max="500" onchange="updateCatField(${i}, 'video_target', parseInt(this.value))" style="width:80px; border:1px solid #e2e8f0;" ${unlimited ? 'disabled' : ''}></td>
            <td><span style="color:#e53e3e; cursor:pointer; font-size:1.2em;" onclick="removeCategory(${i})">&times;</span></td>
        </tr>
    `).join('');
    updateCatSummary();
}

function updateCatField(i, field, value) {
    if (!config?.categories) return;
    config.categories[i][field] = value;
    updateCatSummary();
}

function toggleUnlimited(checked) {
    config = config || {};
    config.unlimited_mode = checked;
    renderCategories(config.categories || []);
}

function updateCatSummary() {
    const cats = config?.categories || [];
    const defaultTarget = config?.target_per_category || 75;
    document.getElementById('cat-total-count').textContent = cats.length;
    if (config?.unlimited_mode) {
        document.getElementById('cat-total-target').textContent = 'All matching';
    } else {
        const imgTotal = cats.reduce((sum, c) => sum + (c.target || defaultTarget), 0);
        const vidTotal = cats.reduce((sum, c) => sum + (c.video_target || 0), 0);
        document.getElementById('cat-total-target').textContent = imgTotal + (vidTotal ? ' + ' + vidTotal + ' videos' : '');
    }
}

function addCategory() {
    if (!config) return;
    if (!config.categories) config.categories = [];
    const n = config.categories.length + 1;
    config.categories.push({
        id: 'custom_' + n,
        display: 'New Category ' + n,
        target: config.target_per_category || 75,
        video_target: 0
    });
    renderCategories(config.categories);
}

function removeCategory(i) {
    if (!config?.categories) return;
    config.categories.splice(i, 1);
    renderCategories(config.categories);
}

async function completeCategoriesStep() {
    // Validate project name
    var projName = document.getElementById('inp-project-name').value.trim();
    if (!projName) {
        validateProjectName();
        document.getElementById('inp-project-name').focus();
        return;
    }
    if (!config) config = {};
    config.project_name = projName;

    // Ensure all categories have an id
    if (config?.categories) {
        config.categories.forEach((c, i) => {
            if (!c.id) c.id = 'cat_' + i;
        });
    }
    await fetch('/api/config', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(config) });
    document.querySelectorAll('.step-dot')[1].classList.add('done');
    goStep(2);
    renderSources();
}

// ── Step 2: Sources ──
function renderSources() {
    const list = document.getElementById('source-list');
    if (!config) config = { sources: [] };
    if (!config.sources) config.sources = [];
    // Normalize: convert plain strings to {path, label} objects
    config.sources = config.sources.map(function(s, i) {
        if (typeof s === 'string') return { path: s, label: 'Source ' + (i + 1) };
        return s;
    });
    var sources = config.sources;
    list.innerHTML = '';
    sources.forEach(function(s, i) {
        var lbl = s.label || ('Source ' + (i + 1));
        var row = document.createElement('div');
        row.className = 'source-item';
        var lblInput = document.createElement('input');
        lblInput.type = 'text';
        lblInput.value = lbl;
        lblInput.placeholder = 'Source ' + (i + 1);
        lblInput.style.maxWidth = '150px';
        lblInput.onchange = function() { updateSource(i, 'label', this.value); };
        var pathInput = document.createElement('input');
        pathInput.type = 'text';
        pathInput.value = s.path || '';
        pathInput.placeholder = 'Full path to folder (e.g. D:\\Photos)';
        pathInput.onchange = function() { updateSource(i, 'path', this.value); };
        var removeSpan = document.createElement('span');
        removeSpan.className = 'remove';
        removeSpan.innerHTML = '&times;';
        removeSpan.onclick = function() { removeSource(i); };
        row.appendChild(lblInput);
        row.appendChild(pathInput);
        row.appendChild(removeSpan);
        list.appendChild(row);
    });
}

function addSource() {
    if (!config) config = { sources: [] };
    if (!config.sources) config.sources = [];
    config.sources.push({ path: '', label: 'Source ' + (config.sources.length + 1) });
    renderSources();
}

function removeSource(i) {
    config.sources.splice(i, 1);
    renderSources();
}

function updateSource(i, field, value) {
    config.sources[i][field] = value;
}

async function completeStep2() {
    // Fill in empty labels before saving
    if (config && config.sources) {
        config.sources.forEach(function(s, i) {
            if (!s.label || !s.label.trim()) s.label = 'Source ' + (i + 1);
        });
    }
    await fetch('/api/config', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(config) });

    document.querySelectorAll('.step-dot')[2].classList.add('done');
    goStep(3);
}

// ── Step 2: Face References ──
let facesVerified = false;

async function loadFaceStep() {
    const res = await fetch('/api/ref-faces');
    const faces = await res.json();
    renderFacePersons(faces);
    // Show verify button if there are persons with photos
    const hasPhotos = faces.some(f => f.photo_count > 0);
    document.getElementById('face-verify-btn-group').style.display = hasPhotos ? 'flex' : 'none';
    if (!facesVerified) document.getElementById('btn-next-3').disabled = hasPhotos;
    document.getElementById('btn-skip-faces').style.display = facesVerified ? 'none' : '';
}

function renderFacePersons(faces) {
    const container = document.getElementById('face-persons');
    if (!faces.length) {
        container.innerHTML = '<p style="color:#a0aec0; font-style:italic;">No reference faces added yet. Add a person above, or skip this step.</p>';
        return;
    }
    container.innerHTML = faces.map(f => `
        <div style="background:#f7fafc; border:1px solid #e2e8f0; border-radius:8px; padding:15px; margin-bottom:12px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <h3 style="color:#2b6cb0; margin:0; text-transform:capitalize;">${esc(f.name)}</h3>
                <div style="display:flex; gap:8px; align-items:center;">
                    <span style="color:#718096;">${f.photo_count} photo(s)</span>
                    <button class="btn btn-secondary" onclick="removePerson('${esc(f.name)}')" style="margin:0; padding:4px 10px; font-size:.8em; background:#fed7d7; color:#c53030;">Remove</button>
                </div>
            </div>
            <div id="face-thumbs-${f.name}" style="display:flex; gap:6px; flex-wrap:wrap; margin-bottom:10px;"></div>
            <div style="display:flex; gap:8px; align-items:center;">
                <label style="margin:0; cursor:pointer;" class="btn btn-secondary" for="upload-${f.name}">+ Add Photos</label>
                <input type="file" id="upload-${f.name}" multiple accept="image/*" style="display:none" onchange="uploadFacePhotos('${esc(f.name)}', this.files)">
                <button class="btn btn-secondary" onclick="verifyPerson('${esc(f.name)}')" style="margin:0">Verify Face</button>
            </div>
            <div id="face-status-${f.name}" style="margin-top:8px;"></div>
        </div>
    `).join('');

    // Load thumbnails for each person
    faces.forEach(f => loadFaceThumbs(f.name));

    // Show face match mode selector when multiple persons have photos
    var personsWithPhotos = faces.filter(f => f.photo_count > 0);
    var matchArea = document.getElementById('face-match-mode');
    if (personsWithPhotos.length > 1) {
        matchArea.style.display = 'block';
        var names = personsWithPhotos.map(function(f) { return f.name.charAt(0).toUpperCase() + f.name.slice(1); });
        document.getElementById('face-match-people').textContent = 'People: ' + names.join(', ');
        updateMatchModeStyle();
    } else if (personsWithPhotos.length === 1) {
        matchArea.style.display = 'none';
    } else {
        matchArea.style.display = 'none';
    }
}

let faceVerifyCache = {};
let replacedPhotos = {};  // {person: Set of filenames}

async function loadFaceThumbs(person) {
    const res = await fetch('/api/ref-faces/' + encodeURIComponent(person) + '/photos');
    const photos = await res.json();
    const container = document.getElementById('face-thumbs-' + person);
    if (!container) return;
    const statuses = faceVerifyCache[person] || {};
    const replaced = replacedPhotos[person] || new Set();
    container.innerHTML = '';
    photos.forEach(function(p) {
        var st = statuses[p.filename];
        var isReplaced = replaced.has(p.filename) && !st;
        var border = st === 'ok' || st === 'ok_multi' ? '3px solid #4caf50' : st === 'no_face' || st === 'encode_fail' || st === 'error' ? '3px solid #f44336' : isReplaced ? '3px dashed #dd6b20' : '2px solid #cbd5e0';
        var hasAnyVerified = Object.keys(statuses).length > 0;

        var wrap = document.createElement('div');
        wrap.style.cssText = 'display:inline-flex; flex-direction:column; align-items:center; gap:2px; margin-right:8px; margin-bottom:6px;';

        // Verify or Replace button
        if (isReplaced && hasAnyVerified) {
            var verifySpan = document.createElement('span');
            verifySpan.textContent = 'Verify';
            verifySpan.style.cssText = 'font-size:.65em; color:#e53e3e; cursor:pointer; padding:1px 4px; background:#fff5f5; border:1px solid #fed7d7; border-radius:3px; white-space:nowrap;';
            (function(pn, fn) { verifySpan.onclick = function() { verifySinglePhoto(pn, fn); }; })(person, p.filename);
            wrap.appendChild(verifySpan);
        } else {
            var replaceLabel = document.createElement('label');
            replaceLabel.textContent = 'Replace';
            replaceLabel.style.cssText = 'font-size:.65em; color:#3182ce; cursor:pointer; padding:1px 4px; background:#ebf8ff; border-radius:3px; white-space:nowrap;';
            replaceLabel.setAttribute('for', 'replace-' + person + '-' + p.filename);
            wrap.appendChild(replaceLabel);
        }

        // Hidden file input for replace
        var fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.id = 'replace-' + person + '-' + p.filename;
        fileInput.accept = 'image/*';
        fileInput.style.display = 'none';
        (function(pn, fn) { fileInput.onchange = function() { replaceFacePhoto(pn, fn, this.files); }; })(person, p.filename);
        wrap.appendChild(fileInput);

        // Image container
        var imgDiv = document.createElement('div');
        imgDiv.style.cssText = 'position:relative;';
        imgDiv.id = 'face-img-' + person + '-' + p.filename.replace(/[^a-zA-Z0-9]/g, '-');
        imgDiv.onmouseenter = function() { var x = this.querySelector('.face-x'); if (x) x.style.display = 'flex'; };
        imgDiv.onmouseleave = function() { var x = this.querySelector('.face-x'); if (x) x.style.display = 'none'; };

        if (p.thumb) {
            var img = document.createElement('img');
            img.src = 'data:image/jpeg;base64,' + p.thumb;
            img.style.cssText = 'width:70px; height:70px; object-fit:cover; border-radius:4px; border:' + border + '; cursor:pointer;';
            img.title = 'Double-click to enlarge';
            (function(pn, fn) { img.ondblclick = function() { openLightbox(pn, fn); }; })(person, p.filename);
            imgDiv.appendChild(img);
        } else {
            var placeholder = document.createElement('div');
            placeholder.style.cssText = 'width:70px; height:70px; background:#e2e8f0; border-radius:4px; border:' + border + '; display:flex; align-items:center; justify-content:center; font-size:.7em; color:#718096;';
            placeholder.textContent = p.filename;
            imgDiv.appendChild(placeholder);
        }

        // Remove X button
        var xBtn = document.createElement('span');
        xBtn.className = 'face-x remove-face-btn';
        xBtn.setAttribute('data-person', person);
        xBtn.setAttribute('data-filename', p.filename);
        xBtn.innerHTML = '&#x2715;';
        xBtn.style.cssText = 'display:none; position:absolute; top:-4px; right:-4px; width:18px; height:18px; background:#e53e3e; color:white; border-radius:50%; font-size:11px; align-items:center; justify-content:center; cursor:pointer; line-height:1; box-shadow:0 1px 3px rgba(0,0,0,.3);';
        imgDiv.appendChild(xBtn);

        // Status label
        if (st === 'no_face') {
            var lbl = document.createElement('span');
            lbl.innerHTML = '&#x2718;';
            lbl.style.cssText = 'position:absolute; bottom:1px; right:3px; font-size:14px; color:#f44336; text-shadow:0 0 3px #fff;';
            imgDiv.appendChild(lbl);
        } else if (st === 'ok' || st === 'ok_multi') {
            var lbl = document.createElement('span');
            lbl.innerHTML = '&#x2714;';
            lbl.style.cssText = 'position:absolute; bottom:1px; right:3px; font-size:14px; color:#4caf50; text-shadow:0 0 3px #fff;';
            imgDiv.appendChild(lbl);
        }

        wrap.appendChild(imgDiv);

        // Rotate buttons
        var rotDiv = document.createElement('div');
        rotDiv.style.cssText = 'display:flex; gap:2px;';
        var rotL = document.createElement('button');
        rotL.innerHTML = '&#x21BA;';
        rotL.title = 'Rotate left';
        rotL.style.cssText = 'font-size:.7em; padding:1px 5px; cursor:pointer; background:#edf2f7; border:1px solid #cbd5e0; border-radius:3px;';
        (function(pn, fn) { rotL.onclick = function() { rotateFace(pn, fn, 'ccw'); }; })(person, p.filename);
        var rotR = document.createElement('button');
        rotR.innerHTML = '&#x21BB;';
        rotR.title = 'Rotate right';
        rotR.style.cssText = 'font-size:.7em; padding:1px 5px; cursor:pointer; background:#edf2f7; border:1px solid #cbd5e0; border-radius:3px;';
        (function(pn, fn) { rotR.onclick = function() { rotateFace(pn, fn, 'cw'); }; })(person, p.filename);
        rotDiv.appendChild(rotL);
        rotDiv.appendChild(rotR);
        wrap.appendChild(rotDiv);

        container.appendChild(wrap);
    });

    // Add "Verify All Replaced" button if multiple unverified replacements
    const unverified = [...replaced].filter(fn => !statuses[fn]);
    if (unverified.length > 1) {
        var btn = document.createElement('div');
        btn.style.marginTop = '6px';
        var b = document.createElement('button');
        b.className = 'btn btn-secondary';
        b.style.cssText = 'font-size:.8em; padding:4px 12px; color:#e53e3e; border-color:#fed7d7;';
        b.textContent = 'Verify ' + unverified.length + ' Replaced Photos';
        b.onclick = function() { verifyReplacedPhotos(person); };
        btn.appendChild(b);
        container.appendChild(btn);
    }
}

// Event delegation for remove-face buttons (avoids inline onclick quote issues)
document.addEventListener('click', function(e) {
    if (e.target.classList.contains('remove-face-btn')) {
        var person = e.target.getAttribute('data-person');
        var filename = e.target.getAttribute('data-filename');
        removeFacePhoto(person, filename);
    }
});

async function removeFacePhoto(person, filename) {
    if (!confirm('Remove "' + filename + '" from ' + person + "'s reference photos?")) return;
    await fetch('/api/ref-faces/' + encodeURIComponent(person) + '/photo/' + encodeURIComponent(filename), {method: 'DELETE'});
    // Clean caches
    if (faceVerifyCache[person]) delete faceVerifyCache[person][filename];
    if (replacedPhotos[person]) replacedPhotos[person].delete(filename);
    await loadFaceThumbs(person);
    await loadFaces();
}

async function verifySinglePhoto(person, filename) {
    // Show spinner on the image
    showFaceLoader(person, filename);
    const res = await fetch('/api/ref-faces/' + encodeURIComponent(person) + '/verify-photos', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ filenames: [filename] })
    });
    const data = await res.json();
    if (data.results && data.results[filename]) {
        if (!faceVerifyCache[person]) faceVerifyCache[person] = {};
        faceVerifyCache[person][filename] = data.results[filename].status;
    }
    // Remove from replaced set since it's now verified
    if (replacedPhotos[person]) replacedPhotos[person].delete(filename);
    await loadFaceThumbs(person);
    // Show result
    const statusEl = document.getElementById('face-status-' + person);
    if (statusEl && data.results && data.results[filename]) {
        const v = data.results[filename];
        const color = v.status === 'ok' ? '#38a169' : v.status === 'ok_multi' ? '#dd6b20' : '#e53e3e';
        statusEl.innerHTML = '<span style="color:' + color + ';">' + esc(filename) + ': ' + esc(v.message) + '</span>';
    }
}

async function verifyReplacedPhotos(person) {
    const replaced = replacedPhotos[person];
    if (!replaced || !replaced.size) return;
    const filenames = [...replaced].filter(fn => !(faceVerifyCache[person] || {})[fn]);
    if (!filenames.length) return;

    const statusEl = document.getElementById('face-status-' + person);
    showLoader(statusEl, 'Verifying ' + filenames.length + ' replaced photos...');

    const res = await fetch('/api/ref-faces/' + encodeURIComponent(person) + '/verify-photos', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ filenames })
    });
    const data = await res.json();
    if (data.results) {
        if (!faceVerifyCache[person]) faceVerifyCache[person] = {};
        for (const [fn, result] of Object.entries(data.results)) {
            faceVerifyCache[person][fn] = result.status;
            if (replacedPhotos[person]) replacedPhotos[person].delete(fn);
        }
    }
    await loadFaceThumbs(person);

    // Show summary
    if (statusEl && data.results) {
        const entries = Object.entries(data.results);
        const ok = entries.filter(([,v]) => v.status === 'ok' || v.status === 'ok_multi').length;
        const fail = entries.length - ok;
        statusEl.innerHTML = '<span style="color:' + (fail ? '#e53e3e' : '#38a169') + ';">' + ok + '/' + entries.length + ' faces detected' + (fail ? ' (' + fail + ' failed)' : '') + '</span>';
    }
}

async function addPerson() {
    const inp = document.getElementById('inp-face-name');
    const name = inp.value.trim().toLowerCase();
    if (!name) return;

    // Create folder via upload with no files (just creates the dir)
    await fetch('/api/ref-faces/upload', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({person: name, photos: []})
    });
    inp.value = '';
    loadFaceStep();
}

async function removePerson(name) {
    if (!confirm('Remove all reference photos for ' + name + '?')) return;
    await fetch('/api/ref-faces/' + encodeURIComponent(name), {method: 'DELETE'});
    loadFaceStep();
}

function showFaceLoader(person, filename) {
    const safeId = 'face-img-' + person + '-' + filename.replace(/[^a-zA-Z0-9]/g, '-');
    const wrapper = document.getElementById(safeId);
    if (!wrapper) return;
    const imgEl = wrapper.querySelector('img') || wrapper.querySelector('div[style*="70px"]');
    if (imgEl) imgEl.style.opacity = '0.3';
    const old = wrapper.querySelector('.face-spinner');
    if (old) old.remove();
    const spinner = document.createElement('div');
    spinner.className = 'face-spinner';
    spinner.style.cssText = 'position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:20px;height:20px;border:3px solid #e2e8f0;border-top:3px solid #3182ce;border-radius:50%;animation:spin 0.8s linear infinite;';
    wrapper.appendChild(spinner);
}

function markFacesDirty() {
    faceVerifyCache = {};
    facesVerified = false;
    document.getElementById('btn-next-3').disabled = true;
    document.getElementById('face-verify-btn-group').style.display = 'flex';
}

async function rotateFace(person, filename, direction) {
    showFaceLoader(person, filename);
    await fetch('/api/ref-faces/' + encodeURIComponent(person) + '/rotate', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({filename, direction})
    });
    markFacesDirty();
    await loadFaceThumbs(person);
}

async function replaceFacePhoto(person, filename, files) {
    if (!files || !files.length) return;
    showFaceLoader(person, filename);

    const formData = new FormData();
    formData.append('filename', filename);
    formData.append('photo', files[0]);

    await fetch('/api/ref-faces/' + encodeURIComponent(person) + '/replace', {method: 'POST', body: formData});

    // Track as replaced, clear old verify status
    if (!replacedPhotos[person]) replacedPhotos[person] = new Set();
    replacedPhotos[person].add(filename);
    if (faceVerifyCache[person]) delete faceVerifyCache[person][filename];

    markFacesDirty();
    await loadFaceThumbs(person);
}

async function uploadFacePhotos(person, files) {
    if (!files.length) return;
    const formData = new FormData();
    formData.append('person', person);
    for (const f of files) formData.append('photos', f);

    const statusEl = document.getElementById('face-status-' + person);
    statusEl.innerHTML = '<span style="color:#3182ce;">Uploading ' + files.length + ' photo(s)...</span>';

    await fetch('/api/ref-faces/upload', {method: 'POST', body: formData});
    statusEl.innerHTML = '<span style="color:#4caf50;">Uploaded! Click <strong>Verify All Faces</strong> to check.</span>';
    facesVerified = false;
    document.getElementById('btn-next-3').disabled = true;
    loadFaceStep();
}

async function verifyPerson(person) {
    const statusEl = document.getElementById('face-status-' + person);
    showLoader(statusEl, 'Verifying face encodings...');

    const res = await fetch('/api/ref-faces/verify', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({person})
    });
    const data = await res.json();

    if (data.error) {
        statusEl.innerHTML = '<div style="color:#f44336;">' + esc(data.error) + '</div>';
        return;
    }

    const p = data.persons[0];
    if (!p) return;

    // Cache statuses and re-render thumbnails with color borders
    const statuses = {};
    p.photos.forEach(ph => { statuses[ph.filename] = ph.status; });
    faceVerifyCache[person] = statuses;
    loadFaceThumbs(person);

    let html = '<div style="margin-top:8px;">';

    // Per-photo results
    html += '<table style="width:100%; font-size:.85em; margin-bottom:10px;"><thead><tr style="color:#718096;"><th style="text-align:left;">Photo</th><th>Size</th><th>Status</th></tr></thead><tbody>';
    for (const photo of p.photos) {
        const color = photo.status === 'ok' ? '#4caf50' : photo.status === 'ok_multi' ? '#ff9800' : '#f44336';
        html += '<tr><td>' + esc(photo.filename) + '</td><td style="color:#718096;">' + (photo.dimensions || '') + '</td><td style="color:' + color + ';">' + esc(photo.message) + '</td></tr>';
    }
    html += '</tbody></table>';

    // Summary
    html += '<div style="display:flex; gap:20px; flex-wrap:wrap; margin-bottom:8px;">';
    html += '<div><strong style="color:#2b6cb0;">' + p.encodings + '</strong> / ' + p.total_photos + ' faces encoded</div>';
    html += '<div>Diversity: <strong style="color:' + (p.diversity_score > 0.6 ? '#4caf50' : p.diversity_score > 0.3 ? '#ff9800' : '#f44336') + ';">' + Math.round(p.diversity_score * 100) + '%</strong></div>';
    html += '<div>Status: <strong style="color:' + (p.ready ? '#4caf50' : '#ff9800') + ';">' + (p.ready ? 'READY' : 'NEEDS WORK') + '</strong></div>';
    html += '</div>';

    // Issues
    if (p.issues.length) {
        html += '<div style="margin-bottom:6px;">';
        p.issues.forEach(i => { html += '<div style="color:#f44336; font-size:.85em;">&#9888; ' + esc(i) + '</div>'; });
        html += '</div>';
    }

    // Tips
    if (p.tips.length) {
        html += '<div>';
        p.tips.forEach(t => { html += '<div style="color:#718096; font-size:.85em;">&#128161; ' + esc(t) + '</div>'; });
        html += '</div>';
    }

    html += '</div>';
    statusEl.innerHTML = html;
}

async function verifyAllFaces() {
    const resultsEl = document.getElementById('face-verify-results');
    showLoader(resultsEl, 'Verifying all faces... this may take a moment');

    const res = await fetch('/api/ref-faces/verify', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({})
    });
    const data = await res.json();

    if (data.error) {
        resultsEl.innerHTML = '<div style="color:#f44336;">' + esc(data.error) + '</div>';
        return;
    }

    // Cache statuses, re-render thumbnails, and update inline per-person details
    for (const p of (data.persons || [])) {
        const statuses = {};
        p.photos.forEach(ph => { statuses[ph.filename] = ph.status; });
        faceVerifyCache[p.person] = statuses;
        loadFaceThumbs(p.person);

        // Update inline status for each person
        const statusEl = document.getElementById('face-status-' + p.person);
        if (statusEl) {
            let h = '<div style="margin-top:8px;">';
            h += '<table style="width:100%; font-size:.85em; margin-bottom:10px;"><thead><tr style="color:#718096;"><th style="text-align:left;">Photo</th><th>Size</th><th>Status</th></tr></thead><tbody>';
            for (const photo of p.photos) {
                const color = photo.status === 'ok' ? '#4caf50' : photo.status === 'ok_multi' ? '#ff9800' : '#f44336';
                h += '<tr><td>' + esc(photo.filename) + '</td><td style="color:#718096;">' + (photo.dimensions || '') + '</td><td style="color:' + color + ';">' + esc(photo.message) + '</td></tr>';
            }
            h += '</tbody></table>';
            h += '<div style="display:flex; gap:20px; flex-wrap:wrap; margin-bottom:8px;">';
            h += '<div><strong style="color:#2b6cb0;">' + p.encodings + '</strong> / ' + p.total_photos + ' faces encoded</div>';
            h += '<div>Diversity: <strong style="color:' + (p.diversity_score > 0.6 ? '#4caf50' : p.diversity_score > 0.3 ? '#ff9800' : '#f44336') + ';">' + Math.round(p.diversity_score * 100) + '%</strong></div>';
            h += '<div>Status: <strong style="color:' + (p.ready ? '#4caf50' : '#ff9800') + ';">' + (p.ready ? 'READY' : 'NEEDS WORK') + '</strong></div>';
            h += '</div>';
            if (p.issues.length) p.issues.forEach(i => { h += '<div style="color:#f44336; font-size:.85em;">&#9888; ' + esc(i) + '</div>'; });
            if (p.tips.length) p.tips.forEach(t => { h += '<div style="color:#718096; font-size:.85em;">&#128161; ' + esc(t) + '</div>'; });
            h += '</div>';
            statusEl.innerHTML = h;
        }
    }

    if (!data.persons.length) {
        resultsEl.innerHTML = '<div style="color:#718096;">No reference faces to verify.</div>';
        return;
    }

    let allReady = data.ready;
    resultsEl.innerHTML = '<div style="padding:15px; border-radius:8px; background:' + (allReady ? '#f0fff4' : '#fff5f5') + '; border:1px solid ' + (allReady ? '#4caf50' : '#f44336') + ';">' +
        '<strong style="color:' + (allReady ? '#4caf50' : '#f44336') + ';">' + (allReady ? '&#10003; All faces verified and ready!' : '&#9888; ' + esc(data.message)) + '</strong></div>';

    return allReady;
}

async function runVerifyAll() {
    const allReady = await verifyAllFaces();
    facesVerified = true;
    document.getElementById('btn-skip-faces').style.display = 'none';
    if (allReady) {
        document.getElementById('btn-next-3').disabled = false;
    } else {
        document.getElementById('btn-next-3').disabled = false;  // allow proceed with warning
    }
}

// Info tooltip hover
(function() {
    var info = document.getElementById('face-mode-info');
    var tip = document.getElementById('face-mode-tooltip');
    if (info && tip) {
        info.addEventListener('mouseenter', function() { tip.style.display = 'block'; });
        info.addEventListener('mouseleave', function() { tip.style.display = 'none'; });
    }
})();

function updateMatchModeStyle() {
    // No visual highlight — just the radio button selection is enough
}

function getFaceMatchMode() {
    var radio = document.querySelector('input[name="face-match"]:checked');
    return radio ? radio.value : 'any';
}

function skipFaces() {
    config.face_names = [];
    fetch('/api/config', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(config) });
    document.querySelectorAll('.step-dot')[3].classList.add('done');
    goStep(4);
}

async function completeFacesStep() {
    // Get list of persons with faces
    const res = await fetch('/api/ref-faces');
    const faces = await res.json();
    const personsWithPhotos = faces.filter(f => f.photo_count > 0);

    var personNames = personsWithPhotos.map(f => f.name);

    if (personNames.length > 0) {
        if (!facesVerified) {
            // Force verify first
            await runVerifyAll();
            return;  // let user see results before proceeding
        }
        // Check if all ready
        const verifyRes = await fetch('/api/ref-faces/verify', {
            method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({})
        });
        const verifyData = await verifyRes.json();
        if (!verifyData.ready) {
            if (!confirm('Some face references need more photos for reliable recognition. Proceed anyway?')) return;
        }
        config.face_names = personNames;
        config.face_match_mode = personNames.length > 1 ? getFaceMatchMode() : 'any';
    } else {
        config.face_names = [];
        config.face_match_mode = 'any';
    }

    await fetch('/api/config', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(config) });
    document.querySelectorAll('.step-dot')[3].classList.add('done');
    goStep(4);
}

// ── Step 4: Scan ──
async function checkExistingScan() {
    // Restore NSFW checkbox from config
    const chk = document.getElementById('chk-nsfw-filter');
    if (chk && config?.nsfw_filter) chk.checked = true;

    const res = await fetch('/api/stats');
    const st = await res.json();
    if (st.has_scan && st.total_images > 0) {
        document.getElementById('btn-next-4').disabled = false;
        document.getElementById('scan-progress').style.display = 'block';
        document.getElementById('scan-progress').innerHTML = '<div class="line" style="color:#38a169;">Existing scan found: ' + st.total_images + ' images from ' + (st.sources?.length || 0) + ' sources. You can proceed or rescan.</div>';
    }
}

function toggleNsfwFilter(checked) {
    config = config || {};
    config.nsfw_filter = checked;
    fetch('/api/config', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(config) });
}

function toggleAgeEstimation(checked) {
    document.getElementById('age-est-options').style.display = checked ? 'block' : 'none';
    if (checked) buildAgeEstFolderList();
}

function showAgeEstFolders() {
    document.querySelector('input[name="age-est-scope"][value="folders"]').checked = true;
    document.getElementById('age-est-folders').style.display = 'block';
    buildAgeEstFolderList();
}

function buildAgeEstFolderList() {
    var container = document.getElementById('age-est-folders');
    var validSources = (config && config.sources) ? config.sources.filter(function(s) { return s.path && s.path.trim(); }) : [];
    if (!validSources.length) {
        container.innerHTML = '<div style="font-size:.8em; color:#a0aec0;">No sources configured yet.</div>';
        return;
    }
    var tbl = document.createElement('table');
    tbl.style.cssText = 'border-collapse:collapse;';
    validSources.forEach(function(src) {
        var tr = document.createElement('tr');
        var td1 = document.createElement('td');
        td1.style.cssText = 'padding:2px 6px 2px 0; vertical-align:middle;';
        var cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.className = 'age-est-folder-cb';
        cb.value = src.path;
        cb.checked = true;
        cb.style.cssText = 'margin:0;';
        td1.appendChild(cb);
        var td2 = document.createElement('td');
        td2.style.cssText = 'padding:2px 0; font-size:.8em; color:#4a5568;';
        td2.textContent = src.path;
        tr.appendChild(td1);
        tr.appendChild(td2);
        tbl.appendChild(tr);
    });
    container.innerHTML = '';
    container.appendChild(tbl);
    // Wire up radio buttons to toggle folder list visibility
    document.querySelectorAll('input[name="age-est-scope"]').forEach(function(r) {
        r.onchange = function() {
            document.getElementById('age-est-folders').style.display = this.value === 'folders' ? 'block' : 'none';
        };
    });
}

function getAgeEstConfig() {
    var chk = document.getElementById('chk-age-estimation');
    if (!chk || !chk.checked) return null;
    var scope = document.querySelector('input[name="age-est-scope"]:checked');
    var result = { enabled: true, scope: scope ? scope.value : 'all' };
    if (result.scope === 'folders') {
        result.folders = [...document.querySelectorAll('.age-est-folder-cb:checked')].map(function(cb) { return cb.value; });
    }
    return result;
}

async function startScan(full) {
    var btn = document.getElementById('btn-start-scan');
    btn.disabled = true;
    btn.innerHTML = '<span style="display:inline-block;width:14px;height:14px;border:2px solid #bee3f8;border-top:2px solid #fff;border-radius:50%;animation:spinA .7s linear infinite;vertical-align:middle;margin-right:6px;"></span>Starting...';
    const nsfwFilter = document.getElementById('chk-nsfw-filter')?.checked || false;
    const ageEst = getAgeEstConfig();
    await fetch('/api/scan/start', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({full, nsfw_filter: nsfwFilter, age_estimation: ageEst}) });
    btn.disabled = false;
    btn.textContent = 'Start Scan';
    showTaskOverlay('scan');
    // Override completion handler
    const origPoll = taskPoll;
    const check = setInterval(async () => {
        try {
            const res = await fetch('/api/scan/status');
            const st = await res.json();
            if (st.done || st.error || st.cancelled) {
                clearInterval(check);
                if (!st.error && !st.cancelled) {
                    document.getElementById('btn-next-4').disabled = false;
                    document.querySelectorAll('.step-dot')[4].classList.add('done');
                }
            }
        } catch(e) {}
    }, 2000);
}

// ── Step 5: Analyze ──
function renderCategoryBars(container, cats, showSelected) {
    let totalAvail = 0, totalTarget = 0, totalSel = 0;
    const rows = cats.map(c => {
        const avail = c.qualified + (c.selected || 0);
        const sel = c.selected || 0;
        const target = c.target || 75;
        totalAvail += avail; totalTarget += target; totalSel += sel;
        const count = showSelected ? sel : avail;
        const pct = Math.min(100, Math.round(count / target * 100));
        const full = count >= target;
        const low = count < target * 0.5;
        const barColor = full ? '#38a169' : low ? '#e53e3e' : '#dd6b20';
        const statusText = full ? 'FULL' : (count === 0 ? 'EMPTY' : 'FILLING');
        const statusColor = full ? '#38a169' : (count === 0 ? '#a0aec0' : '#dd6b20');
        return `
        <div style="display:flex; align-items:center; gap:12px; padding:8px 0; border-bottom:1px solid #f0f0f0;">
            <div style="min-width:140px; font-weight:500; color:#2d3748; font-size:.9em;">${esc(c.display)}</div>
            <div style="flex:1; height:8px; background:#e2e8f0; border-radius:4px; overflow:hidden;">
                <div style="height:100%; width:${pct}%; background:${barColor}; border-radius:4px; transition:width .5s;"></div>
            </div>
            <div style="min-width:80px; font-size:.85em; color:#4a5568; text-align:right; font-weight:500;">${count} / ${target}</div>
            <div style="min-width:55px; font-size:.75em; font-weight:700; color:${statusColor}; text-align:center;">${statusText}</div>
        </div>`;
    }).join('');

    const overallCount = showSelected ? totalSel : totalAvail;
    const overallPct = Math.min(100, Math.round(overallCount / (totalTarget || 1) * 100));
    const overallFull = overallCount >= totalTarget;

    container.innerHTML = `
        <div style="display:flex; gap:16px; margin-bottom:16px; flex-wrap:wrap;">
            <div style="flex:1; min-width:100px; text-align:center; background:#ebf8ff; border:1px solid #bee3f8; border-radius:8px; padding:12px;">
                <div style="font-size:1.8em; font-weight:bold; color:#2b6cb0;">${totalAvail}</div>
                <div style="font-size:.8em; color:#4a5568;">Available</div>
            </div>
            ${showSelected ? `<div style="flex:1; min-width:100px; text-align:center; background:#f0fff4; border:1px solid #c6f6d5; border-radius:8px; padding:12px;">
                <div style="font-size:1.8em; font-weight:bold; color:#38a169;">${totalSel}</div>
                <div style="font-size:.8em; color:#4a5568;">Selected</div>
            </div>` : ''}
            <div style="flex:1; min-width:100px; text-align:center; background:#fff5f5; border:1px solid #fed7d7; border-radius:8px; padding:12px;">
                <div style="font-size:1.8em; font-weight:bold; color:#e53e3e;">${totalTarget}</div>
                <div style="font-size:.8em; color:#4a5568;">Target</div>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:16px;">
            <div style="font-weight:600; color:#2d3748; min-width:70px;">Overall:</div>
            <div style="flex:1; height:12px; background:#e2e8f0; border-radius:6px; overflow:hidden;">
                <div style="height:100%; width:${overallPct}%; background:${overallFull ? '#38a169' : '#3182ce'}; border-radius:6px;"></div>
            </div>
            <div style="font-weight:600; color:${overallFull ? '#38a169' : '#2d3748'}; min-width:110px; text-align:right;">${overallCount} / ${totalTarget} (${overallPct}%)</div>
        </div>
        <h3 style="color:#2d3748; margin:0 0 8px; font-size:1em;">Category Fill Status</h3>
        ${rows}`;
}

async function runAnalysis() {
    const el = document.getElementById('a-quick');
    showLoader(el, 'Loading category stats...');
    const res = await fetch('/api/categories/summary');
    const cats = await res.json();
    if (!cats.length) { el.innerHTML = '<div style="color:#e53e3e;">No scan data found. Run a scan first.</div>'; return; }
    renderCategoryBars(el, cats, true);
    document.querySelectorAll('.step-dot')[5].classList.add('done');
}

async function runFullAnalysis() {
    const el = document.getElementById('a-full-results');
    showLoader(el, 'Running full analysis (this may take a moment)...');
    document.getElementById('btn-full-analysis').disabled = true;
    const res = await fetch('/api/analyze');
    document.getElementById('btn-full-analysis').disabled = false;
    if (!res.ok) { el.innerHTML = '<div style="color:#e53e3e;">Analysis failed.</div>'; return; }
    const a = await res.json();
    el.style.display = 'block';
    let html = '<h3 style="color:#718096; margin:0 0 8px;">Recommendations</h3>';
    html += (a.recommendations || []).map(r => `
        <div class="rec-card ${r.type}">
            <div class="title">${esc(r.title)}</div>
            <div class="detail">${esc(r.detail)}</div>
        </div>`).join('');
    html += '<h3 style="color:#718096; margin:16px 0 8px;">What makes a great collection</h3>';
    html += (a.priorities || []).map((p, i) => `<div style="color:#718096; padding:3px 0;">${i+1}. ${esc(p)}</div>`).join('');
    el.innerHTML = html;
}

async function resetAllSelections() {
    if (!confirm('Reset all selections? All 900 selected images will go back to the available pool.')) return;
    showLoader('a-quick', 'Resetting selections...');
    await fetch('/api/selections/reset', { method: 'POST' });
    await runAnalysis();
}

// ── Step 6: Select ──
let selCats = [];
let selActiveCat = null;
let selImages = [];
let selOffset = 0;
const SEL_PAGE = 100;
async function loadSelCategories() {
    const res = await fetch('/api/categories/summary');
    selCats = await res.json();
    renderSelCatList();
}

function renderSelCatList() {
    const el = document.getElementById('sel-cat-list');
    el.innerHTML = '<div style="font-weight:600; margin-bottom:8px; color:#2d3748;">Categories</div>';
    let totalSel = 0, totalTarget = 0;
    selCats.forEach(c => {
        totalSel += c.selected;
        totalTarget += c.target;
        const pct = c.target > 0 ? Math.min(100, Math.round(c.selected / c.target * 100)) : 0;
        const full = c.selected >= c.target;
        const active = selActiveCat === c.id;
        el.innerHTML += `
            <div onclick="selectCategory('${c.id}')" style="padding:8px 10px; cursor:pointer; border-radius:6px; margin-bottom:4px;
                background:${active ? '#ebf4ff' : '#fff'}; border:1px solid ${active ? '#3182ce' : '#e2e8f0'};
                ${full ? 'border-left:3px solid #38a169;' : ''}">
                <div style="font-size:.9em; font-weight:${active ? '600' : '500'}; color:#2d3748;">${esc(c.display)}</div>
                <div style="font-size:.8em; color:#718096; margin-top:2px;">
                    ${c.selected}/${c.target} selected
                    <span style="color:${full ? '#38a169' : '#e53e3e'};">(${pct}%)</span>
                </div>
                <div style="height:3px; background:#e2e8f0; border-radius:2px; margin-top:4px;">
                    <div style="height:100%; width:${pct}%; background:${full ? '#38a169' : '#3182ce'}; border-radius:2px;"></div>
                </div>
            </div>`;
    });
    el.innerHTML += `<div style="margin-top:8px; padding:8px 10px; font-size:.85em; font-weight:600; color:#4a5568; border-top:1px solid #e2e8f0;">
        Total: ${totalSel}/${totalTarget}</div>`;
}

async function selectCategory(catId) {
    selActiveCat = catId;
    selOffset = 0;
    renderSelCatList();
    const cat = selCats.find(c => c.id === catId);
    document.getElementById('sel-cat-title').textContent = cat ? cat.display : catId;
    document.getElementById('sel-target-edit').style.display = 'flex';
    document.getElementById('sel-target-input').value = cat ? cat.target : 75;
    document.getElementById('sel-filter-bar').style.display = 'flex';
    await loadSelImages();
}

async function loadSelImages() {
    const grid = document.getElementById('sel-grid');
    showLoader(grid, 'Loading images...');
    // Load both selected and qualified for this category
    const [rSel, rQual] = await Promise.all([
        fetch(`/api/images?category=${selActiveCat}&status=selected&limit=500`),
        fetch(`/api/images?category=${selActiveCat}&status=qualified&limit=500`)
    ]);
    const dSel = await rSel.json();
    const dQual = await rQual.json();
    // Mark them so we know which are selected
    selImages = [
        ...dSel.images.map(i => ({...i, _sel: true})),
        ...dQual.images.map(i => ({...i, _sel: false}))
    ];
    renderSelGrid();
    updateSelCounter();
}

function updateSelCounter() {
    const cat = selCats.find(c => c.id === selActiveCat);
    const nSel = selImages.filter(i => i._sel).length;
    const target = cat ? cat.target : 0;
    document.getElementById('sel-cat-counter').textContent =
        `${nSel} selected / ${target} target / ${selImages.length} available`;
}

function renderSelGrid() {
    const grid = document.getElementById('sel-grid');
    const showSelected = document.getElementById('sel-show-selected').checked;
    let visible = showSelected ? selImages : selImages.filter(i => !i._sel);
    // Pagination
    const total = visible.length;
    const paged = visible.slice(selOffset, selOffset + SEL_PAGE);
    const pagingEl = document.getElementById('sel-grid-paging');
    if (total > SEL_PAGE) {
        pagingEl.style.display = 'block';
        document.getElementById('sel-page-info').textContent =
            ` ${selOffset + 1}-${Math.min(selOffset + SEL_PAGE, total)} of ${total} `;
        document.getElementById('sel-prev').disabled = selOffset === 0;
        document.getElementById('sel-next').disabled = selOffset + SEL_PAGE >= total;
    } else {
        pagingEl.style.display = 'none';
    }
    grid.innerHTML = '';
    paged.forEach(img => {
        const div = document.createElement('div');
        div.style.cssText = `width:100px; height:100px; border-radius:6px; overflow:hidden; cursor:pointer; position:relative;
            border:3px solid ${img._sel ? '#3182ce' : '#e2e8f0'}; flex-shrink:0;`;
        if (img._sel) div.style.boxShadow = '0 0 0 2px #bee3f8';
        const thumbSrc = img.thumb ? 'data:image/jpeg;base64,' + img.thumb : '';
        const isVid = img.media_type === 'video';
        const vidBadge = isVid ? '<div style="position:absolute; bottom:2px; left:2px; background:rgba(0,0,0,.7); color:#fff; border-radius:4px; padding:1px 5px; font-size:10px; font-weight:600;">&#9654; VID</div>' : '';
        div.innerHTML = `<img src="${thumbSrc}" style="width:100%; height:100%; object-fit:cover;">
            ${vidBadge}
            ${img._sel ? '<div style="position:absolute; top:2px; right:2px; background:#3182ce; color:#fff; border-radius:50%; width:18px; height:18px; font-size:12px; display:flex; align-items:center; justify-content:center;">&#10003;</div>' : ''}`;
        div.onclick = () => toggleSelImage(img);
        div.ondblclick = (e) => { e.stopPropagation(); showSelLightbox(img); };
        grid.appendChild(div);
    });
    if (paged.length === 0) {
        grid.innerHTML = '<div style="color:#718096; padding:20px;">No photos or videos in this category.</div>';
    }
}

function selPage(dir) {
    selOffset += dir * SEL_PAGE;
    if (selOffset < 0) selOffset = 0;
    renderSelGrid();
}

async function toggleSelImage(img) {
    const action = img._sel ? 'deselect' : 'select';
    img._sel = !img._sel;
    renderSelGrid();
    updateSelCounter();
    // Update server
    await fetch('/api/images/select', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ hashes: [img.hash], action })
    });
    // Update sidebar count
    const cat = selCats.find(c => c.id === selActiveCat);
    if (cat) {
        cat.selected += (action === 'select' ? 1 : -1);
        renderSelCatList();
    }
}

async function selectAllVisible() {
    const unsel = selImages.filter(i => !i._sel);
    if (unsel.length === 0) return;
    const hashes = unsel.map(i => i.hash);
    unsel.forEach(i => i._sel = true);
    renderSelGrid(); updateSelCounter();
    await fetch('/api/images/select', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ hashes, action: 'select' })
    });
    const cat = selCats.find(c => c.id === selActiveCat);
    if (cat) { cat.selected = selImages.filter(i => i._sel).length; renderSelCatList(); }
}

async function deselectAllVisible() {
    const sel = selImages.filter(i => i._sel);
    if (sel.length === 0) return;
    const hashes = sel.map(i => i.hash);
    sel.forEach(i => i._sel = false);
    renderSelGrid(); updateSelCounter();
    await fetch('/api/images/select', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ hashes, action: 'deselect' })
    });
    const cat = selCats.find(c => c.id === selActiveCat);
    if (cat) { cat.selected = 0; renderSelCatList(); }
}

async function saveTarget() {
    const val = parseInt(document.getElementById('sel-target-input').value) || 0;
    await fetch('/api/categories/update-target', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ id: selActiveCat, target: val })
    });
    const cat = selCats.find(c => c.id === selActiveCat);
    if (cat) cat.target = val;
    renderSelCatList();
    updateSelCounter();
}

function showSelLightbox(img) {
    const overlay = document.createElement('div');
    overlay.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.85);z-index:9999;display:flex;flex-direction:column;align-items:center;justify-content:center;';
    overlay.onclick = (e) => { if (e.target === overlay) overlay.remove(); };

    const isVid = img.media_type === 'video';
    var mediaEl;
    if (isVid) {
        mediaEl = document.createElement('video');
        mediaEl.src = '/api/images/serve/' + img.hash;
        mediaEl.controls = true;
        mediaEl.autoplay = true;
        mediaEl.style.cssText = 'max-width:90vw;max-height:80vh;border-radius:8px;box-shadow:0 4px 30px rgba(0,0,0,.5);';
        mediaEl.onclick = (e) => e.stopPropagation();
    } else {
        mediaEl = document.createElement('img');
        mediaEl.src = '/api/images/serve/' + img.hash;
        mediaEl.style.cssText = 'max-width:90vw;max-height:85vh;border-radius:8px;box-shadow:0 4px 30px rgba(0,0,0,.5);';
    }

    const close = document.createElement('div');
    close.innerHTML = '&times;';
    close.style.cssText = 'position:absolute;top:20px;right:30px;color:#fff;font-size:36px;cursor:pointer;z-index:10;';
    close.onclick = () => { if (isVid) mediaEl.pause(); overlay.remove(); };

    const info = document.createElement('div');
    info.style.cssText = 'position:absolute;bottom:20px;left:50%;transform:translateX(-50%);color:#fff;font-size:.85em;background:rgba(0,0,0,.6);padding:6px 16px;border-radius:6px;';
    var label = (img.filename || '');
    if (isVid && img.duration) label += ' | ' + Math.round(img.duration) + 's';
    if (img.date_taken) label += ' | ' + img.date_taken;
    if (img.source_label) label += ' | ' + img.source_label;
    info.textContent = label;

    overlay.appendChild(mediaEl);
    overlay.appendChild(close);
    overlay.appendChild(info);
    document.body.appendChild(overlay);
}

async function runQuickFill() {
    await fetch('/api/quick-fill', { method:'POST' });
    showTaskOverlay('auto-select');
    const check = setInterval(async () => {
        try {
            const res = await fetch('/api/scan/status');
            const st = await res.json();
            if (st.done || st.error || st.cancelled) {
                clearInterval(check);
                if (!st.error && !st.cancelled) {
                    document.querySelectorAll('.step-dot')[6].classList.add('done');
                }
                await loadSelCategories();
                if (selActiveCat) await loadSelImages();
            }
        } catch(e) {}
    }, 1500);
}

async function runAutoSelect() {
    await fetch('/api/auto-select', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({ strategy: 'balanced', sim_threshold: 0.85 })
    });
    showTaskOverlay('auto-select');
    const check = setInterval(async () => {
        try {
            const res = await fetch('/api/scan/status');
            const st = await res.json();
            if (st.done || st.error || st.cancelled) {
                clearInterval(check);
                if (!st.error && !st.cancelled) {
                    document.querySelectorAll('.step-dot')[6].classList.add('done');
                }
                await loadSelCategories();
                if (selActiveCat) await loadSelImages();
            }
        } catch(e) {}
    }, 2000);
}

// ── Step 7: Review ──
function openGallery() {
    window.open('/api/report', '_blank');
}

// ── Step 8: Export ──
async function loadExportStats() {
    const el = document.getElementById('export-stats');
    showLoader(el, 'Loading stats...');
    const res = await fetch('/api/stats');
    const st = await res.json();
    var dirInput = document.getElementById('inp-export-dir');
    if (!dirInput.value && st.default_export_dir) dirInput.value = st.default_export_dir;
    if (st.has_scan) {
        const selected = st.selected || 0;
        const qualified = st.qualified || 0;
        el.innerHTML = `
            <div class="export-summary">
                <div class="big-num">${selected}</div>
                <div>images selected for export</div>
                ${selected === 0 ? '<div style="color:#e53e3e; font-size:.9em; margin-top:8px;">No images selected yet. Go back to the Select step to pick images.</div>' : ''}
                <div style="color:#718096; font-size:.85em; margin-top:5px;">${qualified} qualified | ${st.total_images} total scanned | ${st.sources?.length || 0} sources</div>
            </div>
        `;
    }
}

async function runExport() {
    const outputDir = document.getElementById('inp-export-dir').value || 'final_collection';

    await fetch('/api/export', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({ output_dir: outputDir })
    });
    showTaskOverlay('export');
    const check = setInterval(async () => {
        try {
            const res = await fetch('/api/scan/status');
            const st = await res.json();
            if (st.done || st.error || st.cancelled) {
                clearInterval(check);
                if (!st.error && !st.cancelled) {
                    document.querySelectorAll('.step-dot')[8].classList.add('done');
                }
            }
        } catch(e) {}
    }, 2000);
}

function esc(s) { const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }

// ── Lightbox ──
function openLightbox(person, filename) {
    document.getElementById('lightbox-img').src = '/api/ref-faces/' + encodeURIComponent(person) + '/photo/' + encodeURIComponent(filename);
    document.getElementById('lightbox').classList.add('open');
}
function closeLightbox() {
    document.getElementById('lightbox').classList.remove('open');
    document.getElementById('lightbox-img').src = '';
}
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });

// ── Projects drawer ──
function toggleDrawer() {
    var drawer = document.getElementById('side-drawer');
    var backdrop = document.getElementById('drawer-backdrop');
    var toggle = document.getElementById('rail-toggle');
    drawer.classList.toggle('open');
    backdrop.classList.toggle('open');
    if (toggle) toggle.classList.toggle('active', drawer.classList.contains('open'));
    if (drawer.classList.contains('open')) {
        loadProjectList();
        loadUserInfo();
    }
}

function openDrawerToProjects() {
    var drawer = document.getElementById('side-drawer');
    if (!drawer.classList.contains('open')) toggleDrawer();
    // Expand projects section
    document.getElementById('projects-section').style.display = 'block';
    document.getElementById('projects-arrow').style.transform = 'rotate(180deg)';
}

function toggleProjectsSection() {
    var sec = document.getElementById('projects-section');
    var arrow = document.getElementById('projects-arrow');
    if (sec.style.display === 'none') {
        sec.style.display = 'block';
        arrow.style.transform = 'rotate(180deg)';
        loadProjectList();
    } else {
        sec.style.display = 'none';
        arrow.style.transform = '';
    }
}

function toggleCleanupSection() {
    var sec = document.getElementById('cleanup-section');
    var arrow = document.getElementById('cleanup-arrow');
    if (sec.style.display === 'none') {
        sec.style.display = 'block';
        arrow.style.transform = 'rotate(180deg)';
    } else {
        sec.style.display = 'none';
        arrow.style.transform = '';
    }
}

function openPhoneImages() {
    alert('Phone Images feature coming soon!\\nConnect your phone via USB to manage and clean up photos directly.');
}

async function loadUserInfo() {
    try {
        const res = await fetch('/api/auth/me');
        const data = await res.json();
        if (data.authenticated) {
            document.getElementById('user-info').textContent = data.user;
        }
    } catch(e) {}
}

async function logout() {
    if (!confirm('Sign out?')) return;
    await fetch('/api/auth/logout', {method: 'POST'});
    window.location.href = '/login';
}

async function loadProjectList() {
    const el = document.getElementById('project-list');
    try {
        const res = await fetch('/api/projects');
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const projects = await res.json();
        if (!Array.isArray(projects) || !projects.length) {
            el.innerHTML = '<div style="padding:18px; color:#a0aec0; font-size:.85em; text-align:center;">No saved projects yet.</div>';
            return;
        }
        el.innerHTML = '';
        projects.forEach(function(p) {
            var modified = p.modified ? new Date(p.modified).toLocaleDateString() + ' ' + new Date(p.modified).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}) : '';
            var info = [p.template || p.event_type || '', 'Step ' + ((p.step || 0) + 1)].filter(Boolean).join(' | ');
            var row = document.createElement('div');
            row.className = 'project-item';
            row.onclick = function() { loadProject(p.dir_name); };
            var nameDiv = document.createElement('div');
            nameDiv.className = 'p-name';
            nameDiv.textContent = p.name;
            var metaDiv = document.createElement('div');
            metaDiv.className = 'p-meta';
            metaDiv.textContent = info + (modified ? ' | ' + modified : '');
            var actDiv = document.createElement('div');
            actDiv.className = 'p-actions';
            var renBtn = document.createElement('button');
            renBtn.className = 'p-del';
            renBtn.style.color = '#2b6cb0';
            renBtn.textContent = 'Rename';
            renBtn.onclick = function(e) { e.stopPropagation(); renameProject(p.dir_name, p.name); };
            actDiv.appendChild(renBtn);
            var delBtn = document.createElement('button');
            delBtn.className = 'p-del';
            delBtn.textContent = 'Delete';
            delBtn.onclick = function(e) { e.stopPropagation(); deleteProject(p.dir_name, p.name); };
            actDiv.appendChild(delBtn);
            row.appendChild(nameDiv);
            row.appendChild(metaDiv);
            row.appendChild(actDiv);
            el.appendChild(row);
        });
    } catch(e) {
        console.error('loadProjectList error:', e);
        el.innerHTML = '<div style="padding:18px; color:#e53e3e; font-size:.85em;">Error loading projects: ' + e.message + '</div>';
    }
}

async function autoSaveDraft() {
    // Auto-save current work as a draft with date-based name
    try {
        var cfg = await (await fetch('/api/config')).json();
        var hasWork = cfg && (cfg.event_type || cfg.sources && cfg.sources.length > 0);
        if (!hasWork) return; // Nothing to save

        var now = new Date();
        var draftName = 'Draft - ' + now.toLocaleDateString() + ' ' + now.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
        var projectName = cfg.project_name || draftName;
        if (!cfg.project_name) projectName = draftName;

        await fetch('/api/projects/save', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ name: projectName, step: currentStep, overwrite: true })
        });
    } catch(e) { /* silent */ }
}

async function saveCurrentProject() {
    var projectName = '';
    if (config && config.project_name) {
        projectName = config.project_name;
    }
    var name = prompt('Project name:', projectName);
    if (!name || !name.trim()) return;
    name = name.trim();
    // Allow overwrite if saving with the same name as current project
    var overwrite = (config && config.project_name && config.project_name === name);
    showLoader('project-list', 'Saving...');
    try {
        var res = await fetch('/api/projects/save', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ name: name, step: currentStep, overwrite: overwrite })
        });
        var data = await res.json();
        if (data.error) { alert(data.error); return; }
        if (config) config.project_name = name;
        await loadProjectList();
    } catch(e) { alert('Save failed: ' + e.message); }
}

async function loadProject(dirName) {
    if (!confirm('Load this project? Current unsaved progress will be lost.')) return;
    showLoader('project-list', 'Loading project...');
    try {
        var res = await fetch('/api/projects/load', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ dir_name: dirName })
        });
        var data = await res.json();
        if (data.error) { alert(data.error); return; }
        toggleDrawer();
        // Reload the app at the saved step
        config = null;
        selectedTemplate = null;
        var savedStep = data.step || 0;
        await loadTemplates();
        var cfg = await (await fetch('/api/config')).json();
        if (cfg && cfg.event_type) {
            config = cfg;
            selectedTemplate = cfg.event_type;
        }
        goStep(savedStep);
    } catch(e) { alert('Load failed: ' + e.message); }
}

async function newProject() {
    if (!confirm('Start a new project? Your current work will be saved as a draft.')) return;
    try {
        await autoSaveDraft();
        await fetch('/api/projects/new', { method: 'POST' });
        if (document.getElementById('side-drawer').classList.contains('open')) toggleDrawer();
        config = null;
        selectedTemplate = null;
        faceVerifyCache = {};
        replacedPhotos = {};
        document.getElementById('inp-project-name').value = '';
        await loadTemplates();
        goStep(0);
    } catch(e) { alert('Error: ' + e.message); }
}

async function deleteProject(dirName, displayName) {
    if (!confirm('Delete project "' + displayName + '"? This cannot be undone.')) return;
    try {
        await fetch('/api/projects/' + encodeURIComponent(dirName), { method: 'DELETE' });
        await loadProjectList();
    } catch(e) { alert('Delete failed: ' + e.message); }
}

async function renameProject(dirName, currentName) {
    var newName = prompt('Rename project:', currentName);
    if (!newName || !newName.trim() || newName.trim() === currentName) return;
    try {
        var res = await fetch('/api/projects/' + encodeURIComponent(dirName) + '/rename', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ name: newName.trim() })
        });
        var data = await res.json();
        if (data.error) { alert(data.error); return; }
        await loadProjectList();
    } catch(e) { alert('Rename failed: ' + e.message); }
}

function validateProjectName() {
    var inp = document.getElementById('inp-project-name');
    var err = document.getElementById('project-name-error');
    if (inp.value.trim()) {
        err.style.display = 'none';
        inp.style.borderColor = '#e2e8f0';
    } else {
        err.style.display = 'block';
        inp.style.borderColor = '#e53e3e';
    }
}

// ── Tutorial system ──

const TOUR_STEPS = [
    {
        target: '.header h1',
        title: 'Welcome!',
        text: 'This is E-z Photo Organizer. It helps you build a curated photo collection for any special event — step by step.',
        position: 'bottom'
    },
    {
        target: '#steps-nav',
        title: 'Step Navigation',
        text: 'These are your 9 steps. Each one guides you through the process — from choosing an event type to exporting your final collection.',
        position: 'bottom'
    },
    {
        target: '.step-dot:first-child',
        title: 'Step 1: Choose Event',
        text: 'Start here. Pick your event type (Bar Mitzva, Wedding, Graduation, etc.) and the app will set up categories and settings for you.',
        position: 'bottom'
    },
    {
        target: '#icon-rail',
        title: 'Side Menu',
        text: 'Quick access to projects, cleanup tools, phone images, tutorial, and more. Click any icon or open the full menu.',
        position: 'right'
    },
    {
        target: '.step-dot:nth-child(3)',
        title: 'Add Your Photo Sources',
        text: 'In the Sources step, add folders from USB drives, cloud backups, phone exports — anywhere your photos live.',
        position: 'bottom'
    },
    {
        target: '.step-dot:nth-child(4)',
        title: 'Face Recognition',
        text: 'Add reference photos of the main person. The app will automatically find them across thousands of photos.',
        position: 'bottom'
    },
    {
        target: '.step-dot:nth-child(5)',
        title: 'Smart Scanning',
        text: 'The scanner extracts dates, detects faces, filters duplicates, and categorizes everything automatically.',
        position: 'bottom'
    },
    {
        target: '.step-dot:nth-child(7)',
        title: 'Auto-Select & Review',
        text: 'Use Quick Fill or Smart Fill to automatically pick the best photos, then manually review and swap as needed.',
        position: 'bottom'
    },
    {
        target: '.step-dot:nth-child(9)',
        title: 'Export',
        text: 'When you are happy with your selection, export everything to a folder — organized and ready for your event!',
        position: 'bottom'
    }
];

let tourIdx = -1;

function startTutorial() {
    tourIdx = 0;
    // Create overlay elements if they don't exist
    if (!document.getElementById('tour-backdrop')) {
        var bd = document.createElement('div');
        bd.id = 'tour-backdrop';
        bd.onclick = function(e) { if (e.target === bd) endTutorial(); };
        document.body.appendChild(bd);

        var hl = document.createElement('div');
        hl.id = 'tour-highlight';
        document.body.appendChild(hl);

        var tt = document.createElement('div');
        tt.id = 'tour-tooltip';
        document.body.appendChild(tt);
    }
    document.getElementById('tour-backdrop').classList.add('active');
    document.getElementById('tour-highlight').style.display = 'block';
    document.getElementById('tour-tooltip').style.display = 'block';
    showTourStep();
}

function showTourStep() {
    var step = TOUR_STEPS[tourIdx];
    var el = document.querySelector(step.target);
    var hl = document.getElementById('tour-highlight');
    var tt = document.getElementById('tour-tooltip');

    tt.classList.remove('show');

    if (!el) { tourNext(); return; }

    // Scroll element into view
    el.scrollIntoView({ behavior: 'smooth', block: 'center' });

    setTimeout(function() {
        var rect = el.getBoundingClientRect();
        var pad = 6;

        // Position highlight
        hl.style.left = (rect.left - pad + window.scrollX) + 'px';
        hl.style.top = (rect.top - pad + window.scrollY) + 'px';
        hl.style.width = (rect.width + pad * 2) + 'px';
        hl.style.height = (rect.height + pad * 2) + 'px';
        hl.classList.add('pulse');

        // Build tooltip content
        var dotsHtml = '';
        for (var i = 0; i < TOUR_STEPS.length; i++) {
            dotsHtml += '<span class="tour-dot' + (i === tourIdx ? ' active' : '') + '"></span>';
        }

        tt.innerHTML = '<h3>' + step.title + '</h3><p>' + step.text + '</p>' +
            '<div class="tour-actions">' +
            '<div class="tour-dots">' + dotsHtml + '</div>' +
            '<div style="display:flex; gap:8px; align-items:center;">' +
            '<button class="tour-skip" onclick="endTutorial()">Skip</button>' +
            '<button class="tour-next" onclick="tourNext()">' + (tourIdx < TOUR_STEPS.length - 1 ? 'Next' : 'Got it!') + '</button>' +
            '</div></div>';

        // Position tooltip
        var pos = step.position || 'bottom';
        var ttW = 340;
        var ttLeft, ttTop;

        if (pos === 'bottom') {
            ttLeft = rect.left + rect.width / 2 - ttW / 2 + window.scrollX;
            ttTop = rect.bottom + 16 + window.scrollY;
        } else if (pos === 'right') {
            ttLeft = rect.right + 16 + window.scrollX;
            ttTop = rect.top + window.scrollY;
        } else if (pos === 'top') {
            ttLeft = rect.left + rect.width / 2 - ttW / 2 + window.scrollX;
            ttTop = rect.top - 140 + window.scrollY;
        }

        // Keep in viewport
        ttLeft = Math.max(12, Math.min(ttLeft, window.innerWidth - ttW - 12));

        tt.style.left = ttLeft + 'px';
        tt.style.top = ttTop + 'px';
        tt.style.width = ttW + 'px';

        // Animate in
        setTimeout(function() { tt.classList.add('show'); }, 50);
    }, 300);
}

function tourNext() {
    tourIdx++;
    if (tourIdx >= TOUR_STEPS.length) {
        endTutorial();
        return;
    }
    showTourStep();
}

function endTutorial() {
    tourIdx = -1;
    var bd = document.getElementById('tour-backdrop');
    var hl = document.getElementById('tour-highlight');
    var tt = document.getElementById('tour-tooltip');
    if (bd) bd.classList.remove('active');
    if (hl) { hl.classList.remove('pulse'); hl.style.display = 'none'; }
    if (tt) { tt.classList.remove('show'); tt.style.display = 'none'; }
    localStorage.setItem('tutorial_seen', '1');
}

// ── Personalized greeting ──
async function loadGreeting() {
    try {
        var res = await fetch('/api/auth/me');
        var data = await res.json();
        if (data.authenticated && data.name) {
            var firstName = data.name.split(' ')[0];
            document.getElementById('header-greeting').textContent = 'Hi ' + firstName + ', how can I help you today?';

            // First visit — auto-start tutorial
            if (!localStorage.getItem('tutorial_seen')) {
                setTimeout(startTutorial, 800);
            }
        }
    } catch(e) {}
}

// ── Cleanup Tool ──
var cleanupOffset = 0;
var cleanupImages = [];

function toggleRailCleanup() {
    var sub = document.getElementById('rail-cleanup-sub');
    if (sub.style.display === 'none') {
        sub.style.display = 'block';
        document.addEventListener('click', _closeRailCleanupOutside, true);
    } else {
        closeRailCleanup();
    }
}

function closeRailCleanup() {
    document.getElementById('rail-cleanup-sub').style.display = 'none';
    document.removeEventListener('click', _closeRailCleanupOutside, true);
}

function _closeRailCleanupOutside(e) {
    var group = document.getElementById('rail-cleanup-group');
    if (group && !group.contains(e.target)) {
        closeRailCleanup();
    }
}

async function openCleanup() {
    await autoSaveDraft();
    document.getElementById('cleanup-overlay').classList.add('active');
    cleanupOffset = 0;
    cleanupImages = [];
    populateCleanupCategories();
    loadCleanupImages();
    updateTrashBadge();
}

function closeCleanup() {
    document.getElementById('cleanup-overlay').classList.remove('active');
}

async function populateCleanupCategories() {
    try {
        var res = await fetch('/api/categories/summary');
        var data = await res.json();
        var sel = document.getElementById('cleanup-filter-cat');
        sel.innerHTML = '<option value="">All Categories</option>';
        (data.categories || []).forEach(function(c) {
            var opt = document.createElement('option');
            opt.value = c.id;
            opt.textContent = c.display || c.id;
            sel.appendChild(opt);
        });
    } catch(e) {}
}

async function loadCleanupImages() {
    cleanupOffset = 0;
    cleanupImages = [];
    var grid = document.getElementById('cleanup-grid');
    grid.innerHTML = '<div style="grid-column:1/-1; text-align:center; color:#a0aec0; padding:40px;"><span style="display:inline-block;width:18px;height:18px;border:3px solid #bee3f8;border-top:3px solid #3182ce;border-radius:50%;animation:spinA .7s linear infinite;vertical-align:middle;margin-right:8px;"></span>Loading...</div>';
    await fetchCleanupPage();
}

async function fetchCleanupPage() {
    var cat = document.getElementById('cleanup-filter-cat').value;
    var status = document.getElementById('cleanup-filter-status').value;
    var media = document.getElementById('cleanup-filter-media').value;
    var trashOnly = document.getElementById('cleanup-show-trash').checked;

    var params = new URLSearchParams({offset: cleanupOffset, limit: 200});
    if (cat) params.set('category', cat);
    if (status) params.set('status', status);
    if (media) params.set('media_type', media);
    if (trashOnly) params.set('trash', '1');

    try {
        var res = await fetch('/api/cleanup/images?' + params.toString());
        var data = await res.json();
        var grid = document.getElementById('cleanup-grid');

        if (cleanupOffset === 0) grid.innerHTML = '';

        if (data.images.length === 0 && cleanupOffset === 0) {
            grid.innerHTML = '<div style="grid-column:1/-1; text-align:center; color:#a0aec0; padding:40px;">No images found. Run a scan first or adjust filters.</div>';
        }

        data.images.forEach(function(img) {
            cleanupImages.push(img);
            var card = document.createElement('div');
            card.className = 'cleanup-card' + (img.trash ? ' trashed' : '');
            card.dataset.hash = img.hash;

            var isVid = img.media_type === 'video';
            var thumbSrc = img.thumb ? ('data:image/jpeg;base64,' + img.thumb) : '';
            var vidBadge = isVid ? '<div style="position:absolute;top:4px;right:4px;background:rgba(0,0,0,.7);color:#fff;font-size:.65em;padding:2px 6px;border-radius:3px;">VID</div>' : '';

            var imgEl = document.createElement('div');
            imgEl.style.position = 'relative';
            var pic = document.createElement('img');
            pic.src = thumbSrc;
            pic.alt = '';
            pic.style.cssText = 'width:100%;height:130px;object-fit:cover;display:block;';
            pic.onerror = function() { this.style.background = '#edf2f7'; };
            imgEl.appendChild(pic);
            if (isVid) {
                var vb = document.createElement('div');
                vb.style.cssText = 'position:absolute;top:4px;right:4px;background:rgba(0,0,0,.7);color:#fff;font-size:.65em;padding:2px 6px;border-radius:3px;';
                vb.textContent = 'VID';
                imgEl.appendChild(vb);
            }
            var infoEl = document.createElement('div');
            infoEl.className = 'card-info';
            infoEl.textContent = img.filename;
            card.appendChild(imgEl);
            card.appendChild(infoEl);

            card.onclick = function() { toggleCleanupTrash(img.hash, card); };
            grid.appendChild(card);
        });

        var loadMore = document.getElementById('cleanup-load-more');
        if (data.total > cleanupOffset + 200) {
            loadMore.style.display = 'block';
            cleanupOffset += 200;
        } else {
            loadMore.style.display = 'none';
        }
    } catch(e) {
        document.getElementById('cleanup-grid').innerHTML = '<div style="grid-column:1/-1; text-align:center; color:#e53e3e; padding:40px;">Failed to load images.</div>';
    }
}

function loadCleanupMore() {
    fetchCleanupPage();
}

async function toggleCleanupTrash(hash, card) {
    var isTrash = card.classList.contains('trashed');
    var endpoint = isTrash ? '/api/cleanup/unmark-trash' : '/api/cleanup/mark-trash';
    try {
        await fetch(endpoint, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({hashes: [hash]})
        });
        card.classList.toggle('trashed');
        updateTrashBadge();
    } catch(e) {}
}

async function updateTrashBadge() {
    try {
        var res = await fetch('/api/cleanup/trash-count');
        var data = await res.json();
        var badge = document.getElementById('cleanup-trash-badge');
        var reviewBtn = document.getElementById('btn-review-trash');
        var bar = document.getElementById('cleanup-bottom-bar');
        var barCount = document.getElementById('cleanup-bar-count');
        var barSize = document.getElementById('cleanup-bar-size');

        if (data.count > 0) {
            badge.textContent = data.count + ' in trash';
            badge.style.display = 'inline-block';
            reviewBtn.style.display = 'inline-block';
            bar.style.display = 'flex';
            barCount.textContent = data.count + ' item' + (data.count !== 1 ? 's' : '') + ' in trash';
            barSize.textContent = data.size_mb > 0 ? '(' + data.size_mb + ' MB)' : '';
        } else {
            badge.style.display = 'none';
            reviewBtn.style.display = 'none';
            bar.style.display = 'none';
        }
    } catch(e) {}
}

function showCleanupTrash() {
    document.getElementById('cleanup-show-trash').checked = true;
    loadCleanupImages();
}

function cleanupSelectAll() {
    var cards = document.querySelectorAll('#cleanup-grid .cleanup-card:not(.trashed)');
    var hashes = [];
    cards.forEach(function(c) { hashes.push(c.dataset.hash); });
    if (hashes.length === 0) return;
    if (!confirm('Mark ' + hashes.length + ' visible images for trash?')) return;
    fetch('/api/cleanup/mark-trash', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({hashes: hashes})
    }).then(function() {
        cards.forEach(function(c) { c.classList.add('trashed'); });
        updateTrashBadge();
    });
}

function cleanupDeselectAll() {
    var cards = document.querySelectorAll('#cleanup-grid .cleanup-card.trashed');
    var hashes = [];
    cards.forEach(function(c) { hashes.push(c.dataset.hash); });
    if (hashes.length === 0) return;
    fetch('/api/cleanup/unmark-trash', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({hashes: hashes})
    }).then(function() {
        cards.forEach(function(c) { c.classList.remove('trashed'); });
        updateTrashBadge();
    });
}

function cleanupClearTrash() {
    fetch('/api/cleanup/images?trash=1&limit=99999').then(function(r) { return r.json(); }).then(function(data) {
        var hashes = data.images.map(function(i) { return i.hash; });
        if (hashes.length === 0) return;
        return fetch('/api/cleanup/unmark-trash', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({hashes: hashes})
        });
    }).then(function() {
        loadCleanupImages();
        updateTrashBadge();
    });
}

async function cleanupConfirmRecycle() {
    var res = await fetch('/api/cleanup/trash-count');
    var data = await res.json();
    if (data.count === 0) { alert('No items in trash.'); return; }
    if (!confirm('Move ' + data.count + ' file' + (data.count !== 1 ? 's' : '') + ' (' + data.size_mb + ' MB) to the Recycle Bin?\\n\\nYou can recover them from the Recycle Bin if needed.')) return;

    var btn = document.querySelector('#cleanup-bottom-bar .btn[style*="e53e3e"]');
    btn.disabled = true;
    btn.textContent = 'Moving...';

    try {
        var r = await fetch('/api/cleanup/confirm-trash', { method: 'POST' });
        var result = await r.json();
        if (result.ok) {
            var msg = result.recycled + ' file' + (result.recycled !== 1 ? 's' : '') + ' moved to Recycle Bin.';
            if (result.failed > 0) msg += '\\n' + result.failed + ' failed.';
            alert(msg);
            loadCleanupImages();
            updateTrashBadge();
        } else {
            alert('Error: ' + (result.error || 'Unknown'));
        }
    } catch(e) {
        alert('Failed to recycle files.');
    }
    btn.disabled = false;
    btn.textContent = 'Move to Recycle Bin';
}

// ── Age Assessment ──
var ageResults = [];

var ageRefPhotos = [];

function openAgeAssessment() {
    document.getElementById('age-overlay').classList.add('active');
    buildAgeFolderList();
    loadAgeResults();
    toggleAgeFaceMode();
    loadExistingAgeRefs();
}

function closeAgeAssessment() {
    document.getElementById('age-overlay').classList.remove('active');
}

function toggleAgeFaceMode() {
    var mode = document.querySelector('input[name="age-face-mode"]:checked');
    document.getElementById('age-face-ref').style.display = (mode && mode.value === 'specific') ? 'block' : 'none';
}

var ageRefVerified = false;

async function loadExistingAgeRefs() {
    try {
        var res = await fetch('/api/ref-faces');
        var faces = await res.json();
        var container = document.getElementById('age-use-existing');
        if (!Array.isArray(faces) || !faces.length) { container.innerHTML = ''; return; }
        var personsWithPhotos = faces.filter(function(f) { return f.photo_count > 0; });
        if (!personsWithPhotos.length) { container.innerHTML = ''; return; }
        container.innerHTML = '<div style="font-size:.8em; color:#718096; margin-bottom:4px;">Or use existing project references:</div>';
        personsWithPhotos.forEach(function(f) {
            var btn = document.createElement('button');
            btn.className = 'btn btn-secondary';
            btn.style.cssText = 'font-size:.78em; padding:4px 10px; margin:0 4px 4px 0;';
            btn.textContent = f.name + ' (' + f.photo_count + ' photos)';
            btn.onclick = function() {
                document.getElementById('age-ref-person-name').value = f.name;
                ageRefVerified = false;
                loadAgeRefThumbs(f.name);
            };
            container.appendChild(btn);
        });
    } catch(e) {}
}

async function uploadAgeRefPhotos(files) {
    var personName = document.getElementById('age-ref-person-name').value.trim();
    if (!personName) { alert('Please enter the person name first.'); return; }

    var status = document.getElementById('age-ref-status');
    status.textContent = 'Uploading ' + files.length + ' photos...';

    var fd = new FormData();
    fd.append('person', personName);
    for (var i = 0; i < files.length; i++) {
        fd.append('photos', files[i]);
    }
    try {
        await fetch('/api/ref-faces/upload', { method: 'POST', body: fd });
    } catch(e) {}
    document.getElementById('age-ref-upload').value = '';
    ageRefVerified = false;
    await loadAgeRefThumbs(personName);
}

async function loadAgeRefThumbs(personName) {
    var thumbs = document.getElementById('age-ref-thumbs');
    var status = document.getElementById('age-ref-status');
    thumbs.innerHTML = '';
    try {
        var res = await fetch('/api/ref-faces/' + encodeURIComponent(personName) + '/photos');
        var photos = await res.json();
        if (!photos.length) {
            status.textContent = 'No photos uploaded yet.';
            document.getElementById('btn-age-verify').style.display = 'none';
            return;
        }
        status.innerHTML = photos.length + ' photo(s) for <strong>' + esc(personName) + '</strong>';
        document.getElementById('btn-age-verify').style.display = '';

        photos.forEach(function(p) {
            var wrap = document.createElement('div');
            wrap.style.cssText = 'position:relative; display:inline-flex; flex-direction:column; align-items:center; gap:2px;';
            wrap.setAttribute('data-filename', p.filename);

            var imgDiv = document.createElement('div');
            imgDiv.style.cssText = 'position:relative;';

            if (p.thumb) {
                var img = document.createElement('img');
                img.src = 'data:image/jpeg;base64,' + p.thumb;
                img.style.cssText = 'width:60px; height:60px; object-fit:cover; border-radius:4px; border:2px solid #cbd5e0;';
                imgDiv.appendChild(img);
            } else {
                var placeholder = document.createElement('div');
                placeholder.style.cssText = 'width:60px; height:60px; background:#e2e8f0; border-radius:4px; display:flex; align-items:center; justify-content:center; font-size:.65em; color:#718096;';
                placeholder.textContent = p.filename;
                imgDiv.appendChild(placeholder);
            }

            // Remove button (X)
            var xBtn = document.createElement('span');
            xBtn.textContent = '\u2715';
            xBtn.style.cssText = 'position:absolute; top:-4px; right:-4px; width:16px; height:16px; background:#e53e3e; color:#fff; border-radius:50%; font-size:10px; display:flex; align-items:center; justify-content:center; cursor:pointer; box-shadow:0 1px 3px rgba(0,0,0,.3);';
            xBtn.title = 'Remove';
            (function(fn) {
                xBtn.onclick = function() { removeAgeRefPhoto(personName, fn); };
            })(p.filename);
            imgDiv.appendChild(xBtn);

            wrap.appendChild(imgDiv);

            // Replace label
            var replaceId = 'age-replace-' + p.filename.replace(/[^a-zA-Z0-9]/g, '-');
            var repLabel = document.createElement('label');
            repLabel.style.cssText = 'font-size:.6em; color:#3182ce; cursor:pointer; padding:1px 4px; background:#ebf8ff; border-radius:3px;';
            repLabel.textContent = 'Replace';
            repLabel.setAttribute('for', replaceId);
            var repInput = document.createElement('input');
            repInput.type = 'file';
            repInput.accept = 'image/*';
            repInput.id = replaceId;
            repInput.style.display = 'none';
            (function(fn) {
                repInput.onchange = function() { replaceAgeRefPhoto(personName, fn, this.files); };
            })(p.filename);
            wrap.appendChild(repLabel);
            wrap.appendChild(repInput);

            thumbs.appendChild(wrap);
        });
    } catch(e) {
        status.textContent = 'Error loading photos.';
    }
}

async function removeAgeRefPhoto(personName, filename) {
    await fetch('/api/ref-faces/' + encodeURIComponent(personName) + '/photo/' + encodeURIComponent(filename), { method: 'DELETE' });
    ageRefVerified = false;
    loadAgeRefThumbs(personName);
}

async function replaceAgeRefPhoto(personName, oldFilename, files) {
    if (!files || !files.length) return;
    // Delete old, upload new
    await fetch('/api/ref-faces/' + encodeURIComponent(personName) + '/photo/' + encodeURIComponent(oldFilename), { method: 'DELETE' });
    var fd = new FormData();
    fd.append('person', personName);
    fd.append('photos', files[0]);
    await fetch('/api/ref-faces/upload', { method: 'POST', body: fd });
    ageRefVerified = false;
    loadAgeRefThumbs(personName);
}

async function verifyAgeRefFaces() {
    var personName = document.getElementById('age-ref-person-name').value.trim();
    if (!personName) { alert('Enter person name first.'); return; }

    var resultDiv = document.getElementById('age-verify-result');
    resultDiv.style.display = 'block';
    resultDiv.style.background = '#f7fafc';
    resultDiv.style.color = '#4a5568';
    resultDiv.textContent = 'Verifying faces...';

    try {
        var res = await fetch('/api/ref-faces/verify', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({person: personName})
        });
        var data = await res.json();
        var personResult = (data.persons && data.persons.length) ? data.persons.find(function(p) { return p.person === personName; }) : null;

        // Apply green/red borders to thumbnails based on per-photo results
        if (personResult && personResult.photos) {
            personResult.photos.forEach(function(ph) {
                var thumbs = document.getElementById('age-ref-thumbs');
                if (!thumbs) return;
                var wraps = thumbs.children;
                for (var i = 0; i < wraps.length; i++) {
                    var imgEl = wraps[i].querySelector('img');
                    var placeholderEl = wraps[i].querySelector('div[style*="60px"]');
                    var target = imgEl || placeholderEl;
                    if (!target) continue;
                    // Match by data-filename attribute
                    if (wraps[i].getAttribute('data-filename') === ph.filename) {
                        var isOk = (ph.status === 'ok' || ph.status === 'ok_multi');
                        target.style.border = '3px solid ' + (isOk ? '#38a169' : '#e53e3e');
                        break;
                    }
                }
            });
        }

        if (data.ready || (personResult && personResult.ready)) {
            resultDiv.style.background = '#f0fff4';
            resultDiv.style.color = '#276749';
            resultDiv.innerHTML = '<strong>Faces verified successfully.</strong> Ready to run age assessment.';
            ageRefVerified = true;
        } else if (personResult) {
            var msg = personResult.issues && personResult.issues.length ? personResult.issues.join('. ') : 'Some photos may not have detectable faces. Try replacing them.';
            resultDiv.style.background = personResult.ok_count > 0 ? '#fffbeb' : '#fff5f5';
            resultDiv.style.color = personResult.ok_count > 0 ? '#92400e' : '#c53030';
            resultDiv.innerHTML = '<strong>' + personResult.ok_count + '/' + personResult.total_photos + ' photos verified.</strong> ' + esc(msg);
            ageRefVerified = false;
        } else {
            resultDiv.style.background = '#fff5f5';
            resultDiv.style.color = '#c53030';
            resultDiv.innerHTML = '<strong>No face data found.</strong> Upload reference photos and try again.';
            ageRefVerified = false;
        }
    } catch(e) {
        resultDiv.style.background = '#fff5f5';
        resultDiv.style.color = '#c53030';
        resultDiv.textContent = 'Verification failed: ' + e.message;
        ageRefVerified = false;
    }
}

function buildAgeFolderList() {
    var container = document.getElementById('age-folder-list');
    container.innerHTML = '';

    // Checkboxes area (above input)
    var cbArea = document.createElement('div');
    cbArea.id = 'age-folder-cbs';
    cbArea.style.cssText = 'margin-bottom:6px;';
    container.appendChild(cbArea);

    // Add source folders from config if available
    if (config && config.sources && config.sources.length) {
        config.sources.forEach(function(src) {
            if (src.path && src.path.trim()) {
                addAgeFolderCheckbox(src.path, false);
            }
        });
    }

    // Manual folder input row
    var addRow = document.createElement('div');
    addRow.style.cssText = 'display:flex; gap:8px; align-items:center;';
    var inp = document.createElement('input');
    inp.type = 'text';
    inp.id = 'age-folder-input';
    inp.placeholder = 'Enter folder path and click Add';
    inp.style.cssText = 'flex:1; padding:6px 10px; border:1px solid #e2e8f0; border-radius:6px; font-size:.85em;';
    var addBtn = document.createElement('button');
    addBtn.className = 'btn btn-secondary';
    addBtn.style.cssText = 'font-size:.8em; padding:6px 12px; margin:0;';
    addBtn.textContent = 'Add Folder';
    addBtn.onclick = function() {
        var val = inp.value.trim();
        if (val) { addAgeFolderCheckbox(val, true); inp.value = ''; }
    };
    addRow.appendChild(inp);
    addRow.appendChild(addBtn);
    container.appendChild(addRow);
}

function addAgeFolderCheckbox(path, checked) {
    var cbArea = document.getElementById('age-folder-cbs');
    var row = document.createElement('div');
    row.style.cssText = 'display:flex; align-items:center; gap:6px; margin-bottom:3px;';
    var cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.className = 'age-folder-cb';
    cb.value = path;
    cb.checked = checked;
    cb.style.cssText = 'margin:0; flex-shrink:0;';
    row.appendChild(cb);
    var txt = document.createElement('span');
    txt.textContent = path;
    txt.style.cssText = 'font-size:.82em; color:#4a5568;';
    row.appendChild(txt);
    cbArea.appendChild(row);
}

async function runAgeAssessment() {
    var folders = [...document.querySelectorAll('.age-folder-cb:checked')].map(function(cb) { return cb.value; });
    if (!folders.length) { alert('Please select at least one folder.'); return; }

    var faceMode = document.querySelector('input[name="age-face-mode"]:checked');
    var mode = faceMode ? faceMode.value : 'all';
    var personName = '';
    if (mode === 'specific') {
        personName = document.getElementById('age-ref-person-name').value.trim();
        if (!personName) { alert('Please enter a person name and upload reference photos.'); return; }
    }

    document.getElementById('btn-run-age').disabled = true;
    document.getElementById('btn-run-age').textContent = 'Running...';
    document.getElementById('btn-stop-age').style.display = '';
    document.getElementById('age-progress').style.display = 'block';
    document.getElementById('age-progress').textContent = 'Starting age assessment...';

    try {
        var res = await fetch('/api/age-assess/start', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ folders: folders, face_mode: mode, person_name: personName })
        });
        var data = await res.json();
        if (data.error) {
            document.getElementById('age-progress').textContent = 'Error: ' + data.error;
            document.getElementById('btn-run-age').disabled = false;
            document.getElementById('btn-run-age').textContent = 'Run Age Assessment';
            document.getElementById('btn-stop-age').style.display = 'none';
            return;
        }
        pollAgeProgress();
    } catch(e) {
        document.getElementById('age-progress').textContent = 'Failed to start: ' + e.message;
        document.getElementById('btn-run-age').disabled = false;
        document.getElementById('btn-run-age').textContent = 'Run Age Assessment';
        document.getElementById('btn-stop-age').style.display = 'none';
    }
}

function pollAgeProgress() {
    var interval = setInterval(async function() {
        try {
            var res = await fetch('/api/scan/status');
            var st = await res.json();
            var prog = document.getElementById('age-progress');
            if (st.lines && st.lines.length) {
                prog.textContent = st.lines[st.lines.length - 1];
            } else if (st.progress) {
                prog.textContent = st.progress;
            }
            if (!st.running || st.done || st.error || st.cancelled) {
                clearInterval(interval);
                document.getElementById('btn-run-age').disabled = false;
                document.getElementById('btn-run-age').textContent = 'Run Age Assessment';
                document.getElementById('btn-stop-age').style.display = 'none';
                if (st.error) {
                    prog.textContent = 'Error: ' + st.error;
                } else if (st.cancelled) {
                    prog.textContent = 'Cancelled.';
                }
                loadAgeResults();
            }
        } catch(e) {
            clearInterval(interval);
        }
    }, 1000);
}

async function stopAgeAssessment() {
    await fetch('/api/task/stop', { method: 'POST' });
    document.getElementById('btn-stop-age').style.display = 'none';
}

async function loadAgeResults() {
    try {
        var res = await fetch('/api/age-assess/results');
        ageResults = await res.json();
        if (ageResults.length) {
            document.getElementById('age-results').style.display = 'block';
            // Populate person filter
            var personSet = new Set();
            ageResults.forEach(function(r) {
                if (r.person) personSet.add(r.person);
            });
            var sel = document.getElementById('age-filter-person');
            sel.innerHTML = '<option value="">All</option>';
            [...personSet].sort().forEach(function(p) {
                var opt = document.createElement('option');
                opt.value = p; opt.textContent = p;
                sel.appendChild(opt);
            });
            renderAgeResults();
        }
    } catch(e) {}
}

function renderAgeResults() {
    var sort = document.getElementById('age-sort').value;
    var filterPerson = document.getElementById('age-filter-person').value;

    var filtered = ageResults;
    if (filterPerson) {
        filtered = filtered.filter(function(r) { return r.person === filterPerson; });
    }

    var sorted = filtered.slice();
    if (sort === 'age-asc') sorted.sort(function(a, b) { return a.estimated_age - b.estimated_age; });
    else if (sort === 'age-desc') sorted.sort(function(a, b) { return b.estimated_age - a.estimated_age; });
    else if (sort === 'name') sorted.sort(function(a, b) { return (a.person || '').localeCompare(b.person || ''); });
    else if (sort === 'date') sorted.sort(function(a, b) { return (a.date || '').localeCompare(b.date || ''); });

    document.getElementById('age-result-count').textContent = '(' + sorted.length + ' images)';

    var grid = document.getElementById('age-grid');
    grid.innerHTML = '';
    sorted.forEach(function(r) {
        var card = document.createElement('div');
        card.className = 'age-card';
        var img = document.createElement('img');
        img.src = r.thumb ? 'data:image/jpeg;base64,' + r.thumb : '';
        img.alt = r.filename;
        img.onerror = function() { this.style.background = '#edf2f7'; this.style.minHeight = '80px'; };
        card.appendChild(img);
        var info = document.createElement('div');
        info.className = 'age-info';
        var badge = document.createElement('span');
        badge.className = 'age-badge';
        badge.textContent = 'Age ~' + r.estimated_age;
        info.appendChild(badge);
        var fn = document.createElement('div');
        fn.style.cssText = 'margin-top:4px; font-size:.9em; color:#718096; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;';
        fn.textContent = r.filename;
        fn.title = r.path || r.filename;
        info.appendChild(fn);
        if (r.folder) {
            var fld = document.createElement('div');
            fld.style.cssText = 'font-size:.85em; color:#a0aec0;';
            fld.textContent = r.folder;
            info.appendChild(fld);
        }
        card.appendChild(info);
        grid.appendChild(card);
    });
}

// ── Init ──
document.getElementById('inp-birthday').max = new Date().toISOString().split('T')[0];
loadTemplates();
loadGreeting();
</script>
</body>
</html>"""

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E-z Photo Organizer Web UI")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    url = f"http://localhost:{args.port}"
    print(f"E-z Photo Organizer running at {url}")

    if not args.no_open:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    app.run(host="127.0.0.1", port=args.port, debug=False)
