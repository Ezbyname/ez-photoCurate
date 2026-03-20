"""
PhotoCurate Web UI — Step-by-step wizard for building photo collections.
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

from flask import Flask, request, jsonify, send_file

sys.stdout.reconfigure(line_buffering=True)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SCAN_DB_PATH = os.path.join(PROJECT_DIR, "scan_db.json")
CONFIG_PATH = os.path.join(PROJECT_DIR, "curate_config.json")

app = Flask(__name__, static_folder=None)

# Lock for scan_db.json reads/writes
_db_lock = threading.Lock()

# ── Background task state ─────────────────────────────────────────────────────

_task = {"running": False, "type": None, "progress": "", "lines": [], "done": False, "error": None}
_task_lock = threading.Lock()


def _reset_task(task_type):
    with _task_lock:
        _task["running"] = True
        _task["type"] = task_type
        _task["progress"] = "Starting..."
        _task["lines"] = []
        _task["done"] = False
        _task["error"] = None


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
                age_days_to_bracket, IMAGE_EXTS, REEF_BIRTHDAY,
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
                ref_encodings = load_reference_faces(face_dir)
                use_faces = bool(ref_encodings)

            # Existing DB for incremental
            existing_db = {}
            if os.path.isfile(SCAN_DB_PATH) and not full:
                old = load_scan_db()
                if old:
                    for img in old.get("images", []):
                        existing_db[img["hash"]] = img
                _update_task(f"Incremental mode: {len(existing_db)} cached images")

            all_images = []
            seen_hashes = set()
            scanned = 0
            skipped = defaultdict(int)

            for source in sources:
                src_path = source.get("path", "")
                src_label = source.get("label", "Unknown")

                if not os.path.isdir(src_path):
                    _update_task(f"Source not found: {src_label}")
                    continue

                _update_task(f"Scanning: {src_label}...")
                src_count = 0

                for dirpath, dirnames, filenames in os.walk(src_path):
                    if any(v in dirpath.lower() for v in ["video", "\u05d5\u05d9\u05d3\u05d0\u05d5"]):
                        continue
                    rel_dir = os.path.relpath(dirpath, src_path)

                    for fname in filenames:
                        ext = os.path.splitext(fname)[1].lower()
                        if ext not in IMAGE_EXTS:
                            continue

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
                            all_images.append(entry)
                            src_count += 1
                            continue

                        try:
                            file_size = os.path.getsize(fpath)
                            if file_size < min_size_kb * 1024:
                                skipped["too_small"] += 1
                                continue
                            img = PILImage.open(fpath)
                            w, h = img.size
                            img.close()
                            if w < min_dim and h < min_dim:
                                skipped["low_res"] += 1
                                continue
                        except Exception:
                            skipped["unreadable"] += 1
                            continue

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
                        if file_size < 500 * 1024:
                            screenshot = is_screenshot(fpath)

                        face_count = 0
                        faces_found = []
                        if use_faces:
                            face_count, faces_found, ok = detect_faces_in_image(
                                fpath, ref_encodings, tolerance)

                        thumb = make_thumbnail_b64(fpath, thumb_size)
                        device = guess_device_source(fname)

                        entry = {
                            "hash": fhash,
                            "path": fpath.replace("\\", "/"),
                            "filename": fname,
                            "source_label": src_label,
                            "device": device,
                            "date": img_date.strftime("%Y-%m-%d") if img_date else None,
                            "age_days": age_days,
                            "category": bracket,
                            "face_count": face_count,
                            "faces_found": faces_found,
                            "has_target_face": any(n in faces_found for n in face_names) if face_names else (face_count > 0),
                            "width": w, "height": h,
                            "size_kb": round(file_size / 1024),
                            "is_screenshot": screenshot,
                            "thumb": thumb,
                        }

                        if screenshot:
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
        })


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

            # Monkey-patch print to capture progress
            import builtins
            _orig_print = builtins.print
            def _capture_print(*args, **kwargs):
                line = " ".join(str(a) for a in args)
                # Skip internal/traceback lines
                if not line.startswith(("File ", "Traceback", "  ", "json.decoder")):
                    _update_task(line)
                _orig_print(*args, **kwargs)
            builtins.print = _capture_print

            from event_agent import auto_select
            db, report = auto_select(db, strategy=strategy, sim_threshold=threshold)

            builtins.print = _orig_print

            save_scan_db(db)

            _update_task(f"Done! Selected {report['total_selected']} images.")
            _finish_task()
        except Exception as e:
            _finish_task(str(e))

    threading.Thread(target=run_select, daemon=True).start()
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
    output_dir = data.get("output_dir", os.path.join(PROJECT_DIR, "final_collection"))
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
                images = [i for i in db["images"] if i.get("status") == "qualified"]

            total = len(images)
            _update_task(f"Exporting {total} images...")

            for i, img in enumerate(images):
                cat = img.get("category", "uncategorized")
                dest_dir = os.path.join(output_dir, cat)
                os.makedirs(dest_dir, exist_ok=True)

                src = img["path"].replace("/", os.sep)
                if not os.path.isfile(src):
                    continue

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

            _update_task(f"Done! {exported} images exported to {output_dir}")
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
        stats["total_images"] = len(images)
        stats["qualified"] = sum(1 for i in images if i.get("status") == "qualified")
        stats["selected"] = sum(1 for i in images if i.get("status") == "selected")
        stats["pool"] = sum(1 for i in images if i.get("status") == "pool")
        stats["sources"] = list(set(i.get("source_label", "") for i in images))

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
<title>PhotoCurate</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background:#f5f7fa; color:#2d3748; min-height:100vh; }

/* ── Layout ── */
.app { max-width:1200px; margin:0 auto; padding:20px; }
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
</style>
</head>
<body>
<div class="app">

<div class="header">
    <h1>PhotoCurate</h1>
    <p>Build a meaningful photo collection for your special event</p>
</div>

<div class="steps" id="steps-nav">
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

<!-- ── STEP 0: Choose Event ── -->
<div class="panel active" id="panel-0">
    <h2>What are you creating?</h2>
    <p>Choose the type of event. This sets up the right categories and targets for your photo collection.</p>
    <div class="template-grid" id="template-grid"></div>

    <div id="event-fields" style="margin-top:20px; display:none;">
        <div class="row">
            <div id="field-birthday" style="display:none">
                <label>Child's Birthday</label>
                <input type="date" id="inp-birthday">
            </div>
            <div id="field-event-date" style="display:none">
                <label>Event Date</label>
                <input type="date" id="inp-event-date">
            </div>
            <div id="field-end-date" style="display:none">
                <label>End Date</label>
                <input type="date" id="inp-end-date">
            </div>
            <div id="field-year" style="display:none">
                <label>Year</label>
                <input type="number" id="inp-year" placeholder="2024" min="2000" max="2030">
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

    <div style="display:flex; gap:20px; margin-bottom:15px; flex-wrap:wrap;">
        <div style="background:#ebf8ff; border:1px solid #bee3f8; border-radius:8px; padding:15px; flex:1; min-width:150px; text-align:center;">
            <div style="font-size:1.8em; font-weight:bold; color:#2b6cb0;" id="cat-total-count">0</div>
            <div style="color:#718096; font-size:.85em;">Categories</div>
        </div>
        <div style="background:#f0fff4; border:1px solid #c6f6d5; border-radius:8px; padding:15px; flex:1; min-width:150px; text-align:center;">
            <div style="font-size:1.8em; font-weight:bold; color:#38a169;" id="cat-total-target">0</div>
            <div style="color:#718096; font-size:.85em;">Total Target Images</div>
        </div>
    </div>

    <table class="analysis-table" id="cat-table">
        <thead>
            <tr>
                <th style="width:40px;">#</th>
                <th>Category Name</th>
                <th style="width:120px;">Target</th>
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

    <div class="btn-group" id="face-verify-btn-group" style="display:none;">
        <button class="btn btn-primary" onclick="runVerifyAll()">Verify All Faces</button>
    </div>

    <div id="face-verify-results" style="display:none; margin-top:20px;"></div>

    <div class="btn-group">
        <button class="btn btn-secondary" onclick="goStep(2)">Back</button>
        <button class="btn btn-secondary" onclick="skipFaces()">Skip (no face recognition)</button>
        <button class="btn btn-primary" id="btn-next-3" disabled onclick="completeFacesStep()">Next: Start Scan</button>
    </div>
</div>

<!-- ── STEP 4: Scan ── -->
<div class="panel" id="panel-4">
    <h2>Scanning your photos</h2>
    <p>This scans all your sources, extracts dates, detects duplicates, and creates thumbnails. This runs once — future scans are incremental.</p>

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
    <button class="btn btn-primary" onclick="runAnalysis()">Run Analysis</button>

    <div id="analysis-results" style="display:none; margin-top:20px;">
        <div class="row" style="margin-bottom:15px;">
            <div style="text-align:center"><div class="export-summary"><div class="big-num" id="a-total">0</div>Total Images</div></div>
            <div style="text-align:center"><div class="export-summary"><div class="big-num" id="a-qualified">0</div>Qualified</div></div>
            <div style="text-align:center"><div class="export-summary"><div class="big-num" id="a-target">0</div>Target</div></div>
        </div>

        <h3 style="color:#718096; margin:15px 0 8px;">Categories</h3>
        <table class="analysis-table">
            <thead><tr><th>Category</th><th>Count</th><th>Target</th><th>Status</th></tr></thead>
            <tbody id="a-categories"></tbody>
        </table>

        <h3 style="color:#718096; margin:20px 0 8px;">Recommendations</h3>
        <div id="a-recommendations"></div>

        <h3 style="color:#718096; margin:20px 0 8px;">What makes a great collection</h3>
        <div id="a-priorities"></div>
    </div>

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
            <div id="sel-filter-bar" style="display:none; margin-bottom:8px; font-size:.85em;">
                <label style="margin-right:8px;">
                    <input type="checkbox" id="sel-show-selected" checked onchange="renderSelGrid()"> Show selected
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
        <div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap;">
            <label style="font-weight:600; color:#2d3748; margin:0;">Auto-fill remaining slots</label>
            <select id="sel-strategy" style="max-width:220px; padding:4px 8px; border:1px solid #cbd5e0; border-radius:4px;">
                <option value="balanced">Balanced</option>
                <option value="quality">Quality</option>
                <option value="diverse">Diverse</option>
            </select>
            <button class="btn btn-primary" id="btn-auto-select" onclick="runAutoSelect()">Auto-Fill</button>
        </div>
        <div class="progress-box" id="select-progress" style="display:none; margin-top:8px;"></div>
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
    <input type="text" id="inp-export-dir" value="final_collection">

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
</div>

<script>
let currentStep = 0;
let selectedTemplate = null;
let templates = [];
let config = null;

// ── Step navigation ──
function goStep(n) {
    currentStep = n;
    document.querySelectorAll('.panel').forEach((p, i) => p.classList.toggle('active', i === n));
    document.querySelectorAll('.step-dot').forEach((d, i) => {
        d.classList.toggle('active', i === n);
    });

    if (n === 1) loadCategoriesStep();
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
    const grid = document.getElementById('template-grid');
    grid.innerHTML = templates.map(t => `
        <div class="template-card" onclick="selectTemplate('${t.event_type}')" id="tpl-${t.event_type}">
            <h3>${t.display_name}</h3>
            <div class="desc">${t.description}</div>
            <div class="meta">${t.num_categories} categories | ${t.categorization}</div>
        </div>
    `).join('');

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
    document.querySelectorAll('.template-card').forEach(c => c.classList.remove('selected'));
    document.getElementById('tpl-' + type).classList.add('selected');
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
    const cats = config?.categories || [];
    renderCategories(cats);
}

function renderCategories(cats) {
    const tbody = document.getElementById('cat-tbody');
    tbody.innerHTML = cats.map((c, i) => `
        <tr>
            <td style="color:#a0aec0;">${i + 1}</td>
            <td><input type="text" value="${esc(c.display || c.id || '')}" onchange="updateCatField(${i}, 'display', this.value)" style="border:1px solid #e2e8f0;"></td>
            <td><input type="number" value="${c.target || config?.target_per_category || 75}" min="1" max="500" onchange="updateCatField(${i}, 'target', parseInt(this.value))" style="width:100px; border:1px solid #e2e8f0;"></td>
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

function updateCatSummary() {
    const cats = config?.categories || [];
    const defaultTarget = config?.target_per_category || 75;
    document.getElementById('cat-total-count').textContent = cats.length;
    const total = cats.reduce((sum, c) => sum + (c.target || defaultTarget), 0);
    document.getElementById('cat-total-target').textContent = total;
}

function addCategory() {
    if (!config) return;
    if (!config.categories) config.categories = [];
    const n = config.categories.length + 1;
    config.categories.push({
        id: 'custom_' + n,
        display: 'New Category ' + n,
        target: config.target_per_category || 75
    });
    renderCategories(config.categories);
}

function removeCategory(i) {
    if (!config?.categories) return;
    config.categories.splice(i, 1);
    renderCategories(config.categories);
}

async function completeCategoriesStep() {
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
    const sources = config?.sources || [];
    list.innerHTML = sources.map((s, i) => `
        <div class="source-item">
            <input type="text" value="${s.label || ''}" placeholder="Label" style="max-width:150px" onchange="updateSource(${i},'label',this.value)">
            <input type="text" value="${s.path || ''}" placeholder="Full path to folder (e.g. D:\\\\Photos)" onchange="updateSource(${i},'path',this.value)">
            <span class="remove" onclick="removeSource(${i})">&times;</span>
        </div>
    `).join('');
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
}

let faceVerifyCache = {};

async function loadFaceThumbs(person) {
    const res = await fetch('/api/ref-faces/' + encodeURIComponent(person) + '/photos');
    const photos = await res.json();
    const container = document.getElementById('face-thumbs-' + person);
    if (!container) return;
    const statuses = faceVerifyCache[person] || {};
    container.innerHTML = photos.map(p => {
        const st = statuses[p.filename];
        const border = st === 'ok' || st === 'ok_multi' ? '3px solid #4caf50' : st === 'no_face' || st === 'encode_fail' || st === 'error' ? '3px solid #f44336' : '2px solid #cbd5e0';
        const label = st === 'no_face' ? '\\u2718' : st === 'ok' || st === 'ok_multi' ? '\\u2714' : '';
        const labelColor = st === 'ok' || st === 'ok_multi' ? '#4caf50' : '#f44336';
        const safeFn = esc(p.filename).replace(/'/g, "\\\\'");
        const safePerson = esc(person).replace(/'/g, "\\\\'");
        let html = '<div style="display:inline-flex; flex-direction:column; align-items:center; gap:2px; margin-right:8px; margin-bottom:6px;">';
        html += '<label style="font-size:.65em; color:#3182ce; cursor:pointer; padding:1px 4px; background:#ebf8ff; border-radius:3px; white-space:nowrap;" for="replace-' + person + '-' + p.filename + '">Replace</label>';
        html += '<input type="file" id="replace-' + person + '-' + p.filename + '" accept="image/*" style="display:none" onchange="replaceFacePhoto(\\'' + safePerson + '\\', \\'' + safeFn + '\\', this.files)">';
        html += '<div style="position:relative;" id="face-img-' + person + '-' + p.filename.replace(/[^a-zA-Z0-9]/g,'-') + '">';
        if (p.thumb) {
            html += '<img src="data:image/jpeg;base64,' + p.thumb + '" style="width:70px; height:70px; object-fit:cover; border-radius:4px; border:' + border + '; cursor:pointer;" title="Double-click to enlarge" ondblclick="openLightbox(\\'' + safePerson + '\\', \\'' + safeFn + '\\')">';
        } else {
            html += '<div style="width:70px; height:70px; background:#e2e8f0; border-radius:4px; border:' + border + '; display:flex; align-items:center; justify-content:center; font-size:.7em; color:#718096;">' + esc(p.filename) + '</div>';
        }
        if (label) {
            html += '<span style="position:absolute; bottom:1px; right:3px; font-size:14px; color:' + labelColor + '; text-shadow:0 0 3px #fff;">' + label + '</span>';
        }
        html += '</div>';
        html += '<div style="display:flex; gap:2px;">';
        html += '<button onclick="rotateFace(\\'' + safePerson + '\\', \\'' + safeFn + '\\', \\'ccw\\')" style="font-size:.7em; padding:1px 5px; cursor:pointer; background:#edf2f7; border:1px solid #cbd5e0; border-radius:3px;" title="Rotate left">&#x21BA;</button>';
        html += '<button onclick="rotateFace(\\'' + safePerson + '\\', \\'' + safeFn + '\\', \\'cw\\')" style="font-size:.7em; padding:1px 5px; cursor:pointer; background:#edf2f7; border:1px solid #cbd5e0; border-radius:3px;" title="Rotate right">&#x21BB;</button>';
        html += '</div>';
        html += '</div>';
        return html;
    }).join('');
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

    markFacesDirty();
    await loadFaceThumbs(person);

    const statusEl = document.getElementById('face-status-' + person);
    if (statusEl) statusEl.innerHTML = '<span style="color:#718096;">Image replaced. Click <strong>Verify All Faces</strong> to re-check.</span>';
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
    statusEl.innerHTML = '<span style="color:#3182ce;">Verifying face encodings... (this may take a moment)</span>';

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
    resultsEl.style.display = 'block';
    resultsEl.innerHTML = '<span style="color:#3182ce;">Verifying all faces...</span>';

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
    if (allReady) {
        document.getElementById('btn-next-3').disabled = false;
    } else {
        document.getElementById('btn-next-3').disabled = false;  // allow proceed with warning
    }
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
    const personNames = faces.filter(f => f.photo_count > 0).map(f => f.name);

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
    } else {
        config.face_names = [];
    }

    await fetch('/api/config', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(config) });
    document.querySelectorAll('.step-dot')[3].classList.add('done');
    goStep(4);
}

// ── Step 4: Scan ──
async function checkExistingScan() {
    const res = await fetch('/api/stats');
    const st = await res.json();
    if (st.has_scan && st.total_images > 0) {
        document.getElementById('btn-next-4').disabled = false;
        document.getElementById('scan-progress').style.display = 'block';
        document.getElementById('scan-progress').innerHTML = '<div class="line" style="color:#38a169;">Existing scan found: ' + st.total_images + ' images from ' + (st.sources?.length || 0) + ' sources. You can proceed or rescan.</div>';
    }
}

let scanPoll = null;

async function startScan(full) {
    document.getElementById('btn-start-scan').disabled = true;
    document.getElementById('scan-progress').style.display = 'block';
    document.getElementById('scan-progress').innerHTML = '<div class="line current">Starting scan...</div>';

    await fetch('/api/scan/start', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({full}) });

    scanPoll = setInterval(async () => {
        const res = await fetch('/api/scan/status');
        const st = await res.json();
        const box = document.getElementById('scan-progress');
        box.innerHTML = st.lines.map((l, i) =>
            '<div class="line' + (i === st.lines.length - 1 ? ' current' : '') + '">' + esc(l) + '</div>'
        ).join('');
        box.scrollTop = box.scrollHeight;

        if (st.done || st.error) {
            clearInterval(scanPoll);
            document.getElementById('btn-start-scan').disabled = false;
            if (st.error) {
                box.innerHTML += '<div class="line" style="color:#f44336">Error: ' + esc(st.error) + '</div>';
            } else {
                document.getElementById('btn-next-4').disabled = false;
                document.querySelectorAll('.step-dot')[4].classList.add('done');
            }
        }
    }, 1500);
}

// ── Step 5: Analyze ──
async function runAnalysis() {
    const res = await fetch('/api/analyze');
    if (!res.ok) return;
    const a = await res.json();

    document.getElementById('analysis-results').style.display = 'block';
    document.getElementById('a-total').textContent = a.total_images;
    document.getElementById('a-qualified').textContent = a.total_qualified;
    document.getElementById('a-target').textContent = a.total_target;

    const tbody = document.getElementById('a-categories');
    tbody.innerHTML = a.categories.map(c => `
        <tr>
            <td>${esc(c.display)}</td>
            <td>${c.count}</td>
            <td>${c.target}</td>
            <td class="status-${c.status}">${c.status.toUpperCase()}</td>
        </tr>
    `).join('');

    const recs = document.getElementById('a-recommendations');
    recs.innerHTML = a.recommendations.map(r => `
        <div class="rec-card ${r.type}">
            <div class="title">${esc(r.title)}</div>
            <div class="detail">${esc(r.detail)}</div>
        </div>
    `).join('');

    const pris = document.getElementById('a-priorities');
    pris.innerHTML = (a.priorities || []).map((p, i) => `<div style="color:#718096; padding:3px 0;">${i+1}. ${esc(p)}</div>`).join('');

    document.querySelectorAll('.step-dot')[5].classList.add('done');
}

// ── Step 6: Select ──
let selCats = [];
let selActiveCat = null;
let selImages = [];
let selOffset = 0;
const SEL_PAGE = 100;
let selectPoll = null;

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
    document.getElementById('sel-filter-bar').style.display = 'block';
    await loadSelImages();
}

async function loadSelImages() {
    const grid = document.getElementById('sel-grid');
    grid.innerHTML = '<div style="color:#718096;">Loading...</div>';
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
        div.innerHTML = `<img src="${thumbSrc}" style="width:100%; height:100%; object-fit:cover;">
            ${img._sel ? '<div style="position:absolute; top:2px; right:2px; background:#3182ce; color:#fff; border-radius:50%; width:18px; height:18px; font-size:12px; display:flex; align-items:center; justify-content:center;">&#10003;</div>' : ''}`;
        div.onclick = () => toggleSelImage(img);
        div.ondblclick = (e) => { e.stopPropagation(); showSelLightbox(img); };
        grid.appendChild(div);
    });
    if (paged.length === 0) {
        grid.innerHTML = '<div style="color:#718096; padding:20px;">No images in this category.</div>';
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
    overlay.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.85);z-index:9999;display:flex;align-items:center;justify-content:center;';
    overlay.onclick = () => overlay.remove();
    const imgEl = document.createElement('img');
    imgEl.src = '/api/images/serve/' + img.hash;
    imgEl.style.cssText = 'max-width:90vw;max-height:85vh;border-radius:8px;box-shadow:0 4px 30px rgba(0,0,0,.5);';
    const close = document.createElement('div');
    close.innerHTML = '&times;';
    close.style.cssText = 'position:absolute;top:20px;right:30px;color:#fff;font-size:36px;cursor:pointer;';
    close.onclick = () => overlay.remove();
    const info = document.createElement('div');
    info.style.cssText = 'position:absolute;bottom:20px;left:50%;transform:translateX(-50%);color:#fff;font-size:.85em;background:rgba(0,0,0,.6);padding:6px 16px;border-radius:6px;';
    info.textContent = (img.filename || '') + (img.date_taken ? ' | ' + img.date_taken : '') + (img.source_label ? ' | ' + img.source_label : '');
    overlay.appendChild(imgEl);
    overlay.appendChild(close);
    overlay.appendChild(info);
    document.body.appendChild(overlay);
}

async function runAutoSelect() {
    const strategy = document.getElementById('sel-strategy').value;
    document.getElementById('btn-auto-select').disabled = true;
    document.getElementById('select-progress').style.display = 'block';
    document.getElementById('select-progress').innerHTML = '<div class="line current">Starting auto-fill...</div>';

    await fetch('/api/auto-select', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({ strategy, sim_threshold: 0.85 })
    });

    selectPoll = setInterval(async () => {
        const res = await fetch('/api/scan/status');
        const st = await res.json();
        const box = document.getElementById('select-progress');
        const clean = st.lines.filter(l => !l.startsWith('File ') && !l.startsWith('Traceback') && !l.startsWith('  ') && !l.startsWith('json.'));
        box.innerHTML = clean.map((l, i) =>
            '<div class="line' + (i === clean.length - 1 ? ' current' : '') + '">' + esc(l) + '</div>'
        ).join('');
        box.scrollTop = box.scrollHeight;

        if (st.done || st.error) {
            clearInterval(selectPoll);
            document.getElementById('btn-auto-select').disabled = false;
            document.querySelectorAll('.step-dot')[6].classList.add('done');
            // Reload categories and current grid
            await loadSelCategories();
            if (selActiveCat) await loadSelImages();
        }
    }, 2000);
}

// ── Step 7: Review ──
function openGallery() {
    window.open('/api/report', '_blank');
}

// ── Step 8: Export ──
async function loadExportStats() {
    const res = await fetch('/api/stats');
    const st = await res.json();
    const el = document.getElementById('export-stats');
    if (st.has_scan) {
        const selected = st.selected || st.qualified || 0;
        el.innerHTML = `
            <div class="export-summary">
                <div class="big-num">${selected}</div>
                <div>images ready to export</div>
                <div style="color:#a0aec0; font-size:.85em; margin-top:5px;">${st.total_images} total scanned | ${st.sources?.length || 0} sources</div>
            </div>
        `;
    }
}

let exportPoll = null;
async function runExport() {
    const outputDir = document.getElementById('inp-export-dir').value || 'final_collection';
    document.getElementById('btn-export').disabled = true;
    document.getElementById('export-progress').style.display = 'block';
    document.getElementById('export-progress').innerHTML = '<div class="line current">Starting export...</div>';

    await fetch('/api/export', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({ output_dir: outputDir })
    });

    exportPoll = setInterval(async () => {
        const res = await fetch('/api/scan/status');
        const st = await res.json();
        const box = document.getElementById('export-progress');
        box.innerHTML = st.lines.map((l, i) =>
            '<div class="line' + (i === st.lines.length - 1 ? ' current' : '') + '">' + esc(l) + '</div>'
        ).join('');
        box.scrollTop = box.scrollHeight;

        if (st.done || st.error) {
            clearInterval(exportPoll);
            document.getElementById('btn-export').disabled = false;
            if (!st.error) {
                document.querySelectorAll('.step-dot')[8].classList.add('done');
                box.innerHTML += '<div class="line" style="color:#4caf50; font-weight:bold;">Your photos are ready!</div>';
            }
        }
    }, 1500);
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

// ── Init ──
loadTemplates();
</script>
</body>
</html>"""

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PhotoCurate Web UI")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    url = f"http://localhost:{args.port}"
    print(f"PhotoCurate running at {url}")

    if not args.no_open:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    app.run(host="127.0.0.1", port=args.port, debug=False)
