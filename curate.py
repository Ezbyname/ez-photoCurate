"""
Image Curation Pipeline - Scan, Categorize, Review, Apply
==========================================================

Full workflow:
  1. python curate.py scan --config curate_config.json
     Scans ALL images from ALL sources, detects faces, extracts dates,
     categorizes by age bracket. Saves results to scan_db.json.

  2. python curate.py report
     Generates interactive HTML gallery from scan_db.json.
     Review images, move between categories, reject to pool.

  3. python curate.py apply changes.json
     Physically copies qualified images to an output folder.

Config file (curate_config.json) example:
{
  "ref_faces_dir": "./ref_faces",
  "face_names": ["reef"],
  "face_tolerance": 0.6,
  "sources": [
    {"path": "D:\\\\reef", "label": "USB Disk"},
    {"path": "C:\\\\...\\\\the presentation", "label": "Current Presentation"},
    {"path": "C:\\\\...\\\\Takeout\\\\Google Photos\\\\reef", "label": "Google Takeout"}
  ],
  "categories": "age",
  "target_per_category": 75,
  "min_size_kb": 80,
  "min_dim": 600
}
"""

import os
import sys
import argparse
import base64
import hashlib
import json
import re
import shutil
import webbrowser
import html as html_mod
from io import BytesIO
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
from PIL import Image, ImageOps

sys.stdout.reconfigure(line_buffering=True)

# ── Constants ──────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".heic", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".webm", ".m4v", ".mpg", ".mpeg", ".3gp"}
MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS
REEF_BIRTHDAY = datetime(2013, 7, 16)
VECTOR_SIZE = 64
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SCAN_DB_PATH = os.path.join(PROJECT_DIR, "scan_db.json")
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_DIR, "curate_config.json")

AGE_BRACKETS = [
    ("00_birth",         0,    14,  "Birth (0-2 weeks)"),
    ("01_month1",       14,    60,  "Month 1 (2-8 weeks)"),
    ("02_months2-3",    60,   120,  "Months 2-3"),
    ("03_months4-6",   120,   210,  "Months 4-6"),
    ("04_months7-9",   210,   300,  "Months 7-9"),
    ("05_months10-12", 300,   365,  "Months 10-12"),
    ("06_year1",       365,   730,  "Age 1-2"),
    ("07_year2",       730,  1095,  "Age 2-3"),
    ("08_year3-4",    1095,  1825,  "Age 3-5"),
    ("09_year5-7",    1825,  2920,  "Age 5-8"),
    ("10_year8-10",   2920,  3650,  "Age 8-10"),
    ("11_year11-12",  3650,  4745,  "Age 11-13"),
    ("12_barmitzva",  4745,  5110,  "Bar Mitzva"),
]

DIR_AGE_MAP = {
    "\u05d1\u05d9\u05ea \u05d7\u05d5\u05dc\u05d9\u05dd": 0,
    "\u05db\u05e9\u05e8\u05d9\u05e3 \u05e0\u05d5\u05dc\u05d3": 0,
    "\u05e8\u05d9\u05e3 \u05de\u05d2\u05d9\u05e2 \u05d4\u05d1\u05d9\u05ea\u05d4": 3,
    "\u05d1\u05e8\u05d9\u05ea \u05d0\u05e8\u05d6": 44,
    "\u05d1\u05e8\u05d9\u05ea": 8, "\u05dc\u05d9\u05d3\u05d4": 3,
    "\u05e8\u05d9\u05e3 \u05d1\u05df \u05d7\u05d5\u05d3\u05e9": 30,
    "\u05d7\u05d5\u05d3\u05e9 1": 30,
    "8.9- 9.9": 55, "28.9": 74,
    "\u05db\u05d9\u05e4\u05d5\u05e8 2013": 90,
    "\u05d7\u05d5\u05d3\u05e9 2-12": 180,
    "\u05e8\u05d9\u05e3 \u05d1\u05df \u05d7\u05d5\u05d3\u05e9\u05d9\u05d9\u05dd": 60,
    "\u05e8\u05d9\u05e3 \u05d1\u05df 3 \u05d7\u05d5\u05d3\u05e9\u05d9\u05dd": 90,
    "\u05e8\u05d9\u05e3 \u05d1\u05df 4 \u05d7\u05d5\u05d3\u05e9\u05d9\u05dd": 120,
    "\u05e8\u05d9\u05e3 \u05d1\u05df 5 \u05d7\u05d5\u05d3\u05e9\u05d9\u05dd": 150,
    "\u05e8\u05d9\u05e3 \u05d1\u05df \u05d7\u05e6\u05d9 \u05e9\u05e0\u05d4": 180,
    "\u05e8\u05d9\u05e3 \u05d1\u05df 7 \u05d7\u05d5\u05d3\u05e9\u05d9\u05dd": 210,
    "\u05e8\u05d9\u05e3 \u05d1\u05df 8 \u05d7\u05d5\u05d3\u05e9\u05d9\u05dd": 240,
    "\u05e8\u05d9\u05e3 \u05d1\u05df 9 \u05d7\u05d5\u05d3\u05e9\u05d9\u05dd": 270,
    "\u05e8\u05d9\u05e3 \u05d1\u05df 10 \u05d7\u05d5\u05d3\u05e9\u05d9\u05dd": 300,
    "\u05e8\u05d9\u05e3 \u05d1\u05df 11 \u05d7\u05d5\u05d3\u05e9\u05d9\u05dd": 330,
    "\u05e8\u05d9\u05e3 \u05d1\u05df \u05e9\u05e0\u05d4": 365,
    "\u05d1\u05d5\u05e7": 365,
    "\u05de\u05e9\u05e4\u05d7\u05d4 \u05d0\u05e8\u05d6": 365,
    "\u05e9\u05e0\u05d4 - \u05e9\u05e0\u05ea\u05d9\u05d9\u05dd": 545,
    "\u05e9\u05e0\u05d4 \u05d5\u05d7\u05d5\u05d3\u05e9": 395,
    "\u05e9\u05e0\u05d4 \u05d5\u05d7\u05d5\u05d3\u05e9\u05d9\u05d9\u05dd": 425,
    "\u05e9\u05e0\u05d4 \u05d5\u05e9\u05dc\u05d5\u05e9": 455,
    "\u05e9\u05e0\u05d4 \u05d5\u05d0\u05e8\u05d1\u05e2": 485,
    "\u05e9\u05e0\u05d4 \u05d5\u05d7\u05de\u05e9": 515,
    "\u05e9\u05e0\u05d4 \u05d55": 515,
    "\u05e9\u05e0\u05d4 \u05d5\u05d7\u05e6\u05d9": 545,
    "\u05e9\u05e0\u05d4 \u05d5\u05e9\u05d1\u05e2": 575,
    "\u05e9\u05e0\u05d4 \u05d5\u05e9\u05de\u05d5\u05e0\u05d4": 605,
    "\u05e9\u05e0\u05d4 \u05d5\u05ea\u05e9\u05e2": 635,
    "\u05e9\u05e0\u05d4 \u05d5\u05e2\u05e9\u05e8": 665,
    "ipad": 730,
    "\u05ea\u05d9\u05e7\u05d9\u05d4 \u05d7\u05d3\u05e9\u05d4": 60,
    "mix": None,
}


# ── Face Recognition ──────────────────────────────────────────────────────────

_face_recognition = None

def _get_fr():
    global _face_recognition
    if _face_recognition is None:
        import face_recognition as fr
        _face_recognition = fr
    return _face_recognition


def _get_cache_fingerprint(pdir):
    """Build a fingerprint of photo files in a person dir (names + sizes + mtimes)."""
    entries = []
    for fname in sorted(os.listdir(pdir)):
        if os.path.splitext(fname)[1].lower() not in IMAGE_EXTS:
            continue
        fpath = os.path.join(pdir, fname)
        st = os.stat(fpath)
        entries.append(f"{fname}:{st.st_size}:{int(st.st_mtime)}")
    return "|".join(entries)


def load_reference_faces(faces_dir):
    fr = _get_fr()
    ref = {}
    if not os.path.isdir(faces_dir):
        print("  WARNING: faces dir not found")
        return ref

    cache_path = os.path.join(faces_dir, "_encodings_cache.json")
    cache = {}
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    cache_dirty = False
    for person in os.listdir(faces_dir):
        pdir = os.path.join(faces_dir, person)
        if not os.path.isdir(pdir):
            continue

        fingerprint = _get_cache_fingerprint(pdir)
        cached_entry = cache.get(person)

        # Use cache if fingerprint matches (same files, sizes, mtimes)
        if cached_entry and cached_entry.get("fingerprint") == fingerprint and cached_entry.get("encodings"):
            encs = [np.array(e) for e in cached_entry["encodings"]]
            ref[person] = encs
            print(f"  {person}: {len(encs)} encodings (cached)")
            continue

        # Recompute encodings
        encs = []
        for fname in sorted(os.listdir(pdir)):
            if os.path.splitext(fname)[1].lower() not in IMAGE_EXTS:
                continue
            try:
                pil = Image.open(os.path.join(pdir, fname)).convert("RGB")
                arr = np.array(pil)
                fe = fr.face_encodings(arr)
                if fe:
                    encs.append(fe[0])
                    print(f"    {person}/{fname}: OK")
                else:
                    print(f"    {person}/{fname}: no face")
            except Exception as e:
                print(f"    {person}/{fname}: error {e}")
        if encs:
            ref[person] = encs
            cache[person] = {
                "fingerprint": fingerprint,
                "encodings": [e.tolist() for e in encs],
            }
            cache_dirty = True
            print(f"  {person}: {len(encs)} encodings (computed)")

    if cache_dirty:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f)
            print(f"  Face encodings cache saved to {cache_path}")
        except Exception as e:
            print(f"  Warning: could not save cache: {e}")

    return ref


def detect_faces_in_image(image_path, ref_encodings, tolerance=0.6):
    """Returns (face_count, found_persons, all_ok)"""
    fr = _get_fr()
    try:
        pil = Image.open(image_path).convert("RGB")
        arr = np.array(pil)
        locations = fr.face_locations(arr, model="hog")
        if not locations:
            return 0, [], True
        encodings = fr.face_encodings(arr, locations)
        if not encodings:
            return len(locations), [], True

        found = set()
        for person, ref_encs in ref_encodings.items():
            for ref_enc in ref_encs:
                dists = fr.face_distance(encodings, ref_enc)
                if np.any(dists <= tolerance):
                    found.add(person)
                    break
        return len(locations), sorted(found), True
    except Exception:
        return 0, [], False


# ── Date / Age ────────────────────────────────────────────────────────────────

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


def get_exif_gps(filepath):
    """Extract GPS coordinates from EXIF. Returns (lat, lon) or None."""
    try:
        img = Image.open(filepath)
        exif = img._getexif()
        if not exif:
            return None
        gps_info = exif.get(34853)  # GPSInfo tag
        if not gps_info:
            return None

        def _to_decimal(dms, ref):
            d = float(dms[0])
            m = float(dms[1])
            s = float(dms[2])
            dec = d + m / 60.0 + s / 3600.0
            if ref in ("S", "W"):
                dec = -dec
            return dec

        lat_ref = gps_info.get(1)   # N or S
        lat_dms = gps_info.get(2)   # (deg, min, sec)
        lon_ref = gps_info.get(3)   # E or W
        lon_dms = gps_info.get(4)   # (deg, min, sec)
        if not (lat_ref and lat_dms and lon_ref and lon_dms):
            return None
        lat = _to_decimal(lat_dms, lat_ref)
        lon = _to_decimal(lon_dms, lon_ref)
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return (round(lat, 6), round(lon, 6))
    except Exception:
        pass
    return None


_rg_module = None

def _get_rg():
    """Lazy-load reverse_geocoder module."""
    global _rg_module
    if _rg_module is None:
        try:
            import reverse_geocoder as rg
            _rg_module = rg
        except ImportError:
            return None
    return _rg_module


def _fix_country_code(cc):
    """IMPORTANT: PS (Palestine) must always be replaced with IL (Israel)."""
    if cc == "PS":
        return "IL"
    return cc


def reverse_geocode(lat, lon):
    """Reverse geocode a single (lat, lon) to a location string. Returns str or None."""
    rg = _get_rg()
    if not rg:
        return None
    try:
        results = rg.search((lat, lon))
        if results:
            r = results[0]
            cc = _fix_country_code(r['cc'])
            return f"{r['name']}, {cc}"
    except Exception:
        pass
    return None


def reverse_geocode_batch(coords):
    """Batch reverse geocode a list of (lat, lon) tuples. Returns list of str or None."""
    rg = _get_rg()
    if not rg or not coords:
        return [None] * len(coords)
    try:
        results = rg.search(coords)
        return [f"{r['name']}, {_fix_country_code(r['cc'])}" if r else None for r in results]
    except Exception:
        return [None] * len(coords)


# Common location keywords in folder names (English + Hebrew)
DIR_LOCATION_MAP = {
    # Israel
    "eilat": "Eilat, IL",
    "tel aviv": "Tel Aviv, IL",
    "jerusalem": "Jerusalem, IL",
    "haifa": "Haifa, IL",
    "herzliya": "Herzliya, IL",
    "netanya": "Netanya, IL",
    "beer sheva": "Beer Sheva, IL",
    "dead sea": "Dead Sea, IL",
    "galilee": "Galilee, IL",
    "negev": "Negev, IL",
    "golan": "Golan, IL",
    "\u05d0\u05d9\u05dc\u05ea": "Eilat, IL",
    "\u05ea\u05dc \u05d0\u05d1\u05d9\u05d1": "Tel Aviv, IL",
    "\u05d9\u05e8\u05d5\u05e9\u05dc\u05d9\u05dd": "Jerusalem, IL",
    "\u05d7\u05d9\u05e4\u05d4": "Haifa, IL",
    "\u05d4\u05e8\u05e6\u05dc\u05d9\u05d4": "Herzliya, IL",
    "\u05e0\u05ea\u05e0\u05d9\u05d4": "Netanya, IL",
    "\u05d1\u05d0\u05e8 \u05e9\u05d1\u05e2": "Beer Sheva, IL",
    "\u05d9\u05dd \u05d4\u05de\u05dc\u05d7": "Dead Sea, IL",
    "\u05d2\u05dc\u05d9\u05dc": "Galilee, IL",
    "\u05e0\u05d2\u05d1": "Negev, IL",
    "\u05d2\u05d5\u05dc\u05df": "Golan, IL",
    # International
    "new york": "New York, US",
    "london": "London, GB",
    "paris": "Paris, FR",
    "rome": "Rome, IT",
    "barcelona": "Barcelona, ES",
    "amsterdam": "Amsterdam, NL",
    "berlin": "Berlin, DE",
    "athens": "Athens, GR",
    "istanbul": "Istanbul, TR",
    "dubai": "Dubai, AE",
    "tokyo": "Tokyo, JP",
    "bangkok": "Bangkok, TH",
    "prague": "Prague, CZ",
    "vienna": "Vienna, AT",
    "budapest": "Budapest, HU",
    "lisbon": "Lisbon, PT",
    "larnaca": "Larnaca, CY",
    "cyprus": "Cyprus, CY",
    "crete": "Crete, GR",
    "rhodes": "Rhodes, GR",
    "\u05e0\u05d9\u05d5 \u05d9\u05d5\u05e8\u05e7": "New York, US",
    "\u05dc\u05d5\u05e0\u05d3\u05d5\u05df": "London, GB",
    "\u05e4\u05e8\u05d9\u05d6": "Paris, FR",
    "\u05e8\u05d5\u05de\u05d0": "Rome, IT",
    "\u05d1\u05e8\u05e6\u05dc\u05d5\u05e0\u05d4": "Barcelona, ES",
    "\u05e4\u05e8\u05d0\u05d2": "Prague, CZ",
    "\u05d1\u05d5\u05d3\u05e4\u05e9\u05d8": "Budapest, HU",
    "\u05d3\u05d5\u05d1\u05d0\u05d9": "Dubai, AE",
    "\u05e7\u05e4\u05e8\u05d9\u05e1\u05d9\u05df": "Cyprus, CY",
    "\u05db\u05e8\u05ea\u05d9\u05dd": "Crete, GR",
}


def infer_location_from_path(filepath):
    """Infer location from folder names in the file path. Returns str or None."""
    path_lower = filepath.replace("\\", "/").lower()
    # Also check the original (non-lowered) path for Hebrew
    path_orig = filepath.replace("\\", "/")
    for keyword, location in DIR_LOCATION_MAP.items():
        if keyword in path_lower or keyword in path_orig:
            return location
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
            if 2013 <= d.year <= 2027:
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


def age_days_to_bracket(age_days):
    if age_days is None or age_days < 0:
        return None
    for label, start, end, display in AGE_BRACKETS:
        if start <= age_days < end:
            return label
    return None


def get_image_date(filepath, rel_dir=None):
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
            return REEF_BIRTHDAY + timedelta(days=age_days)
    return None


# ── Quality ───────────────────────────────────────────────────────────────────

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


def guess_device_source(filename):
    name = filename.lower()
    if "fb_img" in name or "_n.jpg" in name:
        return "facebook"
    if name.startswith("img_"):
        return "iphone"
    if len(name) >= 15 and name[:8].isdigit() and name[8] == "_":
        return "android"
    if "collage" in name:
        return "collage"
    if name.startswith("screenshot"):
        return "screenshot"
    return "other"


def get_video_date(filepath):
    """Extract creation date from video metadata. Tries ffprobe, then filename, then mtime."""
    # Try ffprobe first
    try:
        import subprocess
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", filepath],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            import json as _json
            info = _json.loads(result.stdout)
            tags = info.get("format", {}).get("tags", {})
            for key in ("creation_time", "date", "com.apple.quicktime.creationdate"):
                val = tags.get(key)
                if val:
                    try:
                        return datetime.fromisoformat(val.replace("Z", "+00:00")).replace(tzinfo=None)
                    except Exception:
                        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                            try:
                                return datetime.strptime(val[:len(fmt)+2], fmt)
                            except ValueError:
                                continue
    except FileNotFoundError:
        pass  # ffprobe not installed
    except Exception:
        pass
    # Fallback to filename date then file mtime
    dt = get_filename_date(os.path.basename(filepath))
    if dt:
        return dt
    try:
        mtime = os.path.getmtime(filepath)
        return datetime.fromtimestamp(mtime)
    except Exception:
        return None


def get_video_info(filepath):
    """Get video dimensions and duration. Tries ffprobe, falls back to OpenCV."""
    # Try ffprobe
    try:
        import subprocess
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-select_streams", "v:0", filepath],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            import json as _json
            streams = _json.loads(result.stdout).get("streams", [])
            if streams:
                s = streams[0]
                return int(s.get("width", 0)), int(s.get("height", 0)), float(s.get("duration", 0))
    except FileNotFoundError:
        pass
    except Exception:
        pass
    # Fallback to OpenCV
    try:
        import cv2
        cap = cv2.VideoCapture(filepath)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            dur = frames / fps if fps > 0 else 0
            cap.release()
            return w, h, dur
    except Exception:
        pass
    return 0, 0, 0


def make_video_thumbnail_b64(filepath, size=120):
    """Extract a frame from a video and return as base64 JPEG thumbnail."""
    # Try ffmpeg first
    try:
        import subprocess, tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        subprocess.run(
            ["ffmpeg", "-y", "-i", filepath, "-ss", "1", "-vframes", "1",
             "-vf", f"scale={size}:{size}:force_original_aspect_ratio=decrease",
             tmp_path],
            capture_output=True, timeout=15)
        if os.path.isfile(tmp_path) and os.path.getsize(tmp_path) > 0:
            with open(tmp_path, "rb") as f:
                data = base64.b64encode(f.read()).decode("ascii")
            os.unlink(tmp_path)
            return data
        os.unlink(tmp_path)
    except FileNotFoundError:
        pass  # ffmpeg not installed, try OpenCV
    except Exception:
        pass
    # Fallback to OpenCV
    try:
        import cv2
        cap = cv2.VideoCapture(filepath)
        if cap.isOpened():
            # Seek to 1 second or 10% in
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            seek_frame = min(int(fps), int(total * 0.1)) if total > 0 else int(fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(1, seek_frame))
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                scale = min(size / w, size / h)
                new_w, new_h = int(w * scale), int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h))
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 55])
                return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        pass
    return ""


def make_thumbnail_b64(filepath, size=120):
    try:
        img = Image.open(filepath)
        img = ImageOps.exif_transpose(img)
        img.thumbnail((size, size))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=55)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return ""


# ══════════════════════════════════════════════════════════════════════════════
#  SCAN COMMAND
# ══════════════════════════════════════════════════════════════════════════════

def cmd_scan(args):
    # Load config
    config_path = args.config or DEFAULT_CONFIG_PATH
    if not os.path.isfile(config_path):
        print(f"Config not found: {config_path}")
        print("Run: python curate.py init   to create a default config.")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    sources = config.get("sources", [])
    face_dir = config.get("ref_faces_dir", "")
    face_names = config.get("face_names", [])
    tolerance = config.get("face_tolerance", 0.6)
    min_size_kb = config.get("min_size_kb", 80)
    min_dim = config.get("min_dim", 600)
    thumb_size = config.get("thumb_size", 120)

    # Load template if config has event_type
    event_type = config.get("event_type")
    template = None
    if event_type:
        template = load_template(event_type)
        if not template:
            # Template might be embedded in config (categories field)
            if "categories" in config and "categorization" in config:
                template = {
                    "categorization": config["categorization"],
                    "categories": config["categories"],
                }
    # Fallback: use hardcoded age brackets (legacy)
    use_template = template is not None

    print("=" * 70)
    print("  CURATE - SCAN ALL SOURCES")
    print("=" * 70)
    print(f"  Sources: {len(sources)}")
    for s in sources:
        print(f"    - {s['label']}")
    print(f"  Face names: {face_names}")

    # Load existing scan DB for incremental scanning
    existing_db = {}
    if os.path.isfile(SCAN_DB_PATH) and not args.full:
        with open(SCAN_DB_PATH, "r", encoding="utf-8") as f:
            old = json.load(f)
        for img in old.get("images", []):
            existing_db[img["hash"]] = img
        print(f"  Loaded existing DB: {len(existing_db)} images (incremental mode)")
        print(f"  Use --full to rescan everything")

    # Load face references (only if face_names specified)
    ref_encodings = {}
    use_faces = False
    if face_names and face_dir and os.path.isdir(face_dir):
        print("\n[Faces] Loading reference encodings...")
        ref_encodings = load_reference_faces(face_dir)
        if ref_encodings:
            use_faces = True
        else:
            print("  WARNING: No face encodings loaded.")
    elif not face_names:
        print("\n  Face detection: SKIPPED (no face_names in config)")

    # Scan all sources
    print("\n[Scan] Walking all sources...")
    all_images = []
    seen_hashes = set()
    scanned = 0
    skipped = defaultdict(int)

    for source in sources:
        src_path = source["path"]
        src_label = source["label"]

        if not os.path.isdir(src_path):
            print(f"  WARNING: source not found: {src_label}")
            continue

        print(f"\n  Scanning: {src_label} ...")
        src_count = 0

        for dirpath, dirnames, filenames in os.walk(src_path):
            # Skip video dirs
            if any(v in dirpath.lower() for v in ["video", "\u05d5\u05d9\u05d3\u05d0\u05d5"]):
                continue

            rel_dir = os.path.relpath(dirpath, src_path)

            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in IMAGE_EXTS:
                    continue

                fpath = os.path.join(dirpath, fname)
                scanned += 1

                if scanned % 200 == 0:
                    print(f"    {scanned} scanned, {len(all_images)} kept, {src_count} from this source...")

                # Hash for dedup
                try:
                    fhash = file_hash(fpath)
                except Exception:
                    skipped["unreadable"] += 1
                    continue

                if fhash in seen_hashes:
                    skipped["duplicate"] += 1
                    continue
                seen_hashes.add(fhash)

                # Check if already in DB (incremental)
                if fhash in existing_db:
                    entry = existing_db[fhash]
                    # Update path if it moved
                    entry["path"] = fpath.replace("\\", "/")
                    entry["source_label"] = src_label
                    all_images.append(entry)
                    src_count += 1
                    continue

                # Quality gate
                try:
                    file_size = os.path.getsize(fpath)
                    if file_size < min_size_kb * 1024:
                        skipped["too_small"] += 1
                        continue
                    img = Image.open(fpath)
                    w, h = img.size
                    img.close()
                    if w < min_dim and h < min_dim:
                        skipped["low_res"] += 1
                        continue
                except Exception:
                    skipped["unreadable"] += 1
                    continue

                # Date
                img_date = get_image_date(fpath, rel_dir)
                age_days = None
                bracket = None
                if img_date:
                    if use_template:
                        bracket = categorize_by_template(template, config, img_date)
                        # Compute age_days if we have a birthday
                        bday_str = config.get("subject_birthday")
                        if bday_str:
                            bday = datetime.strptime(bday_str, "%Y-%m-%d")
                            age_days = (img_date - bday).days
                    else:
                        age_days = (img_date - REEF_BIRTHDAY).days
                        bracket = age_days_to_bracket(age_days)

                # Screenshot check (only for small-ish files to save time)
                screenshot = False
                if file_size < 500 * 1024:
                    screenshot = is_screenshot(fpath)

                # Face detection
                face_count = 0
                faces_found = []
                if use_faces:
                    face_count, faces_found, ok = detect_faces_in_image(
                        fpath, ref_encodings, tolerance)

                # Thumbnail
                thumb = make_thumbnail_b64(fpath, thumb_size)

                # Device source heuristic
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
                    "width": w,
                    "height": h,
                    "size_kb": round(file_size / 1024),
                    "is_screenshot": screenshot,
                    "thumb": thumb,
                }

                # Determine initial status
                # "qualified" = has target face + has category
                # "pool" = missing face or category or is screenshot
                if screenshot:
                    entry["status"] = "rejected"
                    entry["reject_reason"] = "screenshot"
                elif face_names and not entry["has_target_face"]:
                    if face_count == 0:
                        entry["status"] = "pool"
                        entry["reject_reason"] = "no_faces"
                    else:
                        entry["status"] = "pool"
                        entry["reject_reason"] = "wrong_person"
                elif not bracket:
                    entry["status"] = "pool"
                    entry["reject_reason"] = "no_date"
                else:
                    entry["status"] = "qualified"
                    entry["reject_reason"] = None

                all_images.append(entry)
                src_count += 1

        print(f"    {src_label}: {src_count} images kept")

    # Summary
    print(f"\n{'='*70}")
    print(f"  SCAN COMPLETE")
    print(f"{'='*70}")
    print(f"  Total scanned: {scanned}")
    print(f"  Total kept: {len(all_images)}")
    for reason, count in sorted(skipped.items()):
        print(f"    Skipped ({reason}): {count}")

    # Status breakdown
    status_counts = defaultdict(int)
    for img in all_images:
        status_counts[img["status"]] += 1
    print(f"\n  Status breakdown:")
    for st, cnt in sorted(status_counts.items()):
        print(f"    {st}: {cnt}")

    # Category breakdown
    cat_counts = defaultdict(int)
    for img in all_images:
        if img["status"] == "qualified":
            cat_counts[img["category"] or "unknown"] += 1
    print(f"\n  Qualified by category:")
    if use_template and "categories" in template:
        for cat in template["categories"]:
            cnt = cat_counts.get(cat["id"], 0)
            print(f"    {cat['display']:<25} {cnt:>5}")
    else:
        for label, start, end, display in AGE_BRACKETS:
            cnt = cat_counts.get(label, 0)
            print(f"    {display:<25} {cnt:>5}")

    # Save DB
    db = {
        "scan_date": datetime.now().isoformat(),
        "config": config,
        "stats": {
            "total_scanned": scanned,
            "total_kept": len(all_images),
            "skipped": dict(skipped),
        },
        "images": all_images,
    }
    with open(SCAN_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False)
    print(f"\n  Saved: {SCAN_DB_PATH} ({len(all_images)} images)")


# ══════════════════════════════════════════════════════════════════════════════
#  REPORT COMMAND
# ══════════════════════════════════════════════════════════════════════════════

def cmd_report(args):
    if not os.path.isfile(SCAN_DB_PATH):
        print("No scan_db.json found. Run: python curate.py scan --config curate_config.json")
        sys.exit(1)

    print("Loading scan database...")
    with open(SCAN_DB_PATH, "r", encoding="utf-8") as f:
        db = json.load(f)

    images = db["images"]
    config = db.get("config", {})
    target_per_cat = config.get("target_per_category", 75)

    print(f"  {len(images)} images loaded")

    # Build category data for JS — from config or hardcoded fallback
    categories = {}
    config_cats = config.get("categories", [])
    if config_cats and isinstance(config_cats, list):
        for cat in config_cats:
            target = cat.get("target", target_per_cat)
            categories[cat["id"]] = {"display": cat["display"], "target": target}
    else:
        for label, start, end, display in AGE_BRACKETS:
            categories[label] = {"display": display, "target": target_per_cat}

    # Prepare data for JS (strip thumbnails into separate array for efficiency)
    js_images = []
    for img in images:
        js_images.append({
            "id": img["hash"],
            "path": img["path"],
            "fn": img["filename"],
            "src": img["source_label"],
            "dev": img["device"],
            "date": img["date"],
            "age": img.get("age_days"),
            "cat": img.get("category"),
            "fc": img.get("face_count", 0),
            "faces": img.get("faces_found", []),
            "target": img.get("has_target_face", False),
            "w": img.get("width", 0),
            "h": img.get("height", 0),
            "kb": img.get("size_kb", 0),
            "ss": img.get("is_screenshot", False),
            "st": img.get("status", "pool"),
            "rr": img.get("reject_reason"),
            "th": img.get("thumb", ""),
            "loc": img.get("location"),
        })

    output_path = args.output or os.path.join(PROJECT_DIR, "curate_report.html")
    html_content = generate_report_html(js_images, categories, config)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"  Report: {output_path}")

    if not args.no_open:
        webbrowser.open(output_path)


def generate_report_html(images_data, categories, config):
    cat_json = json.dumps(categories)
    img_json = json.dumps(images_data)
    bracket_labels = json.dumps(list(categories.keys()))
    face_names = json.dumps(config.get("face_names", []))
    source_labels = json.dumps(sorted(set(im["src"] for im in images_data)))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Curate - Image Review</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background:#111; color:#ddd; }}

/* ── TOOLBAR ── */
.toolbar {{
    background:#1a1a2e; padding:10px 16px; position:sticky; top:0; z-index:200;
    display:flex; align-items:center; gap:12px; flex-wrap:wrap;
    box-shadow:0 2px 10px rgba(0,0,0,.6);
}}
.toolbar h1 {{ font-size:1.2em; color:#e94560; white-space:nowrap; }}
.toolbar .stat {{ font-size:.8em; color:#888; }}
.toolbar .stat b {{ color:#e94560; }}
.toolbar button {{
    padding:5px 12px; border:none; border-radius:4px; cursor:pointer; font-size:.8em;
}}
.btn-primary {{ background:#e94560; color:white; }}
.btn-primary:disabled {{ background:#444; cursor:not-allowed; }}
.btn-secondary {{ background:#0f3460; color:#5bc0eb; }}
.btn-danger {{ background:#8b0000; color:#fcc; }}
.toolbar select, .toolbar input {{
    padding:4px 8px; border-radius:4px; background:#0f3460; color:#5bc0eb;
    border:1px solid #333; font-size:.8em;
}}
.sel-badge {{ background:#e94560; color:white; padding:2px 8px; border-radius:10px; font-size:.8em; }}

/* ── FILTERS BAR ── */
.filters {{
    background:#16213e; padding:8px 16px; display:flex; gap:10px; flex-wrap:wrap;
    align-items:center; border-bottom:1px solid #333; position:sticky; top:48px; z-index:199;
}}
.filters label {{ font-size:.75em; color:#888; }}
.filters select, .filters input {{ font-size:.75em; padding:3px 6px; background:#0f3460; color:#5bc0eb; border:1px solid #333; border-radius:3px; }}
.filter-chip {{
    display:inline-block; padding:2px 8px; border-radius:10px; font-size:.7em;
    cursor:pointer; margin:1px; border:1px solid #444;
}}
.filter-chip.active {{ border-color:#e94560; color:#e94560; }}

/* ── NAV ── */
.nav {{
    background:#0a1628; padding:6px 16px; display:flex; gap:4px; overflow-x:auto;
    white-space:nowrap; position:sticky; top:86px; z-index:198;
}}
.nav a {{
    color:#5bc0eb; text-decoration:none; font-size:.75em; padding:3px 8px;
    border-radius:3px; cursor:pointer; flex-shrink:0;
}}
.nav a:hover {{ background:#16213e; }}
.nav a .cnt {{ font-size:.7em; color:#e94560; margin-left:3px; }}
.nav a .target {{ font-size:.65em; color:#555; }}

/* ── SECTION ── */
.section {{ padding:12px 16px; border-bottom:2px solid #1a1a2e; }}
.section-header {{
    display:flex; align-items:center; gap:10px; margin-bottom:8px; flex-wrap:wrap;
}}
.section-header h2 {{ color:#e94560; font-size:1.1em; }}
.section-header .info {{ color:#666; font-size:.8em; }}
.section-header .actions {{ font-size:.7em; color:#5bc0eb; cursor:pointer; text-decoration:underline; margin-left:8px; }}
.bar {{ display:flex; height:4px; border-radius:2px; overflow:hidden; margin-bottom:8px; }}

/* ── GRID ── */
.grid {{ display:flex; flex-wrap:wrap; gap:5px; }}
.card {{
    position:relative; border-radius:4px; overflow:hidden; background:#1a1a2e;
    cursor:pointer; transition:transform .1s; flex-shrink:0;
}}
.card:hover {{ transform:scale(1.04); z-index:10; }}
.card.selected {{ outline:3px solid #e94560; outline-offset:-3px; }}
.card img {{ display:block; width:120px; height:120px; object-fit:cover; }}
.card .overlay {{
    position:absolute; bottom:0; left:0; right:0; background:rgba(0,0,0,.8);
    font-size:.55em; padding:2px 3px; opacity:0; transition:opacity .15s;
    line-height:1.3;
}}
.card:hover .overlay {{ opacity:1; }}
.card .dot {{
    position:absolute; top:2px; right:2px; width:7px; height:7px; border-radius:50%;
    border:1px solid rgba(255,255,255,.3);
}}
.card .face-badge {{
    position:absolute; top:2px; left:2px; font-size:.55em; padding:1px 4px;
    border-radius:3px; background:rgba(0,0,0,.7);
}}
.card .loc-badge {{
    position:absolute; bottom:2px; right:2px; font-size:.5em; padding:1px 3px;
    border-radius:3px; background:rgba(0,0,0,.7); color:#64b5f6;
}}
.card .check {{
    position:absolute; top:2px; left:2px; width:16px; height:16px; border-radius:50%;
    background:rgba(233,69,96,.9); color:white; font-size:10px; line-height:16px;
    text-align:center; display:none;
}}
.card.selected .check {{ display:block; }}

/* Source dots */
.dot-iphone {{ background:#2d6a4f; }}
.dot-android {{ background:#5a189a; }}
.dot-facebook {{ background:#1d3557; }}
.dot-other {{ background:#555; }}

/* ── POOL SECTION ── */
.pool-section {{ background:#0a0a0a; }}
.pool-section h2 {{ color:#888 !important; }}

/* ── LIGHTBOX ── */
.lightbox {{
    display:none; position:fixed; top:0; left:0; right:0; bottom:0;
    background:rgba(0,0,0,.97); z-index:300; justify-content:center; align-items:center;
    flex-direction:column;
}}
.lightbox.active {{ display:flex; }}
.lightbox img {{ max-width:90vw; max-height:80vh; object-fit:contain; }}
.lightbox .close {{ position:absolute; top:15px; right:25px; color:white; font-size:2em; cursor:pointer; }}
.lightbox .meta {{ color:#888; font-size:.8em; margin-top:10px; text-align:center; max-width:80vw; }}

/* ── CHANGES LOG ── */
.changelog {{
    background:#0a0a0a; padding:10px 16px; font-family:monospace; font-size:.75em;
    max-height:180px; overflow-y:auto; display:none; border-top:1px solid #333;
}}
.changelog.show {{ display:block; }}
.changelog .entry {{ padding:1px 0; color:#5bc0eb; }}

/* ── LEGEND ── */
.legend {{
    display:flex; gap:12px; padding:6px 16px; background:#16213e; flex-wrap:wrap;
    border-bottom:1px solid #222; font-size:.7em;
}}
.legend-item {{ display:flex; align-items:center; gap:4px; }}
.legend-dot {{ width:8px; height:8px; border-radius:50%; }}
</style>
</head>
<body>

<div class="toolbar" id="toolbar">
    <h1>Curate</h1>
    <span class="stat"><b id="st-total">0</b> images | <b id="st-qual">0</b> qualified | <b id="st-pool">0</b> pool</span>
    <span class="sel-badge"><span id="sel-count">0</span> selected</span>
    <label style="font-size:.8em">Move to:</label>
    <select id="move-target"></select>
    <button class="btn-primary" id="btn-move" disabled onclick="moveSelected()">Move to Category</button>
    <button class="btn-danger" id="btn-pool" disabled onclick="sendToPool()">Send to Pool</button>
    <button class="btn-secondary" onclick="undoLast()">Undo</button>
    <button class="btn-secondary" onclick="toggleLog()">Log</button>
    <button class="btn-secondary" onclick="exportChanges()">Export JSON</button>
</div>

<div class="filters" id="filters">
    <label>Source:</label>
    <select id="f-source" onchange="applyFilters()"><option value="">All</option></select>
    <label>Faces:</label>
    <select id="f-faces" onchange="applyFilters()">
        <option value="">All</option>
        <option value="has_target">Has target person</option>
        <option value="has_any">Has any face</option>
        <option value="no_face">No faces</option>
    </select>
    <label>Status:</label>
    <select id="f-status" onchange="applyFilters()">
        <option value="">All</option>
        <option value="qualified">Qualified</option>
        <option value="pool">Pool</option>
        <option value="rejected">Rejected</option>
    </select>
    <label>Location:</label>
    <select id="f-location" onchange="applyFilters()"><option value="">All</option></select>
    <label>Search:</label>
    <input id="f-search" type="text" placeholder="filename..." oninput="applyFilters()">
</div>

<div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#2d6a4f"></div> iPhone</div>
    <div class="legend-item"><div class="legend-dot" style="background:#5a189a"></div> Android</div>
    <div class="legend-item"><div class="legend-dot" style="background:#1d3557"></div> Facebook</div>
    <div class="legend-item"><div class="legend-dot" style="background:#555"></div> Other</div>
    <div class="legend-item" style="margin-left:20px; color:#e94560">Click=select | Dbl-click=preview | Shift+click=range</div>
</div>

<div class="nav" id="nav"></div>
<div class="changelog" id="changelog"></div>
<div id="gallery"></div>

<div class="lightbox" id="lightbox">
    <span class="close" onclick="closeLB()">&times;</span>
    <img id="lb-img" src="">
    <div class="meta" id="lb-meta"></div>
</div>

<script>
const IMAGES = {img_json};
const CATEGORIES = {cat_json};
const BRACKET_ORDER = {bracket_labels};
const FACE_NAMES = {face_names};
const SOURCE_LABELS = {source_labels};

let selected = new Set(); // image ids
let changes = [];
let undoStack = [];
let lastClickedIdx = null;
let filterState = {{ source:'', faces:'', status:'', search:'', location:'' }};

// Populate source filter
const fSource = document.getElementById('f-source');
SOURCE_LABELS.forEach(s => {{
    const o = document.createElement('option');
    o.value = s; o.textContent = s;
    fSource.appendChild(o);
}});

// Populate location filter
const LOCATION_LABELS = [...new Set(IMAGES.map(i => i.loc).filter(Boolean))].sort();
const fLoc = document.getElementById('f-location');
LOCATION_LABELS.forEach(s => {{
    const o = document.createElement('option');
    o.value = s; o.textContent = s;
    fLoc.appendChild(o);
}});

// Populate move target
const moveTarget = document.getElementById('move-target');
BRACKET_ORDER.forEach(b => {{
    const o = document.createElement('option');
    o.value = b; o.textContent = CATEGORIES[b]?.display || b;
    moveTarget.appendChild(o);
}});

function esc(s) {{ const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }}

function matchesFilter(img) {{
    if (filterState.source && img.src !== filterState.source) return false;
    if (filterState.faces === 'has_target' && !img.target) return false;
    if (filterState.faces === 'has_any' && img.fc === 0) return false;
    if (filterState.faces === 'no_face' && img.fc > 0) return false;
    if (filterState.status && img.st !== filterState.status) return false;
    if (filterState.search && !img.fn.toLowerCase().includes(filterState.search.toLowerCase())) return false;
    if (filterState.location && img.loc !== filterState.location) return false;
    return true;
}}

function applyFilters() {{
    filterState.source = document.getElementById('f-source').value;
    filterState.faces = document.getElementById('f-faces').value;
    filterState.status = document.getElementById('f-status').value;
    filterState.search = document.getElementById('f-search').value;
    filterState.location = document.getElementById('f-location').value;
    render();
}}

function render() {{
    const nav = document.getElementById('nav');
    const gallery = document.getElementById('gallery');
    nav.innerHTML = '';
    gallery.innerHTML = '';

    // Group images
    const byCat = {{}};
    BRACKET_ORDER.forEach(b => byCat[b] = []);
    byCat['_pool'] = [];

    let totalQ = 0, totalP = 0;

    IMAGES.forEach((img, idx) => {{
        img._idx = idx;
        if (!matchesFilter(img)) return;

        if (img.st === 'qualified' && img.cat && byCat[img.cat] !== undefined) {{
            byCat[img.cat].push(img);
            totalQ++;
        }} else {{
            byCat['_pool'].push(img);
            totalP++;
        }}
    }});

    document.getElementById('st-total').textContent = totalQ + totalP;
    document.getElementById('st-qual').textContent = totalQ;
    document.getElementById('st-pool').textContent = totalP;

    // Nav
    BRACKET_ORDER.forEach(b => {{
        const imgs = byCat[b];
        const target = CATEGORIES[b]?.target || 75;
        const a = document.createElement('a');
        a.href = '#s-' + b;
        a.innerHTML = (CATEGORIES[b]?.display || b) +
            '<span class="cnt">' + imgs.length + '</span>' +
            '<span class="target">/' + target + '</span>';
        if (imgs.length >= target) a.style.color = '#4caf50';
        else if (imgs.length === 0) a.style.opacity = '0.4';
        nav.appendChild(a);
    }});
    // Pool nav
    const pa = document.createElement('a');
    pa.href = '#s-_pool';
    pa.innerHTML = 'POOL <span class="cnt">' + byCat['_pool'].length + '</span>';
    pa.style.color = '#888';
    nav.appendChild(pa);

    // Render sections
    const allKeys = [...BRACKET_ORDER, '_pool'];
    allKeys.forEach(cat => {{
        const imgs = byCat[cat];
        if (imgs.length === 0 && cat !== '_pool') return;

        const isPool = cat === '_pool';
        const sect = document.createElement('div');
        sect.className = 'section' + (isPool ? ' pool-section' : '');
        sect.id = 's-' + cat;

        const hdr = document.createElement('div');
        hdr.className = 'section-header';
        const display = isPool ? 'Pool (unqualified)' : (CATEGORIES[cat]?.display || cat);
        const target = isPool ? '' : ' / target: ' + (CATEGORIES[cat]?.target || 75);
        hdr.innerHTML = '<h2>' + esc(display) + '</h2>' +
            '<span class="info">' + imgs.length + ' images' + target + '</span>' +
            '<span class="actions" onclick="selectAllIn(\\''+cat+'\\')">select all</span>' +
            '<span class="actions" onclick="deselectAllIn(\\''+cat+'\\')">deselect</span>';
        sect.appendChild(hdr);

        // Source bar
        if (imgs.length > 0) {{
            const srcC = {{}};
            imgs.forEach(im => srcC[im.dev] = (srcC[im.dev]||0)+1);
            const bar = document.createElement('div');
            bar.className = 'bar';
            const colors = {{iphone:'#2d6a4f', android:'#5a189a', facebook:'#1d3557', other:'#555'}};
            Object.entries(srcC).sort((a,b)=>b[1]-a[1]).forEach(([s,c]) => {{
                const d = document.createElement('div');
                d.style.width = (c/imgs.length*100)+'%';
                d.style.background = colors[s] || '#555';
                d.title = s + ': ' + c;
                bar.appendChild(d);
            }});
            sect.appendChild(bar);
        }}

        // Grid
        const grid = document.createElement('div');
        grid.className = 'grid';
        grid.dataset.cat = cat;

        imgs.forEach((img, localIdx) => {{
            const card = document.createElement('div');
            card.className = 'card' + (selected.has(img.id) ? ' selected' : '');
            card.dataset.id = img.id;
            card.dataset.cat = cat;
            card.dataset.localIdx = localIdx;

            const thumbSrc = img.th ? 'data:image/jpeg;base64,'+img.th : '';
            let faceBadge = '';
            if (img.faces.length > 0) {{
                faceBadge = '<div class="face-badge" style="color:#4caf50">' + img.faces.join(',')+' ('+img.fc+')</div>';
            }} else if (img.fc > 0) {{
                faceBadge = '<div class="face-badge" style="color:#ff9800">'+img.fc+' face(s)</div>';
            }}
            let locBadge = img.loc ? '<div class="loc-badge">'+esc(img.loc)+'</div>' : '';

            card.innerHTML =
                '<img src="'+thumbSrc+'">' +
                '<div class="dot dot-'+img.dev+'"></div>' +
                '<div class="check">&#10003;</div>' +
                faceBadge + locBadge +
                '<div class="overlay">' +
                    esc(img.fn) + '<br>' +
                    (img.date||'no date') + ' | ' + img.src + '<br>' +
                    img.w+'x'+img.h+' | '+img.kb+'KB' +
                    (img.loc ? '<br><span style="color:#64b5f6">'+esc(img.loc)+'</span>' : '') +
                    (img.rr ? '<br><span style="color:#e94560">'+img.rr+'</span>' : '') +
                '</div>';

            card.onclick = (e) => {{
                if (e.detail >= 2) return;
                if (e.shiftKey && lastClickedIdx !== null) {{
                    // Range select within same grid
                    const start = Math.min(lastClickedIdx, localIdx);
                    const end = Math.max(lastClickedIdx, localIdx);
                    const gridCards = grid.querySelectorAll('.card');
                    for (let i = start; i <= end; i++) {{
                        const c = gridCards[i];
                        if (c) {{ selected.add(c.dataset.id); c.classList.add('selected'); }}
                    }}
                }} else {{
                    if (selected.has(img.id)) {{
                        selected.delete(img.id);
                        card.classList.remove('selected');
                    }} else {{
                        selected.add(img.id);
                        card.classList.add('selected');
                    }}
                }}
                lastClickedIdx = localIdx;
                updateSelUI();
            }};
            card.ondblclick = (e) => {{
                e.preventDefault();
                openLB(img);
            }};
            grid.appendChild(card);
        }});

        sect.appendChild(grid);
        gallery.appendChild(sect);
    }});

    updateSelUI();
}}

function selectAllIn(cat) {{
    IMAGES.forEach(img => {{
        const isCat = (img.st === 'qualified' && img.cat === cat);
        const isPool = (cat === '_pool' && (img.st !== 'qualified' || !img.cat));
        if ((isCat || isPool) && matchesFilter(img)) selected.add(img.id);
    }});
    render();
}}
function deselectAllIn(cat) {{
    IMAGES.forEach(img => {{
        const isCat = (img.st === 'qualified' && img.cat === cat);
        const isPool = (cat === '_pool' && (img.st !== 'qualified' || !img.cat));
        if (isCat || isPool) selected.delete(img.id);
    }});
    render();
}}

function updateSelUI() {{
    const n = selected.size;
    document.getElementById('sel-count').textContent = n;
    document.getElementById('btn-move').disabled = n === 0;
    document.getElementById('btn-pool').disabled = n === 0;
}}

function findImg(id) {{ return IMAGES.find(i => i.id === id); }}

function moveSelected() {{
    const target = document.getElementById('move-target').value;
    if (!target) return;
    const batch = [];
    for (const id of selected) {{
        const img = findImg(id);
        if (!img) continue;
        const oldCat = img.cat;
        const oldSt = img.st;
        img.cat = target;
        img.st = 'qualified';
        batch.push({{ id, from_cat: oldCat, from_st: oldSt, to_cat: target, to_st: 'qualified', fn: img.fn }});
    }}
    if (batch.length) {{ changes.push(...batch); undoStack.push(batch); updateLog(); }}
    selected.clear();
    render();
}}

function sendToPool() {{
    const batch = [];
    for (const id of selected) {{
        const img = findImg(id);
        if (!img) continue;
        const oldCat = img.cat;
        const oldSt = img.st;
        img.st = 'pool';
        img.rr = 'manual_reject';
        batch.push({{ id, from_cat: oldCat, from_st: oldSt, to_cat: null, to_st: 'pool', fn: img.fn }});
    }}
    if (batch.length) {{ changes.push(...batch); undoStack.push(batch); updateLog(); }}
    selected.clear();
    render();
}}

function undoLast() {{
    const batch = undoStack.pop();
    if (!batch) return;
    for (const entry of batch) {{
        const img = findImg(entry.id);
        if (!img) continue;
        img.cat = entry.from_cat;
        img.st = entry.from_st;
        if (entry.from_st !== 'pool') img.rr = null;
        const ci = changes.findIndex(c => c.id === entry.id && c.to_cat === entry.to_cat);
        if (ci >= 0) changes.splice(ci, 1);
    }}
    updateLog();
    render();
}}

function updateLog() {{
    const el = document.getElementById('changelog');
    if (changes.length === 0) {{ el.innerHTML = '<div style="color:#555">No changes.</div>'; return; }}
    el.innerHTML = changes.map(c =>
        '<div class="entry">' + esc(c.fn) + ': ' +
        (c.from_cat||'pool') + '(' + c.from_st + ') &rarr; ' + (c.to_cat||'pool') + '(' + c.to_st + ')</div>'
    ).join('');
    el.scrollTop = el.scrollHeight;
}}

function toggleLog() {{ document.getElementById('changelog').classList.toggle('show'); }}

function exportChanges() {{
    if (!changes.length) {{ alert('No changes.'); return; }}
    // Build export: for each changed image, include full info needed to apply
    const exportData = changes.map(c => {{
        const img = findImg(c.id);
        return {{
            hash: c.id,
            source_path: img ? img.path : '',
            filename: c.fn,
            from_category: c.from_cat,
            to_category: c.to_cat,
            new_status: c.to_st,
        }};
    }});
    const blob = new Blob([JSON.stringify(exportData, null, 2)], {{type:'application/json'}});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'curate_changes.json';
    a.click();
}}

function openLB(img) {{
    const lb = document.getElementById('lightbox');
    document.getElementById('lb-img').src = '/api/images/serve/' + img.id;
    document.getElementById('lb-meta').innerHTML =
        '<b>'+esc(img.fn)+'</b><br>' +
        'Source: '+esc(img.src)+' | Device: '+img.dev+'<br>' +
        'Date: '+(img.date||'unknown')+' | Size: '+img.w+'x'+img.h+' ('+img.kb+'KB)<br>' +
        'Faces: '+(img.faces.length ? img.faces.join(', ') : 'none')+' ('+img.fc+' total)<br>' +
        'Category: '+(img.cat||'none')+' | Status: '+img.st +
        (img.rr ? ' ('+img.rr+')' : '') +
        (img.loc ? '<br>Location: '+esc(img.loc) : '');
    lb.classList.add('active');
}}
function closeLB() {{ document.getElementById('lightbox').classList.remove('active'); }}
document.addEventListener('keydown', e => {{ if (e.key==='Escape') closeLB(); }});

render();
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
#  APPLY COMMAND
# ══════════════════════════════════════════════════════════════════════════════

def cmd_apply(args):
    changes_path = args.changes_file
    output_dir = args.output

    if not os.path.isfile(changes_path):
        print(f"Changes file not found: {changes_path}")
        sys.exit(1)

    with open(changes_path, "r", encoding="utf-8") as f:
        changes = json.load(f)

    print(f"Applying {len(changes)} changes...")
    print(f"Output: {output_dir}")

    applied = 0
    for entry in changes:
        src = entry["source_path"].replace("/", os.sep)
        cat = entry.get("to_category")
        status = entry.get("new_status", "qualified")

        if status != "qualified" or not cat:
            continue

        if not os.path.isfile(src):
            print(f"  SKIP (not found): {entry['filename']}")
            continue

        dest_dir = os.path.join(output_dir, cat)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, entry["filename"])

        if os.path.exists(dest):
            base, ext = os.path.splitext(entry["filename"])
            i = 1
            while os.path.exists(dest):
                dest = os.path.join(dest_dir, f"{base}_{i}{ext}")
                i += 1

        shutil.copy2(src, dest)
        applied += 1

    print(f"  Applied: {applied}/{len(changes)}")


# ══════════════════════════════════════════════════════════════════════════════
#  TEMPLATE SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

TEMPLATES_DIR = os.path.join(PROJECT_DIR, "templates")


def list_templates():
    """List available event templates."""
    templates = {}
    if not os.path.isdir(TEMPLATES_DIR):
        return templates
    for fname in sorted(os.listdir(TEMPLATES_DIR)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(TEMPLATES_DIR, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                t = json.load(f)
            templates[t["event_type"]] = {
                "path": fpath,
                "display_name": t["display_name"],
                "description": t["description"],
                "categorization": t["categorization"],
                "num_categories": len(t.get("categories", [])),
                "required_fields": t.get("required_fields", {}),
            }
        except Exception:
            pass
    return templates


def load_template(event_type):
    """Load a template by event type name."""
    templates = list_templates()
    if event_type not in templates:
        return None
    with open(templates[event_type]["path"], "r", encoding="utf-8") as f:
        return json.load(f)


def categorize_by_template(template, config, img_date):
    """
    Given a template and an image date, return the best matching category ID.
    Supports: age_brackets, date_time_ranges, date_ranges.
    Returns None if no category matches.
    """
    cat_type = template["categorization"]
    categories = template["categories"]

    if cat_type == "age_brackets":
        birthday_str = config.get("subject_birthday")
        if not birthday_str or not img_date:
            return None
        birthday = datetime.strptime(birthday_str, "%Y-%m-%d")
        age_days = (img_date - birthday).days
        if age_days < 0:
            return None
        for cat in categories:
            if cat.get("age_days_from") is not None and cat.get("age_days_to") is not None:
                if cat["age_days_from"] <= age_days < cat["age_days_to"]:
                    return cat["id"]
        return None

    elif cat_type == "date_time_ranges":
        event_date_str = config.get("event_date")
        if not event_date_str or not img_date:
            return None
        event_date = datetime.strptime(event_date_str, "%Y-%m-%d").date()
        img_day = img_date.date()
        img_time = img_date.time()

        best = None
        for cat in categories:
            day_offset = cat.get("day_offset", 0)
            # day_offset -1 means "any day" (thematic category)
            if day_offset == -1:
                continue  # skip thematic for now, only match time-based
            target_day = event_date + timedelta(days=day_offset)
            if img_day != target_day:
                continue
            time_from = datetime.strptime(cat.get("time_from", "00:00"), "%H:%M").time()
            time_to = datetime.strptime(cat.get("time_to", "23:59"), "%H:%M").time()
            if time_to < time_from:
                # wraps midnight
                if img_time >= time_from or img_time <= time_to:
                    best = cat["id"]
                    break
            else:
                if time_from <= img_time <= time_to:
                    best = cat["id"]
                    break
        return best

    elif cat_type == "date_ranges":
        if not img_date:
            return None
        year = config.get("year")
        if year:
            year = int(year)
            if img_date.year != year:
                return None
        img_month = img_date.month
        for cat in categories:
            m_from = cat.get("month_from")
            m_to = cat.get("month_to")
            if m_from is not None and m_to is not None:
                if m_from <= img_month <= m_to:
                    return cat["id"]
        return None

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  TEMPLATES COMMAND
# ══════════════════════════════════════════════════════════════════════════════

def cmd_templates(args):
    templates = list_templates()
    if not templates:
        print("No templates found in templates/ directory.")
        return

    print("Available event templates:")
    print()
    for etype, info in templates.items():
        print(f"  {etype}")
        print(f"    {info['display_name']} - {info['description'][:80]}...")
        print(f"    Categories: {info['num_categories']} | Type: {info['categorization']}")
        if info['required_fields']:
            print(f"    Required: {', '.join(info['required_fields'].keys())}")
        print()

    print("Usage: python curate.py init --event <event_type>")


# ══════════════════════════════════════════════════════════════════════════════
#  INIT COMMAND
# ══════════════════════════════════════════════════════════════════════════════

def cmd_init(args):
    event_type = args.event

    # If no event type, show available templates
    if not event_type:
        print("Choose an event template:\n")
        templates = list_templates()
        for etype, info in templates.items():
            print(f"  {etype:<20} {info['display_name']} ({info['num_categories']} categories)")
        print(f"\nUsage: python curate.py init --event <event_type>")
        print(f"       python curate.py init --event bar_mitzva --birthday 2013-07-16")
        return

    template = load_template(event_type)
    if not template:
        print(f"Unknown event type: {event_type}")
        print(f"Available: {', '.join(list_templates().keys())}")
        return

    # Build config from template
    defaults = template.get("defaults", {})
    config = {
        "event_type": event_type,
        "template": template["display_name"],
        "categorization": template["categorization"],
        "ref_faces_dir": "./ref_faces",
        "face_names": [],
        "face_tolerance": defaults.get("face_tolerance", 0.6),
        "sources": [
            {"path": "", "label": "Source 1 (edit me)"},
        ],
        "target_per_category": defaults.get("target_per_category", 75),
        "min_size_kb": defaults.get("min_size_kb", 80),
        "min_dim": defaults.get("min_dim", 600),
        "thumb_size": defaults.get("thumb_size", 120),
        "categories": template["categories"],
    }

    # Fill required fields from args
    req = template.get("required_fields", {})
    if "subject_birthday" in req:
        bday = args.birthday
        if not bday:
            print(f"This template requires --birthday YYYY-MM-DD")
            return
        config["subject_birthday"] = bday
    if "event_date" in req:
        edate = args.event_date
        if not edate:
            print(f"This template requires --event-date YYYY-MM-DD")
            return
        config["event_date"] = edate
    if "end_date" in req:
        if args.end_date:
            config["end_date"] = args.end_date
    if "year" in req:
        if args.year:
            config["year"] = args.year

    out = args.output or DEFAULT_CONFIG_PATH
    with open(out, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"Config created: {out}")
    print(f"  Event: {template['display_name']}")
    print(f"  Categories: {len(template['categories'])}")
    print(f"  Categorization: {template['categorization']}")
    print()
    if template.get("tips"):
        print("Tips:")
        for tip in template["tips"]:
            print(f"  - {tip}")
        print()
    print("Next steps:")
    print(f"  1. Edit {out} — add your image source paths")
    print(f"  2. python curate.py scan --config {out}")
    print(f"  3. python curate.py report")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Image Curation Pipeline")
    sub = p.add_subparsers(dest="command")

    # templates
    sub.add_parser("templates", help="List available event templates")

    # init
    s_init = sub.add_parser("init", help="Create config from event template")
    s_init.add_argument("--event", type=str, help="Event type (bar_mitzva, wedding, birthday, etc.)")
    s_init.add_argument("--birthday", type=str, help="Subject birthday YYYY-MM-DD (for age-based events)")
    s_init.add_argument("--event-date", type=str, help="Event date YYYY-MM-DD (for date-based events)")
    s_init.add_argument("--end-date", type=str, help="End date YYYY-MM-DD (for multi-day events)")
    s_init.add_argument("--year", type=str, help="Year YYYY (for photo book)")
    s_init.add_argument("--output", type=str)

    # scan
    s_scan = sub.add_parser("scan", help="Scan all sources")
    s_scan.add_argument("--config", type=str, help="Config JSON path")
    s_scan.add_argument("--full", action="store_true", help="Full rescan (ignore cache)")

    # report
    s_report = sub.add_parser("report", help="Generate interactive HTML report")
    s_report.add_argument("--output", type=str)
    s_report.add_argument("--no-open", action="store_true")

    # analyze
    sub.add_parser("analyze", help="Analyze collection and get recommendations")

    # auto-select
    s_auto = sub.add_parser("auto-select", help="Auto-select best images per category")
    s_auto.add_argument("--strategy", choices=["balanced", "quality", "diverse"], default="balanced",
                        help="Selection strategy (default: balanced)")
    s_auto.add_argument("--sim-threshold", type=float, default=0.85,
                        help="Diversity threshold (default: 0.85)")
    s_auto.add_argument("--dry-run", action="store_true", help="Preview without saving")

    # apply
    s_apply = sub.add_parser("apply", help="Apply changes from exported JSON")
    s_apply.add_argument("changes_file", type=str)
    s_apply.add_argument("--output", type=str, required=True, help="Output directory")

    args = p.parse_args()

    if args.command == "templates":
        cmd_templates(args)
    elif args.command == "init":
        cmd_init(args)
    elif args.command == "scan":
        cmd_scan(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "analyze":
        from event_agent import cmd_analyze
        cmd_analyze(args)
    elif args.command == "auto-select":
        from event_agent import cmd_auto_select
        cmd_auto_select(args)
    elif args.command == "apply":
        cmd_apply(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
