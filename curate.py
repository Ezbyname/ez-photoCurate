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
        # Guard: old cache format stored raw lists, new format is {"fingerprint":..., "encodings":...}
        if not isinstance(cached_entry, dict):
            # Old format or corrupt — skip cache, recompute
            if cached_entry is not None:
                print(f"  {person}: cache entry is {type(cached_entry).__name__}, recomputing...")
                cache_dirty = True
        elif cached_entry.get("fingerprint") == fingerprint and cached_entry.get("encodings"):
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
    """Lazy-load reverse_geocoder module (single-process mode for thread safety)."""
    global _rg_module
    if _rg_module is None:
        try:
            import reverse_geocoder as rg
            # Pre-load the KD-tree in single-process mode to avoid
            # multiprocessing deadlocks when called from Flask threads
            rg.search([(0, 0)], mode=2)
            _rg_module = rg
        except ImportError:
            return None
    return _rg_module


def _fix_country_code(cc):
    """IMPORTANT: PS, SY, LB in GPS geocoding must always be replaced with IL (Israel).
    The GeoNames database incorrectly labels Israeli locations (Golan Heights,
    northern border, Jerusalem/Old City) with these country codes."""
    if cc in ("PS", "SY", "LB"):
        return "IL"
    return cc


def reverse_geocode(lat, lon):
    """Reverse geocode a single (lat, lon) to a location string. Returns str or None."""
    rg = _get_rg()
    if not rg:
        return None
    try:
        results = rg.search((lat, lon), mode=2)
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
        results = rg.search(coords, mode=2)
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


def extract_video_frames(filepath, n_frames=5):
    """
    Extract evenly-spaced frames from a video as PIL RGB images.
    Returns list of (frame_pil, timestamp_sec) tuples.
    Tries OpenCV (works on all platforms without ffmpeg).
    """
    frames = []
    try:
        import cv2
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return frames
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames < 1:
            cap.release()
            return frames

        # Pick frame positions: skip first/last 5%, sample evenly in between
        start = max(1, int(total_frames * 0.05))
        end = max(start + 1, int(total_frames * 0.95))
        step = max(1, (end - start) // n_frames)
        positions = list(range(start, end, step))[:n_frames]

        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if ret and frame is not None:
                # Convert BGR (OpenCV) → RGB (PIL)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                timestamp = round(pos / fps, 2) if fps > 0 else 0
                frames.append((pil_img, timestamp))
        cap.release()
    except Exception:
        pass
    return frames


def analyze_video_frames(filepath, ref_encodings=None, face_names=None,
                         tolerance=0.6, face_match_mode="any", n_frames=5):
    """
    Analyze a video by extracting key frames and running image analysis on them.

    Returns dict with:
        face_count, faces_found, has_target_face, face_distance,
        photo_grade, image_vector, dhash, best_thumb (base64)
    Returns None on failure.
    """
    frames = extract_video_frames(filepath, n_frames=n_frames)
    if not frames:
        return None

    fr = None
    best_face_count = 0
    all_found_persons = set()
    best_face_distance = None
    best_grade_frame = None
    best_grade_composite = -1
    best_sharpness_frame = None
    best_sharpness = -1

    # ── Analyze each frame ──
    for pil_img, ts in frames:
        w, h = pil_img.size

        # Face detection (if reference encodings provided)
        if ref_encodings:
            if fr is None:
                fr = _get_fr()
            try:
                # Resize for speed (face_recognition works fine at 800px)
                detect_img = pil_img.copy()
                max_dim = 800
                if w > max_dim or h > max_dim:
                    detect_img.thumbnail((max_dim, max_dim), Image.LANCZOS)
                arr = np.array(detect_img)
                locations = fr.face_locations(arr, model="hog")
                if locations:
                    fc = len(locations)
                    if fc > best_face_count:
                        best_face_count = fc
                    encodings = fr.face_encodings(arr, locations)
                    if encodings:
                        for person, ref_encs in ref_encodings.items():
                            for ref_enc in ref_encs:
                                dists = fr.face_distance(encodings, ref_enc)
                                min_d = float(np.min(dists))
                                if best_face_distance is None or min_d < best_face_distance:
                                    best_face_distance = min_d
                                if min_d <= tolerance:
                                    all_found_persons.add(person)
            except Exception:
                pass

        # Sharpness check (pick sharpest frame for vector/grade)
        try:
            import cv2 as _cv2
            gray = np.array(pil_img.convert("L"))
            sharpness = _cv2.Laplacian(gray, _cv2.CV_64F).var()
            if sharpness > best_sharpness:
                best_sharpness = sharpness
                best_sharpness_frame = pil_img
        except Exception:
            if best_sharpness_frame is None:
                best_sharpness_frame = pil_img

    # Use the sharpest frame for quality grading, vector, and dHash
    if best_sharpness_frame is None:
        best_sharpness_frame = frames[0][0]

    result = {
        "face_count": best_face_count,
        "faces_found": sorted(all_found_persons),
        "has_target_face": False,
        "face_distance": round(best_face_distance, 3) if best_face_distance is not None and best_face_distance < 999 else None,
        "photo_grade": None,
        "image_vector": None,
        "dhash": None,
        "best_thumb": None,
        "_analyzed_frames": len(frames),
    }

    # Determine has_target_face
    if face_names and all_found_persons:
        if face_match_mode == "all":
            result["has_target_face"] = all(n in all_found_persons for n in face_names)
        else:
            result["has_target_face"] = any(n in all_found_persons for n in face_names)

    # ── Quality grading from best frame ──
    try:
        import cv2 as _cv2
        w, h = best_sharpness_frame.size
        megapixels = (w * h) / 1_000_000

        # Resolution score
        if megapixels >= 20:
            resolution = 100
        elif megapixels >= 12:
            resolution = 90
        elif megapixels >= 5:
            resolution = 70
        elif megapixels >= 2:
            resolution = 50
        else:
            resolution = max(20, megapixels / 0.5 * 20)

        # Sharpness
        gray = np.array(best_sharpness_frame.convert("L"))
        sharpness_var = _cv2.Laplacian(gray, _cv2.CV_64F).var()
        sharpness_score = min(100, sharpness_var / 8.0)

        # Noise
        blur_k = _cv2.GaussianBlur(gray, (5, 5), 0)
        high_pass = gray.astype(float) - blur_k.astype(float)
        noise_std = np.std(high_pass)
        noise_score = max(0, min(100, 100 - noise_std * 5))

        # Compression (estimate from frame data size)
        file_size_kb = os.path.getsize(filepath) / 1024
        # For video, estimate per-frame size from total file / estimated frames
        try:
            _, _, dur = get_video_info(filepath)
            fps_est = 30
            est_frames = max(1, dur * fps_est)
            kb_per_frame = file_size_kb / est_frames
            kb_per_mp = kb_per_frame / max(megapixels, 0.1)
        except Exception:
            kb_per_mp = 500  # reasonable default
        compression_score = min(100, kb_per_mp / 15)

        # Color
        rgb_arr = np.array(best_sharpness_frame.convert("RGB"))
        hsv = _cv2.cvtColor(rgb_arr, _cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1].mean()
        sat_score = min(100, sat / 1.5)
        brightness = hsv[:, :, 2].astype(float)
        p5, p95 = np.percentile(brightness, [5, 95])
        range_score = min(100, (p95 - p5) / 2.0)
        color_score = sat_score * 0.4 + range_score * 0.6

        # Exposure
        mean_bright = brightness.mean()
        std_bright = brightness.std()
        exposure_score = 80
        if mean_bright < 60 or mean_bright > 210:
            exposure_score -= 30
        elif mean_bright < 80 or mean_bright > 180:
            exposure_score -= 10
        if std_bright < 20:
            exposure_score -= 15
        elif std_bright > 90:
            exposure_score -= 10
        exposure_score = max(0, min(100, exposure_score))

        # Focus (center vs edges)
        ch, cw = gray.shape
        center = gray[ch // 4: 3 * ch // 4, cw // 4: 3 * cw // 4]
        focus_score = min(100, _cv2.Laplacian(center, _cv2.CV_64F).var() / 8.0)

        # Composite
        composite = (
            resolution * 0.10 + sharpness_score * 0.20 + noise_score * 0.10 +
            compression_score * 0.05 + color_score * 0.10 + exposure_score * 0.15 +
            focus_score * 0.20 + 80 * 0.10  # distortion fixed at 80 for video frames
        )

        result["photo_grade"] = {
            "resolution": round(resolution, 1),
            "sharpness": round(sharpness_score, 1),
            "noise": round(noise_score, 1),
            "compression": round(compression_score, 1),
            "color": round(color_score, 1),
            "exposure": round(exposure_score, 1),
            "focus": round(focus_score, 1),
            "distortion": 80.0,  # N/A for video frames
            "composite": round(composite, 1),
            "blur_score": round(sharpness_var, 1),
        }
    except Exception:
        pass

    # ── Image vector from best frame ──
    try:
        size = 32
        gray = best_sharpness_frame.convert("L").resize((size, size), Image.LANCZOS)
        gray_vec = np.array(gray, dtype=np.float32).flatten()
        rgb = best_sharpness_frame.convert("RGB")
        r, g, b = rgb.split()
        hist_r = np.array(r.histogram()[:256], dtype=np.float32)
        hist_g = np.array(g.histogram()[:256], dtype=np.float32)
        hist_b = np.array(b.histogram()[:256], dtype=np.float32)
        hist_r = np.add.reduceat(hist_r, range(0, 256, 32))
        hist_g = np.add.reduceat(hist_g, range(0, 256, 32))
        hist_b = np.add.reduceat(hist_b, range(0, 256, 32))
        vec = np.concatenate([gray_vec, hist_r, hist_g, hist_b])
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        result["image_vector"] = vec
    except Exception:
        pass

    # ── dHash from best frame ──
    try:
        hash_size = 8
        dh_img = best_sharpness_frame.convert("L").resize(
            (hash_size + 1, hash_size), Image.LANCZOS
        )
        pixels = list(dh_img.getdata())
        bits = []
        for row in range(hash_size):
            row_start = row * (hash_size + 1)
            for col in range(hash_size):
                bits.append(1 if pixels[row_start + col] > pixels[row_start + col + 1] else 0)
        result["dhash"] = int("".join(str(b) for b in bits), 2)
    except Exception:
        pass

    # ── Best thumbnail (sharpest frame) ──
    try:
        thumb = best_sharpness_frame.copy()
        thumb.thumbnail((120, 120), Image.LANCZOS)
        if thumb.mode in ("RGBA", "P"):
            thumb = thumb.convert("RGB")
        buf = BytesIO()
        thumb.save(buf, format="JPEG", quality=55)
        result["best_thumb"] = base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        pass

    return result


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
            "blur": img.get("blur_score"),
            "grade": img.get("photo_grade"),
            "st": img.get("status", "pool"),
            "rr": img.get("reject_reason"),
            "th": img.get("thumb", ""),
            "loc": img.get("location"),
            "pref": img.get("preference"),
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
.btn-save {{ background:#38a169; color:white; font-weight:600; }}
.btn-save:hover {{ background:#48bb78; }}
.btn-save:disabled {{ background:#444; cursor:not-allowed; }}
.btn-save-exit {{ background:#2b6cb0; color:white; font-weight:600; }}
.btn-save-exit:hover {{ background:#3182ce; }}
.btn-save-exit:disabled {{ background:#444; cursor:not-allowed; }}
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
.card .blur-badge {{
    position:absolute; top:2px; right:2px; font-size:.55em; padding:1px 4px;
    border-radius:3px; background:rgba(233,69,96,.85); color:white;
}}
.card .grade-badge {{
    position:absolute; bottom:2px; left:2px; font-size:.6em; padding:1px 5px;
    border-radius:3px; color:white; font-weight:bold; opacity:.9;
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
.lightbox video {{ max-width:90vw; max-height:80vh; border-radius:8px; }}
.lightbox .close {{ position:absolute; top:15px; right:25px; color:white; font-size:2em; cursor:pointer; z-index:310; }}
.lightbox .meta {{ color:#888; font-size:.8em; margin-top:10px; text-align:center; max-width:80vw; }}

/* ── TUTORIAL OVERLAY ── */
.help-btn {{
    position:fixed; bottom:20px; right:20px; width:44px; height:44px; border-radius:50%;
    background:#e94560; color:white; border:none; font-size:1.4em; font-weight:700;
    cursor:pointer; z-index:250; box-shadow:0 2px 12px rgba(233,69,96,.5);
    display:flex; align-items:center; justify-content:center;
}}
.help-btn:hover {{ background:#f05a73; transform:scale(1.1); }}
.tut-overlay {{
    display:none; position:fixed; top:0; left:0; right:0; bottom:0;
    background:rgba(0,0,0,.85); z-index:400; justify-content:center; align-items:center;
}}
.tut-overlay.active {{ display:flex; }}
.tut-card {{
    background:#1a1a2e; border:1px solid #333; border-radius:12px; padding:28px 32px;
    max-width:520px; width:90vw; color:#ddd; position:relative;
    box-shadow:0 8px 40px rgba(0,0,0,.6);
}}
.tut-card h3 {{ color:#e94560; margin-bottom:6px; font-size:1.1em; }}
.tut-card .step-label {{ color:#888; font-size:.75em; margin-bottom:12px; }}
.tut-card p {{ font-size:.9em; line-height:1.6; margin-bottom:16px; }}
.tut-card p b {{ color:#5bc0eb; }}
.tut-card p code {{ background:#0f3460; padding:1px 6px; border-radius:3px; font-size:.85em; color:#e94560; }}
.tut-btns {{ display:flex; justify-content:space-between; align-items:center; gap:10px; }}
.tut-btns button {{ padding:8px 20px; border:none; border-radius:6px; cursor:pointer; font-size:.85em; }}
.tut-btns .btn-next {{ background:#e94560; color:white; }}
.tut-btns .btn-next:hover {{ background:#f05a73; }}
.tut-btns .btn-prev {{ background:#0f3460; color:#5bc0eb; }}
.tut-btns .btn-skip {{ background:none; border:none; color:#888; text-decoration:underline; cursor:pointer; font-size:.8em; }}
.tut-dots {{ display:flex; gap:6px; }}
.tut-dots span {{ width:8px; height:8px; border-radius:50%; background:#333; }}
.tut-dots span.active {{ background:#e94560; }}

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

/* ── PREFERENCE (like/dislike) ── */
.card.pref-like {{ outline:2px solid #4caf50; outline-offset:-2px; }}
.card.pref-dislike {{ outline:2px solid #e53935; outline-offset:-2px; }}
.card.pref-like.selected {{ outline:3px solid #4caf50; }}
.card.pref-dislike.selected {{ outline:3px solid #e53935; }}
.card .pref-btns {{
    position:absolute; bottom:2px; left:50%; transform:translateX(-50%);
    display:flex; gap:4px; opacity:0; transition:opacity .15s; z-index:5;
}}
.card:hover .pref-btns {{ opacity:1; }}
.pref-btn {{
    width:22px; height:22px; border-radius:50%; border:none; cursor:pointer;
    font-size:12px; line-height:22px; text-align:center; padding:0;
    background:rgba(0,0,0,.7); color:#888; transition:all .12s;
}}
.pref-btn:hover {{ transform:scale(1.2); }}
.pref-btn.active-like {{ background:#4caf50; color:white; }}
.pref-btn.active-dislike {{ background:#e53935; color:white; }}

/* Lightbox preference buttons */
.lb-pref-btns {{
    display:flex; gap:12px; margin-top:8px; justify-content:center;
}}
.lb-pref-btn {{
    padding:6px 16px; border-radius:6px; border:1px solid #444; cursor:pointer;
    font-size:.85em; background:#1a1a2e; color:#888; transition:all .15s;
}}
.lb-pref-btn:hover {{ border-color:#888; }}
.lb-pref-btn.active-like {{ background:#4caf50; color:white; border-color:#4caf50; }}
.lb-pref-btn.active-dislike {{ background:#e53935; color:white; border-color:#e53935; }}

/* Preference stats line */
.pref-stats {{
    font-size:.75em; color:#888; padding:0 4px;
}}
.pref-stats .liked {{ color:#4caf50; }}
.pref-stats .disliked {{ color:#e53935; }}
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
    <label style="font-size:.75em;margin-left:8px">Sort:</label>
    <select id="sort-mode" onchange="render()" style="font-size:.75em;padding:2px 4px">
        <option value="date">Date</option>
        <option value="grade_desc">Grade (best first)</option>
        <option value="grade_asc">Grade (worst first)</option>
    </select>
    <span style="flex:1"></span>
    <span id="save-status" style="font-size:.75em;color:#888;display:none"></span>
    <button class="btn-save" onclick="saveChanges(false)">Save</button>
    <button class="btn-save-exit" onclick="saveChanges(true)">Save &amp; Exit</button>
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
    <label>Preference:</label>
    <select id="f-pref" onchange="applyFilters()">
        <option value="">All</option>
        <option value="like">Liked</option>
        <option value="dislike">Disliked</option>
        <option value="unrated">Unrated</option>
    </select>
    <label>Grade:</label>
    <select id="f-grade" onchange="applyFilters()">
        <option value="">All</option>
        <option value="high">High (70+)</option>
        <option value="medium">Medium (40-70)</option>
        <option value="low">Low (&lt;40)</option>
    </select>
    <label>Search:</label>
    <input id="f-search" type="text" placeholder="filename..." oninput="applyFilters()">
    <span class="pref-stats" id="pref-stats"></span>
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
    <video id="lb-video" controls style="display:none" onclick="event.stopPropagation()"></video>
    <div class="meta" id="lb-meta"></div>
    <div class="lb-pref-btns" id="lb-pref-btns">
        <button class="lb-pref-btn" id="lb-like-btn" onclick="toggleLBPref('like')">&#x1F44D; Like</button>
        <button class="lb-pref-btn" id="lb-dislike-btn" onclick="toggleLBPref('dislike')">&#x1F44E; Dislike</button>
    </div>
</div>

<button class="help-btn" onclick="startTutorial()" title="Help / Tutorial">?</button>

<div class="tut-overlay" id="tut-overlay" onclick="if(event.target===this)closeTutorial()">
    <div class="tut-card">
        <div class="step-label" id="tut-step-label"></div>
        <h3 id="tut-title"></h3>
        <p id="tut-body"></p>
        <div class="tut-btns">
            <button class="btn-prev" id="tut-prev" onclick="tutStep(-1)">Back</button>
            <div class="tut-dots" id="tut-dots"></div>
            <button class="btn-next" id="tut-next" onclick="tutStep(1)">Next</button>
        </div>
    </div>
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
let filterState = {{ source:'', faces:'', status:'', search:'', location:'', pref:'', grade:'' }};
let currentLBImg = null;  // track which image is open in lightbox

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
    if (filterState.pref === 'like' && img.pref !== 'like') return false;
    if (filterState.pref === 'dislike' && img.pref !== 'dislike') return false;
    if (filterState.pref === 'unrated' && (img.pref === 'like' || img.pref === 'dislike')) return false;
    if (filterState.grade) {{
        const gc = img.grade ? img.grade.composite : -1;
        if (filterState.grade === 'high' && gc < 70) return false;
        if (filterState.grade === 'medium' && (gc < 40 || gc >= 70)) return false;
        if (filterState.grade === 'low' && (gc >= 40 || gc < 0)) return false;
    }}
    return true;
}}

function applyFilters() {{
    filterState.source = document.getElementById('f-source').value;
    filterState.faces = document.getElementById('f-faces').value;
    filterState.status = document.getElementById('f-status').value;
    filterState.search = document.getElementById('f-search').value;
    filterState.location = document.getElementById('f-location').value;
    filterState.pref = document.getElementById('f-pref').value;
    filterState.grade = document.getElementById('f-grade').value;
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

        if ((img.st === 'qualified' || img.st === 'selected') && img.cat && byCat[img.cat] !== undefined) {{
            byCat[img.cat].push(img);
            totalQ++;
        }} else if (img.st !== 'rejected') {{
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

    // Sort images within each category
    const sortMode = document.getElementById('sort-mode')?.value || 'date';
    const sortFn = sortMode === 'grade_desc'
        ? (a, b) => ((b.grade?.composite||0) - (a.grade?.composite||0))
        : sortMode === 'grade_asc'
        ? (a, b) => ((a.grade?.composite||0) - (b.grade?.composite||0))
        : (a, b) => ((a.date||'') < (b.date||'') ? -1 : (a.date||'') > (b.date||'') ? 1 : 0);
    Object.values(byCat).forEach(arr => arr.sort(sortFn));

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
            let blurBadge = (img.blur != null && img.blur < 50) ? '<div class="blur-badge">⚠ Blurry</div>' : '';
            let gradeBadge = '';
            if (img.grade) {{
                const g = img.grade.composite;
                const gc = g >= 70 ? '#4caf50' : g >= 40 ? '#ff9800' : '#e94560';
                gradeBadge = '<div class="grade-badge" style="background:'+gc+'">'+g.toFixed(0)+'</div>';
            }}

            const likeActive = img.pref === 'like' ? ' active-like' : '';
            const dislikeActive = img.pref === 'dislike' ? ' active-dislike' : '';
            const prefClass = img.pref === 'like' ? ' pref-like' : (img.pref === 'dislike' ? ' pref-dislike' : '');

            card.innerHTML =
                '<img src="'+thumbSrc+'">' +
                '<div class="dot dot-'+img.dev+'"></div>' +
                '<div class="check">&#10003;</div>' +
                faceBadge + locBadge + blurBadge + gradeBadge +
                '<div class="pref-btns">' +
                    '<button class="pref-btn'+likeActive+'" data-action="like" title="Like">&#x1F44D;</button>' +
                    '<button class="pref-btn'+dislikeActive+'" data-action="dislike" title="Dislike">&#x1F44E;</button>' +
                '</div>' +
                '<div class="overlay">' +
                    esc(img.fn) + '<br>' +
                    (img.date||'no date') + ' | ' + img.src + '<br>' +
                    img.w+'x'+img.h+' | '+img.kb+'KB' +
                    (img.grade ? '<br>Grade: <b>'+img.grade.composite.toFixed(0)+'</b>/100' : '') +
                    (img.loc ? '<br><span style="color:#64b5f6">'+esc(img.loc)+'</span>' : '') +
                    (img.rr ? '<br><span style="color:#e94560">'+img.rr+'</span>' : '') +
                '</div>';
            if (prefClass) card.className += prefClass;

            card.querySelectorAll('.pref-btn').forEach(btn => {{
                btn.onclick = (e) => {{
                    e.stopPropagation();
                    const action = btn.dataset.action;
                    const newPref = img.pref === action ? null : action;
                    setPref(img, newPref);
                }};
            }});

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
        const inCat = ((img.st === 'qualified' || img.st === 'selected') && img.cat === cat);
        const inPool = (cat === '_pool' && img.st !== 'qualified' && img.st !== 'selected' && img.st !== 'rejected');
        if ((inCat || inPool) && matchesFilter(img)) selected.add(img.id);
    }});
    render();
}}
function deselectAllIn(cat) {{
    IMAGES.forEach(img => {{
        const inCat = ((img.st === 'qualified' || img.st === 'selected') && img.cat === cat);
        const inPool = (cat === '_pool' && img.st !== 'qualified' && img.st !== 'selected' && img.st !== 'rejected');
        if (inCat || inPool) selected.delete(img.id);
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

async function saveChanges(exitAfter) {{
    // Build current state of ALL images (not just changed ones)
    const payload = IMAGES.map(img => ({{
        hash: img.id,
        category: img.cat || null,
        status: img.st
    }}));

    const statusEl = document.getElementById('save-status');
    statusEl.style.display = 'inline';
    statusEl.style.color = '#f6e05e';
    statusEl.textContent = 'Saving...';

    // Disable buttons during save
    document.querySelectorAll('.btn-save, .btn-save-exit').forEach(b => b.disabled = true);

    try {{
        const res = await fetch('/api/curate/save', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{ changes: payload }})
        }});
        const data = await res.json();
        if (data.ok) {{
            statusEl.style.color = '#48bb78';
            statusEl.textContent = 'Saved! (' + data.updated + ' images updated)';
            changes = [];
            undoStack = [];
            updateLog();
            if (exitAfter) {{
                statusEl.textContent = 'Saved! Returning...';
                setTimeout(() => {{ window.close(); }}, 800);
            }} else {{
                setTimeout(() => {{ statusEl.style.display = 'none'; }}, 3000);
            }}
        }} else {{
            statusEl.style.color = '#fc8181';
            statusEl.textContent = 'Error: ' + (data.error || 'Save failed');
        }}
    }} catch (err) {{
        statusEl.style.color = '#fc8181';
        statusEl.textContent = 'Network error: ' + err.message;
    }}
    document.querySelectorAll('.btn-save, .btn-save-exit').forEach(b => b.disabled = false);
}}

function isVideo(fn) {{
    return /\.(mp4|mov|avi|mkv|webm|m4v)$/i.test(fn);
}}

function setPref(img, pref) {{
    img.pref = pref;
    // Fire and forget API call
    fetch('/api/images/preference', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ hash: img.id, preference: pref }})
    }}).catch(err => console.warn('Preference save failed:', err));
    updatePrefStats();
    render();
    // If lightbox is open for this image, update its buttons too
    if (currentLBImg && currentLBImg.id === img.id) updateLBPrefUI();
}}

function toggleLBPref(action) {{
    if (!currentLBImg) return;
    const newPref = currentLBImg.pref === action ? null : action;
    setPref(currentLBImg, newPref);
}}

function updateLBPrefUI() {{
    if (!currentLBImg) return;
    const likeBtn = document.getElementById('lb-like-btn');
    const dislikeBtn = document.getElementById('lb-dislike-btn');
    likeBtn.className = 'lb-pref-btn' + (currentLBImg.pref === 'like' ? ' active-like' : '');
    dislikeBtn.className = 'lb-pref-btn' + (currentLBImg.pref === 'dislike' ? ' active-dislike' : '');
}}

function updatePrefStats() {{
    let liked = 0, disliked = 0, unrated = 0;
    IMAGES.forEach(img => {{
        if (img.pref === 'like') liked++;
        else if (img.pref === 'dislike') disliked++;
        else unrated++;
    }});
    const el = document.getElementById('pref-stats');
    if (el) el.innerHTML = '<span class="liked">' + liked + ' liked</span> / <span class="disliked">' + disliked + ' disliked</span> / ' + unrated + ' unrated';
}}

function openLB(img) {{
    currentLBImg = img;
    const lb = document.getElementById('lightbox');
    const imgEl = document.getElementById('lb-img');
    const vidEl = document.getElementById('lb-video');
    const url = '/api/images/serve/' + img.id;

    if (isVideo(img.fn)) {{
        imgEl.style.display = 'none';
        imgEl.src = '';
        vidEl.src = url;
        vidEl.style.display = 'block';
    }} else {{
        vidEl.style.display = 'none';
        vidEl.pause();
        vidEl.src = '';
        imgEl.src = url;
        imgEl.style.display = 'block';
    }}

    document.getElementById('lb-meta').innerHTML =
        '<b>'+esc(img.fn)+'</b><br>' +
        'Source: '+esc(img.src)+' | Device: '+img.dev+'<br>' +
        'Date: '+(img.date||'unknown')+' | Size: '+img.w+'x'+img.h+' ('+img.kb+'KB)<br>' +
        'Faces: '+(img.faces.length ? img.faces.join(', ') : 'none')+' ('+img.fc+' total)<br>' +
        'Category: '+(img.cat||'none')+' | Status: '+img.st +
        (img.rr ? ' ('+img.rr+')' : '') +
        (img.grade ? '<br><b>Grade: '+img.grade.composite.toFixed(0)+'/100</b> — ' +
            'Sharp:'+img.grade.sharpness.toFixed(0)+' Focus:'+img.grade.focus.toFixed(0)+
            ' Noise:'+img.grade.noise.toFixed(0)+' Exp:'+img.grade.exposure.toFixed(0)+
            ' Color:'+img.grade.color.toFixed(0)+' Res:'+img.grade.resolution.toFixed(0)+
            ' Comp:'+img.grade.compression.toFixed(0)+' Dist:'+img.grade.distortion.toFixed(0) : '') +
        (img.loc ? '<br>Location: '+esc(img.loc) : '');
    updateLBPrefUI();
    lb.classList.add('active');
}}
function closeLB() {{
    currentLBImg = null;
    const vidEl = document.getElementById('lb-video');
    vidEl.pause(); vidEl.src = '';
    document.getElementById('lightbox').classList.remove('active');
}}
document.addEventListener('keydown', e => {{ if (e.key==='Escape') closeLB(); }});

// ── Tutorial ──
const TUT_STEPS = [
    {{
        title: 'Welcome to the Curate Gallery',
        body: 'This page lets you <b>review, organize, and move</b> your scanned images between categories. Here\\'s a quick tour of how everything works.'
    }},
    {{
        title: 'Stats Bar (top-left)',
        body: 'Shows your totals at a glance:<br>- <b>images</b>: total visible<br>- <b>qualified</b>: assigned to a category<br>- <b>pool</b>: unassigned, waiting for you to sort'
    }},
    {{
        title: 'Selecting Images',
        body: '<b>Click</b> a thumbnail to select it (blue checkmark appears).<br><b>Shift+click</b> to select a range of images.<br><b>Double-click</b> to open full-size preview.<br><br>The red <code>selected</code> badge in the toolbar shows how many you\\'ve picked.'
    }},
    {{
        title: 'Moving Images to a Category',
        body: '1. Select the images you want to move<br>2. Pick the target from the <b>Move to</b> dropdown<br>3. Click <b>Move to Category</b><br><br>This assigns them to that category and marks them as qualified.'
    }},
    {{
        title: 'Sending to Pool',
        body: 'Select images and click <b>Send to Pool</b> to remove them from their category. They go back to the <b>POOL</b> section at the bottom where unassigned images live.'
    }},
    {{
        title: 'Filters',
        body: 'Use the filter bar to narrow what\\'s shown:<br>- <b>Source</b>: which folder<br>- <b>Faces</b>: has target person, any face, or none<br>- <b>Status</b>: qualified, pool, or rejected<br>- <b>Location</b>: filter by GPS city/country<br>- <b>Search</b>: find by filename'
    }},
    {{
        title: 'Category Tabs',
        body: 'The colored bar below the filters shows each category with its <b>count</b> and <b>target</b>.<br><br>Example: <code>in israel 309/200</code> means 309 images assigned, target is 200.<br>Green text = target reached. Click a tab to jump to that section.'
    }},
    {{
        title: 'Device Legend',
        body: 'Each thumbnail has a small colored dot indicating the source device:<br>- <span style="color:#2d6a4f">Green</span> = iPhone<br>- <span style="color:#5a189a">Purple</span> = Android<br>- <span style="color:#1d3557">Blue</span> = Facebook<br>- <span style="color:#555">Gray</span> = Other'
    }},
    {{
        title: 'Undo, Log & Export',
        body: '<b>Undo</b>: reverses your last action.<br><b>Log</b>: shows a history of all moves you\\'ve made.<br><b>Export JSON</b>: downloads your changes as a file \\u2014 useful as a backup.<br><br>Use <b>select all</b> / <b>deselect</b> links in each section header for bulk operations.'
    }},
    {{
        title: 'Save & Save and Exit',
        body: 'The green <b>Save</b> button (top-right) saves all your changes to the server so nothing is lost.<br><br>The blue <b>Save & Exit</b> button saves and closes the gallery, returning you to the main app to continue to the next step (Export).<br><br>Always save before closing!'
    }},
    {{
        title: 'You\\'re Ready!',
        body: 'Start by scrolling through your categories, or use <b>filters</b> to find specific photos. Move them around until each category has the right images.<br><br>When done, go back to the main app and proceed to <b>Export</b>.<br><br>Click the <b>?</b> button anytime to see this guide again.'
    }}
];

let tutIdx = 0;

function startTutorial() {{
    tutIdx = 0;
    renderTut();
    document.getElementById('tut-overlay').classList.add('active');
}}
function closeTutorial() {{
    document.getElementById('tut-overlay').classList.remove('active');
}}
function tutStep(dir) {{
    tutIdx += dir;
    if (tutIdx >= TUT_STEPS.length) {{ closeTutorial(); return; }}
    if (tutIdx < 0) tutIdx = 0;
    renderTut();
}}
function renderTut() {{
    const s = TUT_STEPS[tutIdx];
    document.getElementById('tut-step-label').textContent = 'Step ' + (tutIdx+1) + ' of ' + TUT_STEPS.length;
    document.getElementById('tut-title').textContent = s.title;
    document.getElementById('tut-body').innerHTML = s.body;
    document.getElementById('tut-prev').style.visibility = tutIdx === 0 ? 'hidden' : 'visible';
    const nextBtn = document.getElementById('tut-next');
    nextBtn.textContent = tutIdx === TUT_STEPS.length - 1 ? 'Got it!' : 'Next';
    // Dots
    const dots = document.getElementById('tut-dots');
    dots.innerHTML = '';
    for (let i = 0; i < TUT_STEPS.length; i++) {{
        const d = document.createElement('span');
        if (i === tutIdx) d.className = 'active';
        d.onclick = () => {{ tutIdx = i; renderTut(); }};
        d.style.cursor = 'pointer';
        dots.appendChild(d);
    }}
}}

render();
updatePrefStats();
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


def _match_by_keywords(categories, fpath):
    """
    Try to match file path against category keywords.
    Returns category ID if a keyword matches, None otherwise.
    """
    if not fpath or not categories:
        return None
    path_lower = fpath.lower().replace("\\", "/")
    best_score = 0
    best_cat = None
    for cat in categories:
        keywords = cat.get("keywords", [])
        if keywords:
            score = sum(1 for kw in keywords if kw.lower() in path_lower)
            if score > best_score:
                best_score = score
                best_cat = cat["id"]
    return best_cat


def _match_thematic_category(thematic_cats, fpath=None):
    """
    Among thematic/catch-all categories, pick the best match using keyword
    matching on file path. Falls back to first category if no keywords match.
    """
    if not thematic_cats:
        return None
    kw_match = _match_by_keywords(thematic_cats, fpath)
    if kw_match:
        return kw_match
    return thematic_cats[0]["id"]


def categorize_by_template(template, config, img_date, fpath=None):
    """
    Given a template and an image date, return the best matching category ID.
    Supports: age_brackets, date_time_ranges, date_ranges.
    Returns None if no category matches.
    fpath is optional — used for keyword matching against thematic categories.
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

        # Separate time-based and thematic categories
        time_cats = [c for c in categories if c.get("day_offset", 0) != -1]
        thematic_cats = [c for c in categories if c.get("day_offset", 0) == -1]

        # Try time-based match first
        for cat in time_cats:
            day_offset = cat.get("day_offset", 0)
            target_day = event_date + timedelta(days=day_offset)
            if img_day != target_day:
                continue
            time_from = datetime.strptime(cat.get("time_from", "00:00"), "%H:%M").time()
            time_to = datetime.strptime(cat.get("time_to", "23:59"), "%H:%M").time()
            if time_to < time_from:
                # wraps midnight
                if img_time >= time_from or img_time <= time_to:
                    return cat["id"]
            else:
                if time_from <= img_time <= time_to:
                    return cat["id"]

        # No time match — fall back to thematic categories
        if thematic_cats:
            return _match_thematic_category(thematic_cats, fpath)

        return None

    elif cat_type == "date_ranges":
        if not img_date:
            return None
        year = config.get("year")
        if year:
            year = int(year)
            if img_date.year != year:
                return None
        img_month = img_date.month

        # Separate: keyword categories (full-year + keywords), specific ranges, catch-all
        specific_cats = []
        keyword_cats = []
        catchall_cats = []
        for cat in categories:
            m_from = cat.get("month_from")
            m_to = cat.get("month_to")
            if m_from is None or m_to is None:
                continue
            is_full_year = (m_from == 1 and m_to == 12)
            if is_full_year and cat.get("keywords"):
                keyword_cats.append(cat)
            elif is_full_year:
                catchall_cats.append(cat)
            else:
                specific_cats.append(cat)

        # 1. Try keyword match first (e.g., folder "family vacation" → Travel)
        if keyword_cats:
            kw_match = _match_by_keywords(keyword_cats, fpath)
            if kw_match:
                return kw_match

        # 2. Match by specific month range
        for cat in specific_cats:
            if cat["month_from"] <= img_month <= cat["month_to"]:
                return cat["id"]

        # 3. Catch-all (full-year without keywords) as last resort
        if catchall_cats:
            return catchall_cats[0]["id"]

        return None

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  CATEGORY CLASSIFICATION RULES ENGINE
#
#  Each event type maps to a dict of category_id → rule.
#  A rule contains:
#    path_keywords: words in folder/filename path that strongly signal this category
#    path_exclude:  words that disqualify from this category
#    face_range:    (min, max) face count — None means no constraint
#    min_sharpness: minimum sharpness score (0-100)
#    max_sharpness: maximum sharpness score (0-100)
#    orientation:   "landscape", "portrait", or None
#    priority:      higher = checked first (default 0)
#    is_default:    True = fallback category when nothing else matches
#    min_tag_hits:  minimum tag keyword matches required (prevents false positives)
# ══════════════════════════════════════════════════════════════════════════════

CATEGORY_RULES = {

    # ── WEDDING ──────────────────────────────────────────────────────────────
    "wedding": {
        "01_prep_bride": {
            "path_keywords": ["bride", "bridal", "getting ready", "makeup", "hair", "dress",
                              "bridesmaids", "bouquet", "veil", "mirror"],
            "path_exclude": ["groom"],
            "face_range": (1, 6),
            "priority": 5,
        },
        "02_prep_groom": {
            "path_keywords": ["groom", "groomsmen", "getting ready", "suit", "tie",
                              "cufflinks", "boutonniere", "best man"],
            "path_exclude": ["bride"],
            "face_range": (1, 6),
            "priority": 5,
        },
        "03_first_look": {
            "path_keywords": ["first look", "pre-ceremony", "pre ceremony", "reveal",
                              "before ceremony", "outdoor", "garden"],
            "face_range": (1, 3),
            "min_sharpness": 40,
        },
        "04_ceremony": {
            "path_keywords": ["ceremony", "altar", "chuppah", "vows", "rings", "aisle",
                              "officiant", "rabbi", "priest", "church", "chapel", "synagogue"],
            "priority": 3,
        },
        "05_couple_portraits": {
            "path_keywords": ["couple", "portrait", "posed", "romantic", "sunset",
                              "golden hour", "bride groom", "formal"],
            "face_range": (2, 2),
            "min_sharpness": 50,
            "priority": 2,
        },
        "06_family_portraits": {
            "path_keywords": ["family", "parents", "grandparents", "siblings", "formal",
                              "group portrait", "family photo"],
            "face_range": (3, 30),
            "min_sharpness": 40,
        },
        "07_cocktail": {
            "path_keywords": ["cocktail", "hour", "appetizer", "drinks", "mingling",
                              "hors d'oeuvres", "reception start"],
        },
        "08_reception_entry": {
            "path_keywords": ["entrance", "first dance", "grand entrance", "introduction",
                              "dance floor", "opening dance"],
            "priority": 2,
        },
        "09_speeches": {
            "path_keywords": ["speech", "toast", "best man", "maid of honor",
                              "microphone", "podium", "tribute"],
            "face_range": (1, 3),
        },
        "10_party": {
            "path_keywords": ["dancing", "party", "dance floor", "dj", "band",
                              "celebration", "reception"],
            "max_sharpness": 60,
            "is_default": True,
        },
        "11_cake": {
            "path_keywords": ["cake", "cutting", "dessert", "sweet", "confetti",
                              "bouquet toss", "garter"],
            "priority": 4,
        },
        "12_candids": {
            "path_keywords": ["candid", "guest", "laugh", "fun", "moment", "reaction",
                              "emotion", "hug", "kiss"],
        },
        "13_details": {
            "path_keywords": ["detail", "decor", "flower", "centerpiece", "table",
                              "ring", "invitation", "venue", "setup", "decoration",
                              "sign", "menu", "place card"],
            "face_range": (0, 0),
            "priority": 3,
        },
    },

    # ── BIRTHDAY PARTY ───────────────────────────────────────────────────────
    # Keywords are aligned with CLIP TAG_VOCABULARY tags where possible.
    # Tag matching uses exact tokens (not substrings).
    "birthday": {
        "01_setup": {
            "path_keywords": ["setup", "decoration", "decor", "banner",
                              "table", "venue", "before",
                              "flowers", "indoors"],
            "face_range": (0, 2),
            "priority": 3,
        },
        "02_arrivals": {
            "path_keywords": ["arrival", "greeting", "welcome", "door",
                              "entrance", "outdoors"],
        },
        "03_activities": {
            "path_keywords": ["game", "activity", "playing sports",
                              "playing basketball", "swimming",
                              "swimming pool", "playground", "trampoline",
                              "running", "jumping", "dancing", "bouncing",
                              "eating", "food"],
            "max_sharpness": 70,
        },
        "04_cake": {
            "path_keywords": ["birthday cake", "blowing candles", "cake",
                              "candles", "singing", "wish", "happy birthday"],
            "min_tag_hits": 2,
            "priority": 3,
        },
        "05_gifts": {
            "path_keywords": ["gift", "opening gifts", "present", "unwrap",
                              "box", "bag", "surprise"],
        },
        "06_group_shots": {
            "path_keywords": ["group photo", "team photo", "group",
                              "everyone", "together", "family", "crowd"],
            "face_range": (4, 50),
            "priority": 2,
        },
        "07_candids": {
            "path_keywords": ["candid", "fun", "moment", "laughing",
                              "smiling", "happy", "casual"],
            "is_default": True,
        },
        "08_portrait": {
            "path_keywords": ["portrait", "selfie", "close up", "posed",
                              "birthday boy", "birthday girl", "formal"],
            "face_range": (1, 2),
            "min_sharpness": 50,
            "priority": 2,
        },
    },

    # ── SPORTS SEASON ────────────────────────────────────────────────────────
    "sports_season": {
        "01_team": {
            "path_keywords": ["team", "squad", "roster", "lineup", "group photo",
                              "team photo", "together"],
            "path_exclude": ["family", "birthday", "vacation", "home", "dinner",
                             "selfie", "restaurant", "beach", "pool"],
            "face_range": (5, 50),
            "orientation": "landscape",
            "priority": 3,
        },
        "02_practice": {
            "path_keywords": ["practice", "training", "warmup", "drill", "workout",
                              "stretching", "conditioning"],
        },
        "03_games": {
            "path_keywords": ["game", "match", "playing sports",
                              "playing basketball", "action", "basketball",
                              "basketball court", "gym", "sports field",
                              "league", "tournament", "competition",
                              "running", "jumping", "action shot"],
            "orientation": "landscape",
            "max_sharpness": 50,
        },
        "04_celebrations": {
            "path_keywords": ["win", "celebration", "cheer", "victory", "champion",
                              "trophy", "medal", "high five", "hug"],
            "face_range": (2, 6),
        },
        "05_portraits": {
            "path_keywords": ["portrait", "headshot", "profile", "posed", "jersey"],
            "face_range": (1, 1),
            "min_sharpness": 50,
            "priority": 2,
        },
        "06_awards": {
            "path_keywords": ["award", "trophy", "medal", "ceremony", "mvp",
                              "champion", "plaque", "certificate"],
            "priority": 5,
        },
        "07_off_field": {
            "path_keywords": ["fun", "party", "off", "trip", "outing", "birthday",
                              "vacation", "holiday", "beach", "pool", "dinner",
                              "restaurant", "family", "home", "selfie", "bus",
                              "travel", "hotel"],
            "orientation": "portrait",
            "is_default": True,
        },
    },

    # ── VACATION / TRIP ──────────────────────────────────────────────────────
    "vacation": {
        "01_departure": {
            "path_keywords": ["airport", "departure", "flight", "plane", "travel",
                              "packing", "suitcase", "taxi", "terminal", "boarding"],
            "priority": 4,
        },
        # Days 1-7 are date-based, no keyword rules needed
        "20_landscapes": {
            "path_keywords": ["landscape", "scenery", "view", "mountain", "ocean",
                              "beach", "sunset", "sunrise", "panorama", "nature",
                              "sky", "lake", "river", "forest", "cliff", "valley"],
            "face_range": (0, 0),
            "orientation": "landscape",
            "priority": 3,
        },
        "21_food": {
            "path_keywords": ["food", "restaurant", "dinner", "lunch", "breakfast",
                              "coffee", "meal", "dish", "cuisine", "market", "menu",
                              "wine", "cocktail", "bar", "cafe"],
            "face_range": (0, 3),
            "priority": 4,
        },
        "22_people": {
            "path_keywords": ["portrait", "people", "selfie", "family", "group",
                              "together", "smile", "pose"],
            "face_range": (1, 10),
            "min_sharpness": 40,
        },
        "23_details": {
            "path_keywords": ["detail", "architecture", "building", "street", "sign",
                              "art", "museum", "culture", "souvenir", "shop", "door",
                              "texture", "pattern", "close"],
            "face_range": (0, 1),
        },
    },

    # ── PHOTO BOOK / YEAR IN REVIEW ──────────────────────────────────────────
    "photo_book": {
        # Seasons are date-based (month ranges), no keyword rules needed
        "05_family": {
            "path_keywords": ["family", "portrait", "together", "group", "pose",
                              "formal", "studio"],
            "face_range": (2, 15),
            "min_sharpness": 40,
        },
        "06_travel": {
            "path_keywords": ["travel", "vacation", "trip", "flight", "hotel",
                              "beach", "mountain", "tour", "abroad", "overseas"],
        },
        "07_celebrations": {
            "path_keywords": ["birthday", "holiday", "christmas", "hanukkah",
                              "thanksgiving", "easter", "new year", "party",
                              "celebration", "anniversary", "passover"],
            "priority": 3,
        },
        "08_everyday": {
            "path_keywords": ["home", "garden", "park", "walk", "school", "morning",
                              "cooking", "backyard", "playground"],
            "is_default": True,
        },
        "09_pets": {
            "path_keywords": ["dog", "cat", "pet", "puppy", "kitten", "animal",
                              "vet", "walk"],
            "face_range": (0, 2),
            "priority": 4,
        },
        "10_favorites": {
            "path_keywords": ["best", "favorite", "highlight", "star", "top",
                              "special"],
            "min_sharpness": 60,
        },
    },

    # ── BABY'S FIRST YEAR ────────────────────────────────────────────────────
    "baby_first_year": {
        # Months are age-bracket based, only special categories need rules
        "13_first_bday": {
            "path_keywords": ["birthday", "first birthday", "party", "cake", "candle",
                              "smash", "one year", "1 year"],
            "priority": 5,
        },
        "14_milestones": {
            "path_keywords": ["first", "milestone", "crawling", "walking", "standing",
                              "sitting", "tooth", "step", "word", "smile", "laugh",
                              "bath", "swimming", "solid food"],
        },
    },

    # ── GRADUATION ───────────────────────────────────────────────────────────
    "graduation": {
        "01_getting_ready": {
            "path_keywords": ["getting ready", "dress", "gown", "cap", "mirror",
                              "preparation", "outfit", "morning"],
        },
        "02_ceremony": {
            "path_keywords": ["ceremony", "stage", "diploma", "commencement",
                              "auditorium", "procession", "march", "dean"],
            "priority": 3,
        },
        "03_cap_toss": {
            "path_keywords": ["cap toss", "toss", "throw", "jubilation", "stage walk",
                              "diploma hand", "handshake"],
            "priority": 4,
        },
        "04_family_photos": {
            "path_keywords": ["family", "parents", "mom", "dad", "grandparents",
                              "siblings", "formal", "portrait"],
            "face_range": (2, 15),
            "min_sharpness": 40,
        },
        "05_friends": {
            "path_keywords": ["friend", "classmate", "buddy", "squad", "group",
                              "yearbook", "selfie"],
            "face_range": (2, 20),
        },
        "06_celebration": {
            "path_keywords": ["dinner", "restaurant", "party", "celebration",
                              "toast", "champagne", "cheers"],
            "is_default": True,
        },
        "07_candids": {
            "path_keywords": ["candid", "moment", "laugh", "hug", "emotion",
                              "tear", "joy"],
        },
    },

    # ── ANNIVERSARY ──────────────────────────────────────────────────────────
    "anniversary": {
        "01_early_years": {
            "path_keywords": ["early", "young", "dating", "engagement", "wedding",
                              "first", "beginning", "old photo", "throwback"],
        },
        "02_milestones": {
            "path_keywords": ["milestone", "baby", "house", "career", "achievement",
                              "promotion", "move", "graduation"],
        },
        "03_family": {
            "path_keywords": ["family", "kids", "children", "growing", "together",
                              "holiday", "vacation"],
            "face_range": (3, 15),
        },
        "04_preparation": {
            "path_keywords": ["setup", "preparation", "decoration", "venue", "flowers",
                              "table", "before"],
            "face_range": (0, 2),
        },
        "05_celebration": {
            "path_keywords": ["celebration", "party", "toast", "dinner", "dance",
                              "cake", "speech"],
            "is_default": True,
        },
        "06_guests": {
            "path_keywords": ["guest", "group", "friends", "colleague", "everyone",
                              "crowd", "table"],
            "face_range": (3, 50),
        },
        "07_portraits": {
            "path_keywords": ["portrait", "couple", "posed", "romantic", "embrace",
                              "kiss", "hug"],
            "face_range": (1, 2),
            "min_sharpness": 50,
        },
    },

    # ── FAMILY REUNION ───────────────────────────────────────────────────────
    "family_reunion": {
        "01_arrivals": {
            "path_keywords": ["arrival", "greeting", "welcome", "hug", "door",
                              "driveway", "entrance"],
        },
        "02_group_shots": {
            "path_keywords": ["group", "everyone", "family photo", "branch",
                              "generations", "whole family", "lineup"],
            "face_range": (5, 50),
            "orientation": "landscape",
            "priority": 3,
        },
        "03_activities": {
            "path_keywords": ["game", "activity", "play", "sport", "ball", "swim",
                              "hike", "craft", "water", "football", "volleyball"],
            "is_default": True,
        },
        "04_food": {
            "path_keywords": ["food", "meal", "bbq", "barbecue", "grill", "table",
                              "eating", "kitchen", "cooking", "potluck", "picnic"],
            "priority": 3,
        },
        "05_kids": {
            "path_keywords": ["kid", "child", "children", "play", "young", "baby",
                              "toddler", "running"],
            "face_range": (1, 6),
        },
        "06_elders": {
            "path_keywords": ["grandma", "grandpa", "elder", "grandfather",
                              "grandmother", "nana", "papa", "senior", "generation"],
            "priority": 4,
        },
        "07_candids": {
            "path_keywords": ["candid", "fun", "moment", "laugh", "silly",
                              "spontaneous"],
        },
    },

    # ── BABY SHOWER / GENDER REVEAL ──────────────────────────────────────────
    "baby_shower": {
        "01_setup": {
            "path_keywords": ["decoration", "setup", "decor", "balloon", "banner",
                              "table", "venue", "theme", "centerpiece"],
            "face_range": (0, 1),
            "priority": 3,
        },
        "02_arrivals": {
            "path_keywords": ["arrival", "greeting", "welcome", "door", "entrance",
                              "hello"],
        },
        "03_games": {
            "path_keywords": ["game", "activity", "play", "quiz", "trivia",
                              "contest", "diaper", "bottle"],
            "is_default": True,
        },
        "04_gifts": {
            "path_keywords": ["gift", "present", "opening", "unwrap", "box",
                              "clothes", "toy", "onesie"],
            "priority": 3,
        },
        "05_reveal": {
            "path_keywords": ["reveal", "gender", "boy", "girl", "pink", "blue",
                              "surprise", "pop", "confetti", "smoke", "balloon pop"],
            "priority": 5,
        },
        "06_group": {
            "path_keywords": ["group", "everyone", "together", "photo"],
            "face_range": (4, 50),
        },
        "07_candids": {
            "path_keywords": ["candid", "fun", "moment", "laugh", "belly",
                              "bump", "mama"],
        },
    },

    # ── QUINCEAÑERA / SWEET 16 ───────────────────────────────────────────────
    "quinceanera": {
        "01_getting_ready": {
            "path_keywords": ["getting ready", "makeup", "hair", "dress", "mirror",
                              "preparation", "tiara", "crown"],
        },
        "02_ceremony": {
            "path_keywords": ["ceremony", "church", "mass", "blessing", "religious",
                              "chapel", "altar"],
            "priority": 3,
        },
        "03_portraits": {
            "path_keywords": ["portrait", "posed", "formal", "studio", "outdoor",
                              "quinceañera", "princess"],
            "face_range": (1, 2),
            "min_sharpness": 50,
            "priority": 2,
        },
        "04_reception": {
            "path_keywords": ["reception", "entrance", "grand entrance", "arrival",
                              "venue", "hall", "ballroom"],
        },
        "05_dances": {
            "path_keywords": ["dance", "waltz", "vals", "chambelan", "father daughter",
                              "first dance", "choreography"],
            "priority": 4,
        },
        "06_cake": {
            "path_keywords": ["cake", "toast", "champagne", "dessert", "cutting",
                              "candle", "last doll"],
            "priority": 4,
        },
        "07_party": {
            "path_keywords": ["party", "dancing", "dj", "music", "crowd",
                              "dance floor", "celebration"],
            "is_default": True,
        },
        "08_guests": {
            "path_keywords": ["guest", "group", "friends", "family", "table",
                              "posed", "everyone"],
            "face_range": (3, 50),
        },
    },

    # ── MEMORIAL / TRIBUTE ───────────────────────────────────────────────────
    "memorial": {
        "01_childhood": {
            "path_keywords": ["child", "baby", "kid", "young", "toddler", "infant",
                              "school", "elementary", "playground"],
            "priority": 2,
        },
        "02_youth": {
            "path_keywords": ["teen", "youth", "high school", "college", "university",
                              "prom", "graduation", "student"],
        },
        "03_family": {
            "path_keywords": ["family", "wedding", "children", "kids", "spouse",
                              "husband", "wife", "parent", "home"],
            "face_range": (2, 15),
            "is_default": True,
        },
        "04_career": {
            "path_keywords": ["work", "career", "office", "job", "professional",
                              "business", "achievement", "award", "retirement"],
            "face_range": (0, 3),
        },
        "05_friends": {
            "path_keywords": ["friend", "buddy", "colleague", "community", "club",
                              "neighbor", "group", "social"],
            "face_range": (2, 20),
        },
        "06_hobbies": {
            "path_keywords": ["hobby", "garden", "fishing", "cooking", "travel",
                              "music", "art", "sport", "reading", "craft", "golf",
                              "hiking", "camping"],
        },
        "07_recent": {
            "path_keywords": ["recent", "last", "final", "later", "senior",
                              "elder", "grandparent"],
        },
        "08_legacy": {
            "path_keywords": ["legacy", "special", "memorable", "best", "favorite",
                              "love", "joy", "moment", "smile"],
            "min_sharpness": 50,
        },
    },

    # ── PET TIMELINE ─────────────────────────────────────────────────────────
    "pet_timeline": {
        "01_first_day": {
            "path_keywords": ["first day", "adoption", "rescue", "shelter", "new home",
                              "welcome", "arrive", "puppy mill"],
            "priority": 5,
        },
        "02_puppy": {
            "path_keywords": ["puppy", "kitten", "baby", "tiny", "small", "young",
                              "little", "newborn", "pup"],
            "priority": 3,
        },
        "03_growing": {
            "path_keywords": ["growing", "bigger", "training", "learn", "sit",
                              "fetch", "walk", "leash"],
            "is_default": True,
        },
        "04_adventures": {
            "path_keywords": ["adventure", "outdoor", "park", "beach", "hike",
                              "trail", "swim", "lake", "mountain", "snow", "camp",
                              "run", "fetch", "woods", "field"],
        },
        "05_family": {
            "path_keywords": ["family", "kid", "child", "cuddle", "couch", "bed",
                              "sofa", "lap", "together", "friend", "play"],
            "face_range": (1, 10),
        },
        "06_silly": {
            "path_keywords": ["silly", "funny", "derp", "costume", "halloween",
                              "hat", "tongue", "mess", "destroy", "oops", "guilty"],
            "priority": 3,
        },
        "07_portraits": {
            "path_keywords": ["portrait", "close", "face", "eyes", "pose", "beauty",
                              "headshot", "profile", "studio"],
            "face_range": (0, 1),
            "min_sharpness": 50,
        },
    },

    # ── RETIREMENT PARTY ─────────────────────────────────────────────────────
    "retirement": {
        "01_career": {
            "path_keywords": ["career", "work", "office", "job", "professional",
                              "desk", "workplace", "years of service", "old photo"],
        },
        "02_colleagues": {
            "path_keywords": ["colleague", "coworker", "team", "department", "staff",
                              "boss", "manager", "work friend"],
            "face_range": (2, 20),
        },
        "03_setup": {
            "path_keywords": ["setup", "decoration", "decor", "venue", "banner",
                              "balloon", "sign", "table"],
            "face_range": (0, 1),
            "priority": 3,
        },
        "04_speeches": {
            "path_keywords": ["speech", "tribute", "toast", "microphone", "podium",
                              "presentation", "award", "plaque", "gift"],
            "priority": 3,
        },
        "05_celebration": {
            "path_keywords": ["celebration", "dinner", "party", "cake", "champagne",
                              "cheers", "dance"],
            "is_default": True,
        },
        "06_group": {
            "path_keywords": ["group", "everyone", "photo", "together",
                              "family", "crowd"],
            "face_range": (4, 50),
        },
        "07_candids": {
            "path_keywords": ["candid", "fun", "moment", "laugh", "hug",
                              "emotion", "tear", "joy"],
        },
    },
}

# ── Tag-family lookup for per-family score caps ─────────────────────────────
_TAG_FAMILY_CACHE = None

def _get_tag_family():
    """Lazy lookup: tag name → family.  Graceful fallback if clip_engine absent."""
    global _TAG_FAMILY_CACHE
    if _TAG_FAMILY_CACHE is not None:
        return _TAG_FAMILY_CACHE
    try:
        from clip_engine import TAG_VOCABULARY
        _TAG_FAMILY_CACHE = {t: f for f, ts in TAG_VOCABULARY.items() for t in ts}
    except ImportError:
        _TAG_FAMILY_CACHE = {}
    return _TAG_FAMILY_CACHE


def score_category_rules(entry, event_type, categories):
    """
    Score an image against all category rules for the given event type.
    Returns the best-matching category ID, or None if no rules exist.

    Scoring (tags = primary signal, filename/path = fallback):
      Tags:  +10 per keyword match (first 2 per tag-family at full weight,
             additional same-family hits at +3 — diminishing returns)
      Path:  +2  per keyword match against filename/path (if tags exist)
             +10 per keyword match against filename/path (if no tags — full fallback)
      -20 per exclusion match (both signals, full strength)
      +5  if face_count in range (with keyword evidence), +2 without
      -8  if face_count outside range
      +3  if orientation matches (keyword evidence required)
      +3  if sharpness in range (keyword evidence required)
      +priority bonus (keyword evidence required)
      +3  for is_default (fallback for unrecognized images)
    """
    rules = CATEGORY_RULES.get(event_type)
    if not rules:
        return None

    cat_ids = {c["id"] for c in categories}
    face_count = entry.get("face_count", 0)
    grade = entry.get("photo_grade") or {}
    sharpness = grade.get("sharpness", 50)
    w = entry.get("width", 0)
    h = entry.get("height", 0)
    is_landscape = w > h * 1.2 if w and h else False

    path_lower = entry.get("path", "").lower().replace("\\", "/")
    fname_lower = entry.get("filename", "").lower()
    tags = entry.get("tags") or []
    has_tags = len(tags) > 0
    tags_text = " ".join(tags)
    path_text = path_lower + " " + fname_lower

    # Scoring weights:
    #   Tags are the primary signal (10 pts/hit).
    #   Filename/path is a fallback — full weight when no tags, reduced when
    #   tags exist so they don't compete with actual image understanding.
    TAG_WEIGHT = 10
    PATH_WEIGHT = 2 if has_tags else 10

    best_score = -999
    best_cat = None
    _debug_scores = {}  # cat_id → {score, tag_hits, path_hits, ...}

    for cat_id, rule in rules.items():
        if cat_id not in cat_ids:
            continue

        score = 0
        keywords = rule.get("path_keywords", [])
        excludes = rule.get("path_exclude", [])

        # Tag keyword matches (primary signal).
        # Exact matching: keyword must equal the full tag or one of its
        # space-separated tokens.  Prevents substring inflation (e.g.,
        # "play" no longer matches "playground" or "playing basketball").
        # Per-TAG counting: each tag contributes at most 1 hit.
        _tag_kw_matched = []    # keyword that matched
        _tag_kw_source = []     # tag that produced the match
        for tag in tags:
            tag_tokens = tag.split()
            for kw in keywords:
                if kw == tag or kw in tag_tokens:
                    _tag_kw_matched.append(kw)
                    _tag_kw_source.append(tag)
                    break  # one hit per tag
        tag_hits = len(_tag_kw_matched)

        # min_tag_hits gate: if the rule requires N tag hits and we have
        # fewer, skip this category entirely (prevents false positives
        # from a single weak signal, e.g., 04_cake on a non-cake image).
        min_th = rule.get("min_tag_hits")
        if min_th and has_tags and tag_hits < min_th:
            _debug_scores[cat_id] = {
                "score": -999, "tag_hits": _tag_kw_matched,
                "path_hits": [], "excl": 0, "face_adj": 0, "bonus": 0,
                "gated": f"min_tag_hits={min_th}, got {tag_hits}",
            }
            continue

        # Per-family diminishing returns:  first TAG_FAM_CAP hits from
        # each tag family score at full weight; additional same-family
        # hits score at TAG_OVERFLOW_WEIGHT.  Prevents related tags
        # (e.g., "playing sports" + "playing basketball" + "jumping")
        # from inflating a category's score beyond the evidence.
        TAG_FAM_CAP = 2
        TAG_OVERFLOW_WEIGHT = 3
        tag_fam = _get_tag_family()
        _fam_counts = {}
        tag_score = 0
        for src_tag in _tag_kw_source:
            fam = tag_fam.get(src_tag)
            n = _fam_counts.get(fam, 0) + 1
            _fam_counts[fam] = n
            tag_score += TAG_WEIGHT if n <= TAG_FAM_CAP else TAG_OVERFLOW_WEIGHT
        score += tag_score

        # Path/filename keyword matches (fallback signal)
        _path_kw_matched = [kw for kw in keywords if kw in path_text]
        path_hits = len(_path_kw_matched)
        score += path_hits * PATH_WEIGHT

        kw_hits = tag_hits + path_hits

        # Exclusions — exact token matching, per-tag
        tag_excl = sum(1 for tag in tags
                       if any(kw == tag or kw in tag.split()
                              for kw in excludes))
        path_excl = sum(1 for kw in excludes if kw in path_text)
        excl_hits = tag_excl + path_excl
        if excl_hits > 0:
            score -= excl_hits * 20

        # Face range — evidence, but weaker without keyword support
        fr = rule.get("face_range")
        face_adj = 0
        if fr:
            if fr[0] <= face_count <= fr[1]:
                face_adj = 5 if kw_hits > 0 else 2
            else:
                face_adj = -8
            score += face_adj

        # Orientation, sharpness, priority — only with keyword evidence
        bonus = 0
        if kw_hits > 0:
            orient = rule.get("orientation")
            if orient == "landscape" and is_landscape:
                bonus += 3
            elif orient == "portrait" and not is_landscape and w and h:
                bonus += 3

            min_s = rule.get("min_sharpness")
            max_s = rule.get("max_sharpness")
            if min_s is not None and sharpness >= min_s:
                bonus += 3
            if max_s is not None and sharpness <= max_s:
                bonus += 3

            bonus += rule.get("priority", 0)
        score += bonus

        # Default fallback — wins over incidental face-range matches
        if rule.get("is_default"):
            score += 3

        # Developer inspection: per-category breakdown
        _debug_scores[cat_id] = {
            "score": score,
            "tag_score": tag_score,
            "tag_hits": _tag_kw_matched,
            "fam_counts": dict(_fam_counts),
            "path_hits": _path_kw_matched,
            "excl": excl_hits,
            "face_adj": face_adj,
            "bonus": bonus,
        }

        if score > best_score:
            best_score = score
            best_cat = cat_id

    # Store inspection data on the entry for developer tooling.
    # Top 3 candidates + winner, sorted by score descending.
    _ranked = sorted(_debug_scores.items(), key=lambda x: -x[1]["score"])
    entry["_cat_debug"] = {
        "winner": best_cat,
        "winner_score": best_score,
        "has_tags": has_tags,
        "tag_weight": TAG_WEIGHT,
        "path_weight": PATH_WEIGHT,
        "top_tags": tags[:5],
        "candidates": {cid: info for cid, info in _ranked[:3]},
    }

    return best_cat


def categorize_sports_heuristic(entry, categories):
    """Delegates to the rules engine for sports_season."""
    result = score_category_rules(entry, "sports_season", categories)
    return result or (categories[0]["id"] if categories else None)


def categorize_heuristic(entry, categories, template=None):
    """
    Generalized heuristic categorization using the rules engine.
    Tries CATEGORY_RULES first (scored matching), falls back to
    keyword/face heuristics for templates without explicit rules.
    Returns a category ID string.
    """
    template_type = template.get("event_type", "") if template else ""

    # Try rules engine first (covers all known event types)
    ruled = score_category_rules(entry, template_type, categories)
    if ruled:
        return ruled

    # Fallback for custom/unknown templates: keyword + face heuristics
    face_count = entry.get("face_count", 0)
    grade = entry.get("photo_grade") or {}
    sharpness = grade.get("sharpness", 50)
    path_lower = entry.get("path", "").lower().replace("\\", "/")
    folder_parts = "/".join(path_lower.split("/")[:-1])
    fname_lower = entry.get("filename", "").lower()
    tags = entry.get("tags") or []
    has_tags = len(tags) > 0
    tags_text = " ".join(tags)
    path_text = folder_parts + " " + fname_lower

    # Step 1: Keyword matching — tags primary, path fallback
    tag_weight = 10
    path_weight = 2 if has_tags else 10
    best_score = 0
    best_cat = None
    for cat in categories:
        keywords = list(cat.get("keywords", []))
        display_words = [w.lower() for w in cat.get("display", "").split()
                         if len(w) > 3 and w.lower() not in ("over", "with", "from", "years", "best")]
        all_kw = keywords + display_words
        # Per-tag scoring: exact token matching, each tag at most 1 hit
        _t_hits = sum(1 for tag in tags
                      if any(kw == tag or kw in tag.split()
                             for kw in all_kw))
        _p_hits = sum(1 for kw in all_kw if kw in path_text)
        score = _t_hits * tag_weight + _p_hits * path_weight
        if score > best_score:
            best_score = score
            best_cat = cat["id"]
    if best_cat:
        return best_cat

    # Step 2: Face-count heuristics
    if face_count in (1, 2) and sharpness > 40:
        for cat in categories:
            d = cat.get("display", "").lower()
            if "portrait" in d or "couple" in d:
                return cat["id"]

    if face_count >= 3:
        for cat in categories:
            d = cat.get("display", "").lower()
            if any(w in d for w in ("family", "group", "team", "colleague", "friend")):
                return cat["id"]

    if face_count == 0:
        for cat in categories:
            d = cat.get("display", "").lower()
            if any(w in d for w in ("landscape", "scenery", "detail", "action", "game", "career")):
                return cat["id"]

    # Step 3: Default category (first one with is_default in template, or first)
    return categories[0]["id"] if categories else None


def refine_thematic_category(entry, thematic_cats):
    """
    Post-Pass-2 refinement of thematic category assignments using face count.
    Returns a new category ID if a better match is found, None otherwise.
    """
    if len(thematic_cats) <= 1:
        return None

    face_count = entry.get("face_count", 0)
    path_lower = entry.get("path", "").lower().replace("\\", "/")

    # Try keyword matching first (now that we have full entry context)
    kw_match = _match_by_keywords(thematic_cats, path_lower)
    if kw_match:
        return kw_match

    # Face-count heuristics
    if face_count in (1, 2):
        for cat in thematic_cats:
            d = cat.get("display", "").lower()
            if "portrait" in d or "couple" in d:
                return cat["id"]

    if face_count >= 3:
        for cat in thematic_cats:
            d = cat.get("display", "").lower()
            if "family" in d or "group" in d or "colleague" in d:
                return cat["id"]

    if face_count == 0:
        for cat in thematic_cats:
            d = cat.get("display", "").lower()
            if any(w in d for w in ("landscape", "scenery", "detail", "career", "milestone")):
                return cat["id"]

    return None


def compute_image_vector(fpath, size=32):
    """
    Compute a compact perceptual vector for visual similarity comparison.
    Returns a 1D numpy array (grayscale thumbnail flattened + color histogram).
    """
    import numpy as np
    try:
        from PIL import Image
        pil = Image.open(fpath)
        # Grayscale thumbnail
        gray = pil.convert("L").resize((size, size), Image.LANCZOS)
        gray_vec = np.array(gray, dtype=np.float32).flatten()
        # RGB color histogram (8 bins per channel)
        rgb = pil.convert("RGB")
        r, g, b = rgb.split()
        hist_r = np.array(r.histogram()[:256], dtype=np.float32)
        hist_g = np.array(g.histogram()[:256], dtype=np.float32)
        hist_b = np.array(b.histogram()[:256], dtype=np.float32)
        # Downsample histograms to 8 bins each
        hist_r = np.add.reduceat(hist_r, range(0, 256, 32))
        hist_g = np.add.reduceat(hist_g, range(0, 256, 32))
        hist_b = np.add.reduceat(hist_b, range(0, 256, 32))
        vec = np.concatenate([gray_vec, hist_r, hist_g, hist_b])
        # Unit normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        pil.close()
        return vec
    except Exception:
        return None


def cosine_similarity(v1, v2):
    """Cosine similarity between two unit-normalized vectors."""
    import numpy as np
    return float(np.dot(v1, v2))


def compute_dhash(fpath, hash_size=8):
    """
    Compute difference hash (dHash) for near-duplicate detection.
    Compares relative brightness between horizontally adjacent pixels.
    Returns an integer hash (hash_size^2 bits, default 64-bit).
    Much more robust than pixel comparison for burst photos, slight crops,
    and minor angle/zoom differences.
    """
    try:
        from PIL import Image
        img = Image.open(fpath).convert("L").resize(
            (hash_size + 1, hash_size), Image.LANCZOS
        )
        pixels = list(img.getdata())
        bits = []
        for row in range(hash_size):
            row_start = row * (hash_size + 1)
            for col in range(hash_size):
                bits.append(1 if pixels[row_start + col] > pixels[row_start + col + 1] else 0)
        img.close()
        return int("".join(str(b) for b in bits), 2)
    except Exception:
        return None


def hamming_distance(hash1, hash2):
    """Hamming distance between two integer hashes (number of differing bits)."""
    return bin(hash1 ^ hash2).count("1")


# ══════════════════════════════════════════════════════════════════════════════
#  SIMILARITY CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

# Default thresholds
CLUSTER_DHASH_THRESHOLD = 5        # dHash Hamming distance ≤ 5 = exact duplicate
CLUSTER_VECTOR_THRESHOLD = 0.92    # Cosine similarity ≥ 0.92 = near-duplicate
CLUSTER_CLIP_THRESHOLD = 0.93      # CLIP cosine ≥ 0.93 = semantically near-identical


def cluster_similar_images(images, vector_threshold=CLUSTER_VECTOR_THRESHOLD,
                           dhash_threshold=CLUSTER_DHASH_THRESHOLD,
                           clip_threshold=CLUSTER_CLIP_THRESHOLD,
                           clip_vectors=None,
                           progress_cb=None):
    """
    Group similar images into clusters. For each cluster, pick the best-quality
    representative and mark the rest as suppressed.

    Modifies images in-place, adding:
      - cluster_id:       str — shared ID for all images in a cluster (None if unclustered)
      - is_representative: bool — True for the best image in the cluster
      - suppressed_by:    str — hash of the representative (None if not suppressed)
      - cluster_size:     int — total images in this cluster (only on representative)

    Uses union-find over up to three similarity signals:
      1. dHash Hamming distance ≤ dhash_threshold (catches burst photos)
      2. Pixel vector cosine similarity ≥ vector_threshold (catches composition similarity)
      3. CLIP vector cosine similarity ≥ clip_threshold (catches semantic near-duplicates)

    Representative selection: highest photo_grade composite wins.

    Returns: dict with diagnostics {clusters, suppressed, largest_cluster, elapsed}
    """
    import numpy as np
    import time as _time

    t0 = _time.monotonic()

    # ── Collect eligible images (non-rejected, with at least one signal) ──
    eligible = []
    for img in images:
        if img.get("status") == "rejected":
            continue
        has_vector = img.get("image_vector") is not None
        has_dhash = img.get("dhash") is not None
        if has_vector or has_dhash:
            eligible.append(img)

    n = len(eligible)
    if n < 2:
        # Nothing to cluster
        for img in images:
            img.pop("cluster_id", None)
            img.pop("is_representative", None)
            img.pop("suppressed_by", None)
            img.pop("cluster_size", None)
        return {"clusters": 0, "suppressed": 0, "largest_cluster": 0, "elapsed": 0}

    if progress_cb:
        progress_cb(f"Clustering {n} images...")

    # ── Union-Find ──
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    # ── Build vectors matrix for batch cosine similarity ──
    vectors = []
    vec_idx_map = {}  # maps eligible index → vectors array index
    for i, img in enumerate(eligible):
        v = img.get("image_vector")
        if v is not None:
            vec_idx_map[i] = len(vectors)
            if isinstance(v, list):
                vectors.append(np.array(v, dtype=np.float32))
            else:
                vectors.append(v.astype(np.float32))

    # ── dHash pass: union exact/near duplicates ──
    dhashes = []
    dhash_idx_map = {}
    for i, img in enumerate(eligible):
        dh = img.get("dhash")
        if dh is not None:
            dhash_idx_map[i] = len(dhashes)
            dhashes.append(dh)

    if dhashes:
        nd = len(dhashes)
        # Reverse map: dhash array index → eligible index
        dhash_to_eligible = {v: k for k, v in dhash_idx_map.items()}
        for i in range(nd):
            for j in range(i + 1, nd):
                dist = bin(dhashes[i] ^ dhashes[j]).count("1")
                if dist <= dhash_threshold:
                    union(dhash_to_eligible[i], dhash_to_eligible[j])

    # ── Vector pass: union visually similar images ──
    if vectors:
        vec_matrix = np.stack(vectors)  # shape (m, dim)
        # Batch cosine similarity via matrix multiply (vectors are unit-normalized)
        # Process in chunks to limit memory for very large sets
        m = len(vectors)
        vec_to_eligible = {v: k for k, v in vec_idx_map.items()}
        CHUNK = 500
        for start in range(0, m, CHUNK):
            end = min(start + CHUNK, m)
            chunk = vec_matrix[start:end]  # (chunk_size, dim)
            sims = chunk @ vec_matrix.T     # (chunk_size, m)
            for ci in range(end - start):
                gi = start + ci
                # Only check upper triangle (j > gi) to avoid double-counting
                row = sims[ci, gi + 1:]
                matches = np.where(row >= vector_threshold)[0]
                ei = vec_to_eligible[gi]
                for offset in matches:
                    gj = gi + 1 + offset
                    ej = vec_to_eligible[gj]
                    union(ei, ej)

            if progress_cb and start > 0:
                progress_cb(f"Clustering: {start}/{m} vectors compared...")

    # ── CLIP semantic pass: union semantically near-identical images ──
    if clip_vectors:
        clip_vecs = []
        clip_idx_map = {}  # eligible index → clip array index
        for i, img in enumerate(eligible):
            h = img.get("hash")
            if h and h in clip_vectors:
                cv = clip_vectors[h]
                if isinstance(cv, list):
                    cv = np.array(cv, dtype=np.float32)
                norm = np.linalg.norm(cv)
                if norm > 0:
                    clip_idx_map[i] = len(clip_vecs)
                    clip_vecs.append(cv / norm)

        if len(clip_vecs) >= 2:
            clip_matrix = np.stack(clip_vecs)
            mc = len(clip_vecs)
            clip_to_eligible = {v: k for k, v in clip_idx_map.items()}
            CLIP_CHUNK = 500
            for start in range(0, mc, CLIP_CHUNK):
                end = min(start + CLIP_CHUNK, mc)
                chunk = clip_matrix[start:end]
                sims = chunk @ clip_matrix.T
                for ci in range(end - start):
                    gi = start + ci
                    row = sims[ci, gi + 1:]
                    matches = np.where(row >= clip_threshold)[0]
                    ei = clip_to_eligible[gi]
                    for offset in matches:
                        gj = gi + 1 + offset
                        ej = clip_to_eligible[gj]
                        union(ei, ej)

            if progress_cb:
                progress_cb(f"Clustering: CLIP pass done ({mc} vectors)")

    # ── Build cluster groups ──
    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    # Only keep groups with 2+ members (actual clusters)
    clusters = {k: v for k, v in groups.items() if len(v) > 1}

    # ── Clear old cluster fields from ALL images ──
    for img in images:
        img.pop("cluster_id", None)
        img.pop("is_representative", None)
        img.pop("suppressed_by", None)
        img.pop("cluster_size", None)

    # ── Assign cluster fields ──
    total_suppressed = 0
    largest = 0

    for cluster_idx, (_, members) in enumerate(clusters.items()):
        cluster_imgs = [eligible[i] for i in members]

        # Representative: highest photo_grade composite, tie-break by size_kb
        def _quality_key(img):
            grade = img.get("photo_grade")
            composite = grade.get("composite", 0) if isinstance(grade, dict) else 0
            return (composite, img.get("size_kb", 0))

        cluster_imgs.sort(key=_quality_key, reverse=True)
        representative = cluster_imgs[0]
        rep_hash = representative.get("hash", f"cluster_{cluster_idx}")
        cid = f"c_{rep_hash[:12]}"

        representative["cluster_id"] = cid
        representative["is_representative"] = True
        representative["cluster_size"] = len(cluster_imgs)

        for img in cluster_imgs[1:]:
            img["cluster_id"] = cid
            img["is_representative"] = False
            img["suppressed_by"] = rep_hash

        total_suppressed += len(cluster_imgs) - 1
        largest = max(largest, len(cluster_imgs))

    elapsed = round(_time.monotonic() - t0, 2)

    return {
        "clusters": len(clusters),
        "suppressed": total_suppressed,
        "largest_cluster": largest,
        "elapsed": elapsed,
    }


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
