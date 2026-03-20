"""
Download Reef's photos from Google Photos shared album.
Focuses on missing ages (2+), selects best quality, skips duplicates.
"""

import os
import sys
import json
import hashlib
import shutil
import requests
import time
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

sys.stdout.reconfigure(line_buffering=True)

# ── Configuration ──────────────────────────────────────────────────────────────
PROJECT_DIR = r"C:\Codes\Reef images for bar mitza"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "sorted")
ONEDRIVE_DIR = r"C:\Users\erezg\OneDrive - Pen-Link Ltd\Desktop\אישי\ריף\בר מצווה"
MANIFEST_FILE = os.path.join(PROJECT_DIR, "manifest.json")
CREDENTIALS_FILE = os.path.join(PROJECT_DIR, "credentials.json")
TOKEN_FILE = os.path.join(PROJECT_DIR, "token.json")
REEF_BIRTHDAY = datetime(2013, 7, 16)
MAX_PER_AGE = 75
TARGET_PER_AGE = 65
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".heic", ".webp"}

SCOPES = ["https://www.googleapis.com/auth/photoslibrary.readonly"]

# Album share link - we'll find it by listing albums
REEF_ALBUM_LINK = "https://photos.app.goo.gl/wvP5hbQBYXqtqi15A"

# ── Age brackets ───────────────────────────────────────────────────────────────
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

BRACKET_DISPLAY = {
    "00_birth":        "Birth (0-2 weeks)",
    "01_month1":       "Month 1",
    "02_months2-3":    "Months 2-3",
    "03_months4-6":    "Months 4-6",
    "04_months7-9":    "Months 7-9",
    "05_months10-12":  "Months 10-12",
    "06_year1":        "Age 1 (1-2 years)",
    "07_year2":        "Age 2 (2-3 years)",
    "08_year3-4":      "Age 3-4",
    "09_year5-7":      "Age 5-7",
    "10_year8-10":     "Age 8-10",
    "11_year11-12":    "Age 11-12",
    "12_barmitzva":    "Bar Mitzva prep (12.5+)",
}

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


def authenticate():
    """Authenticate with Google Photos API via OAuth2."""
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing expired token...")
            creds.refresh(Request())
        else:
            print("Opening browser for Google sign-in...")
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
        print("Authentication successful!")
    return creds


def find_reef_album(creds):
    """Find Reef's album by listing all albums."""
    print("Searching for Reef's album...")
    headers = {"Authorization": f"Bearer {creds.token}"}
    albums = []
    next_page = None

    while True:
        params = {"pageSize": 50}
        if next_page:
            params["pageToken"] = next_page
        resp = requests.get(
            "https://photoslibrary.googleapis.com/v1/albums",
            headers=headers, params=params
        )
        resp.raise_for_status()
        data = resp.json()
        for album in data.get("albums", []):
            title = album.get("title", "")
            count = album.get("mediaItemsCount", "0")
            albums.append(album)
            if "reef" in title.lower() or "ריף" in title.lower():
                print(f"  Found: '{title}' ({count} items)")
                return album
        next_page = data.get("nextPageToken")
        if not next_page:
            break

    # Also check shared albums
    print("Checking shared albums...")
    next_page = None
    while True:
        params = {"pageSize": 50}
        if next_page:
            params["pageToken"] = next_page
        resp = requests.get(
            "https://photoslibrary.googleapis.com/v1/sharedAlbums",
            headers=headers, params=params
        )
        resp.raise_for_status()
        data = resp.json()
        for album in data.get("sharedAlbums", []):
            title = album.get("title", "")
            count = album.get("mediaItemsCount", "0")
            share_info = album.get("shareInfo", {})
            share_url = share_info.get("shareableUrl", "")
            print(f"  Shared: '{title}' ({count} items)")
            if "reef" in title.lower() or "ריף" in title.lower():
                print(f"  -> Match! Using this album.")
                return album
        next_page = data.get("nextPageToken")
        if not next_page:
            break

    # If not found by name, list all for user to see
    print("\nAll albums found:")
    for a in albums:
        print(f"  {a.get('title', 'Untitled')} ({a.get('mediaItemsCount', '?')} items)")
    return None


def list_album_items(creds, album_id):
    """List all media items in an album."""
    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
    }
    items = []
    next_page = None
    page_num = 0

    while True:
        page_num += 1
        body = {
            "albumId": album_id,
            "pageSize": 100,
        }
        if next_page:
            body["pageToken"] = next_page

        resp = requests.post(
            "https://photoslibrary.googleapis.com/v1/mediaItems:search",
            headers=headers, json=body
        )
        resp.raise_for_status()
        data = resp.json()

        page_items = data.get("mediaItems", [])
        items.extend(page_items)

        if page_num % 10 == 0:
            print(f"  Fetched {len(items)} items so far...")

        next_page = data.get("nextPageToken")
        if not next_page:
            break

        # Rate limiting
        time.sleep(0.1)

    return items


def item_to_date(item):
    """Extract creation date from a media item."""
    meta = item.get("mediaMetadata", {})
    creation_time = meta.get("creationTime", "")
    if creation_time:
        # Format: 2013-07-19T11:22:08Z
        try:
            return datetime.fromisoformat(creation_time.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            pass
    return None


def age_days_to_bracket(age_days):
    if age_days is None or age_days < 0:
        return None
    for label, start, end in AGE_BRACKETS:
        if start <= age_days < end:
            return label
    return None


def download_image(url, dest_path):
    """Download image from Google Photos base URL."""
    # Append =d for original quality download
    download_url = url + "=d"
    resp = requests.get(download_url, stream=True, timeout=30)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
    return os.path.getsize(dest_path)


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


def get_existing_counts():
    """Count existing images per bracket folder."""
    counts = {}
    for bracket_label, folder_name in BRACKET_FOLDER.items():
        folder_path = os.path.join(OUTPUT_DIR, folder_name)
        if os.path.exists(folder_path):
            count = sum(1 for f in os.listdir(folder_path)
                       if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS)
            counts[bracket_label] = count
        else:
            counts[bracket_label] = 0
    return counts


def main():
    print("=" * 60)
    print("Reef Google Photos Selector")
    print("=" * 60)

    # Load existing data
    manifest = load_manifest()
    existing_google_ids = set()
    for entry in manifest.values():
        gid = entry.get("google_id")
        if gid:
            existing_google_ids.add(gid)
    print(f"Manifest entries: {len(manifest)}")
    print(f"Already downloaded from Google: {len(existing_google_ids)}")

    existing_counts = get_existing_counts()
    print("\nCurrent image counts per age:")
    for bracket_label, _, _ in AGE_BRACKETS:
        count = existing_counts.get(bracket_label, 0)
        display = BRACKET_DISPLAY[bracket_label]
        needed = max(0, TARGET_PER_AGE - count)
        status = "FULL" if count >= TARGET_PER_AGE else f"need ~{needed} more"
        print(f"  {display:<25} {count:>4} images  ({status})")

    # Authenticate
    creds = authenticate()

    # Find album
    album = find_reef_album(creds)
    if not album:
        print("\nCould not find Reef's album. Make sure you shared it as an album.")
        return

    album_id = album["id"]
    album_title = album.get("title", "Unknown")
    album_count = album.get("mediaItemsCount", "?")
    print(f"\nUsing album: '{album_title}' ({album_count} items)")

    # List all items
    print("Fetching album contents (this may take a while for 8900+ items)...")
    items = list_album_items(creds, album_id)
    print(f"Total items in album: {len(items)}")

    # Filter to photos only (not videos)
    photos = []
    for item in items:
        meta = item.get("mediaMetadata", {})
        mime = item.get("mimeType", "")
        if "video" in mime.lower():
            continue
        if "video" in meta:
            continue
        # Check it's an image
        if not any(mime.lower().endswith(ext.strip(".")) for ext in IMAGE_EXTENSIONS):
            if "image" not in mime.lower():
                continue
        photos.append(item)
    print(f"Photos (excluding videos): {len(photos)}")

    # Categorize by age bracket
    by_bracket = defaultdict(list)
    no_date = 0
    out_of_range = 0

    for photo in photos:
        # Skip if already downloaded
        if photo["id"] in existing_google_ids:
            continue

        dt = item_to_date(photo)
        if not dt:
            no_date += 1
            continue

        age_days = (dt - REEF_BIRTHDAY).days
        bracket = age_days_to_bracket(age_days)
        if not bracket:
            out_of_range += 1
            continue

        # Quality score from metadata
        meta = photo.get("mediaMetadata", {})
        width = int(meta.get("width", 0))
        height = int(meta.get("height", 0))
        resolution = width * height

        by_bracket[bracket].append({
            "id": photo["id"],
            "filename": photo.get("filename", f"reef_{photo['id'][:8]}.jpg"),
            "baseUrl": photo.get("baseUrl", ""),
            "date": dt,
            "age_days": age_days,
            "width": width,
            "height": height,
            "resolution": resolution,
        })

    print(f"\nNew candidates by age bracket:")
    for bracket_label, _, _ in AGE_BRACKETS:
        count = len(by_bracket.get(bracket_label, []))
        print(f"  {BRACKET_DISPLAY[bracket_label]:<25} {count:>5} candidates")
    if no_date:
        print(f"  (skipped {no_date} with no date, {out_of_range} out of age range)")

    # Select and download for brackets that need more images
    print("\n" + "=" * 60)
    print("Downloading best images for each age bracket...")
    print("=" * 60)

    total_downloaded = 0
    summary = {}

    for bracket_label, _, _ in AGE_BRACKETS:
        display = BRACKET_DISPLAY[bracket_label]
        folder_name = BRACKET_FOLDER[bracket_label]
        existing = existing_counts.get(bracket_label, 0)
        needed = max(0, MAX_PER_AGE - existing)
        candidates = by_bracket.get(bracket_label, [])

        if needed == 0:
            print(f"\n{display}: already full ({existing} images), skipping")
            summary[bracket_label] = {"existing": existing, "added": 0, "total": existing}
            continue

        if not candidates:
            print(f"\n{display}: no new candidates available")
            summary[bracket_label] = {"existing": existing, "added": 0, "total": existing}
            continue

        # Sort by resolution (best quality first)
        candidates.sort(key=lambda x: x["resolution"], reverse=True)

        # Filter: minimum 800x600
        candidates = [c for c in candidates if c["width"] >= 800 or c["height"] >= 800]

        to_download = candidates[:needed]
        print(f"\n{display}: downloading {len(to_download)} of {len(candidates)} candidates (have {existing})...")

        dest_dir = os.path.join(OUTPUT_DIR, folder_name)
        onedrive_dest = os.path.join(ONEDRIVE_DIR, folder_name)
        os.makedirs(dest_dir, exist_ok=True)
        os.makedirs(onedrive_dest, exist_ok=True)

        added = 0
        for i, photo in enumerate(to_download):
            if not photo["baseUrl"]:
                continue

            dest_path = unique_dest_path(dest_dir, photo["filename"])
            try:
                fsize = download_image(photo["baseUrl"], dest_path)
                if fsize < 50 * 1024:  # Skip if downloaded file is too small
                    os.remove(dest_path)
                    continue

                # Also copy to OneDrive
                onedrive_path = unique_dest_path(onedrive_dest, photo["filename"])
                shutil.copy2(dest_path, onedrive_path)

                # Update manifest
                fhash = file_hash(dest_path)
                manifest[fhash] = {
                    "source": f"google_photos:{photo['id']}",
                    "dest": dest_path,
                    "date": photo["date"].isoformat(),
                    "date_source": "google_photos",
                    "folder": folder_name,
                    "google_id": photo["id"],
                    "resolution": f"{photo['width']}x{photo['height']}",
                }
                added += 1
                total_downloaded += 1

                if (i + 1) % 20 == 0:
                    print(f"  ... {i + 1}/{len(to_download)} downloaded")
                    save_manifest(manifest)  # Save periodically

            except Exception as e:
                print(f"  WARN: failed to download {photo['filename']}: {e}")
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                continue

            # Rate limit
            time.sleep(0.05)

        summary[bracket_label] = {"existing": existing, "added": added, "total": existing + added}
        print(f"  Done: +{added} images (total now: {existing + added})")

    save_manifest(manifest)

    # ── Final Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"{'Age Bracket':<25} {'Before':>8} {'Added':>8} {'Total':>8}")
    print("-" * 60)

    total_all = 0
    missing = []
    low = []

    for bracket_label, _, _ in AGE_BRACKETS:
        display = BRACKET_DISPLAY[bracket_label]
        s = summary.get(bracket_label, {"existing": 0, "added": 0, "total": 0})
        total = s["total"]
        total_all += total
        marker = ""
        if total == 0:
            missing.append(display)
            marker = "  << MISSING"
        elif total < 50:
            low.append((display, total))
            marker = "  << LOW"
        print(f"{display:<25} {s['existing']:>8} {s['added']:>8} {total:>8}{marker}")

    print("-" * 60)
    print(f"{'TOTAL':<25} {'':>8} {total_downloaded:>8} {total_all:>8}")

    if missing:
        print(f"\n!! MISSING AGES (0 images): {', '.join(missing)}")
    if low:
        print(f"\n!! LOW COVERAGE (<50 images):")
        for display, count in low:
            print(f"   {display}: only {count} images")

    print(f"\nTotal downloaded this run: {total_downloaded}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"OneDrive: {ONEDRIVE_DIR}")


if __name__ == "__main__":
    main()
