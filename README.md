# PhotoCurate

An intelligent photo curation pipeline for building meaningful image collections for special events — bar/bat mitzvas, weddings, birthdays, photo books, and more.

Scans multiple image sources, deduplicates, detects faces, categorizes by date/age, and generates an interactive HTML gallery for reviewing and curating your final selection.

## Features

- **Multi-source scanning** — USB drives, cloud exports (Google Takeout), local folders, OneDrive
- **Smart deduplication** — hash-based + visual similarity detection
- **Face recognition** — identify specific people using reference photos (dlib-based)
- **Auto-categorization** — assign images to age brackets, time periods, or custom categories
- **Interactive HTML gallery** — review thousands of images with filters, move between categories, reject bad ones
- **Incremental scanning** — re-scan is fast, only processes new images
- **Quality gates** — auto-skip tiny images, low-res, screenshots

## Quick Start

```bash
# 1. Install dependencies
pip install pillow numpy face_recognition

# 2. Create config
python curate.py init

# 3. Edit curate_config.json — set your sources, face references, targets

# 4. Scan all sources
python curate.py scan --config curate_config.json

# 5. Generate interactive review gallery
python curate.py report

# 6. Review in browser, export changes, then apply
python curate.py apply curate_changes.json --output ./final_collection
```

## Configuration

`curate_config.json`:
```json
{
  "ref_faces_dir": "./ref_faces",
  "face_names": ["person_name"],
  "face_tolerance": 0.6,
  "sources": [
    {"path": "D:\\photos", "label": "USB Drive"},
    {"path": "C:\\Users\\me\\Pictures", "label": "Local Pictures"},
    {"path": "C:\\takeout\\Google Photos", "label": "Google Takeout"}
  ],
  "target_per_category": 75,
  "min_size_kb": 80,
  "min_dim": 600,
  "thumb_size": 120
}
```

### Face Recognition Setup

Create reference face directories:
```
ref_faces/
  person_name/
    photo1.jpg    # Clear face photo
    photo2.jpg    # Multiple angles improve accuracy
  another_person/
    photo1.jpg
```

## Commands

| Command | Description |
|---------|-------------|
| `curate.py init` | Create default config file |
| `curate.py scan --config <file>` | Scan all sources, build image database |
| `curate.py scan --config <file> --full` | Full rescan (ignore cache) |
| `curate.py report` | Generate interactive HTML gallery |
| `curate.py report --no-open` | Generate without opening browser |
| `curate.py apply <changes.json> --output <dir>` | Copy curated images to output |

## Interactive Gallery

The HTML report provides:

- **Thumbnails** for all images grouped by category
- **Filters** — by source, face detection status, qualification status, filename search
- **Selection** — click to select, shift+click for range, select/deselect all per category
- **Move** — promote pool images into categories or reject bad ones to pool
- **Undo** — revert last action
- **Export** — save changes as JSON for applying to filesystem
- **Lightbox** — double-click any image for full-size preview with metadata

## Pipeline Architecture

```
Sources (USB, cloud, local)
    |
    v
[SCAN] -> scan_db.json (cached)
  - Hash dedup
  - Quality gate (size, resolution)
  - Date extraction (EXIF > JSON sidecar > filename > directory)
  - Face detection & recognition
  - Auto-categorization (age brackets)
  - Thumbnail generation
    |
    v
[REPORT] -> curate_report.html
  - Interactive review gallery
  - Move/reject/promote images
  - Export changes JSON
    |
    v
[APPLY] -> final_collection/
  - Copy qualified images to category folders
```

## Legacy Scripts

The project includes earlier single-purpose scripts that were used during initial development:

| Script | Purpose |
|--------|---------|
| `sort_images.py` | Initial date-based sorting |
| `select_from_disk.py` | Select from USB drive |
| `select_from_google_photos.py` | Download from Google Photos API |
| `select_from_takeout.py` | Import from Google Takeout |
| `select_extras.py` | Pick top 500 extras |
| `cleanup_presentation.py` | Remove screenshots & burst duplicates |
| `dedup_by_vectors.py` | Visual dedup v1 (cosine similarity) |
| `dedup_by_vectors_v2.py` | Visual dedup v2 (per-bracket) |
| `curate_presentation.py` | First-pass curation |
| `fill_from_disk.py` | Fill gaps from USB |
| `strict_dedup.py` | Strict two-pass dedup |
| `refill_and_backup.py` | Refill + create backup pool |
| `filter_no_faces.py` | OpenCV face detection filter |
| `image_selector.py` | CLI tool with face recognition |
| `image_grader.py` | Image quality grading |
| `scan_and_grade_all.py` | Scan and grade all sources |
| `gallery_report.py` | Simple HTML gallery (superseded by curate.py report) |

## Dependencies

- Python 3.10+
- `pillow` — image processing, EXIF extraction
- `numpy` — vector operations
- `face_recognition` — face detection & recognition (requires dlib)

Optional:
- `opencv-python` — used by legacy scripts for Haar cascade detection

## License

MIT
