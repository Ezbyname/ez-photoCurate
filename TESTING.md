# Sanity Test Suite

Pre-push sanity tests for E-z Photo Organizer.

## Quick Start

```bash
pytest test_sanity.py -v
```

## What's Covered (81 tests)

| Area | Tests |
|------|-------|
| **Auth - Signup** | Missing name, weak password, password mismatch, full signup+verify flow |
| **Auth - Login** | Successful login, wrong password, logout, `/api/auth/me` endpoint |
| **Auth - Redirect** | Unauthenticated users redirected to `/login`, API returns 401 |
| **Password Validation** | Exactly 8 chars, too short, no uppercase, no special char, all special chars accepted |
| **Contact Validation** | Valid email, valid phone, invalid email, short phone, email case normalization |
| **Wizard UI** | Main page loads, all 9 steps present, tutorial elements, greeting, sidebar logout |
| **API Endpoints** | `/api/config`, `/api/templates`, `/api/stats`, `/api/images`, `/api/categories/summary`, `/api/ref-faces`, `/api/projects`, `/api/scan/status` |
| **Config** | Save/load config, unlimited mode toggle |
| **Image Serving** | 404 for missing image, serve real image via hash |
| **Video Support** | VIDEO_EXTS defined, thumbnail generation (OpenCV), video info extraction |
| **Face References** | Upload (multipart), list after upload, PNG transparency, delete single photo, delete person |
| **Project Save/Load** | Save project, list projects, load project (verify state restored), delete project, missing name |
| **Report** | No scan data (graceful error), report with data |
| **Category Targets** | Update target, update with no config |
| **Image Move** | Move between categories, move to pool (reject reason) |
| **Scan & Fill** | Scan with no config, scan empty folder, concurrent task rejection (409), task stop, quick fill |
| **Export** | Export with no data, export image, mixed media export (image + video) |
| **Selection** | Select/deselect images, reset selections |
| **NSFW** | `_check_nsfw` function exists and handles edge cases |
| **Curate Module** | IMAGE_EXTS, REEF_BIRTHDAY, age bracket mapping, thumbnail generation |
| **Resilience** | Missing scan_db, corrupt/malformed scan_db, image serve with no DB |

## Pre-Push Usage

Run before every push:

```bash
pytest test_sanity.py -v && git push
```

Or chain it:

```bash
pytest test_sanity.py -v --tb=short && echo "All tests passed!" || echo "TESTS FAILED - do not push"
```

## Requirements

- Python 3.10+
- pytest (`pip install pytest`)
- All project dependencies (Flask, Pillow, OpenCV, etc.)

## Notes

- Tests use a **temporary SQLite database** — your real `users.db` is never touched.
- Tests that need images create tiny synthetic ones via Pillow.
- Tests that need videos create synthetic AVI files via OpenCV.
- The test suite runs in ~22 seconds.
- If a test is skipped, it means an optional dependency (Pillow, OpenCV) is missing.
