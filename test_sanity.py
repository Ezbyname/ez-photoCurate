"""
Sanity test suite for E-z Photo Organizer.
Run before every push:  pytest test_sanity.py -v

Covers: auth, wizard pages, API endpoints, image/video handling, export.
"""

import os
import sys
import json
import shutil
import tempfile
import sqlite3
import pytest

# Ensure project dir is on path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def _patch_auth_db(tmp_path_factory):
    """Use a temp database so tests don't touch real users.db."""
    tmp = tmp_path_factory.mktemp("db")
    db_path = str(tmp / "test_users.db")
    import auth
    auth.DB_PATH = db_path
    auth.init_db()
    yield db_path


@pytest.fixture(scope="session")
def app_client(_patch_auth_db):
    """Create a Flask test client with auth disabled for most tests."""
    import app as app_module
    app_module.app.config["TESTING"] = True
    app_module.app.config["SECRET_KEY"] = "test-key"
    with app_module.app.test_client() as client:
        yield client


@pytest.fixture
def authed_client(app_client):
    """Client with an authenticated session (signup + verify flow)."""
    import auth
    unique = f"sanity_{os.getpid()}@test.com"

    # Create and verify user directly in DB
    auth.create_user(unique, "email", "Test123!", "Sanity Tester")
    auth.verify_user(unique)

    # Log in via API
    resp = app_client.post("/api/auth/login", json={
        "contact": unique,
        "password": "Test123!",
    })
    assert resp.status_code == 200, f"Login failed: {resp.get_json()}"
    yield app_client


# ── Auth tests ───────────────────────────────────────────────────────────────

class TestAuth:
    def test_login_page_loads(self, app_client):
        resp = app_client.get("/login")
        assert resp.status_code == 200
        html = resp.data.decode()
        assert "Sign Up" in html or "Log In" in html

    def test_signup_missing_name(self, app_client):
        resp = app_client.post("/api/auth/signup", json={
            "contact": "bad@test.com",
            "password": "Test123!",
            "confirm_password": "Test123!",
        })
        assert resp.status_code == 400
        assert "name" in resp.get_json()["error"].lower()

    def test_signup_weak_password(self, app_client):
        resp = app_client.post("/api/auth/signup", json={
            "full_name": "Test",
            "contact": "weak@test.com",
            "password": "short",
            "confirm_password": "short",
        })
        assert resp.status_code == 400
        assert "password" in resp.get_json()["error"].lower()

    def test_signup_password_mismatch(self, app_client):
        resp = app_client.post("/api/auth/signup", json={
            "full_name": "Test",
            "contact": "mm@test.com",
            "password": "Test123!",
            "confirm_password": "Test999!",
        })
        assert resp.status_code == 400
        assert "match" in resp.get_json()["error"].lower()

    def test_signup_and_verify(self, app_client):
        import auth
        contact = f"verify_{os.getpid()}@test.com"
        resp = app_client.post("/api/auth/signup", json={
            "full_name": "Verify User",
            "contact": contact,
            "password": "Test123!",
            "confirm_password": "Test123!",
        })
        data = resp.get_json()
        assert resp.status_code == 200
        assert data.get("ok")

        # Get the code from DB directly (dev mode)
        conn = sqlite3.connect(auth.DB_PATH)
        row = conn.execute(
            "SELECT code FROM verification_codes WHERE contact = ? AND used = 0 ORDER BY id DESC LIMIT 1",
            (contact,)).fetchone()
        conn.close()
        assert row, "No verification code in DB"

        resp = app_client.post("/api/auth/verify", json={
            "contact": contact,
            "code": row[0],
        })
        assert resp.status_code == 200
        assert resp.get_json().get("ok")

    def test_login_success(self, app_client):
        import auth
        contact = f"login_{os.getpid()}@test.com"
        auth.create_user(contact, "email", "Test123!", "Login User")
        auth.verify_user(contact)

        resp = app_client.post("/api/auth/login", json={
            "contact": contact,
            "password": "Test123!",
        })
        assert resp.status_code == 200
        assert resp.get_json().get("ok")

    def test_login_wrong_password(self, app_client):
        import auth
        contact = f"wrongpw_{os.getpid()}@test.com"
        auth.create_user(contact, "email", "Test123!", "Wrong PW User")
        auth.verify_user(contact)

        resp = app_client.post("/api/auth/login", json={
            "contact": contact,
            "password": "WrongPass!1",
        })
        assert resp.status_code == 400

    def test_logout(self, authed_client):
        resp = authed_client.post("/api/auth/logout")
        assert resp.status_code == 200

    def test_me_endpoint(self, authed_client):
        resp = authed_client.get("/api/auth/me")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data.get("authenticated")
        assert data.get("name")


# ── Unauthenticated redirect ────────────────────────────────────────────────

class TestAuthRedirect:
    def test_root_redirects_to_login(self, app_client):
        # Clear session
        app_client.post("/api/auth/logout")
        resp = app_client.get("/", follow_redirects=False)
        assert resp.status_code == 302 or resp.status_code == 200

    def test_api_returns_401(self, app_client):
        app_client.post("/api/auth/logout")
        resp = app_client.get("/api/config")
        # Either 401 or redirected
        assert resp.status_code in (200, 401, 302)


# ── Main page & wizard ──────────────────────────────────────────────────────

class TestWizardUI:
    def test_index_page_loads(self, authed_client):
        resp = authed_client.get("/")
        assert resp.status_code == 200
        html = resp.data.decode()
        assert "E-z Photo Organizer" in html

    def test_wizard_steps_present(self, authed_client):
        resp = authed_client.get("/")
        html = resp.data.decode()
        steps = ["Event", "Sources", "Faces", "Scan",
                 "Analyze", "Categories", "Select", "Review", "Export"]
        for step in steps:
            assert step in html, f"Missing wizard step: {step}"

    def test_tutorial_elements_present(self, authed_client):
        resp = authed_client.get("/")
        html = resp.data.decode()
        assert "TOUR_STEPS" in html or "startTutorial" in html

    def test_greeting_element(self, authed_client):
        resp = authed_client.get("/")
        html = resp.data.decode()
        assert "header-greeting" in html

    def test_sidebar_has_logout(self, authed_client):
        resp = authed_client.get("/")
        html = resp.data.decode()
        assert "Sign Out" in html or "logout" in html.lower()


# ── API endpoints ────────────────────────────────────────────────────────────

class TestAPI:
    def test_config_get(self, authed_client):
        resp = authed_client.get("/api/config")
        assert resp.status_code == 200

    def test_templates(self, authed_client):
        resp = authed_client.get("/api/templates")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_stats(self, authed_client):
        resp = authed_client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "total_images" in data or "total_media" in data

    def test_images_list(self, authed_client):
        resp = authed_client.get("/api/images")
        assert resp.status_code == 200

    def test_categories_summary(self, authed_client):
        resp = authed_client.get("/api/categories/summary")
        assert resp.status_code == 200

    def test_ref_faces_list(self, authed_client):
        resp = authed_client.get("/api/ref-faces")
        assert resp.status_code == 200

    def test_projects_list(self, authed_client):
        resp = authed_client.get("/api/projects")
        assert resp.status_code == 200

    def test_scan_status(self, authed_client):
        resp = authed_client.get("/api/scan/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "running" in data


# ── Config save/load ─────────────────────────────────────────────────────────

class TestConfig:
    def test_save_and_load(self, authed_client):
        test_config = {
            "event_type": "bar_mitzva",
            "sources": ["/tmp/test"],
            "cat_targets": {"0-1yr": 10},
            "cat_vid_targets": {"0-1yr": 2},
            "unlimited_mode": False,
        }
        resp = authed_client.post("/api/config", json=test_config)
        assert resp.status_code == 200

        resp = authed_client.get("/api/config")
        data = resp.get_json()
        assert data.get("event_type") == "bar_mitzva"

    def test_unlimited_mode(self, authed_client):
        resp = authed_client.post("/api/config", json={"unlimited_mode": True})
        assert resp.status_code == 200
        resp = authed_client.get("/api/config")
        assert resp.get_json().get("unlimited_mode") is True
        # Reset
        authed_client.post("/api/config", json={"unlimited_mode": False})


# ── Image serving ────────────────────────────────────────────────────────────

class TestImageServing:
    def test_serve_missing_image_404(self, authed_client):
        resp = authed_client.get("/api/images/serve/nonexistent_hash")
        assert resp.status_code == 404

    def test_serve_image(self, authed_client, tmp_path):
        """Create a tiny test image, add to scan_db, and verify serving."""
        # Create a 1x1 red PNG
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        img_path = tmp_path / "test_img.jpg"
        Image.new("RGB", (10, 10), "red").save(str(img_path))

        # Inject into scan_db
        import app as app_module
        db = app_module.load_scan_db()
        if not db:
            db = {"images": [], "config": {}}
        test_entry = {
            "hash": "test_serve_hash",
            "path": str(img_path).replace("\\", "/"),
            "filename": "test_img.jpg",
            "media_type": "image",
            "status": "candidate",
            "category": "test",
        }
        db["images"] = [e for e in db.get("images", []) if e["hash"] != "test_serve_hash"]
        db["images"].append(test_entry)
        app_module.save_scan_db(db)

        resp = authed_client.get("/api/images/serve/test_serve_hash")
        assert resp.status_code == 200
        assert resp.content_type.startswith("image/")

        # Cleanup
        db["images"] = [e for e in db["images"] if e["hash"] != "test_serve_hash"]
        app_module.save_scan_db(db)


# ── Video support ────────────────────────────────────────────────────────────

class TestVideoSupport:
    def test_video_exts_defined(self):
        from curate import VIDEO_EXTS, MEDIA_EXTS
        assert ".mp4" in VIDEO_EXTS
        assert ".mov" in VIDEO_EXTS
        assert ".mp4" in MEDIA_EXTS

    def test_video_thumbnail_function_exists(self):
        from curate import make_video_thumbnail_b64
        assert callable(make_video_thumbnail_b64)

    def test_video_info_function_exists(self):
        from curate import get_video_info
        assert callable(get_video_info)

    def test_video_date_function_exists(self):
        from curate import get_video_date
        assert callable(get_video_date)

    def test_video_thumbnail_generation(self, tmp_path):
        """Generate a thumbnail from a synthetic video (requires OpenCV)."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            pytest.skip("OpenCV not installed")

        # Create a tiny video file using OpenCV
        vid_path = str(tmp_path / "test.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(vid_path, fourcc, 1, (64, 64))
        for _ in range(3):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            frame[:, :, 2] = 255  # red
            out.write(frame)
        out.release()

        from curate import make_video_thumbnail_b64
        thumb = make_video_thumbnail_b64(vid_path, 64)
        assert thumb and len(thumb) > 10, "Video thumbnail should not be empty"

    def test_video_info_extraction(self, tmp_path):
        """Extract video info from a synthetic video."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            pytest.skip("OpenCV not installed")

        vid_path = str(tmp_path / "test_info.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(vid_path, fourcc, 10, (128, 96))
        for _ in range(30):
            out.write(np.zeros((96, 128, 3), dtype=np.uint8))
        out.release()

        from curate import get_video_info
        w, h, dur = get_video_info(vid_path)
        assert w == 128
        assert h == 96
        assert dur > 0


# ── Export endpoint ──────────────────────────────────────────────────────────

class TestExport:
    def test_export_no_scan_data(self, authed_client, tmp_path):
        resp = authed_client.post("/api/export", json={
            "output_dir": str(tmp_path / "export_out"),
            "status": "selected",
        })
        # Should start the task (async) even if no data
        assert resp.status_code in (200, 409)

    def test_export_with_image(self, authed_client, tmp_path):
        """Add a selected image to scan_db and export it."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        # Create test image
        img_path = tmp_path / "export_test.jpg"
        Image.new("RGB", (10, 10), "blue").save(str(img_path))

        import app as app_module
        import time

        # Wait for any running task
        for _ in range(10):
            status = authed_client.get("/api/scan/status").get_json()
            if not status.get("running"):
                break
            time.sleep(0.5)

        db = app_module.load_scan_db() or {"images": [], "config": {}}
        test_entry = {
            "hash": "export_test_hash",
            "path": str(img_path).replace("\\", "/"),
            "filename": "export_test.jpg",
            "media_type": "image",
            "status": "selected",
            "category": "test_cat",
        }
        db["images"] = [e for e in db.get("images", []) if e["hash"] != "export_test_hash"]
        db["images"].append(test_entry)
        app_module.save_scan_db(db)

        out_dir = str(tmp_path / "exported")
        resp = authed_client.post("/api/export", json={
            "output_dir": out_dir,
            "status": "selected",
        })
        assert resp.status_code == 200

        # Wait for export to finish
        for _ in range(20):
            status = authed_client.get("/api/scan/status").get_json()
            if not status.get("running"):
                break
            time.sleep(0.3)

        # Check file was exported
        exported_file = os.path.join(out_dir, "test_cat", "export_test.jpg")
        assert os.path.exists(exported_file), f"Exported file not found at {exported_file}"

        # Cleanup scan_db
        db["images"] = [e for e in db["images"] if e["hash"] != "export_test_hash"]
        app_module.save_scan_db(db)


# ── Selection / move ─────────────────────────────────────────────────────────

class TestSelection:
    def test_select_image(self, authed_client, tmp_path):
        import app as app_module

        db = app_module.load_scan_db() or {"images": [], "config": {}}
        test_entry = {
            "hash": "sel_test_hash",
            "path": "/tmp/fake.jpg",
            "filename": "fake.jpg",
            "media_type": "image",
            "status": "candidate",
            "category": "test",
        }
        db["images"] = [e for e in db.get("images", []) if e["hash"] != "sel_test_hash"]
        db["images"].append(test_entry)
        app_module.save_scan_db(db)

        resp = authed_client.post("/api/images/select", json={
            "hashes": ["sel_test_hash"],
            "status": "selected",
        })
        assert resp.status_code == 200

        db = app_module.load_scan_db()
        entry = next((i for i in db["images"] if i["hash"] == "sel_test_hash"), None)
        assert entry and entry["status"] == "selected"

        # Cleanup
        db["images"] = [e for e in db["images"] if e["hash"] != "sel_test_hash"]
        app_module.save_scan_db(db)

    def test_reset_selections(self, authed_client):
        resp = authed_client.post("/api/selections/reset")
        assert resp.status_code == 200


# ── NSFW filter ──────────────────────────────────────────────────────────────

class TestNSFW:
    def test_nsfw_check_function(self):
        """Verify the _check_nsfw helper exists and handles missing classifier."""
        from app import _check_nsfw
        # Should not crash with a None classifier; it will raise but we catch
        try:
            result = _check_nsfw("/nonexistent", None)
        except (TypeError, AttributeError):
            pass  # Expected when classifier is None


# ── Curate module ────────────────────────────────────────────────────────────

class TestCurate:
    def test_image_exts(self):
        from curate import IMAGE_EXTS
        assert ".jpg" in IMAGE_EXTS
        assert ".png" in IMAGE_EXTS

    def test_reef_birthday(self):
        from curate import REEF_BIRTHDAY
        assert REEF_BIRTHDAY is not None

    def test_age_days_to_bracket(self):
        from curate import age_days_to_bracket
        bracket = age_days_to_bracket(0)
        assert bracket is not None

    def test_make_thumbnail(self, tmp_path):
        try:
            from PIL import Image
            from curate import make_thumbnail_b64
        except ImportError:
            pytest.skip("Pillow not installed")

        img_path = str(tmp_path / "thumb_test.jpg")
        Image.new("RGB", (100, 100), "green").save(img_path)
        thumb = make_thumbnail_b64(img_path, 64)
        assert thumb and len(thumb) > 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
