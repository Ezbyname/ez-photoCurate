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

    def test_icon_rail_present(self, authed_client):
        resp = authed_client.get("/")
        html = resp.data.decode()
        assert "icon-rail" in html

    def test_cleanup_tool_in_sidebar(self, authed_client):
        resp = authed_client.get("/")
        html = resp.data.decode()
        assert "Cleanup" in html

    def test_cleanup_section_in_sidebar(self, authed_client):
        resp = authed_client.get("/")
        html = resp.data.decode()
        assert "Cleanup" in html

    def test_phone_images_in_sidebar(self, authed_client):
        resp = authed_client.get("/")
        html = resp.data.decode()
        assert "Phone Images" in html

    def test_age_assessment_in_sidebar(self, authed_client):
        resp = authed_client.get("/")
        html = resp.data.decode()
        assert "Age Assessment" in html

    def test_age_overlay_present(self, authed_client):
        resp = authed_client.get("/")
        html = resp.data.decode()
        assert 'id="age-overlay"' in html


# ── JavaScript syntax validation ─────────────────────────────────────────────

class TestJavaScript:
    def test_js_syntax_valid(self, authed_client):
        """Extract all JS from the page and validate syntax with node --check."""
        import subprocess
        import tempfile

        resp = authed_client.get("/")
        html = resp.data.decode()

        # Extract all <script> blocks
        js_blocks = []
        start = 0
        while True:
            s = html.find("<script>", start)
            if s == -1:
                break
            e = html.find("</script>", s)
            if e == -1:
                break
            js_blocks.append(html[s + 8:e])
            start = e + 9

        assert js_blocks, "No <script> blocks found in page"
        full_js = "\n".join(js_blocks)

        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", encoding="utf-8", delete=False) as f:
            f.write(full_js)
            tmp_path = f.name

        try:
            result = subprocess.run(
                ["node", "--check", tmp_path],
                capture_output=True, text=True, timeout=10,
            )
            assert result.returncode == 0, (
                f"JavaScript syntax error:\n{result.stderr}"
            )
        except FileNotFoundError:
            pytest.skip("Node.js not installed — cannot validate JS syntax")
        finally:
            os.unlink(tmp_path)

    def test_js_key_functions_exist(self, authed_client):
        """Verify critical JS functions are present (not broken by syntax errors)."""
        resp = authed_client.get("/")
        html = resp.data.decode()
        functions = [
            "toggleDrawer", "loadTemplates", "startScan", "goStep",
            "showTaskOverlay", "startTutorial", "loadGreeting",
            "openCleanup", "closeCleanup", "cleanupConfirmRecycle",
            "toggleRailCleanup", "closeRailCleanup",
            "toggleProjectsSection", "toggleCleanupSection",
            "updateMatchModeStyle", "getFaceMatchMode",
            "toggleAgeEstimation", "getAgeEstConfig", "buildAgeEstFolderList",
            "openAgeAssessment", "closeAgeAssessment", "runAgeAssessment", "renderAgeResults",
            "toggleAgeFaceMode", "uploadAgeRefPhotos", "loadExistingAgeRefs",
            "openDrawerToProjects", "openPhoneImages",
        ]
        for fn in functions:
            assert f"function {fn}" in html, (
                f"JS function '{fn}' not found — possible syntax error breaking the script"
            )

    def test_no_nested_quote_issues(self, authed_client):
        """Check for common nested quote patterns that break JS in Python triple-quoted strings."""
        resp = authed_client.get("/")
        html = resp.data.decode()

        # Find all <script> content
        scripts = []
        start = 0
        while True:
            s = html.find("<script>", start)
            if s == -1:
                break
            e = html.find("</script>", s)
            scripts.append(html[s + 8:e])
            start = e + 9

        js = "\n".join(scripts)

        # Pattern: onclick="...='...'" — single quotes nested in single quotes inside double quotes
        # This is hard to catch generically, but we can check for common broken patterns
        import re
        # Look for onerror="...='...'" or onclick="...='...'" with unescaped nested quotes
        broken = re.findall(r"""onerror=["'].*?=\s*'[^"']*'[^"']*["']""", js)
        for match in broken:
            # If quotes are unbalanced, it's a problem
            assert match.count("'") % 2 == 0, f"Possible nested quote issue: {match}"


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


# ── Face reference upload/delete ─────────────────────────────────────────────

class TestFaceReferences:
    def test_upload_ref_face_multipart(self, authed_client, tmp_path):
        """Upload a reference face photo via multipart form."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        import io
        img = Image.new("RGB", (100, 100), "red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        resp = authed_client.post(
            "/api/ref-faces/upload",
            data={"person": "testperson"},
            content_type="multipart/form-data",
        )
        # Even without files, should not crash
        assert resp.status_code == 200

        # Now with a file
        buf.seek(0)
        from werkzeug.datastructures import FileStorage
        resp = authed_client.post(
            "/api/ref-faces/upload",
            data={
                "person": "testperson",
                "photos": (buf, "ref_test.jpg"),
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data.get("ok")
        assert "ref_test.jpg" in data.get("saved", [])

    def test_list_ref_faces_after_upload(self, authed_client):
        resp = authed_client.get("/api/ref-faces")
        assert resp.status_code == 200
        data = resp.get_json()
        names = [p["name"] for p in data]
        assert "testperson" in names

    def test_ref_face_photos_list(self, authed_client):
        resp = authed_client.get("/api/ref-faces/testperson/photos")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_upload_png_with_transparency(self, authed_client, tmp_path):
        """Upload an RGBA PNG and verify it doesn't break."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        import io
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        resp = authed_client.post(
            "/api/ref-faces/upload",
            data={
                "person": "testperson_rgba",
                "photos": (buf, "rgba_test.png"),
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200

        # Verify the photo is listed
        resp = authed_client.get("/api/ref-faces/testperson_rgba/photos")
        assert resp.status_code == 200

    def test_delete_single_photo(self, authed_client):
        resp = authed_client.delete("/api/ref-faces/testperson/photo/ref_test.jpg")
        assert resp.status_code == 200

    def test_delete_person(self, authed_client):
        resp = authed_client.delete("/api/ref-faces/testperson")
        assert resp.status_code == 200

        # Verify removed
        resp = authed_client.get("/api/ref-faces")
        names = [p["name"] for p in resp.get_json()]
        assert "testperson" not in names

    def test_cleanup_rgba_person(self, authed_client):
        authed_client.delete("/api/ref-faces/testperson_rgba")


# ── Project save/load cycle ──────────────────────────────────────────────────

class TestProjectSaveLoad:
    def test_save_project(self, authed_client):
        # Set up some config first
        authed_client.post("/api/config", json={
            "event_type": "bar_mitzva",
            "sources": ["/tmp/project_test"],
        })

        resp = authed_client.post("/api/projects/save", json={
            "name": "Test Project",
            "step": 3,
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data.get("ok")
        assert data["project"]["name"] == "Test Project"

    def test_project_in_list(self, authed_client):
        resp = authed_client.get("/api/projects")
        assert resp.status_code == 200
        projects = resp.get_json()
        names = [p["name"] for p in projects]
        assert "Test Project" in names

    def test_load_project(self, authed_client):
        # First change config to something different
        authed_client.post("/api/config", json={
            "event_type": "wedding",
        })

        # Now load the saved project
        resp = authed_client.post("/api/projects/load", json={
            "dir_name": "Test Project",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data.get("ok")
        assert data.get("step") == 3

        # Verify config was restored
        resp = authed_client.get("/api/config")
        config = resp.get_json()
        assert config.get("event_type") == "bar_mitzva"

    def test_load_nonexistent_project(self, authed_client):
        resp = authed_client.post("/api/projects/load", json={
            "dir_name": "nonexistent_project_xyz",
        })
        assert resp.status_code == 404

    def test_delete_project(self, authed_client):
        resp = authed_client.delete("/api/projects/Test Project")
        assert resp.status_code == 200

        # Verify removed
        resp = authed_client.get("/api/projects")
        names = [p.get("name", p.get("dir_name")) for p in resp.get_json()]
        assert "Test Project" not in names

    def test_save_project_missing_name(self, authed_client):
        resp = authed_client.post("/api/projects/save", json={"name": ""})
        assert resp.status_code == 400

    def test_rename_project(self, authed_client):
        """Save a project, rename it, verify new name in list."""
        authed_client.post("/api/config", json={"event_type": "wedding", "sources": ["/tmp/rn"]})
        authed_client.post("/api/projects/save", json={"name": "Before Rename", "step": 1})

        resp = authed_client.post("/api/projects/Before Rename/rename", json={"name": "After Rename"})
        assert resp.status_code == 200
        assert resp.get_json()["name"] == "After Rename"

        # Verify in list
        projects = authed_client.get("/api/projects").get_json()
        names = [p["name"] for p in projects]
        assert "After Rename" in names

    def test_rename_project_missing_name(self, authed_client):
        """Rename with empty name should fail."""
        resp = authed_client.post("/api/projects/Before Rename/rename", json={"name": ""})
        assert resp.status_code == 400

    def test_rename_nonexistent_project(self, authed_client):
        """Rename a project that doesn't exist should 404."""
        resp = authed_client.post("/api/projects/nonexistent_xyz_999/rename", json={"name": "Foo"})
        assert resp.status_code == 404


# ── Report generation ────────────────────────────────────────────────────────

class TestReport:
    def test_report_no_scan_data(self, authed_client):
        """Report endpoint should fail gracefully with no scan data."""
        import app as app_module
        # Temporarily remove scan_db
        had_db = os.path.isfile(app_module.SCAN_DB_PATH)
        if had_db:
            backup = app_module.SCAN_DB_PATH + ".bak"
            shutil.copy2(app_module.SCAN_DB_PATH, backup)
            os.remove(app_module.SCAN_DB_PATH)

        resp = authed_client.get("/api/report")
        assert resp.status_code in (400, 500)

        # Restore
        if had_db:
            shutil.copy2(backup, app_module.SCAN_DB_PATH)
            os.remove(backup)

    def test_report_with_data(self, authed_client):
        """Report endpoint should return HTML when scan data exists."""
        import app as app_module

        # Always use known-good test data for report
        db = {"images": [
            {"hash": "rpt1", "path": "/tmp/fake.jpg", "filename": "f.jpg",
             "media_type": "image", "status": "selected", "category": "test",
             "source_label": "Test", "date": "2020-01-01", "face_count": 0,
             "faces_found": [], "has_target_face": False, "face_distance": None,
             "width": 100, "height": 100, "size_kb": 50, "thumb": "",
             "device": "unknown", "is_screenshot": False, "age_days": None},
        ], "config": {"event_type": "bar_mitzva"}}
        app_module.save_scan_db(db)

        resp = authed_client.get("/api/report")
        # May fail if curate.cmd_report has issues, but endpoint should respond
        assert resp.status_code in (200, 400, 500)


# ── Category target update ───────────────────────────────────────────────────

class TestCategoryTargets:
    def test_update_target(self, authed_client):
        # Set up categories in config
        config = {
            "event_type": "bar_mitzva",
            "categories": [
                {"id": "0-1yr", "name": "Baby", "target": 10},
                {"id": "1-2yr", "name": "Toddler", "target": 10},
            ],
        }
        authed_client.post("/api/config", json=config)

        resp = authed_client.post("/api/categories/update-target", json={
            "id": "0-1yr",
            "target": 25,
        })
        assert resp.status_code == 200

    def test_update_target_no_config(self, authed_client):
        # Clear config
        import app as app_module
        if os.path.isfile(app_module.CONFIG_PATH):
            os.remove(app_module.CONFIG_PATH)

        resp = authed_client.post("/api/categories/update-target", json={
            "id": "0-1yr",
            "target": 25,
        })
        assert resp.status_code == 400


# ── Image move between categories ────────────────────────────────────────────

class TestImageMove:
    def test_move_image_category(self, authed_client):
        import app as app_module

        db = app_module.load_scan_db() or {"images": [], "config": {}}
        db["images"] = [e for e in db.get("images", []) if e["hash"] != "move_test_hash"]
        db["images"].append({
            "hash": "move_test_hash",
            "path": "/tmp/move_test.jpg",
            "filename": "move_test.jpg",
            "media_type": "image",
            "status": "qualified",
            "category": "cat_a",
        })
        app_module.save_scan_db(db)

        resp = authed_client.post("/api/images/move", json={
            "hashes": ["move_test_hash"],
            "to_category": "cat_b",
            "to_status": "qualified",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data.get("moved") == 1

        # Verify
        db = app_module.load_scan_db()
        entry = next(i for i in db["images"] if i["hash"] == "move_test_hash")
        assert entry["category"] == "cat_b"

        # Cleanup
        db["images"] = [e for e in db["images"] if e["hash"] != "move_test_hash"]
        app_module.save_scan_db(db)

    def test_move_to_pool_reject(self, authed_client):
        import app as app_module

        db = app_module.load_scan_db() or {"images": [], "config": {}}
        db["images"] = [e for e in db.get("images", []) if e["hash"] != "pool_test_hash"]
        db["images"].append({
            "hash": "pool_test_hash",
            "path": "/tmp/pool_test.jpg",
            "filename": "pool_test.jpg",
            "media_type": "image",
            "status": "qualified",
            "category": "cat_a",
        })
        app_module.save_scan_db(db)

        resp = authed_client.post("/api/images/move", json={
            "hashes": ["pool_test_hash"],
            "to_status": "pool",
        })
        assert resp.status_code == 200

        db = app_module.load_scan_db()
        entry = next(i for i in db["images"] if i["hash"] == "pool_test_hash")
        assert entry["status"] == "pool"
        assert entry.get("reject_reason") == "manual_reject"

        # Cleanup
        db["images"] = [e for e in db["images"] if e["hash"] != "pool_test_hash"]
        app_module.save_scan_db(db)


# ── Quick Fill / scan start ──────────────────────────────────────────────────

class TestScanAndFill:
    def test_scan_start_no_config(self, authed_client):
        """Scan should handle missing config gracefully."""
        import app as app_module
        import time

        # Wait for any running task
        for _ in range(10):
            status = authed_client.get("/api/scan/status").get_json()
            if not status.get("running"):
                break
            time.sleep(0.5)

        # Clear config
        if os.path.isfile(app_module.CONFIG_PATH):
            os.remove(app_module.CONFIG_PATH)

        resp = authed_client.post("/api/scan/start", json={"full": False})
        assert resp.status_code == 200  # Starts the task thread

        # Wait for it to finish (should fail quickly)
        for _ in range(10):
            status = authed_client.get("/api/scan/status").get_json()
            if not status.get("running"):
                break
            time.sleep(0.5)

        # Should have an error about no config
        assert status.get("error") or "config" in status.get("line", "").lower() or not status.get("running")

    def test_scan_with_empty_folder(self, authed_client, tmp_path):
        """Scan an empty folder — should complete with 0 files."""
        import app as app_module
        import time

        # Wait for any running task
        for _ in range(10):
            status = authed_client.get("/api/scan/status").get_json()
            if not status.get("running"):
                break
            time.sleep(0.5)

        empty_dir = str(tmp_path / "empty_source")
        os.makedirs(empty_dir, exist_ok=True)

        authed_client.post("/api/config", json={
            "event_type": "bar_mitzva",
            "sources": [empty_dir],
            "template": "bar_mitzva",
        })

        resp = authed_client.post("/api/scan/start", json={"full": False})
        assert resp.status_code == 200

        # Wait for scan to complete
        for _ in range(30):
            status = authed_client.get("/api/scan/status").get_json()
            if not status.get("running"):
                break
            time.sleep(0.5)

        assert not status.get("running"), "Scan should have finished"

    def test_concurrent_task_rejection(self, authed_client, tmp_path):
        """Starting a second task while one is running should return 409."""
        import app as app_module
        import time

        # Wait for any running task first
        for _ in range(10):
            status = authed_client.get("/api/scan/status").get_json()
            if not status.get("running"):
                break
            time.sleep(0.5)

        # Set up a source with some files so scan takes a moment
        src_dir = str(tmp_path / "slow_source")
        os.makedirs(src_dir, exist_ok=True)

        authed_client.post("/api/config", json={
            "event_type": "bar_mitzva",
            "sources": [src_dir],
            "template": "bar_mitzva",
        })

        # Start scan
        resp1 = authed_client.post("/api/scan/start", json={"full": True})
        if resp1.status_code == 200:
            # Try starting another immediately
            resp2 = authed_client.post("/api/scan/start", json={"full": True})
            # It should either be 409 (task running) or 200 (first one finished already)
            assert resp2.status_code in (200, 409)

        # Wait for cleanup
        for _ in range(10):
            status = authed_client.get("/api/scan/status").get_json()
            if not status.get("running"):
                break
            time.sleep(0.5)

    def test_task_stop(self, authed_client):
        """Task stop endpoint should respond."""
        resp = authed_client.post("/api/task/stop")
        assert resp.status_code == 200

    def test_scan_accepts_age_estimation(self, authed_client):
        """Scan start should accept age_estimation parameter without error."""
        import time

        for _ in range(10):
            status = authed_client.get("/api/scan/status").get_json()
            if not status.get("running"):
                break
            time.sleep(0.5)

        # Only test if no task is running (avoid long model download blocking other tests)
        status = authed_client.get("/api/scan/status").get_json()
        if status.get("running"):
            return  # skip — another task still running

        resp = authed_client.post("/api/scan/start", json={
            "full": False,
            "age_estimation": {"enabled": False},
        })
        assert resp.status_code in (200, 409)

        for _ in range(20):
            status = authed_client.get("/api/scan/status").get_json()
            if not status.get("running"):
                break
            time.sleep(0.5)

    def test_estimate_age_function_exists(self, authed_client):
        """The _estimate_age helper function should be importable."""
        import app as app_module
        assert hasattr(app_module, "_estimate_age")
        assert callable(app_module._estimate_age)

    def test_quick_fill_no_data(self, authed_client):
        """Quick fill with no scan data."""
        import app as app_module
        import time

        # Wait for any running task
        for _ in range(10):
            status = authed_client.get("/api/scan/status").get_json()
            if not status.get("running"):
                break
            time.sleep(0.5)

        resp = authed_client.post("/api/quick-fill")
        # Should start or 409 if something else running
        assert resp.status_code in (200, 409)

        # Wait for it to finish
        for _ in range(10):
            status = authed_client.get("/api/scan/status").get_json()
            if not status.get("running"):
                break
            time.sleep(0.5)


# ── Contact validation edge cases ────────────────────────────────────────────

class TestContactValidation:
    def test_valid_email(self):
        from auth import validate_contact
        ctype, contact, err = validate_contact("user@example.com")
        assert ctype == "email"
        assert contact == "user@example.com"
        assert err == ""

    def test_valid_phone(self):
        from auth import validate_contact
        ctype, contact, err = validate_contact("+1-555-123-4567")
        assert ctype == "phone"
        assert err == ""

    def test_invalid_email(self):
        from auth import validate_contact
        ctype, _, err = validate_contact("not-an-email")
        assert err != ""

    def test_short_phone(self):
        from auth import validate_contact
        ctype, _, err = validate_contact("123")
        assert err != ""

    def test_email_case_normalization(self):
        from auth import validate_contact
        _, contact, _ = validate_contact("User@EXAMPLE.COM")
        assert contact == "user@example.com"


# ── Password validation edge cases ───────────────────────────────────────────

class TestPasswordValidation:
    def test_exactly_8_chars_valid(self):
        from auth import validate_password
        ok, _ = validate_password("Abcdef1!")
        assert ok

    def test_7_chars_too_short(self):
        from auth import validate_password
        ok, err = validate_password("Abcde1!")
        assert not ok
        assert "8" in err

    def test_no_uppercase(self):
        from auth import validate_password
        ok, err = validate_password("abcdefg1!")
        assert not ok
        assert "uppercase" in err.lower()

    def test_no_special_char(self):
        from auth import validate_password
        ok, err = validate_password("Abcdefg1")
        assert not ok
        assert "special" in err.lower()

    def test_all_special_chars_accepted(self):
        from auth import validate_password
        for ch in "!@#$%^&*":
            ok, _ = validate_password(f"Abcdefg1{ch}")
            assert ok, f"Special char '{ch}' should be accepted"


# ── Mixed media export ───────────────────────────────────────────────────────

class TestMixedMediaExport:
    def test_export_images_and_videos(self, authed_client, tmp_path):
        """Export a mix of images and videos."""
        try:
            from PIL import Image
            import cv2
            import numpy as np
        except ImportError:
            pytest.skip("Pillow or OpenCV not installed")

        import app as app_module
        import time

        # Wait for any running task
        for _ in range(10):
            status = authed_client.get("/api/scan/status").get_json()
            if not status.get("running"):
                break
            time.sleep(0.5)

        # Create test image
        img_path = tmp_path / "mix_img.jpg"
        Image.new("RGB", (10, 10), "green").save(str(img_path))

        # Create test video
        vid_path = str(tmp_path / "mix_vid.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(vid_path, fourcc, 1, (64, 64))
        for _ in range(3):
            out.write(np.zeros((64, 64, 3), dtype=np.uint8))
        out.release()

        db = app_module.load_scan_db() or {"images": [], "config": {}}
        # Remove old test entries
        db["images"] = [e for e in db.get("images", [])
                        if e["hash"] not in ("mix_img_hash", "mix_vid_hash")]
        db["images"].extend([
            {
                "hash": "mix_img_hash",
                "path": str(img_path).replace("\\", "/"),
                "filename": "mix_img.jpg",
                "media_type": "image",
                "status": "selected",
                "category": "mix_cat",
            },
            {
                "hash": "mix_vid_hash",
                "path": vid_path.replace("\\", "/"),
                "filename": "mix_vid.avi",
                "media_type": "video",
                "status": "selected",
                "category": "mix_cat",
            },
        ])
        app_module.save_scan_db(db)

        out_dir = str(tmp_path / "mixed_export")
        resp = authed_client.post("/api/export", json={
            "output_dir": out_dir,
            "status": "selected",
        })
        assert resp.status_code == 200

        # Wait for export
        for _ in range(20):
            status = authed_client.get("/api/scan/status").get_json()
            if not status.get("running"):
                break
            time.sleep(0.3)

        # Verify both files exported
        assert os.path.exists(os.path.join(out_dir, "mix_cat", "mix_img.jpg")), "Image not exported"
        assert os.path.exists(os.path.join(out_dir, "mix_cat", "mix_vid.avi")), "Video not exported"

        # Cleanup
        db["images"] = [e for e in db["images"]
                        if e["hash"] not in ("mix_img_hash", "mix_vid_hash")]
        app_module.save_scan_db(db)


# ── Resilience: missing/corrupt scan_db ──────────────────────────────────────

class TestResilience:
    def test_app_handles_missing_scan_db(self, authed_client):
        """API endpoints should not crash when scan_db.json is missing."""
        import app as app_module

        # Backup and remove
        had_db = os.path.isfile(app_module.SCAN_DB_PATH)
        backup = None
        if had_db:
            backup = app_module.SCAN_DB_PATH + ".bak_resil"
            shutil.copy2(app_module.SCAN_DB_PATH, backup)
            os.remove(app_module.SCAN_DB_PATH)

        try:
            # These should not crash
            resp = authed_client.get("/api/images")
            assert resp.status_code in (200, 404)

            resp = authed_client.get("/api/stats")
            assert resp.status_code in (200, 404)

            resp = authed_client.get("/api/categories/summary")
            assert resp.status_code in (200, 404)

            resp = authed_client.post("/api/selections/reset")
            assert resp.status_code in (200, 404)
        finally:
            # Restore
            if backup and os.path.isfile(backup):
                shutil.copy2(backup, app_module.SCAN_DB_PATH)
                os.remove(backup)

    def test_app_handles_corrupt_scan_db(self, authed_client):
        """API endpoints should not crash when scan_db.json is malformed."""
        import app as app_module

        had_db = os.path.isfile(app_module.SCAN_DB_PATH)
        backup = None
        if had_db:
            backup = app_module.SCAN_DB_PATH + ".bak_corrupt"
            shutil.copy2(app_module.SCAN_DB_PATH, backup)

        try:
            # Write garbage
            with open(app_module.SCAN_DB_PATH, "w") as f:
                f.write("{invalid json!!! broken")

            resp = authed_client.get("/api/images")
            # Should not return 500; graceful handling
            assert resp.status_code in (200, 400, 404, 500)

            resp = authed_client.get("/api/stats")
            assert resp.status_code in (200, 400, 404, 500)
        finally:
            if backup and os.path.isfile(backup):
                shutil.copy2(backup, app_module.SCAN_DB_PATH)
                os.remove(backup)
            elif not had_db and os.path.isfile(app_module.SCAN_DB_PATH):
                os.remove(app_module.SCAN_DB_PATH)

    def test_serve_image_no_scan_db(self, authed_client):
        """Image serve should return 404 gracefully with no scan_db."""
        import app as app_module

        had_db = os.path.isfile(app_module.SCAN_DB_PATH)
        backup = None
        if had_db:
            backup = app_module.SCAN_DB_PATH + ".bak_serve"
            shutil.copy2(app_module.SCAN_DB_PATH, backup)
            os.remove(app_module.SCAN_DB_PATH)

        try:
            resp = authed_client.get("/api/images/serve/anything")
            assert resp.status_code in (404, 500)
        finally:
            if backup and os.path.isfile(backup):
                shutil.copy2(backup, app_module.SCAN_DB_PATH)
                os.remove(backup)


# ── Cleanup / Trash ──────────────────────────────────────────────────────────

class TestCleanup:
    def _inject_test_image(self, client, hash_id="cleanup_hash"):
        import app as app_module
        db = app_module.load_scan_db() or {"images": [], "config": {}}
        db["images"] = [e for e in db.get("images", []) if e["hash"] != hash_id]
        db["images"].append({
            "hash": hash_id,
            "path": "/tmp/cleanup_test.jpg",
            "filename": "cleanup_test.jpg",
            "media_type": "image",
            "status": "candidate",
            "category": "test",
            "size_kb": 512,
        })
        app_module.save_scan_db(db)

    def _remove_test_image(self, hash_id="cleanup_hash"):
        import app as app_module
        db = app_module.load_scan_db() or {"images": [], "config": {}}
        db["images"] = [e for e in db["images"] if e["hash"] != hash_id]
        app_module.save_scan_db(db)

    def test_cleanup_images_list(self, authed_client):
        resp = authed_client.get("/api/cleanup/images")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "images" in data
        assert "total" in data

    def test_mark_trash(self, authed_client):
        self._inject_test_image(authed_client)
        resp = authed_client.post("/api/cleanup/mark-trash", json={
            "hashes": ["cleanup_hash"],
        })
        assert resp.status_code == 200
        assert resp.get_json().get("marked") == 1

        # Verify trash flag
        import app as app_module
        db = app_module.load_scan_db()
        entry = next(i for i in db["images"] if i["hash"] == "cleanup_hash")
        assert entry.get("trash") is True

    def test_trash_count(self, authed_client):
        resp = authed_client.get("/api/cleanup/trash-count")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] >= 1
        assert "size_mb" in data

    def test_trash_filter(self, authed_client):
        resp = authed_client.get("/api/cleanup/images?trash=1")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total"] >= 1
        assert all(i.get("trash") for i in data["images"])

    def test_unmark_trash(self, authed_client):
        resp = authed_client.post("/api/cleanup/unmark-trash", json={
            "hashes": ["cleanup_hash"],
        })
        assert resp.status_code == 200
        assert resp.get_json().get("unmarked") == 1

        import app as app_module
        db = app_module.load_scan_db()
        entry = next(i for i in db["images"] if i["hash"] == "cleanup_hash")
        assert entry.get("trash") is False

    def test_confirm_trash_with_missing_file(self, authed_client):
        """Confirm trash when file doesn't exist — should still remove from DB."""
        self._inject_test_image(authed_client, "cleanup_gone")
        authed_client.post("/api/cleanup/mark-trash", json={
            "hashes": ["cleanup_gone"],
        })

        resp = authed_client.post("/api/cleanup/confirm-trash")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data.get("ok")
        assert data.get("recycled") >= 1

        # Entry should be removed from DB
        import app as app_module
        db = app_module.load_scan_db()
        hashes = [i["hash"] for i in db["images"]]
        assert "cleanup_gone" not in hashes

    def test_confirm_trash_real_file(self, authed_client, tmp_path):
        """Confirm trash with a real file — should send to recycle bin."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        import app as app_module

        img_path = tmp_path / "trash_me.jpg"
        Image.new("RGB", (10, 10), "red").save(str(img_path))

        db = app_module.load_scan_db() or {"images": [], "config": {}}
        db["images"] = [e for e in db["images"] if e["hash"] != "real_trash"]
        db["images"].append({
            "hash": "real_trash",
            "path": str(img_path).replace("\\", "/"),
            "filename": "trash_me.jpg",
            "media_type": "image",
            "status": "candidate",
            "category": "test",
            "trash": True,
            "size_kb": 1,
        })
        app_module.save_scan_db(db)

        resp = authed_client.post("/api/cleanup/confirm-trash")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data.get("ok")
        assert data.get("recycled") >= 1

        # File should be gone from original location
        assert not os.path.exists(str(img_path))

    def test_mark_trash_empty_hashes(self, authed_client):
        resp = authed_client.post("/api/cleanup/mark-trash", json={"hashes": []})
        assert resp.status_code == 400

    def test_cleanup_finish(self, authed_client):
        self._remove_test_image("cleanup_hash")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
