"""
Tests for Video Frame Analysis.
Run: python -m pytest test_video_analysis.py -v

Note: Tests that require actual video files are marked with @pytest.mark.skipif
and will only run when test videos are available.
Tests for the analysis functions use synthetic PIL images directly.
"""
import numpy as np
import pytest
from PIL import Image
from io import BytesIO
import os
import tempfile

from curate import (
    extract_video_frames, analyze_video_frames,
    compute_image_vector, compute_dhash,
)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _make_test_image(w=640, h=480, color=(128, 100, 80), seed=0):
    """Create a test PIL image with some texture (not flat) for meaningful analysis."""
    rng = np.random.RandomState(seed)
    arr = np.full((h, w, 3), color, dtype=np.uint8)
    # Add some texture/noise for non-zero sharpness
    noise = rng.randint(-20, 20, (h, w, 3), dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _make_test_video(filepath, n_frames=30, w=320, h=240, fps=30, seed=0):
    """Create a minimal test video file using OpenCV."""
    try:
        import cv2
    except ImportError:
        pytest.skip("OpenCV not available for video creation")
    rng = np.random.RandomState(seed)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, fps, (w, h))
    for i in range(n_frames):
        # Gradually changing color + texture
        base_color = [80 + i * 2, 100, 120 - i]
        frame = np.full((h, w, 3), base_color, dtype=np.uint8)
        noise = rng.randint(-15, 15, (h, w, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        out.write(frame)
    out.release()
    return filepath


def _make_test_video_pattern(filepath, n_frames=30, w=320, h=240, fps=30, pattern="checkerboard"):
    """Create a test video with a specific spatial pattern for distinct visual content."""
    try:
        import cv2
    except ImportError:
        pytest.skip("OpenCV not available for video creation")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, fps, (w, h))
    for i in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        if pattern == "checkerboard":
            # 32x32 pixel checkerboard — very distinct spatial frequency
            block = 32
            for r in range(0, h, block):
                for c in range(0, w, block):
                    if ((r // block) + (c // block)) % 2 == 0:
                        frame[r:r+block, c:c+block] = [220, 220, 220]
                    else:
                        frame[r:r+block, c:c+block] = [30, 30, 30]
        elif pattern == "diagonal":
            # Diagonal stripes — fundamentally different from checkerboard
            for r in range(h):
                for c in range(w):
                    if ((r + c) // 20) % 2 == 0:
                        frame[r, c] = [200, 50, 50]
                    else:
                        frame[r, c] = [50, 50, 200]
        out.write(frame)
    out.release()
    return filepath


def _make_test_video_colored(filepath, n_frames=30, w=320, h=240, fps=30, color=(128, 128, 128)):
    """Create a test video with a specific dominant color + gradient pattern."""
    try:
        import cv2
    except ImportError:
        pytest.skip("OpenCV not available for video creation")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, fps, (w, h))
    for i in range(n_frames):
        # Create a gradient pattern from the color to its complement
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        for row in range(h):
            frac = row / max(h - 1, 1)
            frame[row, :] = [
                int(color[0] * (1 - frac) + (255 - color[0]) * frac),
                int(color[1] * (1 - frac) + (255 - color[1]) * frac),
                int(color[2] * (1 - frac) + (255 - color[2]) * frac),
            ]
        out.write(frame)
    out.release()
    return filepath


# ══════════════════════════════════════════════════════════════════════════════
#  UNIT TESTS: extract_video_frames
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractFrames:

    def test_extracts_correct_number_of_frames(self):
        """Should extract requested number of frames (or fewer if video is short)."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_test_video(path, n_frames=60, fps=30)
            frames = extract_video_frames(path, n_frames=5)
            assert len(frames) == 5
            # Each should be (PIL Image, timestamp)
            for pil_img, ts in frames:
                assert isinstance(pil_img, Image.Image)
                assert isinstance(ts, (int, float))
                assert ts >= 0
        finally:
            os.unlink(path)

    def test_short_video_returns_fewer_frames(self):
        """A 10-frame video asked for 5 should return up to 5."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_test_video(path, n_frames=10, fps=30)
            frames = extract_video_frames(path, n_frames=5)
            # Should get some frames (at least 1, up to 5 depending on 5%/95% range)
            assert 1 <= len(frames) <= 5
        finally:
            os.unlink(path)

    def test_nonexistent_file_returns_empty(self):
        """Missing file should return empty list, not crash."""
        frames = extract_video_frames("/nonexistent/video.mp4")
        assert frames == []

    def test_corrupt_file_returns_empty(self):
        """Corrupt file should return empty list."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"this is not a video")
            path = f.name
        try:
            frames = extract_video_frames(path)
            assert frames == []
        finally:
            os.unlink(path)

    def test_frames_are_rgb(self):
        """Extracted frames should be RGB PIL images."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_test_video(path, n_frames=30)
            frames = extract_video_frames(path, n_frames=3)
            for pil_img, _ in frames:
                assert pil_img.mode == "RGB"
        finally:
            os.unlink(path)


# ══════════════════════════════════════════════════════════════════════════════
#  UNIT TESTS: analyze_video_frames
# ══════════════════════════════════════════════════════════════════════════════

class TestAnalyzeVideoFrames:

    def test_returns_all_fields(self):
        """Analysis should return all expected fields."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_test_video(path, n_frames=30)
            result = analyze_video_frames(path, n_frames=3)
            assert result is not None
            assert "face_count" in result
            assert "faces_found" in result
            assert "has_target_face" in result
            assert "face_distance" in result
            assert "photo_grade" in result
            assert "image_vector" in result
            assert "dhash" in result
            assert "best_thumb" in result
            assert "_analyzed_frames" in result
        finally:
            os.unlink(path)

    def test_produces_photo_grade(self):
        """Should compute photo_grade with all sub-scores."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_test_video(path, n_frames=30, w=640, h=480)
            result = analyze_video_frames(path, n_frames=3)
            assert result is not None
            grade = result["photo_grade"]
            assert grade is not None
            for key in ("resolution", "sharpness", "noise", "compression",
                        "color", "exposure", "focus", "composite"):
                assert key in grade, f"Missing grade key: {key}"
                assert 0 <= grade[key] <= 100, f"Grade {key}={grade[key]} out of range"
        finally:
            os.unlink(path)

    def test_produces_image_vector(self):
        """Should compute a valid image vector."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_test_video(path, n_frames=30)
            result = analyze_video_frames(path, n_frames=3)
            assert result is not None
            vec = result["image_vector"]
            assert vec is not None
            vec = np.array(vec) if isinstance(vec, list) else vec
            assert vec.shape == (1048,)  # 32*32 + 24 histogram bins
            # Should be unit-normalized
            norm = np.linalg.norm(vec)
            assert abs(norm - 1.0) < 0.01, f"Vector not unit-normalized: norm={norm}"
        finally:
            os.unlink(path)

    def test_produces_dhash(self):
        """Should compute a valid dHash."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_test_video(path, n_frames=30)
            result = analyze_video_frames(path, n_frames=3)
            assert result is not None
            dh = result["dhash"]
            assert dh is not None
            assert isinstance(dh, int)
            assert 0 <= dh < (1 << 64)
        finally:
            os.unlink(path)

    def test_produces_thumbnail(self):
        """Should produce a base64 thumbnail."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_test_video(path, n_frames=30)
            result = analyze_video_frames(path, n_frames=3)
            assert result is not None
            thumb = result["best_thumb"]
            assert thumb is not None
            assert len(thumb) > 100  # non-trivial base64 data
        finally:
            os.unlink(path)

    def test_no_faces_without_refs(self):
        """Without reference encodings, face count should be 0."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_test_video(path, n_frames=30)
            result = analyze_video_frames(path, ref_encodings=None, n_frames=3)
            assert result is not None
            assert result["face_count"] == 0
            assert result["has_target_face"] is False
        finally:
            os.unlink(path)

    def test_nonexistent_video_returns_none(self):
        """Missing video should return None."""
        result = analyze_video_frames("/nonexistent/video.mp4")
        assert result is None

    def test_analyzed_frames_count(self):
        """_analyzed_frames should reflect actual frames processed."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_test_video(path, n_frames=60)
            result = analyze_video_frames(path, n_frames=5)
            assert result is not None
            assert result["_analyzed_frames"] == 5
        finally:
            os.unlink(path)


# ══════════════════════════════════════════════════════════════════════════════
#  INTEGRATION: Video vectors work with clustering
# ══════════════════════════════════════════════════════════════════════════════

class TestVideoClusteringIntegration:

    def test_similar_videos_cluster(self):
        """Two videos from the same content should produce similar vectors and cluster."""
        from curate import cluster_similar_images

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f1, \
             tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f2:
            path1, path2 = f1.name, f2.name
        try:
            # Same seed = same content = should be very similar
            _make_test_video(path1, n_frames=30, seed=42)
            _make_test_video(path2, n_frames=30, seed=42)

            r1 = analyze_video_frames(path1, n_frames=3)
            r2 = analyze_video_frames(path2, n_frames=3)
            assert r1 is not None and r2 is not None

            v1 = np.array(r1["image_vector"], dtype=np.float32)
            v2 = np.array(r2["image_vector"], dtype=np.float32)
            sim = float(np.dot(v1, v2))
            assert sim > 0.9, f"Same-content videos should be similar, got {sim}"

            images = [
                {"hash": "vid1", "status": "qualified", "media_type": "video",
                 "photo_grade": {"composite": 60}, "size_kb": 1000,
                 "image_vector": r1["image_vector"], "dhash": r1["dhash"]},
                {"hash": "vid2", "status": "qualified", "media_type": "video",
                 "photo_grade": {"composite": 50}, "size_kb": 800,
                 "image_vector": r2["image_vector"], "dhash": r2["dhash"]},
            ]
            result = cluster_similar_images(images)
            assert result["clusters"] == 1
            assert result["suppressed"] == 1
        finally:
            os.unlink(path1)
            os.unlink(path2)

    def test_different_videos_dont_cluster(self):
        """Videos with very different content should not cluster."""
        from curate import cluster_similar_images

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f1, \
             tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f2:
            path1, path2 = f1.name, f2.name
        try:
            # Create videos with fundamentally different spatial patterns
            _make_test_video_pattern(path1, n_frames=30, pattern="checkerboard")
            _make_test_video_pattern(path2, n_frames=30, pattern="diagonal")

            r1 = analyze_video_frames(path1, n_frames=3)
            r2 = analyze_video_frames(path2, n_frames=3)
            assert r1 is not None and r2 is not None

            v1 = np.array(r1["image_vector"], dtype=np.float32)
            v2 = np.array(r2["image_vector"], dtype=np.float32)
            sim = float(np.dot(v1, v2))

            images = [
                {"hash": "vid1", "status": "qualified", "media_type": "video",
                 "photo_grade": {"composite": 60}, "size_kb": 1000,
                 "image_vector": r1["image_vector"], "dhash": r1["dhash"]},
                {"hash": "vid2", "status": "qualified", "media_type": "video",
                 "photo_grade": {"composite": 50}, "size_kb": 800,
                 "image_vector": r2["image_vector"], "dhash": r2["dhash"]},
            ]
            result = cluster_similar_images(images)
            assert result["clusters"] == 0, (
                f"Different-pattern videos clustered at sim={sim:.3f}"
            )
        finally:
            os.unlink(path1)
            os.unlink(path2)


# ══════════════════════════════════════════════════════════════════════════════
#  SANITY TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestSanity:

    def test_vector_compatible_with_image_vectors(self):
        """Video vectors should be the same dimensionality as image vectors."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_test_video(path, n_frames=30)
            result = analyze_video_frames(path, n_frames=3)
            assert result is not None
            vid_vec = np.array(result["image_vector"])

            # Create a test image and compute its vector
            img = _make_test_image(640, 480, seed=0)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as img_f:
                img_path = img_f.name
            img.save(img_path, "JPEG")
            img_vec = compute_image_vector(img_path)
            os.unlink(img_path)

            assert vid_vec.shape == img_vec.shape, (
                f"Video vector dim {vid_vec.shape} != image vector dim {img_vec.shape}"
            )
            # Both should be unit-normalized
            assert abs(np.linalg.norm(vid_vec) - 1.0) < 0.01
            assert abs(np.linalg.norm(img_vec) - 1.0) < 0.01
            # Cross-similarity should be a valid float
            sim = float(np.dot(vid_vec, img_vec))
            assert -1.0 <= sim <= 1.0
        finally:
            os.unlink(path)

    def test_dhash_compatible_with_image_dhash(self):
        """Video dHash should be same format as image dHash (64-bit int)."""
        from curate import hamming_distance

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_test_video(path, n_frames=30)
            result = analyze_video_frames(path, n_frames=3)
            assert result is not None
            vid_dhash = result["dhash"]

            # Create a test image dHash
            img = _make_test_image(640, 480, seed=0)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as img_f:
                img_path = img_f.name
            img.save(img_path, "JPEG")
            img_dhash = compute_dhash(img_path)
            os.unlink(img_path)

            # Should be comparable (both 64-bit ints)
            assert isinstance(vid_dhash, int)
            assert isinstance(img_dhash, int)
            dist = hamming_distance(vid_dhash, img_dhash)
            assert 0 <= dist <= 64
        finally:
            os.unlink(path)

    def test_grade_composite_reasonable_range(self):
        """Video composite grade should be in a reasonable range (not always 0 or 100)."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_test_video(path, n_frames=30, w=640, h=480)
            result = analyze_video_frames(path, n_frames=3)
            assert result is not None
            composite = result["photo_grade"]["composite"]
            assert 10 < composite < 90, f"Composite {composite} seems unreasonable"
        finally:
            os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
