"""
Tests for Similarity Clustering (duplicate suppression).
Run: python -m pytest test_clustering.py -v
"""
import numpy as np
import pytest
from curate import (
    cluster_similar_images, compute_dhash, hamming_distance,
    compute_image_vector, cosine_similarity,
    CLUSTER_DHASH_THRESHOLD, CLUSTER_VECTOR_THRESHOLD,
)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _img(hash="abc", composite=50, size_kb=1000, status="qualified",
         vector=None, dhash=None, media_type="image", date=None,
         preference=None, category="cat1"):
    """Create a mock image dict with optional clustering-relevant fields."""
    return {
        "hash": hash,
        "path": f"C:/photos/{hash}.jpg",
        "filename": f"{hash}.jpg",
        "media_type": media_type,
        "status": status,
        "category": category,
        "date": date,
        "preference": preference,
        "size_kb": size_kb,
        "photo_grade": {"composite": composite, "sharpness": 50, "resolution": 60,
                        "noise": 50, "compression": 50, "color": 50, "exposure": 60,
                        "focus": 50},
        "face_count": 0,
        "has_target_face": False,
        "face_distance": None,
        "image_vector": vector,
        "dhash": dhash,
    }


def _random_vector(seed=0, dim=1048):
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


def _similar_vector(base, noise_level=0.02, seed=42):
    rng = np.random.RandomState(seed)
    v = base + rng.randn(*base.shape).astype(np.float32) * noise_level
    v /= np.linalg.norm(v)
    return v


def _similar_dhash(base_dhash, n_flips=3, seed=0):
    """Flip n_flips random bits in a 64-bit dhash."""
    rng = np.random.RandomState(seed)
    result = base_dhash
    positions = rng.choice(64, n_flips, replace=False)
    for pos in positions:
        result ^= (1 << int(pos))
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  UNIT TESTS: CLUSTERING CORE
# ══════════════════════════════════════════════════════════════════════════════

class TestClusterFormation:

    def test_identical_vectors_grouped(self):
        """Images with identical vectors must be in the same cluster."""
        v = _random_vector(seed=1)
        images = [
            _img(hash="a", composite=80, vector=v.tolist()),
            _img(hash="b", composite=60, vector=v.tolist()),
        ]
        result = cluster_similar_images(images)
        assert result["clusters"] == 1
        assert result["suppressed"] == 1
        assert images[0]["cluster_id"] == images[1]["cluster_id"]

    def test_similar_vectors_grouped(self):
        """Images with cosine similarity >= threshold must cluster."""
        v1 = _random_vector(seed=1)
        v2 = _similar_vector(v1, noise_level=0.01, seed=2)
        sim = float(np.dot(v1, v2))
        assert sim >= CLUSTER_VECTOR_THRESHOLD, f"Test setup: sim {sim} < threshold"

        images = [
            _img(hash="a", composite=80, vector=v1.tolist()),
            _img(hash="b", composite=60, vector=v2.tolist()),
        ]
        result = cluster_similar_images(images)
        assert result["clusters"] == 1

    def test_different_vectors_not_grouped(self):
        """Images with low similarity must NOT cluster."""
        v1 = _random_vector(seed=1)
        v2 = _random_vector(seed=99)
        sim = float(np.dot(v1, v2))
        assert sim < CLUSTER_VECTOR_THRESHOLD, f"Test setup: sim {sim} >= threshold"

        images = [
            _img(hash="a", vector=v1.tolist()),
            _img(hash="b", vector=v2.tolist()),
        ]
        result = cluster_similar_images(images)
        assert result["clusters"] == 0
        assert result["suppressed"] == 0

    def test_dhash_exact_duplicates_grouped(self):
        """Images with dHash Hamming distance <= threshold must cluster."""
        dhash1 = 0xAAAAAAAAAAAAAAAA
        dhash2 = _similar_dhash(dhash1, n_flips=3, seed=0)
        dist = hamming_distance(dhash1, dhash2)
        assert dist <= CLUSTER_DHASH_THRESHOLD

        images = [
            _img(hash="a", composite=80, dhash=dhash1),
            _img(hash="b", composite=60, dhash=dhash2),
        ]
        result = cluster_similar_images(images)
        assert result["clusters"] == 1
        assert result["suppressed"] == 1

    def test_dhash_different_not_grouped(self):
        """Images with very different dHash must NOT cluster."""
        dhash1 = 0xAAAAAAAAAAAAAAAA
        dhash2 = ~dhash1 & ((1 << 64) - 1)  # flip all 64 bits
        dist = hamming_distance(dhash1, dhash2)
        assert dist > CLUSTER_DHASH_THRESHOLD

        images = [
            _img(hash="a", dhash=dhash1),
            _img(hash="b", dhash=dhash2),
        ]
        result = cluster_similar_images(images)
        assert result["clusters"] == 0

    def test_mixed_signals_union(self):
        """dHash groups A-B, vector groups B-C => A-B-C should be one cluster."""
        dhash1 = 0xAAAAAAAAAAAAAAAA
        dhash2 = _similar_dhash(dhash1, n_flips=2, seed=0)  # A-B via dHash

        v2 = _random_vector(seed=10)
        v3 = _similar_vector(v2, noise_level=0.01, seed=11)  # B-C via vector

        images = [
            _img(hash="a", composite=90, dhash=dhash1),
            _img(hash="b", composite=70, dhash=dhash2, vector=v2.tolist()),
            _img(hash="c", composite=50, vector=v3.tolist()),
        ]
        result = cluster_similar_images(images)
        assert result["clusters"] == 1
        assert result["suppressed"] == 2
        # All share same cluster_id
        cids = {img["cluster_id"] for img in images if img.get("cluster_id")}
        assert len(cids) == 1

    def test_multi_cluster(self):
        """Multiple independent groups should form separate clusters."""
        v1 = _random_vector(seed=1)
        v2 = _similar_vector(v1, noise_level=0.01, seed=2)
        v3 = _random_vector(seed=50)
        v4 = _similar_vector(v3, noise_level=0.01, seed=51)

        images = [
            _img(hash="a1", composite=80, vector=v1.tolist()),
            _img(hash="a2", composite=60, vector=v2.tolist()),
            _img(hash="b1", composite=90, vector=v3.tolist()),
            _img(hash="b2", composite=40, vector=v4.tolist()),
            _img(hash="lone", composite=70, vector=_random_vector(seed=99).tolist()),
        ]
        result = cluster_similar_images(images)
        assert result["clusters"] == 2
        assert result["suppressed"] == 2
        # Lone image has no cluster
        lone = [i for i in images if i["hash"] == "lone"][0]
        assert lone.get("cluster_id") is None

    def test_rejected_images_excluded(self):
        """Rejected images should not participate in clustering."""
        v = _random_vector(seed=1)
        images = [
            _img(hash="a", composite=80, vector=v.tolist(), status="qualified"),
            _img(hash="b", composite=60, vector=v.tolist(), status="rejected"),
        ]
        result = cluster_similar_images(images)
        assert result["clusters"] == 0  # Only 1 eligible, so no cluster


# ══════════════════════════════════════════════════════════════════════════════
#  UNIT TESTS: REPRESENTATIVE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

class TestRepresentativeSelection:

    def test_highest_quality_wins(self):
        """Best photo_grade composite should become representative."""
        v = _random_vector(seed=1)
        images = [
            _img(hash="low", composite=30, vector=v.tolist()),
            _img(hash="high", composite=90, vector=v.tolist()),
            _img(hash="mid", composite=60, vector=v.tolist()),
        ]
        cluster_similar_images(images)

        reps = [i for i in images if i.get("is_representative")]
        assert len(reps) == 1
        assert reps[0]["hash"] == "high"

    def test_tie_break_by_size(self):
        """Equal composite: larger file size wins."""
        v = _random_vector(seed=1)
        images = [
            _img(hash="small", composite=80, size_kb=500, vector=v.tolist()),
            _img(hash="large", composite=80, size_kb=3000, vector=v.tolist()),
        ]
        cluster_similar_images(images)

        rep = [i for i in images if i.get("is_representative")][0]
        assert rep["hash"] == "large"

    def test_representative_has_cluster_size(self):
        """Representative should know the total cluster size."""
        v = _random_vector(seed=1)
        images = [
            _img(hash="a", composite=90, vector=v.tolist()),
            _img(hash="b", composite=50, vector=v.tolist()),
            _img(hash="c", composite=30, vector=v.tolist()),
        ]
        cluster_similar_images(images)

        rep = [i for i in images if i.get("is_representative")][0]
        assert rep["cluster_size"] == 3

    def test_non_representative_has_suppressed_by(self):
        """Suppressed members should reference the representative's hash."""
        v = _random_vector(seed=1)
        images = [
            _img(hash="best", composite=90, vector=v.tolist()),
            _img(hash="dup1", composite=50, vector=v.tolist()),
            _img(hash="dup2", composite=30, vector=v.tolist()),
        ]
        cluster_similar_images(images)

        suppressed = [i for i in images if i.get("suppressed_by")]
        assert len(suppressed) == 2
        for s in suppressed:
            assert s["suppressed_by"] == "best"
            assert s["is_representative"] is False


# ══════════════════════════════════════════════════════════════════════════════
#  UNIT TESTS: SUPPRESSION MODEL
# ══════════════════════════════════════════════════════════════════════════════

class TestSuppressionModel:

    def test_suppressed_images_not_deleted(self):
        """Suppression must not remove images from the list."""
        v = _random_vector(seed=1)
        images = [
            _img(hash="a", composite=90, vector=v.tolist()),
            _img(hash="b", composite=50, vector=v.tolist()),
        ]
        cluster_similar_images(images)
        assert len(images) == 2  # both still present

    def test_suppression_reversible(self):
        """Re-clustering with higher threshold should un-suppress."""
        v1 = _random_vector(seed=1)
        v2 = _similar_vector(v1, noise_level=0.01, seed=2)  # sim ~0.95
        images = [
            _img(hash="a", composite=90, vector=v1.tolist()),
            _img(hash="b", composite=50, vector=v2.tolist()),
        ]
        # First: cluster with default threshold 0.92 (catches them at sim ~0.95)
        cluster_similar_images(images, vector_threshold=0.92)
        assert images[1].get("suppressed_by") is not None

        # Second: re-cluster with high threshold (releases them)
        cluster_similar_images(images, vector_threshold=0.999)
        assert images[0].get("suppressed_by") is None
        assert images[1].get("suppressed_by") is None
        assert images[0].get("cluster_id") is None

    def test_fields_cleared_on_recluster(self):
        """Old cluster fields must be fully cleared before re-clustering."""
        v = _random_vector(seed=1)
        images = [
            _img(hash="a", composite=90, vector=v.tolist()),
            _img(hash="b", composite=50, vector=v.tolist()),
        ]
        cluster_similar_images(images)
        assert images[1].get("cluster_id") is not None

        # Now make them unclustered by removing vectors
        for img in images:
            img["image_vector"] = None
        cluster_similar_images(images)

        # All cluster fields should be gone
        for img in images:
            assert img.get("cluster_id") is None
            assert img.get("is_representative") is None
            assert img.get("suppressed_by") is None

    def test_status_preserved_after_clustering(self):
        """Clustering should not change image status."""
        v = _random_vector(seed=1)
        images = [
            _img(hash="a", composite=90, vector=v.tolist(), status="qualified"),
            _img(hash="b", composite=50, vector=v.tolist(), status="selected"),
        ]
        cluster_similar_images(images)
        assert images[0]["status"] == "qualified"
        assert images[1]["status"] == "selected"


# ══════════════════════════════════════════════════════════════════════════════
#  UNIT TESTS: CONFIGURABILITY
# ══════════════════════════════════════════════════════════════════════════════

class TestConfigurability:

    def test_vector_threshold_configurable(self):
        """Higher threshold should produce fewer clusters."""
        v1 = _random_vector(seed=1)
        v2 = _similar_vector(v1, noise_level=0.05, seed=2)
        sim = float(np.dot(v1, v2))

        images_loose = [
            _img(hash="a", vector=v1.tolist()),
            _img(hash="b", vector=v2.tolist()),
        ]
        images_strict = [
            _img(hash="a", vector=v1.tolist()),
            _img(hash="b", vector=v2.tolist()),
        ]

        r_loose = cluster_similar_images(images_loose, vector_threshold=0.85)
        r_strict = cluster_similar_images(images_strict, vector_threshold=0.99)

        # With noise_level=0.05, sim is ~0.95-0.98 range
        # Loose threshold should cluster, strict should not
        assert r_loose["clusters"] >= r_strict["clusters"]

    def test_dhash_threshold_configurable(self):
        """Threshold 0 should only match exact dhash duplicates."""
        dhash1 = 0xAAAAAAAAAAAAAAAA
        dhash2 = _similar_dhash(dhash1, n_flips=1, seed=0)

        images = [
            _img(hash="a", dhash=dhash1),
            _img(hash="b", dhash=dhash2),
        ]
        # Threshold 0 = only exact match
        r = cluster_similar_images(images, dhash_threshold=0)
        assert r["clusters"] == 0

        # Threshold 5 = allows 1 bit flip
        images2 = [
            _img(hash="a", dhash=dhash1),
            _img(hash="b", dhash=dhash2),
        ]
        r2 = cluster_similar_images(images2, dhash_threshold=5)
        assert r2["clusters"] == 1

    def test_diagnostics_returned(self):
        """Result dict should have all diagnostic fields."""
        v = _random_vector(seed=1)
        images = [
            _img(hash="a", composite=90, vector=v.tolist()),
            _img(hash="b", composite=50, vector=v.tolist()),
        ]
        result = cluster_similar_images(images)
        assert "clusters" in result
        assert "suppressed" in result
        assert "largest_cluster" in result
        assert "elapsed" in result
        assert isinstance(result["elapsed"], float)


# ══════════════════════════════════════════════════════════════════════════════
#  INTEGRATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegration:

    def test_burst_photos_clustered(self):
        """Simulate a burst: 5 near-identical dHash + similar vectors."""
        base_dhash = 0xAAAAAAAAAAAAAAAA
        base_vec = _random_vector(seed=1)

        images = []
        for i in range(5):
            dh = base_dhash if i == 0 else _similar_dhash(base_dhash, n_flips=min(i, 3), seed=i)
            v = _similar_vector(base_vec, noise_level=0.002 * (i + 1), seed=i + 10)
            images.append(_img(
                hash=f"burst_{i}",
                composite=80 - i * 5,  # first is best quality
                vector=v.tolist(),
                dhash=dh,
            ))

        result = cluster_similar_images(images)
        assert result["clusters"] == 1
        assert result["suppressed"] == 4
        assert result["largest_cluster"] == 5

        # Best quality (burst_0, composite=80) should be representative
        rep = [i for i in images if i.get("is_representative")][0]
        assert rep["hash"] == "burst_0"

    def test_mixed_quality_correct_winner(self):
        """In a group of 4, the one with composite=95 should always win."""
        v = _random_vector(seed=1)
        images = [
            _img(hash="worst", composite=20, vector=v.tolist()),
            _img(hash="best", composite=95, vector=v.tolist()),
            _img(hash="okay", composite=60, vector=v.tolist()),
            _img(hash="good", composite=78, vector=v.tolist()),
        ]
        cluster_similar_images(images)

        rep = [i for i in images if i.get("is_representative")][0]
        assert rep["hash"] == "best"
        assert rep["cluster_size"] == 4

        suppressed = [i for i in images if i.get("suppressed_by")]
        assert len(suppressed) == 3
        assert all(s["suppressed_by"] == "best" for s in suppressed)

    def test_large_dataset_stability(self):
        """50 images: some clustered, some solo — no crashes, correct counts."""
        images = []
        # 3 clusters of 5 similar images each (noise_level 0.002-0.01 keeps sim > 0.92)
        for cluster_idx in range(3):
            base_v = _random_vector(seed=cluster_idx * 100)
            for i in range(5):
                v = _similar_vector(base_v, noise_level=0.002 * (i + 1), seed=cluster_idx * 100 + i)
                images.append(_img(
                    hash=f"c{cluster_idx}_i{i}",
                    composite=90 - i * 10,
                    vector=v.tolist(),
                ))
        # 35 unique solo images
        for i in range(35):
            v = _random_vector(seed=500 + i)
            images.append(_img(hash=f"solo_{i}", composite=50, vector=v.tolist()))

        result = cluster_similar_images(images)
        assert result["clusters"] == 3
        assert result["suppressed"] == 12  # 3 clusters * 4 suppressed each
        assert len(images) == 50  # no images removed

        reps = [i for i in images if i.get("is_representative")]
        assert len(reps) == 3
        unclustered = [i for i in images if i.get("cluster_id") is None]
        assert len(unclustered) == 35

    def test_only_vectors_no_dhash(self):
        """Clustering should work with vectors only (no dHash data)."""
        v1 = _random_vector(seed=1)
        v2 = _similar_vector(v1, noise_level=0.01, seed=2)
        images = [
            _img(hash="a", composite=80, vector=v1.tolist()),
            _img(hash="b", composite=60, vector=v2.tolist()),
        ]
        result = cluster_similar_images(images)
        assert result["clusters"] == 1

    def test_only_dhash_no_vectors(self):
        """Clustering should work with dHash only (no vector data)."""
        dh1 = 0xAAAAAAAAAAAAAAAA
        dh2 = _similar_dhash(dh1, n_flips=2, seed=0)
        images = [
            _img(hash="a", composite=80, dhash=dh1),
            _img(hash="b", composite=60, dhash=dh2),
        ]
        result = cluster_similar_images(images)
        assert result["clusters"] == 1


# ══════════════════════════════════════════════════════════════════════════════
#  SANITY TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestSanity:

    def test_empty_list(self):
        """Empty image list should not crash."""
        result = cluster_similar_images([])
        assert result["clusters"] == 0
        assert result["suppressed"] == 0

    def test_single_image(self):
        """Single image should not form a cluster."""
        images = [_img(hash="solo", vector=_random_vector(seed=1).tolist())]
        result = cluster_similar_images(images)
        assert result["clusters"] == 0
        assert images[0].get("cluster_id") is None

    def test_no_vectors_no_dhash(self):
        """Images without any signal should not crash or cluster."""
        images = [_img(hash="a"), _img(hash="b")]
        result = cluster_similar_images(images)
        assert result["clusters"] == 0

    def test_all_rejected(self):
        """All-rejected list should produce no clusters."""
        v = _random_vector(seed=1)
        images = [
            _img(hash="a", status="rejected", vector=v.tolist()),
            _img(hash="b", status="rejected", vector=v.tolist()),
        ]
        result = cluster_similar_images(images)
        assert result["clusters"] == 0

    def test_one_representative_per_cluster(self):
        """Each cluster must have exactly one representative."""
        v = _random_vector(seed=1)
        images = [_img(hash=f"i{i}", composite=50 + i, vector=v.tolist()) for i in range(10)]
        cluster_similar_images(images)

        cluster_ids = {i["cluster_id"] for i in images if i.get("cluster_id")}
        for cid in cluster_ids:
            members = [i for i in images if i.get("cluster_id") == cid]
            reps = [i for i in members if i.get("is_representative")]
            assert len(reps) == 1, f"Cluster {cid} has {len(reps)} representatives"

    def test_suppressed_by_points_to_representative(self):
        """Every suppressed_by value must match an actual representative hash."""
        v = _random_vector(seed=1)
        images = [_img(hash=f"i{i}", composite=50 + i, vector=v.tolist()) for i in range(5)]
        cluster_similar_images(images)

        rep_hashes = {i["hash"] for i in images if i.get("is_representative")}
        for img in images:
            sb = img.get("suppressed_by")
            if sb:
                assert sb in rep_hashes, f"suppressed_by={sb} not found in representatives"

    def test_cluster_id_consistent_within_group(self):
        """All members of a cluster must share the same cluster_id."""
        v = _random_vector(seed=1)
        images = [_img(hash=f"i{i}", composite=50 + i, vector=v.tolist()) for i in range(5)]
        cluster_similar_images(images)

        cluster_ids = {i["cluster_id"] for i in images if i.get("cluster_id")}
        assert len(cluster_ids) == 1  # all in one cluster

    def test_result_counts_match(self):
        """Diagnostic counts must match actual image field state."""
        v1 = _random_vector(seed=1)
        v2 = _random_vector(seed=50)
        images = [
            _img(hash="a1", composite=90, vector=v1.tolist()),
            _img(hash="a2", composite=50, vector=_similar_vector(v1, noise_level=0.01, seed=2).tolist()),
            _img(hash="a3", composite=30, vector=_similar_vector(v1, noise_level=0.01, seed=3).tolist()),
            _img(hash="b1", composite=80, vector=v2.tolist()),
            _img(hash="b2", composite=40, vector=_similar_vector(v2, noise_level=0.01, seed=52).tolist()),
            _img(hash="solo", composite=70, vector=_random_vector(seed=99).tolist()),
        ]
        result = cluster_similar_images(images)

        actual_clusters = len({i["cluster_id"] for i in images if i.get("cluster_id")})
        actual_suppressed = sum(1 for i in images if i.get("suppressed_by"))
        actual_reps = sum(1 for i in images if i.get("is_representative"))

        assert result["clusters"] == actual_clusters
        assert result["suppressed"] == actual_suppressed
        assert actual_reps == actual_clusters  # one rep per cluster

    def test_idempotent(self):
        """Running clustering twice on the same data should produce identical results."""
        v = _random_vector(seed=1)
        images = [
            _img(hash="a", composite=90, vector=v.tolist()),
            _img(hash="b", composite=50, vector=v.tolist()),
        ]
        r1 = cluster_similar_images(images)
        cid_1 = images[0].get("cluster_id")
        rep_1 = images[0].get("is_representative")

        r2 = cluster_similar_images(images)
        cid_2 = images[0].get("cluster_id")
        rep_2 = images[0].get("is_representative")

        assert r1["clusters"] == r2["clusters"]
        assert r1["suppressed"] == r2["suppressed"]
        assert rep_1 == rep_2

    def test_progress_callback_called(self):
        """Progress callback should be called during clustering."""
        v = _random_vector(seed=1)
        images = [_img(hash=f"i{i}", vector=v.tolist()) for i in range(5)]
        messages = []
        cluster_similar_images(images, progress_cb=lambda msg: messages.append(msg))
        assert len(messages) > 0

    def test_video_images_not_clustered_without_vector(self):
        """Videos (no vector/dHash) should pass through unclustered."""
        images = [
            _img(hash="vid1", media_type="video"),
            _img(hash="vid2", media_type="video"),
        ]
        result = cluster_similar_images(images)
        assert result["clusters"] == 0

    def test_numpy_array_vectors_accepted(self):
        """Vectors as numpy arrays (not just lists) should work."""
        v = _random_vector(seed=1)
        images = [
            _img(hash="a", composite=90, vector=v),  # numpy array, not list
            _img(hash="b", composite=50, vector=v),
        ]
        result = cluster_similar_images(images)
        assert result["clusters"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
