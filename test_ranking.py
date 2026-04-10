"""
Tests for the Ranking Engine.
Run: python -m pytest test_ranking.py -v
"""
import numpy as np
import pytest
from ranking_engine import RankingEngine, MetadataPredictor


# ══════════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

def _img(
    hash="abc", preference=None, composite=50, sharpness=50,
    face_count=0, has_target_face=False, face_distance=None,
    size_kb=1000, media_type="image", date=None, location=None,
):
    """Create a mock image dict."""
    return {
        "hash": hash,
        "preference": preference,
        "photo_grade": {
            "composite": composite, "sharpness": sharpness,
            "resolution": 60, "noise": 50, "compression": 50,
            "color": 50, "exposure": 60, "focus": sharpness,
        },
        "face_count": face_count,
        "has_target_face": has_target_face,
        "face_distance": face_distance,
        "size_kb": size_kb,
        "media_type": media_type,
        "date": date,
        "location": location,
        "filename": f"{hash}.jpg",
        "path": f"C:/photos/{hash}.jpg",
    }


def _random_vector(seed=0):
    rng = np.random.RandomState(seed)
    v = rng.randn(1048).astype(np.float32)  # 32x32 + 24 hist bins
    v /= np.linalg.norm(v)
    return v


def _similar_vector(base, noise_level=0.05, seed=42):
    rng = np.random.RandomState(seed)
    v = base + rng.randn(*base.shape).astype(np.float32) * noise_level
    v /= np.linalg.norm(v)
    return v


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1: PREFERENCE FEEDBACK
# ══════════════════════════════════════════════════════════════════════════════

class TestPreferenceScoring:

    def test_liked_image_scores_highest(self):
        """Liked images should always score higher than unrated ones."""
        r = RankingEngine()
        liked = _img(hash="liked", preference="like", composite=30)
        unrated = _img(hash="unrated", composite=90)

        s_liked, _ = r.score(liked)
        s_unrated, _ = r.score(unrated)

        assert s_liked > s_unrated, f"Liked ({s_liked}) should beat unrated ({s_unrated})"

    def test_disliked_image_scores_lowest(self):
        """Disliked images should score lower than unrated ones."""
        r = RankingEngine()
        disliked = _img(hash="disliked", preference="dislike", composite=90)
        unrated = _img(hash="unrated", composite=30)

        s_disliked, _ = r.score(disliked)
        s_unrated, _ = r.score(unrated)

        assert s_disliked < s_unrated, f"Disliked ({s_disliked}) should be below unrated ({s_unrated})"

    def test_liked_beats_disliked(self):
        """Liked always beats disliked regardless of quality."""
        r = RankingEngine()
        liked = _img(hash="liked", preference="like", composite=10)
        disliked = _img(hash="disliked", preference="dislike", composite=100)

        s_liked, _ = r.score(liked)
        s_disliked, _ = r.score(disliked)

        assert s_liked > s_disliked

    def test_visual_similarity_to_liked(self):
        """Images visually similar to liked ones should score higher."""
        r = RankingEngine()
        v_liked = _random_vector(seed=1)
        v_similar = _similar_vector(v_liked, noise_level=0.02, seed=2)
        v_different = _random_vector(seed=99)

        # Learn from a liked image
        liked = _img(hash="liked", preference="like")
        r.learn_from_feedback([liked], vector_lookup={"liked": v_liked})

        similar = _img(hash="similar")
        different = _img(hash="different")

        s_similar, _ = r.score(similar, vector=v_similar)
        s_different, _ = r.score(different, vector=v_different)

        assert s_similar > s_different, "Similar to liked should rank higher"

    def test_visual_similarity_to_disliked_penalizes(self):
        """Images similar to disliked ones should be penalized."""
        r = RankingEngine()
        v_disliked = _random_vector(seed=1)
        v_similar = _similar_vector(v_disliked, noise_level=0.02, seed=2)
        v_different = _random_vector(seed=99)

        disliked = _img(hash="disliked", preference="dislike")
        r.learn_from_feedback([disliked], vector_lookup={"disliked": v_disliked})

        similar = _img(hash="similar")
        different = _img(hash="different")

        s_similar, _ = r.score(similar, vector=v_similar)
        s_different, _ = r.score(different, vector=v_different)

        assert s_similar < s_different, "Similar to disliked should rank lower"

    def test_session_update_incremental(self):
        """update_feedback should shift preference without full retrain."""
        r = RankingEngine()
        v1 = _random_vector(seed=1)
        v2 = _similar_vector(v1, noise_level=0.01, seed=2)

        assert r._session_vector is None

        r.update_feedback(_img(), "like", vector=v1)
        assert r._session_vector is not None
        assert r._n_likes == 1

        r.update_feedback(_img(), "like", vector=v2)
        assert r._n_likes == 2


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2: QUALITY SEPARATION
# ══════════════════════════════════════════════════════════════════════════════

class TestQualitySeparation:

    def test_quality_does_not_override_preference(self):
        """High quality should NOT beat liked low quality."""
        r = RankingEngine()
        liked_bad = _img(hash="lb", preference="like", composite=20)
        unrated_great = _img(hash="ug", composite=95)

        s_liked, bd_liked = r.score(liked_bad)
        s_unrated, bd_unrated = r.score(unrated_great)

        assert s_liked > s_unrated
        # Verify quality component is separate in breakdown
        assert bd_liked["quality"] < bd_unrated["quality"]
        assert bd_liked["preference"] > bd_unrated["preference"]

    def test_quality_score_range(self):
        """Quality score should be 0-100."""
        r = RankingEngine()
        low = _img(composite=10)
        high = _img(composite=95)

        _, bd_low = r.score(low)
        _, bd_high = r.score(high)

        assert 0 <= bd_low["quality"] <= 100
        assert 0 <= bd_high["quality"] <= 100
        assert bd_high["quality"] > bd_low["quality"]


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 3: DUPLICATE SUPPRESSION
# ══════════════════════════════════════════════════════════════════════════════

class TestDuplicateSuppression:

    def test_true_duplicate_killed(self):
        """dHash distance ≤ 5 should get maximum penalty."""
        r = RankingEngine()
        dhash1 = 0b10110010_11001010_10101010_11110000_00001111_10101010_01010101_11001100
        # Flip 3 bits (distance = 3)
        dhash2 = 0b10110010_11001010_10101010_11110000_00001111_10101010_01010101_11001001

        img1 = _img(hash="img1")
        img2 = _img(hash="img2")

        r.score(img1, dhash=dhash1)
        r.register_selected(img1, dhash=dhash1)

        _, bd = r.score(img2, dhash=dhash2)

        assert bd["duplicate_penalty"] >= 100, "True duplicate should be killed"

    def test_near_duplicate_penalized(self):
        """dHash distance 6-10 should get graduated penalty."""
        r = RankingEngine()
        dhash1 = 0b10110010_11001010_10101010_11110000_00001111_10101010_01010101_11001100
        # Flip 8 bits
        dhash2 = dhash1 ^ 0xFF

        img1 = _img(hash="img1")
        r.register_selected(img1, dhash=dhash1)

        _, bd = r.score(_img(hash="img2"), dhash=dhash2)

        assert 0 < bd["duplicate_penalty"] < 100, "Near-duplicate should get partial penalty"

    def test_different_image_no_penalty(self):
        """Very different images should get no duplicate penalty."""
        r = RankingEngine()
        dhash1 = 0b10110010_11001010_10101010_11110000_00001111_10101010_01010101_11001100
        dhash2 = ~dhash1 & ((1 << 64) - 1)  # flip all bits

        r.register_selected(_img(), dhash=dhash1)

        _, bd = r.score(_img(), dhash=dhash2)
        assert bd["duplicate_penalty"] == 0

    def test_vector_similarity_penalty(self):
        """High vector cosine similarity should penalize."""
        r = RankingEngine()
        v1 = _random_vector(seed=1)
        v_dup = _similar_vector(v1, noise_level=0.005, seed=2)  # very similar
        v_diff = _random_vector(seed=99)

        r.register_selected(_img(), vector=v1)

        _, bd_dup = r.score(_img(), vector=v_dup)
        _, bd_diff = r.score(_img(), vector=v_diff)

        assert bd_dup["duplicate_penalty"] > bd_diff["duplicate_penalty"]


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 4: DIVERSITY + EXPLORATION
# ══════════════════════════════════════════════════════════════════════════════

class TestDiversityExploration:

    def test_diversity_penalizes_same_date_cluster(self):
        """Too many images from same date should reduce diversity score."""
        r = RankingEngine()

        # Select 5 images from same date
        for i in range(5):
            r.register_selected(_img(date="2025-06-15"))

        # New image from same date vs different date
        _, bd_same = r.score(_img(date="2025-06-15"))
        _, bd_diff = r.score(_img(date="2025-03-01"))

        assert bd_same["diversity"] < bd_diff["diversity"]

    def test_exploration_for_unrated(self):
        """Unrated images should get exploration bonus."""
        r = RankingEngine()
        unrated = _img(preference=None)
        rated = _img(preference="like")

        _, bd_unrated = r.score(unrated)
        _, bd_rated = r.score(rated)

        assert bd_unrated["exploration"] > bd_rated["exploration"]

    def test_diversity_penalizes_visual_similarity_to_selected(self):
        """Images similar to already-selected should get lower diversity."""
        r = RankingEngine()
        v1 = _random_vector(seed=1)
        v_similar = _similar_vector(v1, noise_level=0.05, seed=2)
        v_diff = _random_vector(seed=99)

        r.register_selected(_img(), vector=v1)

        _, bd_sim = r.score(_img(), vector=v_similar)
        _, bd_diff = r.score(_img(), vector=v_diff)

        assert bd_sim["diversity"] <= bd_diff["diversity"]


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 5: FINAL SCORE FORMULA
# ══════════════════════════════════════════════════════════════════════════════

class TestFinalScoreFormula:

    def test_score_breakdown_has_all_components(self):
        """Breakdown should contain all scoring components."""
        r = RankingEngine()
        _, bd = r.score(_img())

        expected_keys = {"preference", "quality", "diversity", "exploration",
                         "duplicate_penalty", "no_face_penalty", "final"}
        assert set(bd.keys()) == expected_keys

    def test_weights_are_configurable(self):
        """Custom weights should change the final score."""
        r_default = RankingEngine()
        r_pref = RankingEngine(weights={"preference": 0.9, "quality": 0.05,
                                         "diversity": 0.025, "exploration": 0.025})

        img = _img(preference="like", composite=30)

        s_default, _ = r_default.score(img)
        s_pref_heavy, _ = r_pref.score(img)

        # With higher preference weight, liked images score even higher
        assert s_pref_heavy > s_default

    def test_preference_weight_dominates(self):
        """With default weights (0.6 preference), preference should dominate quality."""
        r = RankingEngine()
        # Low quality but liked
        liked_bad = _img(preference="like", composite=20)
        # High quality but disliked
        disliked_good = _img(preference="dislike", composite=95)

        s_liked, _ = r.score(liked_bad)
        s_disliked, _ = r.score(disliked_good)

        assert s_liked > s_disliked, "Preference must dominate quality at default weights"


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 6: METADATA PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

class TestMetadataPredictor:

    def test_trains_on_rated(self):
        """Should train when ≥ 3 rated images exist."""
        pred = MetadataPredictor()
        images = [
            _img(preference="like", composite=80, sharpness=90),
            _img(preference="like", composite=75, sharpness=85),
            _img(preference="dislike", composite=20, sharpness=15),
            _img(preference="dislike", composite=25, sharpness=20),
        ]
        pred.train(images)
        assert pred._trained

    def test_no_train_with_few_ratings(self):
        """Should not train with < 3 ratings."""
        pred = MetadataPredictor()
        pred.train([_img(preference="like")])
        assert not pred._trained

    def test_predicts_like_for_similar(self):
        """High quality unrated should predict closer to 'like' after training on likes."""
        pred = MetadataPredictor()
        images = [
            _img(preference="like", composite=85, sharpness=90, has_target_face=True, face_distance=0.3),
            _img(preference="like", composite=80, sharpness=85, has_target_face=True, face_distance=0.35),
            _img(preference="like", composite=75, sharpness=80, has_target_face=True, face_distance=0.38),
            _img(preference="dislike", composite=20, sharpness=15, has_target_face=False),
            _img(preference="dislike", composite=25, sharpness=20, has_target_face=False),
        ]
        pred.train(images)

        # Similar to liked
        good = _img(composite=82, sharpness=88, has_target_face=True, face_distance=0.32)
        p_good = pred.predict_probability(good)

        # Similar to disliked
        bad = _img(composite=22, sharpness=18, has_target_face=False)
        p_bad = pred.predict_probability(bad)

        assert p_good > p_bad, f"Good ({p_good}) should predict higher than bad ({p_bad})"
        assert p_good > 0.5, f"Good should predict > 0.5, got {p_good}"


# ══════════════════════════════════════════════════════════════════════════════
#  INTEGRATION: END-TO-END RANKING
# ══════════════════════════════════════════════════════════════════════════════

class TestEndToEnd:

    def test_full_ranking_order(self):
        """Full pipeline: liked > predicted-like > neutral > predicted-dislike > disliked."""
        r = RankingEngine()

        # Training data
        train_imgs = [
            _img(hash="t1", preference="like", composite=80, sharpness=85, has_target_face=True, face_distance=0.3),
            _img(hash="t2", preference="like", composite=75, sharpness=80, has_target_face=True, face_distance=0.35),
            _img(hash="t3", preference="dislike", composite=20, sharpness=15),
            _img(hash="t4", preference="dislike", composite=25, sharpness=20),
        ]
        r.learn_from_feedback(train_imgs)

        # Test images
        liked = _img(hash="liked", preference="like", composite=50)
        good_unrated = _img(hash="good", composite=80, sharpness=85, has_target_face=True, face_distance=0.32)
        neutral = _img(hash="neutral", composite=50)
        bad_unrated = _img(hash="bad", composite=22, sharpness=18)
        disliked = _img(hash="disliked", preference="dislike", composite=50)

        s_liked, _ = r.score(liked)
        s_good, _ = r.score(good_unrated)
        s_neutral, _ = r.score(neutral)
        s_bad, _ = r.score(bad_unrated)
        s_disliked, _ = r.score(disliked)

        assert s_liked > s_good, "Liked > predicted-like"
        assert s_good > s_neutral, "Predicted-like > neutral"
        assert s_neutral > s_bad, "Neutral > predicted-dislike"
        assert s_bad > s_disliked, "Predicted-dislike > disliked"

    def test_dedup_across_selection(self):
        """Selecting an image should penalize near-duplicates for subsequent scoring."""
        r = RankingEngine()
        v1 = _random_vector(seed=1)
        v_dup = _similar_vector(v1, noise_level=0.01, seed=2)

        img1 = _img(hash="img1", composite=80)
        img2 = _img(hash="img2", composite=80)

        # Score before any selection — should be similar
        s1_before, _ = r.score(img1, vector=v1)
        s2_before, _ = r.score(img2, vector=v_dup)

        # Select img1
        r.register_selected(img1, vector=v1)

        # Re-score img2 — should be penalized
        s2_after, bd = r.score(img2, vector=v_dup)

        assert s2_after < s2_before, "Near-duplicate should score lower after original selected"
        assert bd["duplicate_penalty"] > 0


# ══════════════════════════════════════════════════════════════════════════════
#  SANITY TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestSanity:
    """Basic invariants that must always hold regardless of configuration."""

    def test_score_is_finite(self):
        """Score must never be NaN or infinity."""
        r = RankingEngine()
        for pref in (None, "like", "dislike"):
            s, bd = r.score(_img(preference=pref))
            assert np.isfinite(s), f"Score not finite for preference={pref}: {s}"
            for key, val in bd.items():
                assert np.isfinite(val), f"Breakdown[{key}] not finite: {val}"

    def test_deterministic_scoring(self):
        """Same input must always produce the same score."""
        r = RankingEngine()
        img = _img(hash="det", composite=65, sharpness=70)
        s1, _ = r.score(img)
        s2, _ = r.score(img)
        assert s1 == s2

    def test_empty_image_does_not_crash(self):
        """Minimal image dict with defaults should not crash."""
        r = RankingEngine()
        s, bd = r.score({"hash": "x", "filename": "x.jpg"})
        assert np.isfinite(s)

    def test_hash_formats(self):
        """Various hash formats: hex, names, empty, unicode — all must work."""
        r = RankingEngine()
        hashes = ["abc123def456", "my_photo", "", "café", "0" * 64, "a"]
        for h in hashes:
            s, bd = r.score(_img(hash=h))
            assert np.isfinite(s), f"Crashed on hash={h!r}"

    def test_video_scores_without_crash(self):
        """Videos should score like images — no special crash path."""
        r = RankingEngine()
        vid = _img(media_type="video", composite=60)
        s, bd = r.score(vid)
        assert np.isfinite(s)
        assert set(bd.keys()) == {"preference", "quality", "diversity",
                                   "exploration", "duplicate_penalty",
                                   "no_face_penalty", "final"}

    def test_component_ranges(self):
        """Each component must stay within its documented range."""
        r = RankingEngine()
        for pref in (None, "like", "dislike"):
            for comp in (10, 50, 95):
                _, bd = r.score(_img(preference=pref, composite=comp))
                assert 0 <= bd["preference"] <= 100, f"preference OOB: {bd['preference']}"
                assert 0 <= bd["quality"] <= 100, f"quality OOB: {bd['quality']}"
                assert -100 <= bd["diversity"] <= 100, f"diversity OOB: {bd['diversity']}"
                assert 0 <= bd["exploration"] <= 100, f"exploration OOB: {bd['exploration']}"
                assert 0 <= bd["duplicate_penalty"] <= 100, f"dup_penalty OOB: {bd['duplicate_penalty']}"

    def test_duplicate_penalty_caps_at_100(self):
        """Even extreme duplicates should not exceed penalty of 100."""
        r = RankingEngine()
        dhash = 0xAAAAAAAAAAAAAAAA
        r.register_selected(_img(), dhash=dhash)
        # Exact same dhash → distance 0
        _, bd = r.score(_img(), dhash=dhash)
        assert bd["duplicate_penalty"] == 100

    def test_zero_vector_does_not_crash(self):
        """A zero vector (degenerate) should not produce NaN or crash."""
        r = RankingEngine()
        zero_v = np.zeros(1048, dtype=np.float32)
        s, bd = r.score(_img(), vector=zero_v)
        assert np.isfinite(s)

    def test_register_selected_updates_state(self):
        """register_selected must grow internal tracking lists."""
        r = RankingEngine()
        assert len(r._selected_vectors) == 0
        assert len(r._selected_dhashes) == 0
        assert len(r._selected_times) == 0

        v = _random_vector(seed=5)
        r.register_selected(_img(date="2025-01-01"), vector=v, dhash=0xFF)

        assert len(r._selected_vectors) == 1
        assert len(r._selected_dhashes) == 1
        assert len(r._selected_times) == 1

    def test_custom_weights_override(self):
        """Passing weights should override defaults, not ignore them."""
        r = RankingEngine(weights={"quality": 0.99})
        assert r.weights["quality"] == 0.99
        # Other defaults remain
        assert r.weights["preference"] == 0.60

    def test_learn_with_no_vectors(self):
        """learn_from_feedback with no vector_lookup should not crash."""
        r = RankingEngine()
        imgs = [
            _img(preference="like", composite=80),
            _img(preference="dislike", composite=20),
        ]
        r.learn_from_feedback(imgs)  # no vector_lookup
        assert r._session_vector is None
        assert r._n_likes == 1

    def test_score_after_many_selections(self):
        """Scoring should remain stable after many register_selected calls."""
        r = RankingEngine()
        for i in range(50):
            v = _random_vector(seed=i)
            r.register_selected(_img(date=f"2025-01-{(i % 28) + 1:02d}"), vector=v, dhash=i)

        s, bd = r.score(_img(composite=60))
        assert np.isfinite(s)
        assert 0 <= bd["duplicate_penalty"] <= 100

    def test_liked_always_above_disliked_any_quality(self):
        """Preference ordering must hold across the full quality spectrum."""
        r = RankingEngine()
        for q_like in (10, 30, 50):
            for q_dislike in (50, 70, 100):
                liked = _img(preference="like", composite=q_like)
                disliked = _img(preference="dislike", composite=q_dislike)
                s_l, _ = r.score(liked)
                s_d, _ = r.score(disliked)
                assert s_l > s_d, (
                    f"liked(q={q_like})={s_l:.1f} should beat "
                    f"disliked(q={q_dislike})={s_d:.1f}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
