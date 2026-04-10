"""
Ranking Engine — Modular image scoring for Quick Fill selection.

Scoring formula:
    final_score = W_PREF * preference_score
                + W_QUAL * quality_score
                + W_DIV  * diversity_score
                + W_EXPL * exploration_bonus
                - duplicate_penalty

Design principles:
    - User feedback dominates ranking
    - Quality is secondary (filter bad, don't let it override taste)
    - Visual similarity to liked/disliked images is a strong signal
    - Near-duplicates get soft penalty (graduated, not hard skip)
    - Diversity prevents repetitive results
    - Exploration surfaces uncertain-but-promising images
    - Session-level learning adapts within a single session
"""

import math
import numpy as np
from collections import defaultdict


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURABLE WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_WEIGHTS = {
    "preference": 0.60,
    "quality":    0.20,
    "diversity":  0.10,
    "exploration": 0.10,
}

# Score ranges (used for normalization)
SCORE_RANGES = {
    "preference": (-100, 100),
    "quality":    (0, 100),
    "diversity":  (-100, 100),
    "exploration": (0, 100),
}


# ══════════════════════════════════════════════════════════════════════════════
#  RANKING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class RankingEngine:
    """
    Modular scoring engine. Computes per-image scores with full breakdown.
    Learns from user feedback and adapts within a session.
    """

    def __init__(self, weights=None, face_names=None, is_manual_cat=False):
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.face_names = face_names or []
        self.is_manual_cat = is_manual_cat

        # Preference state
        self._pref_predictor = MetadataPredictor()
        self._session_vector = None       # running average of liked image vectors
        self._session_neg_vector = None   # running average of disliked image vectors
        self._liked_vectors = []          # visual vectors of liked images
        self._disliked_vectors = []       # visual vectors of disliked images
        self._n_likes = 0
        self._n_dislikes = 0

        # Diversity state (built up as images are selected)
        self._selected_vectors = []
        self._selected_clip_vectors = []  # CLIP 512-dim semantic vectors
        self._selected_times = []         # date strings of selected images
        self._selected_locations = []     # location strings of selected images

        # Duplicate state
        self._selected_dhashes = []

        # Debug log
        self._score_log = []

    # ──────────────────────────────────────────────────────────────────────
    #  INITIALIZATION — call before scoring
    # ──────────────────────────────────────────────────────────────────────

    def learn_from_feedback(self, images, vector_lookup=None):
        """
        Learn user preferences from all rated images.
        vector_lookup: dict mapping image hash → numpy vector (optional).
        """
        liked = [i for i in images if i.get("preference") == "like"]
        disliked = [i for i in images if i.get("preference") == "dislike"]

        self._n_likes = len(liked)
        self._n_dislikes = len(disliked)

        # Train metadata predictor (logistic regression on photo features)
        self._pref_predictor.train(images)

        # Build visual preference vectors (average of liked/disliked embeddings)
        if vector_lookup:
            like_vecs = []
            for img in liked:
                v = vector_lookup.get(img.get("hash"))
                if v is not None:
                    like_vecs.append(v)
            if like_vecs:
                self._session_vector = np.mean(like_vecs, axis=0)
                norm = np.linalg.norm(self._session_vector)
                if norm > 0:
                    self._session_vector /= norm
                self._liked_vectors = like_vecs

            dislike_vecs = []
            for img in disliked:
                v = vector_lookup.get(img.get("hash"))
                if v is not None:
                    dislike_vecs.append(v)
            if dislike_vecs:
                self._session_neg_vector = np.mean(dislike_vecs, axis=0)
                norm = np.linalg.norm(self._session_neg_vector)
                if norm > 0:
                    self._session_neg_vector /= norm
                self._disliked_vectors = dislike_vecs

    def update_feedback(self, img, preference, vector=None):
        """
        Lightweight incremental update after a single like/dislike.
        Updates session vectors without full retrain.
        """
        if preference == "like":
            self._n_likes += 1
            if vector is not None:
                self._liked_vectors.append(vector)
                # Incremental mean update
                if self._session_vector is None:
                    self._session_vector = vector.copy()
                else:
                    n = len(self._liked_vectors)
                    self._session_vector = (self._session_vector * (n - 1) + vector) / n
                    norm = np.linalg.norm(self._session_vector)
                    if norm > 0:
                        self._session_vector /= norm

        elif preference == "dislike":
            self._n_dislikes += 1
            if vector is not None:
                self._disliked_vectors.append(vector)
                if self._session_neg_vector is None:
                    self._session_neg_vector = vector.copy()
                else:
                    n = len(self._disliked_vectors)
                    self._session_neg_vector = (self._session_neg_vector * (n - 1) + vector) / n
                    norm = np.linalg.norm(self._session_neg_vector)
                    if norm > 0:
                        self._session_neg_vector /= norm

    # ──────────────────────────────────────────────────────────────────────
    #  MAIN SCORING — call per image
    # ──────────────────────────────────────────────────────────────────────

    def score(self, img, vector=None, dhash=None, clip_vector=None,
              quality_weights=None):
        """
        Compute final score with full breakdown.
        Returns (final_score, breakdown_dict).
        """
        breakdown = {}

        # ── Component 1: Preference Score (0-100 normalized) ──
        pref = self._compute_preference(img, vector)
        breakdown["preference"] = pref

        # ── Component 2: Quality Score (0-100 normalized) ──
        qual = self._compute_quality(img, quality_weights or {})
        breakdown["quality"] = qual

        # ── Component 3: Diversity Score (-100 to 100) ──
        div = self._compute_diversity(img, vector, clip_vector)
        breakdown["diversity"] = div

        # ── Component 4: Exploration Bonus (0-100) ──
        expl = self._compute_exploration(img)
        breakdown["exploration"] = expl

        # ── Duplicate Penalty (0-100) ──
        dup = self._compute_duplicate_penalty(vector, dhash)
        breakdown["duplicate_penalty"] = dup

        # ── No-face penalty for manual templates ──
        face_pen = 0
        if img.get("_no_face_penalty"):
            face_pen = 40  # strong but not absolute
        breakdown["no_face_penalty"] = face_pen

        # ── Weighted final score ──
        w = self.weights
        final = (
            w["preference"]  * pref
            + w["quality"]   * qual
            + w["diversity"] * div
            + w["exploration"] * expl
            - dup
            - face_pen
        )

        breakdown["final"] = final
        img["_score"] = final
        img["_score_breakdown"] = breakdown

        return final, breakdown

    def register_selected(self, img, vector=None, dhash=None, clip_vector=None):
        """Call after an image is selected, to update diversity/dedup state."""
        if vector is not None:
            self._selected_vectors.append(vector)
        if clip_vector is not None:
            self._selected_clip_vectors.append(clip_vector)
        if dhash is not None:
            self._selected_dhashes.append(dhash)
        date_str = img.get("date")
        if date_str:
            self._selected_times.append(date_str)
        loc = img.get("location")
        if loc:
            self._selected_locations.append(loc)

    # ──────────────────────────────────────────────────────────────────────
    #  COMPONENT SCORERS
    # ──────────────────────────────────────────────────────────────────────

    def _compute_preference(self, img, vector=None):
        """
        Preference score on 0-100 scale (50 = neutral).
        Combines: explicit like/dislike, visual similarity to liked/disliked,
        metadata prediction, face relevance.
        """
        pref = img.get("preference")

        # Explicit feedback: strongest signal
        if pref == "like":
            return 95.0
        if pref == "dislike":
            return 5.0

        score = 50.0  # neutral baseline

        # ── Visual similarity to liked/disliked images ──
        if vector is not None:
            # Similarity to session "like" vector
            if self._session_vector is not None:
                sim_like = float(np.dot(vector, self._session_vector))
                # sim_like is -1..1, scale to -20..+20
                score += sim_like * 20.0

            # Penalty for similarity to session "dislike" vector
            if self._session_neg_vector is not None:
                sim_dislike = float(np.dot(vector, self._session_neg_vector))
                # Only penalize positive similarity to disliked
                if sim_dislike > 0:
                    score -= sim_dislike * 15.0

            # Bonus for being very similar to any individual liked image
            if self._liked_vectors:
                max_like_sim = max(float(np.dot(vector, lv)) for lv in self._liked_vectors)
                if max_like_sim > 0.85:
                    score += (max_like_sim - 0.85) * 60.0  # strong boost for very similar

            # Penalty for being similar to any individual disliked image
            if self._disliked_vectors:
                max_dislike_sim = max(float(np.dot(vector, dv)) for dv in self._disliked_vectors)
                if max_dislike_sim > 0.80:
                    score -= (max_dislike_sim - 0.80) * 50.0  # strong penalty

        # ── Metadata-based prediction (logistic regression) ──
        meta_pred = self._pref_predictor.predict_probability(img)
        if meta_pred is not None:
            # meta_pred is 0..1, scale to -10..+10
            score += (meta_pred - 0.5) * 20.0

        # ── Face relevance bonus ──
        if self.face_names:
            fd = img.get("face_distance")
            if fd is not None and img.get("has_target_face"):
                # Close face match = strong preference signal
                # fd 0.0 → +10, fd 0.5 → +1
                score += max(0, (0.55 - fd)) * 18.0
            elif img.get("has_target_face"):
                score += 5.0

        return max(0.0, min(100.0, score))

    def _compute_quality(self, img, quality_weights):
        """
        Quality score on 0-100 scale. Pure technical quality — no preference.
        """
        grade = img.get("photo_grade")
        if grade and isinstance(grade, dict):
            composite = grade.get("composite", 0)
            # Composite is already 0-100
            score = float(composite)
        else:
            # Fallback: resolution + file size
            w = img.get("width") or img.get("w", 0)
            h = img.get("height") or img.get("h", 0)
            megapixels = (w * h) / 1_000_000
            res_score = min(megapixels / 12.0, 1.0) * 60.0

            kb = img.get("size_kb") or img.get("kb", 0)
            size_score = min(kb / 3000, 1.0) * 40.0

            score = res_score + size_score

        # Face presence bonus (quality indicator — photos with faces tend to be more curated)
        has_target = img.get("has_target_face") or img.get("target", False)
        face_count = img.get("face_count") or img.get("fc", 0)
        if has_target:
            score = min(100, score + 10)
        elif face_count > 0:
            score = min(100, score + 5)

        # Taste quiz adjustments (small, quality-related)
        quiz = quality_weights.get("_taste_quiz") or {}
        if quiz:
            avoid = quiz.get("avoid", [])
            if "no_face" in avoid and face_count == 0:
                score = max(0, score - 10)
            if "dark" in avoid:
                g = img.get("photo_grade")
                if g and g.get("exposure", 100) < 30:
                    score = max(0, score - 8)
            if quiz.get("sharpness") == "high":
                g = img.get("photo_grade")
                if g and g.get("sharpness", 0) < 40:
                    score = max(0, score - 10)

        return max(0.0, min(100.0, score))

    def _compute_diversity(self, img, vector=None, clip_vector=None):
        """
        Diversity score on -100..100 scale.
        Penalizes images too similar to already-selected ones.
        Considers: pixel similarity, CLIP semantic similarity, time, location.
        """
        if (not self._selected_vectors and not self._selected_clip_vectors
                and not self._selected_times):
            return 50.0  # neutral — no selection context yet

        penalty = 0.0

        # ── Pixel visual similarity to already-selected ──
        if vector is not None and self._selected_vectors:
            max_sim = max(float(np.dot(vector, sv)) for sv in self._selected_vectors)
            if max_sim > 0.75:
                # Graduated penalty: 0.75→0, 0.85→-25, 0.95→-50
                penalty += (max_sim - 0.75) * 200.0

        # ── CLIP semantic similarity to already-selected ──
        # More conservative than pixel similarity — semantic overlap is broader
        if clip_vector is not None and self._selected_clip_vectors:
            max_clip_sim = max(
                float(np.dot(clip_vector, sv))
                for sv in self._selected_clip_vectors)
            if max_clip_sim > 0.90:
                # Graduated penalty: 0.90→0, 0.95→-10, 1.0→-20
                penalty += (max_clip_sim - 0.90) * 200.0

        # ── Time clustering penalty ──
        date_str = img.get("date")
        if date_str and self._selected_times:
            same_date_count = sum(1 for t in self._selected_times if t == date_str)
            if same_date_count > 3:
                penalty += min(30, (same_date_count - 3) * 5)

        # ── Location clustering penalty ──
        loc = img.get("location")
        if loc and self._selected_locations:
            same_loc_count = sum(1 for l in self._selected_locations if l == loc)
            if same_loc_count > 5:
                penalty += min(20, (same_loc_count - 5) * 4)

        # Return as centered score: 50 = neutral, 0 = very repetitive
        return max(-100.0, min(100.0, 50.0 - penalty))

    def _compute_exploration(self, img):
        """
        Exploration bonus on 0-100 scale.
        Highest for uncertain images (model isn't sure → worth showing to user).
        Also gives a small random boost for variety.
        """
        pref = img.get("preference")
        if pref in ("like", "dislike"):
            return 0.0  # already rated, no need to explore

        score = 0.0

        # Uncertainty-based exploration
        meta_pred = self._pref_predictor.predict_probability(img)
        if meta_pred is not None:
            # Most uncertain when prediction is near 0.5
            confidence = abs(meta_pred - 0.5) * 2.0  # 0 = uncertain, 1 = confident
            score += (1.0 - confidence) * 70.0  # up to 70 for max uncertainty
        else:
            score += 40.0  # no model → moderate exploration

        # Small deterministic variety based on image hash
        # (avoids true randomness but ensures different images get chances)
        h = img.get("hash", "")
        if h:
            hash_variety = (hash(h) % 100) / 100.0
            score += hash_variety * 30.0  # up to 30 points

        return min(100.0, score)

    def _compute_duplicate_penalty(self, vector=None, dhash=None):
        """
        Duplicate penalty on 0-100 scale.
        Hard penalty for true duplicates, soft penalty for near-similar.
        """
        if not self._selected_dhashes and not self._selected_vectors:
            return 0.0

        penalty = 0.0

        # ── dHash: catches burst photos ──
        if dhash is not None and self._selected_dhashes:
            min_dist = min(_hamming_distance(dhash, sdh) for sdh in self._selected_dhashes)
            if min_dist <= 5:
                penalty = 100.0  # true duplicate — kill it
            elif min_dist <= 10:
                # Near-duplicate: graduated penalty
                penalty = max(penalty, (10 - min_dist) * 15.0)  # 75→0

        # ── Vector similarity: catches similar composition ──
        if vector is not None and self._selected_vectors:
            max_sim = max(float(np.dot(vector, sv)) for sv in self._selected_vectors)
            if max_sim > 0.95:
                penalty = max(penalty, 100.0)  # essentially identical
            elif max_sim > 0.88:
                # Graduated: 0.88→0, 0.92→30, 0.95→100
                penalty = max(penalty, (max_sim - 0.88) * (100.0 / 0.07))

        return min(100.0, penalty)

    # ──────────────────────────────────────────────────────────────────────
    #  DEBUG
    # ──────────────────────────────────────────────────────────────────────

    def get_score_summary(self, top_n=10):
        """Return readable summary of top-scored images for debugging."""
        entries = []
        for img in sorted(self._score_log, key=lambda x: x.get("_score", 0), reverse=True)[:top_n]:
            bd = img.get("_score_breakdown", {})
            entries.append(
                f"  {img.get('filename', '?'):30s}  "
                f"final={bd.get('final', 0):6.1f}  "
                f"pref={bd.get('preference', 0):5.1f}  "
                f"qual={bd.get('quality', 0):5.1f}  "
                f"div={bd.get('diversity', 0):5.1f}  "
                f"expl={bd.get('exploration', 0):5.1f}  "
                f"dup={bd.get('duplicate_penalty', 0):5.1f}"
            )
        return "\n".join(entries)


# ══════════════════════════════════════════════════════════════════════════════
#  METADATA PREDICTOR (logistic regression on photo features)
# ══════════════════════════════════════════════════════════════════════════════

class MetadataPredictor:
    """Lightweight logistic regression trained on likes/dislikes."""

    def __init__(self):
        self._weights = None
        self._bias = 0.0
        self._trained = False

    @staticmethod
    def _features(img):
        g = img.get("photo_grade") or {}
        return [
            g.get("resolution", 50) / 100.0,
            g.get("sharpness", 50) / 100.0,
            g.get("noise", 50) / 100.0,
            g.get("compression", 50) / 100.0,
            g.get("color", 50) / 100.0,
            g.get("exposure", 50) / 100.0,
            g.get("focus", 50) / 100.0,
            g.get("composite", 50) / 100.0,
            min(img.get("face_count", 0) / 5.0, 1.0),
            1.0 if img.get("has_target_face") else 0.0,
            (1.0 - img.get("face_distance", 0.5)) if img.get("face_distance") is not None else 0.5,
            min(img.get("size_kb", 500) / 5000.0, 1.0),
            1.0 if img.get("media_type") == "video" else 0.0,
        ]

    def train(self, images):
        rated = [i for i in images if i.get("preference") in ("like", "dislike")]
        if len(rated) < 3:
            self._trained = False
            return

        X = [self._features(i) for i in rated]
        Y = [1.0 if i["preference"] == "like" else 0.0 for i in rated]
        n_feat = len(X[0])
        n = len(X)

        w = [0.0] * n_feat
        b = 0.0
        lr = 0.5

        for _ in range(80):
            dw = [0.0] * n_feat
            db = 0.0
            for i in range(n):
                z = b + sum(w[j] * X[i][j] for j in range(n_feat))
                z = max(-10.0, min(10.0, z))
                pred = 1.0 / (1.0 + math.exp(-z))
                err = pred - Y[i]
                for j in range(n_feat):
                    dw[j] += err * X[i][j]
                db += err
            for j in range(n_feat):
                w[j] -= lr * dw[j] / n
            b -= lr * db / n

        self._weights = w
        self._bias = b
        self._trained = True

    def predict_probability(self, img):
        """Returns probability 0..1 that user would like this image, or None."""
        if not self._trained or not self._weights:
            return None
        x = self._features(img)
        z = self._bias + sum(self._weights[j] * x[j] for j in range(len(x)))
        z = max(-10.0, min(10.0, z))
        return 1.0 / (1.0 + math.exp(-z))


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def _hamming_distance(h1, h2):
    return bin(h1 ^ h2).count("1")
