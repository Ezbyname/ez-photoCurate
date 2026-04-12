"""
Event Agent — Intelligent photo collection curator
====================================================

Analyzes a scanned image database and makes smart curation decisions:
- Understands what each event type needs for a meaningful collection
- Auto-selects the best diverse images per category to hit targets
- Identifies gaps, imbalances, and quality issues
- Generates actionable recommendations

Usage (standalone):
    python event_agent.py analyze
    python event_agent.py auto-select
    python event_agent.py auto-select --strategy quality
    python event_agent.py recommend

Or integrated via curate.py:
    python curate.py analyze
    python curate.py auto-select
"""

import os
import sys
import json
import math
import argparse
import hashlib
from datetime import datetime
from collections import defaultdict

import numpy as np
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SCAN_DB_PATH = os.path.join(PROJECT_DIR, "scan_db.json")
VECTOR_SIZE = 64

# ── Event Knowledge Base ──────────────────────────────────────────────────────
# What makes a GREAT collection for each event type

EVENT_KNOWLEDGE = {
    "bar_mitzva": {
        "total_target": (500, 1000),
        "priorities": [
            "Even age coverage — don't overload toddler years and starve age 8-12",
            "Mix of posed and candid for each age",
            "Include family group shots across different ages",
            "Show personality growth — hobbies, school, sports at different ages",
            "Birth/baby photos are emotional anchors — quality over quantity",
            "Recent years (10-13) need strong representation for guests who know the child now",
        ],
        "balance_rules": {
            "early_heavy": "First 2 years often have too many photos. Be selective — pick only the best.",
            "recent_light": "Ages 8-13 often have fewer photos. Cast a wider net across sources.",
            "faces_matter": "Every image should ideally include the child. Group shots are great too.",
        },
        "quality_weights": {
            "has_target_face": 3.0,
            "has_any_face": 1.5,
            "resolution": 1.0,
            "file_size": 0.5,
        },
    },
    "wedding": {
        "total_target": (300, 500),
        "priorities": [
            "Ceremony is the centerpiece — allocate the most images here",
            "Couple portraits should be the highest quality shots",
            "Don't skip details/decor — they set the scene and mood",
            "Balance professional shots with candid guest moments",
            "Include both families adequately",
            "The 'getting ready' sequence tells a story — don't cut it too short",
        ],
        "balance_rules": {
            "ceremony_dominant": "Ceremony should be ~15% of total. It's the emotional core.",
            "party_overflow": "Dancing/party photos look similar. Be aggressive with diversity filter.",
            "details_forgotten": "Decor/detail shots are often overlooked but essential for the story.",
        },
        "quality_weights": {
            "has_target_face": 2.0,
            "has_any_face": 1.0,
            "resolution": 1.5,
            "file_size": 0.5,
        },
    },
    "birthday": {
        "total_target": (100, 200),
        "priorities": [
            "Cake moment is the signature — multiple angles and reactions",
            "Group shot with all guests is essential",
            "Capture the birthday person's reactions throughout",
            "Activities/games show the energy of the event",
            "Don't over-index on setup — a few establishing shots suffice",
        ],
        "balance_rules": {
            "cake_central": "Cake/singing should be well-covered but not overdone (15-20 shots max).",
            "candid_heavy": "Candids make birthdays feel alive. Prioritize natural moments.",
        },
        "quality_weights": {
            "has_target_face": 2.5,
            "has_any_face": 1.5,
            "resolution": 1.0,
            "file_size": 0.3,
        },
    },
    "baby_first_year": {
        "total_target": (300, 500),
        "priorities": [
            "Every month should be represented — this IS the structure",
            "Milestone 'firsts' are the emotional highlights",
            "Monthly consistency (same spot/setup) creates a powerful progression",
            "Include family members — this is also their year",
            "Mix closeups with wider context shots",
        ],
        "balance_rules": {
            "month_balance": "Each month should have roughly equal coverage.",
            "newborn_flood": "Newborn period has tons of photos. Be very selective on quality.",
        },
        "quality_weights": {
            "has_target_face": 2.0,
            "has_any_face": 1.5,
            "resolution": 1.0,
            "file_size": 0.5,
        },
    },
    "photo_book": {
        "total_target": (150, 300),
        "priorities": [
            "Seasonal balance gives the book a natural rhythm",
            "Lead each season with a strong establishing image",
            "Mix landscapes/scenes with people shots",
            "Include everyday moments — they become the most cherished",
            "Travel and celebrations are natural chapter breaks",
        ],
        "balance_rules": {
            "seasonal_balance": "Each season should have similar coverage unless one was exceptional.",
            "people_vs_places": "Aim for 60% people, 40% places/things.",
        },
        "quality_weights": {
            "has_any_face": 1.0,
            "resolution": 1.5,
            "file_size": 0.5,
        },
    },
    "vacation": {
        "total_target": (100, 250),
        "priorities": [
            "Each day should be represented — it's a daily journey",
            "Mix wide establishing shots with close details",
            "Include food/dining — it's part of the travel experience",
            "People shots at landmarks are better than empty landmarks",
            "Capture the vibe, not just the sights",
        ],
        "balance_rules": {
            "day_balance": "Each day should have similar coverage.",
            "landmark_overload": "Don't take 20 shots of the same landmark. Pick the best 2-3.",
        },
        "quality_weights": {
            "has_any_face": 1.0,
            "resolution": 1.5,
            "file_size": 0.5,
        },
    },
}


# ── Image Scoring ─────────────────────────────────────────────────────────────

def compute_quality_score(img, weights):
    """Score an image based on weighted quality factors.

    Uses photo_grade (comprehensive multi-dimensional grading) when available,
    with user preference as the most important signal.
    """
    score = 0.0

    # ── Photo grade composite (0-100 → 0-5 points) ──
    grade = img.get("photo_grade")
    if grade and isinstance(grade, dict):
        composite = grade.get("composite", 0)
        # Scale 0-100 grade to 0-5 points (major factor)
        score += (composite / 100.0) * 5.0
    else:
        # Fallback: basic resolution + file size scoring
        w = img.get("width") or img.get("w", 0)
        h = img.get("height") or img.get("h", 0)
        megapixels = (w * h) / 1_000_000
        score += min(megapixels / 12.0, 1.0) * weights.get("resolution", 1.0)

        kb = img.get("size_kb") or img.get("kb", 0)
        score += min(kb / 3000, 1.0) * weights.get("file_size", 0.5)

    # ── Face scores ──
    has_target = img.get("has_target_face") or img.get("target", False)
    face_count = img.get("face_count") or img.get("fc", 0)

    if has_target:
        score += weights.get("has_target_face", 2.0)
    elif face_count > 0:
        score += weights.get("has_any_face", 1.0)

    if face_count >= 3:
        score += 0.5

    # ── User preference (MOST IMPORTANT signal) ──
    pref = img.get("preference")
    if pref == "like":
        score += 8.0   # Massive boost — user taste overrides all
    elif pref == "dislike":
        score -= 6.0   # Strong penalty — push to bottom

    # ── Taste quiz adjustments ──
    quiz = weights.get("_taste_quiz") or {}
    if quiz:
        face_count = img.get("face_count") or img.get("fc", 0)
        # "avoid" penalties
        avoid = quiz.get("avoid", [])
        if "no_face" in avoid and face_count == 0:
            score -= 2.0
        if "dark" in avoid:
            g = img.get("photo_grade")
            if g and g.get("exposure", 100) < 30:
                score -= 1.5
        # "faces" preference
        faces_pref = quiz.get("faces")
        if faces_pref == "many" and face_count >= 3:
            score += 1.0
        elif faces_pref == "few" and 1 <= face_count <= 2:
            score += 0.5
        # "closeups" — boost images with large face (low face_distance = close match)
        if quiz.get("closeups") == "yes":
            fd = img.get("face_distance")
            if fd is not None and fd < 0.4:
                score += 0.5
        # "colorful" boost
        if quiz.get("colorful") == "yes":
            g = img.get("photo_grade")
            if g and g.get("color", 0) > 70:
                score += 0.5
        # "sharpness" strict
        if quiz.get("sharpness") == "high":
            g = img.get("photo_grade")
            if g and g.get("sharpness", 0) < 40:
                score -= 1.0

    return score


class PreferencePredictor:
    """
    Lightweight logistic regression trained on user likes/dislikes.
    Predicts preference scores for unrated images during Quick Fill.
    """

    def __init__(self):
        self._weights = None
        self._bias = 0.0
        self._trained = False

    @staticmethod
    def _features(img):
        """Extract feature vector from image metadata."""
        g = img.get("photo_grade") or {}
        return [
            (g.get("resolution", 50)) / 100.0,
            (g.get("sharpness", 50)) / 100.0,
            (g.get("noise", 50)) / 100.0,
            (g.get("compression", 50)) / 100.0,
            (g.get("color", 50)) / 100.0,
            (g.get("exposure", 50)) / 100.0,
            (g.get("focus", 50)) / 100.0,
            (g.get("composite", 50)) / 100.0,
            min((img.get("face_count", 0)) / 5.0, 1.0),
            1.0 if img.get("has_target_face") else 0.0,
            (1.0 - img.get("face_distance", 0.5)) if img.get("face_distance") is not None else 0.5,
            min((img.get("size_kb", 500)) / 5000.0, 1.0),
            1.0 if img.get("media_type") == "video" else 0.0,
        ]

    def train(self, images):
        """Train from all rated images using logistic regression (gradient descent)."""
        rated = [i for i in images if i.get("preference") in ("like", "dislike")]
        if len(rated) < 3:
            self._trained = False
            return

        X = [self._features(i) for i in rated]
        Y = [1.0 if i.get("preference") == "like" else 0.0 for i in rated]
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

    def score(self, img):
        """
        Returns (preference_score, exploration_bonus) for an image.
        Liked/disliked images return (0, 0) — their boost is already in compute_quality_score.
        Unrated with model: (predicted_score scaled -5..+5, exploration_bonus 0..2)
        Unrated without model: (0, 1) — small exploration bonus
        """
        pref = img.get("preference")
        if pref in ("like", "dislike"):
            return 0.0, 0.0  # already scored in compute_quality_score

        if not self._trained or not self._weights:
            return 0.0, 1.0  # no model yet, small exploration boost

        x = self._features(img)
        z = self._bias + sum(self._weights[j] * x[j] for j in range(len(x)))
        z = max(-10.0, min(10.0, z))
        p = 1.0 / (1.0 + math.exp(-z))

        # Scale prediction to score: p=1.0→+5, p=0.0→-5
        pref_score = (p - 0.5) * 10.0

        # Exploration bonus: highest when uncertain (p near 0.5), zero when confident
        confidence = abs(p - 0.5) * 2.0  # 0=uncertain, 1=confident
        exploration = max(0.0, 2.0 * (1.0 - confidence))

        return pref_score, exploration


def compute_image_vector(image_path):
    """Compute a visual feature vector for diversity comparison."""
    try:
        img = Image.open(image_path).convert("RGB")
        gray = img.convert("L").resize((VECTOR_SIZE, VECTOR_SIZE), Image.LANCZOS)
        pixels = np.array(gray, dtype=np.float32).flatten()
        img_small = img.resize((128, 128), Image.LANCZOS)
        arr = np.array(img_small)
        hist_r = np.histogram(arr[:, :, 0], bins=16, range=(0, 256))[0].astype(np.float32)
        hist_g = np.histogram(arr[:, :, 1], bins=16, range=(0, 256))[0].astype(np.float32)
        hist_b = np.histogram(arr[:, :, 2], bins=16, range=(0, 256))[0].astype(np.float32)
        vector = np.concatenate([pixels, hist_r * 10, hist_g * 10, hist_b * 10])
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector
    except Exception:
        return None


def compute_phash(image_path, hash_size=16):
    """Compute difference-hash for near-duplicate detection. Returns binary array."""
    try:
        img = Image.open(image_path).convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
        pixels = np.array(img, dtype=np.float64)
        diff = pixels[:, 1:] > pixels[:, :-1]
        return diff.flatten().astype(np.uint8)
    except Exception:
        return None


# ── Analysis ──────────────────────────────────────────────────────────────────

def load_scan_db(path=None):
    db_path = path or SCAN_DB_PATH
    if not os.path.isfile(db_path):
        print(f"No scan database found at {db_path}")
        print("Run: python curate.py scan --config curate_config.json")
        sys.exit(1)
    with open(db_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_event_type(config):
    return config.get("event_type", "bar_mitzva")


def get_categories_from_config(config, images=None):
    """Extract ordered list of category dicts from config, or infer from data."""
    cats = config.get("categories", [])
    if isinstance(cats, list) and cats and isinstance(cats[0], dict):
        return cats

    # Try loading from template
    event_type = config.get("event_type")
    if event_type:
        tpl_path = os.path.join(PROJECT_DIR, "templates", f"{event_type}.json")
        if os.path.isfile(tpl_path):
            with open(tpl_path, "r", encoding="utf-8") as f:
                tpl = json.load(f)
            return tpl.get("categories", [])

    # Fallback: infer categories from image data
    if images:
        target = config.get("target_per_category", 75)
        seen = {}
        for img in images:
            cat = img.get("category")
            if cat and cat not in seen:
                seen[cat] = {"id": cat, "display": cat, "target": target}
        return [seen[k] for k in sorted(seen.keys())]

    return []


def analyze_collection(db):
    """Deep analysis of the scanned collection. Returns structured report."""
    config = db.get("config", {})
    images = db.get("images", [])
    event_type = get_event_type(config)
    knowledge = EVENT_KNOWLEDGE.get(event_type, EVENT_KNOWLEDGE.get("photo_book"))
    categories = get_categories_from_config(config, images)
    target_per_cat = config.get("target_per_category", 75)

    # Build category lookup
    cat_lookup = {}
    for cat in categories:
        cid = cat["id"]
        cat_lookup[cid] = {
            "display": cat.get("display", cid),
            "target": cat.get("target", target_per_cat),
            "images": [],
            "pool_candidates": [],
        }

    # Classify images
    total_qualified = 0
    total_pool = 0
    source_counts = defaultdict(int)
    face_stats = {"with_target": 0, "with_any": 0, "no_face": 0}

    for img in images:
        src = img.get("source_label", "unknown")
        source_counts[src] += 1

        fc = img.get("face_count", 0)
        has_target = img.get("has_target_face", False)
        if has_target:
            face_stats["with_target"] += 1
        elif fc > 0:
            face_stats["with_any"] += 1
        else:
            face_stats["no_face"] += 1

        cat = img.get("category")
        status = img.get("status", "pool")

        if status == "qualified" and cat and cat in cat_lookup:
            cat_lookup[cat]["images"].append(img)
            total_qualified += 1
        else:
            # Check if this image could fit any category (for gap filling)
            if cat and cat in cat_lookup:
                cat_lookup[cat]["pool_candidates"].append(img)
            total_pool += 1

    # Category analysis
    cat_analysis = []
    total_target = sum(c["target"] for c in cat_lookup.values())
    total_selected = 0

    for cid, cdata in cat_lookup.items():
        count = len(cdata["images"])
        target = cdata["target"]
        pool_available = len(cdata["pool_candidates"])
        ratio = count / target if target > 0 else 0
        total_selected += min(count, target)

        status = "ok"
        if count == 0:
            status = "empty"
        elif ratio < 0.3:
            status = "critical"
        elif ratio < 0.7:
            status = "low"
        elif ratio < 1.0:
            status = "close"
        elif ratio > 3.0:
            status = "overflow"
        else:
            status = "ok"

        cat_analysis.append({
            "id": cid,
            "display": cdata["display"],
            "count": count,
            "target": target,
            "ratio": ratio,
            "status": status,
            "pool_available": pool_available,
            "surplus": max(0, count - target),
            "deficit": max(0, target - count),
        })

    # Recommendations
    recommendations = []

    # Overall balance
    total_min, total_max = knowledge.get("total_target", (100, 1000))
    if total_qualified < total_min:
        recommendations.append({
            "type": "warning",
            "title": "Collection too small",
            "detail": f"You have {total_qualified} qualified images but {event_type} typically needs {total_min}-{total_max}. Add more sources or relax filters.",
        })
    elif total_qualified > total_max * 2:
        recommendations.append({
            "type": "info",
            "title": "Very large pool to choose from",
            "detail": f"You have {total_qualified} qualified images. Use auto-select to pick the best {total_target} with diversity.",
        })

    # Category gaps
    empty_cats = [c for c in cat_analysis if c["status"] == "empty"]
    critical_cats = [c for c in cat_analysis if c["status"] == "critical"]
    overflow_cats = [c for c in cat_analysis if c["status"] == "overflow"]

    if empty_cats:
        names = ", ".join(c["display"] for c in empty_cats)
        recommendations.append({
            "type": "critical",
            "title": f"{len(empty_cats)} empty categories",
            "detail": f"No images in: {names}. Check if your sources cover these periods, or manually assign from pool.",
        })

    if critical_cats:
        for c in critical_cats:
            recommendations.append({
                "type": "warning",
                "title": f"'{c['display']}' critically low",
                "detail": f"Only {c['count']}/{c['target']} images. {c['pool_available']} pool candidates available.",
            })

    if overflow_cats:
        for c in overflow_cats:
            recommendations.append({
                "type": "info",
                "title": f"'{c['display']}' has excess images",
                "detail": f"{c['count']} images vs target {c['target']} ({c['surplus']} surplus). Auto-select will pick the best {c['target']}.",
            })

    # Event-specific advice
    for rule_name, advice in knowledge.get("balance_rules", {}).items():
        recommendations.append({
            "type": "tip",
            "title": rule_name.replace("_", " ").title(),
            "detail": advice,
        })

    # Source diversity
    if len(source_counts) == 1:
        recommendations.append({
            "type": "warning",
            "title": "Single source",
            "detail": "All images from one source. Adding more sources (cloud, USB, shared albums) improves coverage and variety.",
        })

    # Face coverage
    if face_stats["no_face"] > total_qualified * 0.5:
        recommendations.append({
            "type": "info",
            "title": "Many images without detected faces",
            "detail": f"{face_stats['no_face']}/{len(images)} images have no faces. Consider enabling face detection to prioritize people shots.",
        })

    return {
        "event_type": event_type,
        "event_display": knowledge.get("display_name", event_type),
        "total_images": len(images),
        "total_qualified": total_qualified,
        "total_pool": total_pool,
        "total_target": total_target,
        "source_counts": dict(source_counts),
        "face_stats": face_stats,
        "categories": cat_analysis,
        "recommendations": recommendations,
        "priorities": knowledge.get("priorities", []),
    }


# ── Auto-Selection ────────────────────────────────────────────────────────────

def auto_select(db, strategy="balanced", sim_threshold=0.85, dry_run=False):
    """
    Automatically select the best images for each category.

    Strategies:
        balanced  — quality-ranked, diversity-filtered, respect targets
        quality   — pure quality ranking, less diversity enforcement
        diverse   — maximize visual variety, accept lower quality

    Returns the updated db and a selection report.
    """
    config = db.get("config", {})
    images = db.get("images", [])
    event_type = get_event_type(config)
    knowledge = EVENT_KNOWLEDGE.get(event_type, EVENT_KNOWLEDGE.get("photo_book"))
    categories = get_categories_from_config(config, images)
    target_per_cat = config.get("target_per_category", 75)
    weights = knowledge.get("quality_weights", {})

    # Adjust strategy params
    if strategy == "quality":
        sim_threshold = 0.95  # very loose diversity
    elif strategy == "diverse":
        sim_threshold = 0.75  # strict diversity

    # Group images by category (only face-matched when face names configured)
    face_names = config.get("face_names", [])
    unlimited = config.get("unlimited_mode", False)
    by_category = defaultdict(list)
    for img in images:
        if img.get("status") == "rejected":
            continue
        if face_names and img.get("media_type") != "video" and not img.get("has_target_face"):
            continue
        cat = img.get("category")
        if cat:
            by_category[cat].append(img)

    # Build category targets
    cat_targets = {}
    cat_vid_targets = {}
    for cat in categories:
        cat_targets[cat["id"]] = cat.get("target", target_per_cat)
        cat_vid_targets[cat["id"]] = cat.get("video_target", 0)

    report = {
        "strategy": strategy,
        "sim_threshold": sim_threshold,
        "selections": [],
        "total_selected": 0,
        "total_available": 0,
    }

    selected_hashes = set()
    total_selected = 0

    for cat in categories:
        cid = cat["id"]
        target = cat_targets.get(cid, target_per_cat)
        pool = by_category.get(cid, [])
        report["total_available"] += len(pool)

        if not pool:
            report["selections"].append({
                "category": cid,
                "display": cat.get("display", cid),
                "target": target,
                "selected": 0,
                "available": 0,
            })
            continue

        # Face distance filtering — reject false positives
        if face_names:
            age_days_to = cat.get("age_days_to", 99999)
            thresholds = config.get("face_distance_thresholds", {})
            if age_days_to <= 365:
                max_dist = thresholds.get("infant", 0.45)
            elif age_days_to <= 1095:
                max_dist = thresholds.get("toddler", 0.50)
            else:
                max_dist = thresholds.get("default", 0.55)
            pool = [i for i in pool
                    if i.get("face_distance") is not None and i.get("face_distance") <= max_dist]

        # Score all candidates
        for img in pool:
            fd = img.get("face_distance")
            base = compute_quality_score(img, weights)
            if fd is not None and face_names:
                base += max(0, (0.6 - fd)) * 5
            img["_score"] = base

        # Sort by score descending
        pool.sort(key=lambda x: x["_score"], reverse=True)

        # Temporal balancing: interleave from different time periods
        if strategy in ("balanced", "diverse") and len(pool) > target * 2:
            dated = [img for img in pool if img.get("date")]
            undated = [img for img in pool if not img.get("date")]
            if len(dated) > target:
                from collections import defaultdict as _dd
                time_buckets = _dd(list)
                for img in dated:
                    month_key = img["date"][:7]  # YYYY-MM
                    time_buckets[month_key].append(img)
                # Round-robin from time buckets (each bucket already sorted by score)
                bucket_keys = sorted(time_buckets.keys())
                bucket_iters = {k: iter(time_buckets[k]) for k in bucket_keys}
                reordered = []
                while len(reordered) < len(dated):
                    added = False
                    for k in bucket_keys:
                        try:
                            reordered.append(next(bucket_iters[k]))
                            added = True
                        except StopIteration:
                            continue
                    if not added:
                        break
                pool = reordered + undated

        # Select with diversity filter
        selected = []
        selected_vectors = []
        selected_phashes = []

        print(f"  {cat.get('display', cid)}: {len(pool)} candidates, target {target}...")

        for img in pool:
            if len(selected) >= target:
                break

            # Diversity check via image vectors + perceptual hash
            if sim_threshold < 1.0 and selected_vectors:
                vec = _get_or_compute_vector(img)
                too_similar = False
                if vec is not None:
                    mat = np.array(selected_vectors, dtype=np.float32)
                    sims = mat @ vec
                    if np.max(sims) >= sim_threshold:
                        too_similar = True

                # Also check perceptual hash (catches near-duplicates that vectors miss)
                if not too_similar and selected_phashes:
                    img_path = img.get("path", "").replace("/", os.sep)
                    phash = img.get("_phash")
                    if phash is None and img_path and os.path.isfile(img_path):
                        phash = compute_phash(img_path)
                        img["_phash"] = phash
                    if phash is not None:
                        for sp in selected_phashes:
                            hamming = float(np.sum(phash != sp)) / len(phash)
                            if hamming < 0.12:
                                too_similar = True
                                break

                if too_similar:
                    continue

                if vec is not None:
                    selected_vectors.append(vec)
                else:
                    selected_vectors.append(np.zeros(VECTOR_SIZE * VECTOR_SIZE + 48))

                # Cache phash
                img_path = img.get("path", "").replace("/", os.sep)
                phash = img.get("_phash")
                if phash is None and img_path and os.path.isfile(img_path):
                    phash = compute_phash(img_path)
                if phash is not None:
                    selected_phashes.append(phash)

            elif sim_threshold < 1.0:
                vec = _get_or_compute_vector(img)
                if vec is not None:
                    selected_vectors.append(vec)
                img_path = img.get("path", "").replace("/", os.sep)
                if img_path and os.path.isfile(img_path):
                    phash = compute_phash(img_path)
                    if phash is not None:
                        selected_phashes.append(phash)

            selected.append(img)
            img_hash = img.get("hash", "")
            selected_hashes.add(img_hash)

        # Update statuses
        if not dry_run:
            for img in pool:
                img_hash = img.get("hash", "")
                if img_hash in selected_hashes:
                    img["status"] = "selected"
                elif img.get("status") != "rejected":
                    img["status"] = "qualified"  # still qualified, just not selected

        cat_selected = len(selected)
        total_selected += cat_selected
        report["selections"].append({
            "category": cid,
            "display": cat.get("display", cid),
            "target": target,
            "selected": cat_selected,
            "available": len(pool),
            "top_score": round(selected[0]["_score"], 2) if selected else 0,
            "min_score": round(selected[-1]["_score"], 2) if selected else 0,
        })

        print(f"    -> selected {cat_selected}/{target} (from {len(pool)})")

    report["total_selected"] = total_selected

    # Clean up temp scores
    for img in images:
        img.pop("_score", None)
        img.pop("_phash", None)

    return db, report


_image_vectors_cache = None

def _get_or_compute_vector(img):
    """Get image vector from npz sidecar, entry, or compute from file."""
    global _image_vectors_cache
    h = img.get("hash")
    # Try npz sidecar first
    if _image_vectors_cache is None:
        npz_path = os.path.join(PROJECT_DIR, "image_vectors.npz")
        if os.path.isfile(npz_path):
            try:
                _image_vectors_cache = dict(np.load(npz_path, allow_pickle=False))
            except Exception as e:
                print(f"[WARN] Could not load {npz_path} (corrupt?): {e}", flush=True)
                _image_vectors_cache = {}
        else:
            _image_vectors_cache = {}
    if h and h in _image_vectors_cache:
        return np.array(_image_vectors_cache[h], dtype=np.float32)
    # Legacy: try entry-embedded vector
    cached = img.get("image_vector")
    if cached is not None:
        return np.array(cached, dtype=np.float32)
    path = img.get("path", "")
    if not path or not os.path.isfile(path.replace("/", os.sep)):
        return None
    return compute_image_vector(path.replace("/", os.sep))


# ── Recommendations Engine ────────────────────────────────────────────────────

def generate_recommendations(analysis, selection_report=None):
    """Generate human-readable recommendations based on analysis."""
    lines = []
    event = analysis["event_type"]
    knowledge = EVENT_KNOWLEDGE.get(event, {})

    lines.append("=" * 70)
    lines.append(f"  COLLECTION ANALYSIS: {analysis.get('event_display', event).upper()}")
    lines.append("=" * 70)

    # Overview
    lines.append(f"\n  Total images scanned: {analysis['total_images']}")
    lines.append(f"  Qualified: {analysis['total_qualified']}")
    lines.append(f"  Pool (unassigned): {analysis['total_pool']}")
    lines.append(f"  Target total: {analysis['total_target']}")

    # Sources
    lines.append(f"\n  Sources:")
    for src, cnt in sorted(analysis["source_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"    {src}: {cnt}")

    # Face stats
    fs = analysis["face_stats"]
    lines.append(f"\n  Faces:")
    lines.append(f"    With target person: {fs['with_target']}")
    lines.append(f"    With any face: {fs['with_any']}")
    lines.append(f"    No faces: {fs['no_face']}")

    # Category breakdown
    lines.append(f"\n  {'Category':<28} {'Count':>6} {'Target':>7} {'Status':>10}")
    lines.append(f"  {'-'*55}")

    status_emoji = {
        "ok": "OK", "close": "CLOSE", "low": "LOW",
        "critical": "!! LOW", "empty": "EMPTY", "overflow": "EXCESS",
    }

    for cat in analysis["categories"]:
        status = status_emoji.get(cat["status"], cat["status"])
        lines.append(f"  {cat['display']:<28} {cat['count']:>6} {cat['target']:>7} {status:>10}")

    # Selection report (if auto-select was run)
    if selection_report:
        lines.append(f"\n  AUTO-SELECTION ({selection_report['strategy']} strategy)")
        lines.append(f"  {'Category':<28} {'Selected':>9} {'Target':>7} {'Available':>10}")
        lines.append(f"  {'-'*58}")
        for sel in selection_report["selections"]:
            lines.append(f"  {sel['display']:<28} {sel['selected']:>9} {sel['target']:>7} {sel['available']:>10}")
        lines.append(f"  {'-'*58}")
        lines.append(f"  {'TOTAL':<28} {selection_report['total_selected']:>9}")

    # Recommendations
    lines.append(f"\n  RECOMMENDATIONS")
    lines.append(f"  {'-'*55}")

    type_prefix = {
        "critical": "[!!]", "warning": "[!]",
        "info": "[i]", "tip": "[*]",
    }

    for rec in analysis["recommendations"]:
        prefix = type_prefix.get(rec["type"], "[-]")
        lines.append(f"\n  {prefix} {rec['title']}")
        # Wrap detail text
        detail = rec["detail"]
        while len(detail) > 65:
            cut = detail[:65].rfind(" ")
            if cut < 20:
                cut = 65
            lines.append(f"      {detail[:cut]}")
            detail = detail[cut:].strip()
        if detail:
            lines.append(f"      {detail}")

    # Event priorities
    priorities = analysis.get("priorities", [])
    if priorities:
        lines.append(f"\n  WHAT MAKES A GREAT {event.upper().replace('_', ' ')} COLLECTION")
        lines.append(f"  {'-'*55}")
        for i, p in enumerate(priorities, 1):
            lines.append(f"  {i}. {p}")

    lines.append("")
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def cmd_analyze(args):
    db = load_scan_db(args.db if hasattr(args, "db") else None)
    analysis = analyze_collection(db)
    report_text = generate_recommendations(analysis)
    print(report_text)

    # Save analysis JSON
    out_path = os.path.join(PROJECT_DIR, "analysis_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f"  Analysis saved: {out_path}")


def cmd_auto_select(args):
    db = load_scan_db(args.db if hasattr(args, "db") else None)
    strategy = getattr(args, "strategy", "balanced")
    threshold = getattr(args, "sim_threshold", 0.85)
    dry = getattr(args, "dry_run", False)

    print("=" * 70)
    print(f"  AUTO-SELECT (strategy: {strategy}, diversity: {threshold})")
    print("=" * 70)

    db, sel_report = auto_select(db, strategy=strategy, sim_threshold=threshold, dry_run=dry)

    # Also run analysis for full report
    analysis = analyze_collection(db)
    report_text = generate_recommendations(analysis, sel_report)
    print(report_text)

    if not dry:
        # Save updated DB
        with open(SCAN_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False)
        print(f"\n  Updated: {SCAN_DB_PATH}")
        print(f"  Total selected: {sel_report['total_selected']}")
        print(f"\n  Next: python curate.py report   (to review selections)")
    else:
        print(f"\n  DRY RUN — no changes saved.")
        print(f"  Would select: {sel_report['total_selected']} images")


def main():
    p = argparse.ArgumentParser(description="Event Agent — Smart photo curation")
    sub = p.add_subparsers(dest="command")

    s_analyze = sub.add_parser("analyze", help="Analyze collection and get recommendations")
    s_analyze.add_argument("--db", type=str, help="Path to scan_db.json")

    s_select = sub.add_parser("auto-select", help="Auto-select best images per category")
    s_select.add_argument("--strategy", choices=["balanced", "quality", "diverse"], default="balanced")
    s_select.add_argument("--sim-threshold", type=float, default=0.85)
    s_select.add_argument("--dry-run", action="store_true")
    s_select.add_argument("--db", type=str, help="Path to scan_db.json")

    args = p.parse_args()

    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "auto-select":
        cmd_auto_select(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
