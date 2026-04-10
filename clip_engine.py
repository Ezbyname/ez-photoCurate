"""
clip_engine.py — CLIP-based semantic image embeddings and auto-tagging.

Computes 512-dim CLIP embeddings and generates human-readable tags from a
controlled vocabulary using zero-shot classification.

Architecture:
  embeddings  = machine representation (512-dim vectors, sidecar .npz file)
  tags        = human-readable descriptors (controlled vocabulary, in scan_db)
  categories  = controlled grouping (handled by curate.py rules engine)

Model: CLIP ViT-B/32 via ONNX Runtime (no PyTorch required).
Dependencies: numpy, Pillow, onnxruntime (already installed for NudeNet).
"""

import hashlib
import json
import os
import gzip
import html
import re
import threading
import urllib.request
from functools import lru_cache

import numpy as np
from PIL import Image

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_DIR, "models", "clip")
VISION_MODEL_PATH = os.path.join(MODEL_DIR, "vision_model.onnx")
TEXT_MODEL_PATH = os.path.join(MODEL_DIR, "text_model.onnx")
BPE_PATH = os.path.join(MODEL_DIR, "bpe_simple_vocab_16e6.txt.gz")
TAG_CACHE_PATH = os.path.join(MODEL_DIR, "tag_text_embeddings.npz")

# ── Model Identity ───────────────────────────────────────────────────────────

MODEL_ID = "clip-vit-b32-onnx"
MODEL_VERSION = "1.0"  # kept for backward compat / tag cache migration
EMBED_VERSION = f"{MODEL_ID}:{MODEL_VERSION}"  # changes when ONNX model changes

# ── CLIP Image Preprocessing Constants ───────────────────────────────────────

CLIP_SIZE = 224
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# ── Tag Selection Defaults ───────────────────────────────────────────────────

TAG_TOP_K = 10            # max tags per image (was 15 — reduced to cut fill noise)
TAG_MIN_THRESHOLD = 0.20  # minimum cosine similarity (was 0.18 — raised to cut weak fills)
TAG_STRONG_THRESHOLD = 0.26  # always-include threshold

# ── Controlled Tag Vocabulary ────────────────────────────────────────────────
# Grouped by semantic family.  Each tag is matched against the image embedding
# via CLIP zero-shot classification.  Tag names are chosen to overlap with
# CATEGORY_RULES path_keywords in curate.py so they improve categorization.

TAG_VOCABULARY = {
    # ── SCENE / SETTING ──────────────────────────────────────────────────
    "scene": [
        "indoors", "outdoors",
        "basketball court", "gym", "sports field", "stadium",
        "swimming pool", "playground",
        "beach", "ocean", "lake", "mountain", "forest",
        "park", "garden",
        "street", "city",
        "home", "kitchen", "bathroom", "bedroom", "living room",
        "restaurant", "hotel",
        "office", "classroom", "school",
        "church", "synagogue",
        # "stage" removed — max 0.223, never discriminative
        "dance floor",
        "airport",
        "party", "ceremony",
    ],
    # ── PEOPLE ────────────────────────────────────────────────────────────
    "people": [
        "baby", "toddler", "child", "teenager",
        "portrait", "selfie", "group photo", "team photo",
        "crowd", "family",
        # "couple" removed — 33% false positive on any 2-person photo
        "bride", "groom", "athlete",
    ],
    # ── ACTIVITY ──────────────────────────────────────────────────────────
    "activity": [
        "playing sports", "playing basketball", "running", "jumping",
        "swimming",
        # "exercising" removed — redundant with specific sports, 21% noise
        "eating", "drinking", "cooking",
        "dancing", "singing", "performing",
        # "posing" removed — 31% noise, nearly all photos involve posing
        "smiling", "laughing",
        "hugging", "kissing",
        "celebrating",
        # "cheering" removed — 31% noise, indistinguishable from smiling
        "reading", "studying", "working",
        "traveling", "walking", "driving",
        # "sleeping" removed — 33% false positive on non-sleeping images
        "sitting",
        "bathing",
        "opening gifts", "blowing candles",
    ],
    # ── OBJECTS ────────────────────────────────────────────────────────────
    "object": [
        "basketball", "trophy",
        # "soccer ball" removed — appeared on baby photos, pure noise
        # "medal" removed — max 0.233, never discriminative
        "birthday cake", "food", "drinks",
        # "balloon" removed — max 0.250, 36% false positive, #1 misclassification driver
        "flowers", "gift",
        "car", "bus", "airplane",
        "phone", "computer",
        # "document" removed — noise on any image with text/paper
        "dog", "cat",
        "wedding dress", "suit", "jersey",
        "musical instrument", "microphone",
        "book",
        # "camera" removed — meta-tag (describes tool, not content)
    ],
    # ── MOOD / STYLE ──────────────────────────────────────────────────────
    "mood": [
        "happy", "formal", "casual",
        # "serious" removed — max 0.226, too weak to be useful
        "action shot", "close up",
        "nighttime", "sunset",
    ],
}

# Flattened index: [(tag_name, family), ...]
_TAG_INDEX = []
for _fam, _tags in TAG_VOCABULARY.items():
    for _tag in _tags:
        _TAG_INDEX.append((_tag, _fam))


def _build_prompt(tag, family):
    """Build the CLIP text prompt for a tag (zero-shot classification)."""
    if family == "activity":
        return f"a photo of a person {tag}"
    if family == "mood":
        return f"a {tag} photograph"
    return f"a photo of a {tag}"


# ── Tagger Version (auto-computed fingerprint) ──────────────────────────────
# Any change to vocabulary, thresholds, or prompt templates auto-bumps this.

def _compute_tagger_fingerprint():
    """Deterministic hash of everything that affects tag output."""
    data = {
        "vocab": TAG_VOCABULARY,
        "top_k": TAG_TOP_K,
        "min_threshold": TAG_MIN_THRESHOLD,
        "strong_threshold": TAG_STRONG_THRESHOLD,
        "prompts": {fam: _build_prompt("__PROBE__", fam)
                    for fam in TAG_VOCABULARY},
    }
    raw = json.dumps(data, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


TAGGER_VERSION = _compute_tagger_fingerprint()


# ── Download URLs ────────────────────────────────────────────────────────────

_HF_BASE = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main"
_DOWNLOAD_URLS = {
    "vision": f"{_HF_BASE}/onnx/vision_model_quantized.onnx",
    "text":   f"{_HF_BASE}/onnx/text_model_quantized.onnx",
    "bpe":    ("https://github.com/openai/CLIP/raw/main"
               "/clip/bpe_simple_vocab_16e6.txt.gz"),
}
# Fallback URLs (non-quantized, larger but guaranteed to exist)
_FALLBACK_URLS = {
    "vision": f"{_HF_BASE}/onnx/vision_model.onnx",
    "text":   f"{_HF_BASE}/onnx/text_model.onnx",
}


# ── Download Utilities ───────────────────────────────────────────────────────

def _download_file(url, dest, progress_fn=None):
    """Download a file from *url* to *dest* with optional progress callback.

    Args:
        progress_fn: callable(downloaded_bytes, total_bytes) or None.
    """
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".tmp"
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "EzPhotoOrganizer/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_fn and total:
                        progress_fn(downloaded, total)
        os.replace(tmp, dest)
    except Exception:
        if os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def ensure_models(progress_fn=None):
    """Download CLIP model files if not already present.

    Args:
        progress_fn: callable(message_str) for progress updates.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    files = [
        ("vision", VISION_MODEL_PATH,
         _DOWNLOAD_URLS["vision"], _FALLBACK_URLS.get("vision")),
        ("text", TEXT_MODEL_PATH,
         _DOWNLOAD_URLS["text"], _FALLBACK_URLS.get("text")),
        ("bpe", BPE_PATH,
         _DOWNLOAD_URLS["bpe"], None),
    ]

    for label, path, url, fallback_url in files:
        if os.path.isfile(path):
            continue
        if progress_fn:
            progress_fn(f"Downloading CLIP {label} model...")

        def _inner_progress(downloaded, total, _label=label):
            pct = downloaded * 100 // total if total else 0
            if progress_fn:
                progress_fn(f"Downloading CLIP {_label}... {pct}%")

        try:
            _download_file(url, path, progress_fn=_inner_progress)
        except Exception:
            if fallback_url:
                if progress_fn:
                    progress_fn(
                        f"Retrying CLIP {label} from fallback URL...")
                _download_file(
                    fallback_url, path, progress_fn=_inner_progress)
            else:
                raise


def is_available():
    """Return True if CLIP models are on disk and ONNX Runtime is importable."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return (os.path.isfile(VISION_MODEL_PATH)
            and os.path.isfile(TEXT_MODEL_PATH)
            and os.path.isfile(BPE_PATH))


# ── CLIP BPE Tokenizer ──────────────────────────────────────────────────────
# Adapted from OpenAI's CLIP (MIT License).  Simplified for ASCII-only tag
# prompts (no ftfy / regex dependency).

@lru_cache()
def _bytes_to_unicode():
    """Map byte values to printable unicode characters for BPE."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\u00a1"), ord("\u00ac") + 1))
          + list(range(ord("\u00ae"), ord("\u00ff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _get_pairs(word):
    """Return set of adjacent character pairs in *word* tuple."""
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


# Simplified pattern (ASCII-focused, sufficient for English tag vocabulary).
_TOKEN_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d|[a-zA-Z]+|[0-9]|[^\s\w]+""",
    re.IGNORECASE,
)


class _SimpleTokenizer:
    """Minimal CLIP BPE tokenizer — no external dependencies."""

    def __init__(self, bpe_path):
        byte_encoder = _bytes_to_unicode()

        with gzip.open(bpe_path, "rt", encoding="utf-8") as f:
            bpe_data = f.read()
        merges = bpe_data.strip().split("\n")[1:]  # skip header
        merges = merges[:48894]  # 49408 vocab = 256 base + 256 eow + 48894 merges + 2 special
        merges = [tuple(m.split()) for m in merges if m]

        vocab = list(byte_encoder.values())
        vocab += [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])

        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.byte_encoder = byte_encoder
        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }
        self.sot = self.encoder["<|startoftext|>"]
        self.eot = self.encoder["<|endoftext|>"]

    def _bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(
                pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                if (word[j] == first
                        and j < len(word) - 1
                        and word[j + 1] == second):
                    new_word.append(first + second)
                    i = j + 2
                else:
                    new_word.append(word[j])
                    i = j + 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        result = " ".join(word)
        self.cache[token] = result
        return result

    def encode(self, text):
        """Encode *text* to a list of BPE token IDs."""
        text = html.unescape(text).strip().lower()
        tokens = []
        for match in _TOKEN_PAT.finditer(text):
            word = match.group()
            encoded = "".join(
                self.byte_encoder[b] for b in word.encode("utf-8"))
            tokens.extend(
                self.encoder[bpe_tok]
                for bpe_tok in self._bpe(encoded).split(" "))
        return tokens


_tokenizer_instance = None
_tok_lock = threading.Lock()


def _get_tokenizer():
    global _tokenizer_instance
    if _tokenizer_instance is None:
        with _tok_lock:
            if _tokenizer_instance is None:
                _tokenizer_instance = _SimpleTokenizer(BPE_PATH)
    return _tokenizer_instance


def _tokenize(texts, context_length=77):
    """Tokenize a list of texts for the CLIP text encoder.

    Returns (input_ids, attention_mask) as int64 numpy arrays.
    """
    tokenizer = _get_tokenizer()
    if isinstance(texts, str):
        texts = [texts]

    input_ids = np.zeros((len(texts), context_length), dtype=np.int64)
    attention_mask = np.zeros((len(texts), context_length), dtype=np.int64)

    for i, text in enumerate(texts):
        toks = [tokenizer.sot] + tokenizer.encode(text) + [tokenizer.eot]
        length = min(len(toks), context_length)
        input_ids[i, :length] = toks[:length]
        attention_mask[i, :length] = 1
        if length == context_length:
            input_ids[i, -1] = tokenizer.eot

    return input_ids, attention_mask


# ── Image Preprocessing ──────────────────────────────────────────────────────

def _preprocess_to_tensor(img):
    """Convert a PIL RGB image to CLIP input tensor [1, 3, 224, 224]."""
    # Resize shortest side to CLIP_SIZE, bicubic interpolation
    w, h = img.size
    scale = CLIP_SIZE / min(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    img = img.resize((new_w, new_h), Image.BICUBIC)

    # Center crop to CLIP_SIZE x CLIP_SIZE
    left = (new_w - CLIP_SIZE) // 2
    top = (new_h - CLIP_SIZE) // 2
    img = img.crop((left, top, left + CLIP_SIZE, top + CLIP_SIZE))

    # To float32 [0,1], normalize with CLIP mean/std
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - CLIP_MEAN) / CLIP_STD

    # HWC -> CHW, add batch dimension
    return arr.transpose(2, 0, 1)[np.newaxis].astype(np.float32)


def preprocess_image(fpath):
    """Load an image from *fpath* and return a CLIP-ready tensor."""
    img = Image.open(fpath).convert("RGB")
    return _preprocess_to_tensor(img)


def preprocess_pil(pil_img):
    """Convert a PIL Image to a CLIP-ready tensor."""
    return _preprocess_to_tensor(pil_img.convert("RGB"))


# ── ONNX Sessions (lazy-loaded, thread-safe) ────────────────────────────────

_vision_session = None
_text_session = None
_session_lock = threading.Lock()


def _get_vision_session():
    global _vision_session
    if _vision_session is None:
        with _session_lock:
            if _vision_session is None:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
                opts.inter_op_num_threads = 1
                opts.intra_op_num_threads = 2
                _vision_session = ort.InferenceSession(
                    VISION_MODEL_PATH, sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
    return _vision_session


def _get_text_session():
    global _text_session
    if _text_session is None:
        with _session_lock:
            if _text_session is None:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
                _text_session = ort.InferenceSession(
                    TEXT_MODEL_PATH, sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
    return _text_session


# ── Tag Text Embeddings (computed once, cached to disk) ──────────────────────

_tag_embeddings_cache = None
_tag_lock = threading.Lock()


def _get_tag_embeddings():
    """Return the (N, 512) text-embedding matrix for the controlled vocabulary.

    Computed once via the CLIP text encoder, then cached to disk.
    """
    global _tag_embeddings_cache
    if _tag_embeddings_cache is not None:
        return _tag_embeddings_cache

    with _tag_lock:
        if _tag_embeddings_cache is not None:
            return _tag_embeddings_cache

        # Try loading from disk cache
        _cache_key = f"{EMBED_VERSION}:{TAGGER_VERSION}"
        if os.path.isfile(TAG_CACHE_PATH):
            try:
                data = np.load(TAG_CACHE_PATH, allow_pickle=False)
                ver = str(data["version"]) if "version" in data else ""
                if ver == _cache_key:
                    _tag_embeddings_cache = data["embeddings"].astype(
                        np.float32)
                    return _tag_embeddings_cache
            except Exception:
                pass

        # Compute fresh via text encoder
        session = _get_text_session()
        input_names = [inp.name for inp in session.get_inputs()]
        has_mask = len(input_names) > 1

        prompts = [_build_prompt(tag, fam) for tag, fam in _TAG_INDEX]

        batch_size = 32
        all_embeds = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            input_ids, attention_mask = _tokenize(batch)
            feeds = {input_names[0]: input_ids}
            if has_mask:
                feeds[input_names[1]] = attention_mask
            outputs = session.run(None, feeds)
            emb = outputs[0]
            if emb.ndim == 3:
                # Model returned sequence; take CLS token
                emb = emb[:, 0, :]
            all_embeds.append(emb)

        embeddings = np.vstack(all_embeds).astype(np.float32)

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-8)

        # Cache to disk
        try:
            os.makedirs(os.path.dirname(TAG_CACHE_PATH), exist_ok=True)
            np.savez_compressed(
                TAG_CACHE_PATH,
                embeddings=embeddings,
                version=np.array(_cache_key),
            )
        except Exception:
            pass

        _tag_embeddings_cache = embeddings
        return embeddings


# ── Core API ─────────────────────────────────────────────────────────────────

def compute_embedding(fpath):
    """Compute a 512-dim CLIP embedding for an image file.

    Returns:
        numpy float32 array of shape (512,), L2-normalized.
        None on failure.
    """
    try:
        session = _get_vision_session()
        pixel_values = preprocess_image(fpath)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: pixel_values})
        emb = outputs[0]
        if emb.ndim == 3:
            emb = emb[:, 0, :]
        if emb.ndim == 2:
            emb = emb[0]
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.astype(np.float32)
    except Exception:
        return None


def compute_embedding_pil(pil_img):
    """Compute a 512-dim CLIP embedding from a PIL Image.

    Returns:
        numpy float32 array of shape (512,), L2-normalized.
        None on failure.
    """
    try:
        session = _get_vision_session()
        pixel_values = preprocess_pil(pil_img)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: pixel_values})
        emb = outputs[0]
        if emb.ndim == 3:
            emb = emb[:, 0, :]
        if emb.ndim == 2:
            emb = emb[0]
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.astype(np.float32)
    except Exception:
        return None


def generate_tags(embedding,
                  top_k=TAG_TOP_K,
                  min_threshold=TAG_MIN_THRESHOLD,
                  strong_threshold=TAG_STRONG_THRESHOLD):
    """Generate human-readable tags from a CLIP embedding.

    Uses zero-shot classification against the controlled vocabulary.
    Tag selection: all tags above *strong_threshold* are always included,
    then fills up to *top_k* with tags above *min_threshold*.

    Args:
        embedding: 512-dim CLIP embedding (L2-normalized).
        top_k: Maximum number of tags.
        min_threshold: Minimum cosine similarity to include a tag.
        strong_threshold: Tags above this are always included.

    Returns:
        (tags, tag_meta) where:
          tags     — list of tag name strings, confidence-descending
          tag_meta — dict with "model", "version", "scores" {tag: float}
    """
    tag_embeddings = _get_tag_embeddings()

    # Cosine similarity (both sides L2-normalized)
    similarities = embedding @ tag_embeddings.T

    # Pair each score with its tag info and sort descending
    scored = sorted(
        zip(_TAG_INDEX, similarities.tolist()),
        key=lambda x: -x[1],
    )

    # ── Selection with per-family dedup for fill slots ─────────────
    # Strong tags (>= strong_threshold) always go in.
    # Fill tags are capped per family to prevent redundant age/person
    # tags (baby, toddler, child) from consuming all visible slots.
    _FILL_CAP = {"people": 2, "activity": 4}  # per-family max fills
    _fill_counts = {}  # family → count of fill slots used

    selected = []
    for (tag_name, family), sim_val in scored:
        if sim_val >= strong_threshold:
            selected.append((tag_name, family, sim_val))
        elif sim_val >= min_threshold and len(selected) < top_k:
            cap = _FILL_CAP.get(family)
            if cap is not None:
                used = _fill_counts.get(family, 0)
                if used >= cap:
                    continue  # skip — family has enough fills
                _fill_counts[family] = used + 1
            selected.append((tag_name, family, sim_val))
        elif sim_val < min_threshold:
            break  # sorted descending — no more above threshold

    tags = [name for name, _, _ in selected]
    scores = {name: round(score, 4) for name, _, score in selected}

    tag_meta = {
        "model": MODEL_ID,
        "embed_version": EMBED_VERSION,
        "tagger_version": TAGGER_VERSION,
        "version": MODEL_VERSION,  # backward compat
        "scores": scores,
    }

    return tags, tag_meta


# ── Vector Storage (sidecar .npz file) ──────────────────────────────────────

def save_vectors(vectors_dict, path):
    """Save CLIP vectors to a compressed numpy archive.

    Crash-safe: writes to a .tmp file first, then does an atomic
    os.replace.  On failure the .tmp is cleaned up and the original
    file (if any) is left intact.

    Args:
        vectors_dict: {image_hash: numpy_array} mapping.
        path: Destination .npz file path.
    """
    if not vectors_dict:
        return
    tmp = path + ".tmp"
    try:
        np.savez_compressed(tmp, **vectors_dict)
        os.replace(tmp, path)
    except Exception:
        if os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def load_vectors(path):
    """Load CLIP vectors from a compressed numpy archive.

    Returns:
        dict {image_hash: numpy_array}, or empty dict if file missing.
    """
    if not os.path.isfile(path):
        return {}
    try:
        data = np.load(path)
        return {key: data[key] for key in data.files}
    except Exception:
        return {}
