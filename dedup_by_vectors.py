"""
Visual dedup using image vectors.
1. Compute a feature vector for each image (resized pixels + color histogram)
2. Find all pairs with cosine similarity > 0.90
3. Group similar images, keep only the best quality from each group
4. Remove duplicates from sorted folders, presentation, and OneDrive
"""

import os
import sys
import json
import re
import numpy as np
from datetime import datetime
from PIL import Image
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)

OUTPUT_DIR = r"C:\Codes\Reef images for bar mitza\sorted"
ONEDRIVE_DIR = r"C:\Users\erezg\OneDrive - Pen-Link Ltd\Desktop\אישי\ריף\בר מצווה"
MANIFEST_FILE = r"C:\Codes\Reef images for bar mitza\manifest.json"
VECTORS_FILE = r"C:\Codes\Reef images for bar mitza\image_vectors.npz"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".heic", ".webp"}
SIMILARITY_THRESHOLD = 0.90
VECTOR_SIZE = 64  # resize to 64x64


def compute_vector(image_path):
    """
    Compute feature vector: resized grayscale pixels + color histogram.
    Returns a normalized float vector.
    """
    try:
        img = Image.open(image_path).convert("RGB")

        # Feature 1: Resized grayscale pixels (structural info)
        gray = img.convert("L").resize((VECTOR_SIZE, VECTOR_SIZE), Image.LANCZOS)
        pixels = np.array(gray, dtype=np.float32).flatten()

        # Feature 2: Color histogram (16 bins per channel = 48 values)
        img_small = img.resize((128, 128), Image.LANCZOS)
        arr = np.array(img_small)
        hist_r = np.histogram(arr[:, :, 0], bins=16, range=(0, 256))[0].astype(np.float32)
        hist_g = np.histogram(arr[:, :, 1], bins=16, range=(0, 256))[0].astype(np.float32)
        hist_b = np.histogram(arr[:, :, 2], bins=16, range=(0, 256))[0].astype(np.float32)

        # Combine: pixels (4096) + color hist (48) = 4144 dimensions
        vector = np.concatenate([pixels, hist_r * 10, hist_g * 10, hist_b * 10])

        # Normalize to unit vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm

        return vector
    except Exception as e:
        return None


def quality_score(image_path):
    """Higher = better quality."""
    try:
        sz = os.path.getsize(image_path)
        img = Image.open(image_path)
        w, h = img.size
        img.close()
        return w * h * (sz / 1024)  # resolution * filesize
    except Exception:
        return 0


def find_groups(similarity_matrix, threshold):
    """
    Given a similarity matrix, find groups of images that are similar.
    Uses union-find for efficient grouping.
    """
    n = similarity_matrix.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Find all pairs above threshold
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] >= threshold:
                union(i, j)
                pairs += 1

    # Build groups
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    # Only return groups with more than 1 member
    return [members for members in groups.values() if len(members) > 1], pairs


def main():
    print("=" * 60)
    print("Visual Dedup by Image Vectors")
    print("=" * 60)

    # ── Step 1: Collect all images from sorted folders ─────────────────────────
    print("\nStep 1: Collecting images...")
    images = []  # (folder_name, filename, full_path)

    for folder in sorted(os.listdir(OUTPUT_DIR)):
        folder_path = os.path.join(OUTPUT_DIR, folder)
        if not os.path.isdir(folder_path) or folder in ("the presentation", "extra images"):
            continue
        for fname in sorted(os.listdir(folder_path)):
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTS:
                images.append((folder, fname, os.path.join(folder_path, fname)))

    print(f"  Total images: {len(images)}")

    # ── Step 2: Compute vectors ────────────────────────────────────────────────
    print("\nStep 2: Computing image vectors...")
    vectors = []
    valid_indices = []

    for i, (folder, fname, fpath) in enumerate(images):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(images)}...")

        vec = compute_vector(fpath)
        if vec is not None:
            vectors.append(vec)
            valid_indices.append(i)
        else:
            print(f"  WARN: Could not process {folder}/{fname}")

    vectors = np.array(vectors, dtype=np.float32)
    print(f"  Computed {len(vectors)} vectors ({vectors.shape[1]} dimensions each)")

    # Save vectors for future use
    np.savez_compressed(VECTORS_FILE,
                       vectors=vectors,
                       indices=np.array(valid_indices))
    print(f"  Saved to {VECTORS_FILE}")

    # ── Step 3: Compute similarity matrix ──────────────────────────────────────
    print("\nStep 3: Computing cosine similarity matrix...")
    # Cosine similarity = dot product of unit vectors
    similarity = vectors @ vectors.T
    print(f"  Matrix shape: {similarity.shape}")

    # Count pairs above threshold
    above = np.sum(similarity > SIMILARITY_THRESHOLD) - len(vectors)  # subtract diagonal
    print(f"  Pairs above {SIMILARITY_THRESHOLD} threshold: {above // 2}")

    # ── Step 4: Find groups ────────────────────────────────────────────────────
    print("\nStep 4: Finding similar image groups...")
    groups, n_pairs = find_groups(similarity, SIMILARITY_THRESHOLD)
    print(f"  Found {len(groups)} groups of similar images ({n_pairs} pairs)")

    # ── Step 5: Decide what to keep/remove ─────────────────────────────────────
    print("\nStep 5: Selecting best from each group...")
    to_remove = []  # (folder, filename, reason)

    for group_idx, members in enumerate(groups):
        # Get image info for each member
        group_images = []
        for idx in members:
            real_idx = valid_indices[idx]
            folder, fname, fpath = images[real_idx]
            score = quality_score(fpath)
            group_images.append((folder, fname, fpath, score, idx))

        # Sort by quality (best first)
        group_images.sort(key=lambda x: x[3], reverse=True)

        # Keep the best, remove the rest
        keep = group_images[0]
        for folder, fname, fpath, score, idx in group_images[1:]:
            # Calculate similarity with the kept image
            sim = float(similarity[keep[4], idx])
            to_remove.append((folder, fname, f"similar to {keep[0]}/{keep[1]} ({sim:.2f})"))

        if len(group_images) <= 5:
            names = [f"{f}/{n}" for f, n, _, _, _ in group_images]
            print(f"  Group {group_idx + 1}: {len(group_images)} images, keep {keep[0]}/{keep[1]}")
            for folder, fname, fpath, score, idx in group_images[1:]:
                sim = float(similarity[keep[4], idx])
                print(f"    remove {folder}/{fname} (sim={sim:.2f})")
        elif (group_idx + 1) % 10 == 0 or group_idx < 3:
            print(f"  Group {group_idx + 1}: {len(group_images)} images, keep {keep[0]}/{keep[1]}, remove {len(group_images) - 1}")

    print(f"\nTotal to remove: {len(to_remove)}")

    if not to_remove:
        print("Nothing to remove!")
        return

    # ── Step 6: Remove duplicates ──────────────────────────────────────────────
    print("\nStep 6: Removing duplicates...")

    pres_sorted = os.path.join(OUTPUT_DIR, "the presentation")
    pres_onedrive = os.path.join(ONEDRIVE_DIR, "the presentation")

    removed_count = 0
    removed_per_folder = defaultdict(int)

    for folder, fname, reason in to_remove:
        # Remove from sorted folder
        src_path = os.path.join(OUTPUT_DIR, folder, fname)
        if os.path.exists(src_path):
            os.remove(src_path)

        # Remove from OneDrive age folder
        od_path = os.path.join(ONEDRIVE_DIR, folder, fname)
        if os.path.exists(od_path):
            os.remove(od_path)

        # Remove from presentation folders
        pres_name = f"{folder}__{fname}"
        for pres_dir in [pres_sorted, pres_onedrive]:
            if pres_dir and os.path.exists(pres_dir):
                pres_path = os.path.join(pres_dir, pres_name)
                if os.path.exists(pres_path):
                    os.remove(pres_path)

        removed_count += 1
        removed_per_folder[folder] += 1

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DEDUP SUMMARY")
    print("=" * 60)
    print(f"Similar groups found: {len(groups)}")
    print(f"Images removed: {removed_count}")
    print(f"\nRemoved per folder:")
    for folder, count in sorted(removed_per_folder.items()):
        remaining = sum(1 for f in os.listdir(os.path.join(OUTPUT_DIR, folder))
                       if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
        print(f"  {folder}: -{count} (now {remaining})")

    # Presentation count
    pres_count = 0
    if os.path.exists(pres_sorted):
        pres_count = sum(1 for f in os.listdir(pres_sorted)
                        if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
    print(f"\n  the presentation: {pres_count} images")

    # Save removal log
    log_file = os.path.join(OUTPUT_DIR, "..", "dedup_log.json")
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump([{"folder": fo, "file": fi, "reason": r} for fo, fi, r in to_remove],
                 f, ensure_ascii=False, indent=2)
    print(f"\nRemoval log saved to {log_file}")


if __name__ == "__main__":
    main()
