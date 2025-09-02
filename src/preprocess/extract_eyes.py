#!/usr/bin/env python3
"""
extract_eyes.py — Per-person eye cropper for MPIIFaceGaze with progress + revised CSV.

Usage examples:
  # Basic (uses label in pXX.txt)
  python -m src.preprocess.extract_eyes --person p00

  # Force both eyes, with progress bar and preview every 1000 crops
  python -m src.preprocess.extract_eyes --person p00 --eye-mode both --preview-every 1000

  # Print checkpoints every 200 lines instead of tqdm
  python -m src.preprocess.extract_eyes --person p00 --no-tqdm --log-every 200

CSV columns (in order):
  out_path, rel_path, file_name, eye_label, x0, y0, x1, y1, out_size, pad_scale
"""

from __future__ import annotations
import argparse
from pathlib import Path
import csv
import sys
from typing import List, Tuple, Optional

import cv2
import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False


# ----------------------------- helpers -----------------------------

def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur, *cur.parents]:
        if (p / "dataset" / "MPIIFaceGaze").exists() or (p / ".git").exists():
            return p
    return cur


def eye_bbox_from_corners(p1, p2, pad_scale=0.40) -> List[int]:
    """Return a square bbox (x0,y0,x1,y1) around the two eye-corner points."""
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)
    c  = (p1 + p2) / 2.0
    w  = np.linalg.norm(p1 - p2)
    side = w * (1 + pad_scale * 2)   # pad on both sides
    half = side / 2.0
    x0, y0 = c - half
    x1, y1 = c + half
    return [int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))]


def clip_bbox(x0, y0, x1, y1, W, H):
    x0 = max(0, min(x0, W-1))
    y0 = max(0, min(y0, H-1))
    x1 = max(0, min(x1, W))
    y1 = max(0, min(y1, H))
    if x1 <= x0: x1 = min(W, x0+1)
    if y1 <= y0: y1 = min(H, y0+1)
    return x0, y0, x1, y1


def crop_eye(img_bgr, bbox, out_size=64) -> Optional[np.ndarray]:
    x0, y0, x1, y1 = bbox
    crop = img_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)              # normalize contrast
    eye  = cv2.resize(gray, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return eye


def parse_landmarks(tokens: List[str]) -> List[Tuple[float, float]]:
    """
    Parse 12 landmark numbers starting at tokens[3:15], as used in your notebook cell.
    Expected order (assumed):
      0: Right eye - left corner
      1: Right eye - right corner
      2: Left  eye - left corner
      3: Left  eye - right corner
      4: Mouth - left corner
      5: Mouth - right corner
    """
    if len(tokens) < 16:
        raise ValueError(f"Annotation too short (len={len(tokens)}): {' '.join(tokens[:8])} ...")
    nums = list(map(float, tokens[3:15]))  # 12 numbers = 6 (x,y) pairs
    return [(nums[i], nums[i+1]) for i in range(0, 12, 2)]


def corners_from_landmarks(landmarks, which: str):
    right_eye_corners = (landmarks[0], landmarks[1])
    left_eye_corners  = (landmarks[2], landmarks[3])
    if which == "left":
        return left_eye_corners
    elif which == "right":
        return right_eye_corners
    else:
        raise ValueError(f"Invalid eye spec: {which}")


# ----------------------------- core -----------------------------

def process_subject(person: str,
                    dataset_root: Path,
                    out_dir: Path,
                    eye_mode: str = "label",  # "label"|"left"|"right"|"both"
                    out_size: int = 64,
                    pad_scale: float = 0.40,
                    max_images: Optional[int] = None,
                    verbose: bool = True,
                    use_tqdm: bool = True,
                    log_every: int = 100,
                    preview_every: Optional[int] = None) -> None:
    """
    Process one subject (e.g., 'p00') using that subject's annotation file.

    CSV columns written per crop:
      out_path, rel_path, file_name, eye_label, x0, y0, x1, y1, out_size, pad_scale
    """
    subj_dir = dataset_root / person
    if not subj_dir.exists():
        raise FileNotFoundError(f"Person folder not found: {subj_dir}")

    annot_path = subj_dir / f"{person}.txt"
    if not annot_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annot_path}")

    eyes_dir = out_dir / "eyes"
    eyes_dir.mkdir(parents=True, exist_ok=True)

    # CSV metadata
    meta_path = out_dir / "meta.csv"
    meta_fh = open(meta_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(meta_fh)
    writer.writerow([
        "out_path", "rel_path", "file_name", "eye_label",
        "x0","y0","x1","y1","out_size","pad_scale"
    ])

    processed = 0
    skipped = 0
    total_lines = 0

    with open(annot_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    iterator = lines
    show_tqdm = use_tqdm and HAS_TQDM and verbose
    if show_tqdm:
        iterator = tqdm(lines, desc=f"Extracting eyes {person}", unit="img")

    for line in iterator:
        total_lines += 1
        tokens = line.split()
        rel_path = tokens[0]
        eye_label = tokens[-1].lower() if tokens else ""

        # Decide which eyes to generate
        if eye_mode == "label":
            eyes_to_do = [eye_label] if eye_label in {"left","right"} else []
        elif eye_mode == "both":
            eyes_to_do = ["left", "right"]
        elif eye_mode in {"left", "right"}:
            eyes_to_do = [eye_mode]
        else:
            eyes_to_do = []

        if not eyes_to_do:
            skipped += 1
            if verbose and not show_tqdm and total_lines % log_every == 0:
                print(f"[skip] {rel_path}: invalid label '{eye_label}' for eye-mode={eye_mode}")
            continue

        img_path = subj_dir / rel_path
        if not img_path.exists():
            skipped += 1
            if verbose and not show_tqdm and total_lines % log_every == 0:
                print(f"[skip] missing image: {img_path}")
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            skipped += 1
            if verbose and not show_tqdm and total_lines % log_every == 0:
                print(f"[skip] failed to read: {img_path}")
            continue

        H, W = img_bgr.shape[:2]

        try:
            landmarks = parse_landmarks(tokens)
        except Exception as e:
            skipped += 1
            if verbose and not show_tqdm and total_lines % log_every == 0:
                print(f"[skip] parse landmarks failed for {rel_path}: {e}")
            continue

        for which in eyes_to_do:
            try:
                eye_corners = corners_from_landmarks(landmarks, which)
                bbox = eye_bbox_from_corners(*eye_corners, pad_scale=pad_scale)
                bbox = clip_bbox(*bbox, W, H)
                eye_img = crop_eye(img_bgr, bbox, out_size=out_size)
                if eye_img is None:
                    skipped += 1
                    if verbose and not show_tqdm and total_lines % log_every == 0:
                        print(f"[skip] empty crop: {rel_path} ({which}) bbox={bbox}")
                    continue

                # Save
                file_name = f"{rel_path.replace('/', '_')}_{which[0].upper()}.png"
                out_path = (eyes_dir / file_name).resolve()
                cv2.imwrite(str(out_path), eye_img)

                # CSV row per new spec
                writer.writerow([
                    str(out_path),          # out_path first
                    rel_path,               # rel_path as in annotation
                    file_name,              # new file name
                    which,                  # eye_label
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    out_size, pad_scale
                ])
                processed += 1

                # Optional preview
                if preview_every and processed % preview_every == 0:
                    preview_path = out_dir / f"preview_{processed}_{which}.jpg"
                    cv2.imwrite(str(preview_path), eye_img)

            except Exception as e:
                skipped += 1
                if verbose and not show_tqdm and total_lines % log_every == 0:
                    print(f"[skip] {rel_path} ({which}): {e}")

        # Lightweight periodic checkpoint if not using tqdm
        if verbose and not show_tqdm and total_lines % log_every == 0:
            print(f"[progress] processed={processed} skipped={skipped} "
                  f"({total_lines}/{len(lines)} lines)")

        if max_images is not None and processed >= max_images:
            if verbose and not show_tqdm:
                print(f"[info] Reached max-images={max_images}, stopping early.")
            break

    meta_fh.close()

    if verbose:
        # final summary (works with tqdm too)
        if show_tqdm:
            tqdm.write(f"Processed: {processed} crops; Skipped: {skipped}; Lines: {total_lines}")
        else:
            print("----- Summary -----")
            print(f"Subject:      {person}")
            print(f"Lines read:   {total_lines}")
            print(f"Processed:    {processed} crops")
            print(f"Skipped:      {skipped} items")
            print(f"Output dir:   {out_dir}")
            print(f"Meta CSV:     {meta_path}")


# ----------------------------- CLI -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Per-person eye extractor for MPIIFaceGaze (with progress)")
    base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    repo_root = find_repo_root(base)

    parser.add_argument("--person", required=True, help="Subject ID like p00, p01, ...")
    parser.add_argument("--dataset-root", type=Path,
                        default=repo_root / "dataset" / "MPIIFaceGaze",
                        help="Path to MPIIFaceGaze root (default: <repo>/dataset/MPIIFaceGaze)")
    parser.add_argument("--out-dir", type=Path,
                        help="Output directory (default: <repo>/outputs/preprocessed/<person>)")
    parser.add_argument("--eye-mode", choices=["label","left","right","both"], default="label",
                        help="Which eye(s) to save. 'label' uses last token in annotation line.")
    parser.add_argument("--out-size", type=int, default=64, help="Eye crop size (pixels).")
    parser.add_argument("--pad-scale", type=float, default=0.40, help="Padding scale around eye corners.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap on number of crops.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Reduce logging.")

    # Progress controls
    parser.add_argument("--no-tqdm", action="store_true",
                        help="Disable tqdm progress bar and use periodic print checkpoints instead.")
    parser.add_argument("--log-every", type=int, default=100,
                        help="When --no-tqdm is set, print a checkpoint every N lines (default: 100).")
    parser.add_argument("--preview-every", type=int, default=None,
                        help="If set (e.g., 1000), saves a preview crop every N processed eyes.")

    args = parser.parse_args()

    if not args.dataset_root.exists():
        sys.exit(f"Dataset root not found: {args.dataset_root}")

    out_dir = args.out_dir or (repo_root / "outputs" / "preprocessed" / args.person)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_tqdm = (not args.no_tqdm)

    process_subject(
        person=args.person,
        dataset_root=args.dataset_root,
        out_dir=out_dir,
        eye_mode=args.eye_mode,
        out_size=args.out_size,
        pad_scale=args.pad_scale,
        max_images=args.max_images,
        verbose=not args.quiet,
        use_tqdm=use_tqdm,
        log_every=args.log_every,
        preview_every=args.preview_every
    )


if __name__ == "__main__":
    main()
