#!/usr/bin/env python3
"""
Baseline OCR evaluation harness for SmartANPR.

Purpose
-------
Measure the EXISTING baseline pipeline's license-plate OCR quality on a FIXED,
hand-labeled evaluation set, so that future improvements are compared on exactly
the same samples. This script does NOT modify or reimplement the detection/OCR
algorithm - it imports `read_license_plate` from the project's own `util.py` and
reproduces the same crop preprocessing used in `main.py`
(grayscale -> fixed threshold 64 -> THRESH_BINARY_INV).

What it does
------------
1. Loads the fixed evaluation samples (eval/ground_truth.csv).
2. For each sample: seeks the labeled frame in the LOCAL video, crops the
   labeled plate bbox, runs the unchanged baseline OCR path.
3. Compares the predicted string against the human ground-truth string.
4. Computes: exact-match accuracy, character-level accuracy, read rate, and
   per-sample OCR latency / throughput (FPS).
5. Distinguishes MISSING/FAILED OCR (model returned nothing) from WRONG OCR
   (model returned a different string).
6. Saves per-sample results (CSV) and an aggregate summary (JSON).

It intentionally computes NO metrics if ground truth is absent - it will not
invent data.

Usage
-----
    python eval/evaluate_ocr.py \
        --video /path/to/local/sample.mp4 \
        --ground-truth eval/ground_truth.csv \
        --out-dir eval/results

Run `python eval/evaluate_ocr.py --help` for all options.
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone

# --- Make the project root importable so we reuse the ORIGINAL util.py --------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# Outcome labels -- kept explicit so "missing" is never confused with "wrong".
OUTCOME_CORRECT = "correct"
OUTCOME_WRONG = "wrong"
OUTCOME_MISSING = "missing"  # OCR returned None: rejected by format filter / unreadable


def levenshtein(a, b):
    """Edit distance between two strings (pure Python, no extra deps)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def char_accuracy(pred, truth):
    """
    Character-level accuracy = 1 - (edit_distance / len(truth)).
    Clamped to [0, 1]. If pred is None/empty it counts as 0 correct chars.
    """
    truth = truth or ""
    pred = pred or ""
    if len(truth) == 0:
        return None  # cannot score against empty ground truth
    dist = levenshtein(pred, truth)
    return max(0.0, 1.0 - dist / len(truth))


def parse_bbox(raw):
    """
    Parse a plate bbox stored as 'x1 y1 x2 y2' (optionally bracketed) into
    four ints. Mirrors the whitespace-separated convention used in test.csv.
    """
    s = raw.strip().strip("[]")
    parts = [p for p in s.replace(",", " ").split() if p]
    if len(parts) != 4:
        raise ValueError(f"bbox must have 4 numbers, got: {raw!r}")
    x1, y1, x2, y2 = (float(p) for p in parts)
    return int(x1), int(y1), int(x2), int(y2)


def load_ground_truth(path):
    """Load and validate the fixed evaluation set. Ignores # comment lines."""
    if not os.path.exists(path):
        return None, f"Ground-truth file not found: {path}"

    rows = []
    with open(path, "r", newline="") as f:
        # Strip comment lines and blank lines before parsing.
        cleaned = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]

    if not cleaned:
        return [], None

    reader = csv.DictReader(cleaned)
    required = {"sample_id", "frame_nmr", "plate_bbox", "true_plate"}
    missing_cols = required - set(reader.fieldnames or [])
    if missing_cols:
        return None, f"Ground-truth is missing required columns: {sorted(missing_cols)}"

    for i, r in enumerate(reader, 1):
        true_plate = (r.get("true_plate") or "").strip().upper()
        if not true_plate:
            return None, f"Row {i} ({r.get('sample_id')}): empty true_plate. Fill in real labels."
        try:
            frame_nmr = int(r["frame_nmr"])
            bbox = parse_bbox(r["plate_bbox"])
        except (ValueError, KeyError) as e:
            return None, f"Row {i} ({r.get('sample_id')}): {e}"
        rows.append(
            {
                "sample_id": (r.get("sample_id") or f"row_{i}").strip(),
                "frame_nmr": frame_nmr,
                "bbox": bbox,
                "true_plate": true_plate,
                "notes": (r.get("notes") or "").strip(),
            }
        )
    return rows, None


def run_baseline_ocr_on_crop(frame, bbox, read_license_plate, cv2):
    """
    Reproduce EXACTLY the baseline crop + preprocessing from main.py, then call
    the project's own read_license_plate(). No algorithm changes.

    Returns (pred_text_or_None, score_or_None, ocr_seconds).
    """
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2, :]
    if crop.size == 0:
        return None, None, 0.0

    # --- identical to main.py baseline path ---
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, crop_thresh = cv2.threshold(crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

    t0 = time.perf_counter()
    text, score = read_license_plate(crop_thresh)
    dt = time.perf_counter() - t0
    return text, score, dt


def evaluate(video_path, gt_rows, out_dir):
    # Imports deferred so --help works without heavy deps installed.
    import cv2
    from util import read_license_plate

    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"ERROR: could not open video: {video_path}")

    per_sample = []
    n_correct = n_wrong = n_missing = 0
    char_acc_sum = 0.0
    char_acc_count = 0
    ocr_time_total = 0.0

    for row in gt_rows:
        cap.set(cv2.CAP_PROP_POS_FRAMES, row["frame_nmr"])
        ret, frame = cap.read()
        if not ret or frame is None:
            # Frame unavailable in this video: record explicitly, do not fabricate.
            per_sample.append(
                {
                    **_flat(row),
                    "pred_plate": "",
                    "pred_score": "",
                    "outcome": OUTCOME_MISSING,
                    "reason": "frame_unavailable",
                    "char_accuracy": "",
                    "ocr_seconds": "",
                }
            )
            n_missing += 1
            continue

        pred, score, dt = run_baseline_ocr_on_crop(frame, row["bbox"], read_license_plate, cv2)
        ocr_time_total += dt

        if pred is None:
            outcome, reason = OUTCOME_MISSING, "ocr_returned_none"
            n_missing += 1
            cacc = 0.0
        elif pred == row["true_plate"]:
            outcome, reason = OUTCOME_CORRECT, ""
            n_correct += 1
            cacc = 1.0
        else:
            outcome, reason = OUTCOME_WRONG, ""
            n_wrong += 1
            cacc = char_accuracy(pred, row["true_plate"])

        # Character accuracy is defined for every scored sample (missing -> 0).
        if cacc is not None:
            char_acc_sum += cacc
            char_acc_count += 1

        per_sample.append(
            {
                **_flat(row),
                "pred_plate": pred or "",
                "pred_score": "" if score is None else round(float(score), 6),
                "outcome": outcome,
                "reason": reason,
                "char_accuracy": "" if cacc is None else round(cacc, 6),
                "ocr_seconds": round(dt, 6),
            }
        )

    cap.release()

    n_total = len(gt_rows)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "video": os.path.abspath(video_path),
        "n_samples": n_total,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "n_missing": n_missing,
        "exact_match_accuracy": _safe_div(n_correct, n_total),
        "read_rate": _safe_div(n_correct + n_wrong, n_total),  # produced any string
        "char_level_accuracy_mean": _safe_div(char_acc_sum, char_acc_count),
        "ocr_seconds_total": round(ocr_time_total, 6),
        "ocr_seconds_per_sample": _safe_div(ocr_time_total, n_total),
        "ocr_fps": _safe_div(n_total, ocr_time_total) if ocr_time_total > 0 else None,
        "note": (
            "exact_match_accuracy = correct / total. "
            "read_rate counts any produced string (correct or wrong). "
            "MISSING = OCR returned None (format filter reject / unreadable / frame gone), "
            "distinct from WRONG."
        ),
    }

    # Write outputs.
    per_sample_path = os.path.join(out_dir, "ocr_per_sample.csv")
    summary_path = os.path.join(out_dir, "ocr_summary.json")

    fieldnames = [
        "sample_id", "frame_nmr", "true_plate", "pred_plate", "pred_score",
        "outcome", "reason", "char_accuracy", "ocr_seconds", "notes",
    ]
    with open(per_sample_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_sample:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary, per_sample_path, summary_path


def _flat(row):
    return {
        "sample_id": row["sample_id"],
        "frame_nmr": row["frame_nmr"],
        "true_plate": row["true_plate"],
        "notes": row["notes"],
    }


def _safe_div(a, b):
    return round(a / b, 6) if b else None


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Evaluate baseline SmartANPR OCR on a fixed labeled set (no algorithm changes).",
    )
    p.add_argument("--video", required=True,
                   help="Path to the LOCAL input video the ground truth was labeled against.")
    p.add_argument("--ground-truth", default=os.path.join(THIS_DIR, "ground_truth.csv"),
                   help="Fixed evaluation set CSV (default: eval/ground_truth.csv).")
    p.add_argument("--out-dir", default=os.path.join(THIS_DIR, "results"),
                   help="Directory to write results into (default: eval/results).")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    gt_rows, err = load_ground_truth(args.ground_truth)
    if err:
        print(f"[ground-truth] {err}")
        print("Nothing to evaluate. Create eval/ground_truth.csv from the template "
              "and add real labels first. No metrics were fabricated.")
        return 2
    if not gt_rows:
        print(f"[ground-truth] {args.ground_truth} contains no labeled samples yet.")
        print("Add real labels (see eval/README.md). No metrics were fabricated.")
        return 2

    if not os.path.exists(args.video):
        print(f"[video] Not found: {args.video}")
        print("Provide the local video the labels were made against (see eval/README.md).")
        return 2

    print(f"Loaded {len(gt_rows)} labeled sample(s) from {args.ground_truth}")
    summary, per_sample_path, summary_path = evaluate(args.video, gt_rows, args.out_dir)

    print("\n=== BASELINE OCR RESULTS ===")
    print(f"  samples              : {summary['n_samples']}")
    print(f"  correct              : {summary['n_correct']}")
    print(f"  wrong                : {summary['n_wrong']}")
    print(f"  missing/failed       : {summary['n_missing']}")
    print(f"  exact-match accuracy : {summary['exact_match_accuracy']}")
    print(f"  char-level accuracy  : {summary['char_level_accuracy_mean']}")
    print(f"  read rate            : {summary['read_rate']}")
    print(f"  OCR sec / sample     : {summary['ocr_seconds_per_sample']}")
    print(f"  OCR FPS              : {summary['ocr_fps']}")
    print(f"\nPer-sample -> {per_sample_path}")
    print(f"Summary    -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
