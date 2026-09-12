# SmartANPR — Evaluation Harness

A small, **fixed**, reproducible evaluation setup for measuring the baseline
SmartANPR OCR quality, so future improvements are compared on exactly the same
samples.

> This harness **does not modify** the detection/OCR algorithm. It imports
> `read_license_plate` from the project's own `util.py` and reproduces the exact
> crop preprocessing used in `main.py` (grayscale → threshold `64` →
> `THRESH_BINARY_INV`). It measures the baseline as-is.

---

## Why you must label ground truth first

There is **no ground-truth plate text in this repository.** The only text data,
`test.csv`, is the **baseline model's own output** — grading the model against
its own predictions would report a meaningless ~100% and would be fake.

The license-plate detector was also trained `single_cls=True` (plate vs. not),
so even the original training dataset has **no character transcriptions** to
borrow. Real, human-verified labels are therefore required.

To keep this feasible on a laptop, you only need a **small** set (see sizing
below). The set is fixed once created so every experiment is comparable.

---

## Files

| File | Purpose |
|---|---|
| `evaluate_ocr.py` | The evaluator. Runs baseline OCR on labeled crops, scores it. |
| `ground_truth.template.csv` | Copy this to `ground_truth.csv` and fill in real labels. |
| `ground_truth.csv` | **You create this.** The fixed evaluation set. |
| `results/ocr_per_sample.csv` | Generated: per-sample predictions + outcomes. |
| `results/ocr_summary.json` | Generated: aggregate metrics. |

`ground_truth.csv` and `results/` are the experiment record — commit
`ground_truth.csv` so the fixed set is shared; `results/` can be regenerated.

---

## Ground-truth format

One row = one license-plate instance you have visually confirmed.

```csv
sample_id,frame_nmr,plate_bbox,true_plate,notes
gt_001,2,"960 1758 1215 1874",NA13NRU,clear frontal
gt_002,94,"1237 1677 1441 1787",GX15OGJ,slight angle
```

- `sample_id` — stable unique id. **Never renumber** existing ids; the set must
  stay fixed.
- `frame_nmr` — 0-based frame index in your **local** video, in the same read
  order `main.py` uses.
- `plate_bbox` — plate box in that frame, pixels, `"x1 y1 x2 y2"` (quoted,
  space-separated — same convention as `test.csv`).
- `true_plate` — the characters **you** read by eye, uppercase, no spaces. This
  is the ground truth. **Do not copy it from `test.csv`.**
- `notes` — optional (e.g. `angled`, `blur`, `occluded`).

Lines starting with `#` and blank lines are ignored.

### How to get `frame_nmr` and `plate_bbox` quickly

The baseline output `test.csv` already lists candidate frames and plate boxes.
Use it only as a **convenience index to locate plates** — then look at the
actual frame and type the **true** characters yourself (which may differ from
what the model guessed). Copy the `license_plate_bbox` value into `plate_bbox`.

---

## Recommended evaluation-set size

Limited compute/labeling time, so keep it small but meaningful:

- **Minimum useful:** ~20–30 distinct plates.
- **Comfortable:** ~40–60 plates covering a mix of easy (frontal, sharp) and
  hard (angled, blurred, small, low-light) cases.

Include hard cases deliberately — they are where improvements will show up.
Label each plate once (its clearest frame) unless you specifically want to test
frame-to-frame consistency later.

---

## Providing the video

The harness reads frames from your **local** video (not committed to the repo;
see `data/README` / runtime docs). Point `--video` at it. The `frame_nmr`
values in `ground_truth.csv` must match this exact file.

---

## Running

From the project root (`SmartANPR/SmartANPR/`):

```bash
python eval/evaluate_ocr.py \
    --video /path/to/your/local/sample.mp4 \
    --ground-truth eval/ground_truth.csv \
    --out-dir eval/results
```

Options:
- `--ground-truth` defaults to `eval/ground_truth.csv`.
- `--out-dir` defaults to `eval/results`.

If ground truth or the video is missing, the script prints what's needed and
exits **without** producing metrics — it never fabricates data.

---

## Metrics reported

- **Exact-match accuracy** = `correct / total`. The primary headline number.
- **Character-level accuracy** = mean of `1 − editDistance(pred, truth)/len(truth)`
  per sample (missing reads count as 0). Rewards partial correctness.
- **Read rate** = fraction that produced *any* string (correct or wrong). Low
  read rate points at the format filter / detection, not OCR quality.
- **OCR latency / FPS** = time spent inside `read_license_plate` per sample.

### Correct vs. Wrong vs. Missing

The per-sample CSV labels every sample as exactly one of:
- `correct` — prediction equals ground truth.
- `wrong` — OCR produced a string, but it differs.
- `missing` — OCR returned `None` (rejected by the 7-char format filter,
  unreadable crop, or the frame was unavailable). The `reason` column
  disambiguates.

This separation matters: a **missing** result is usually the rigid format
filter rejecting a valid plate, whereas a **wrong** result is a genuine OCR
error. Future improvements target these differently.

---

## What this harness deliberately does NOT do

- It does not change thresholds, the format validator, or OCR settings.
- It does not (yet) score plate-detection precision/recall — that needs
  ground-truth detection boxes over full frames, which the current labeling
  format does not capture. It can be added later without changing this file's
  outputs.
