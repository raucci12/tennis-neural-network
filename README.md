# CoachCV

A computer vision app that analyzes Olympic weightlifting Clean & Jerk
videos, detects each lift phase (Setting, Cleaning, Hold, Pressing,
Complete, Release), and generates natural-language coaching feedback
using a locally-running LLM -- no API key, no cost.

Originally built as a final project for an Augmented Intelligence course.
See `CoachCV Poster.pdf` for the original project overview.

## Demo

<!-- TODO: paste your recorded video link here, e.g.:
https://github.com/user-attachments/assets/your-video-id-here
-->

A short walkthrough analyzing a real Clean & Jerk video, from raw footage
through detected phases to the final coaching feedback.

## What's in this repo

- `best.pt` -- pre-trained YOLOv8 model, included so you can run the app
  immediately without training anything yourself
- `analyze_video.py` -- run THIS to evaluate a video
- `coach_feedback.py` -- used automatically by analyze_video.py; also
  where you can edit the coach's personality/tone (the `SYSTEM_PROMPT`)
- `requirements.txt` -- pinned dependency versions
- `train.py`, `data.yaml` -- only needed if you want to retrain the model
  yourself (see "Retraining" below)

## Setup

Requires Python 3.10-3.13.

**If you're also trying my other projects (like the tennis prediction
app) on the same machine, use an isolated virtual environment for each
one** -- this repo and the tennis project have some overlapping-but-
different dependency requirements (notably around numpy), and installing
everything into one shared global Python environment can cause one
project's setup to silently break another's. A virtual environment keeps
each project's packages completely separate.

```bash
git clone <this-repo-url>
cd coach-cv

python -m venv venv

# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

You'll know it worked if your terminal prompt now shows `(venv)` at the
start of the line. From here on, run everything below from inside this
activated environment. To leave it when you're done: `deactivate`.

You'll also need [Ollama](https://ollama.com) installed and running
locally (this is separate from the Python virtual environment, since it's
its own application, not a Python package), with a model pulled:
```bash
ollama pull llama3.2
```

## Quick start (use the pre-trained model)

```bash
python analyze_video.py path/to/your_video.mp4
```

**For best results, keep recording 2-3+ seconds after the lift
completes** -- the Release phase needs that extra footage to be detected
reliably.

You'll see the detected phases with timestamps, followed by coaching
feedback. If the lift was completed successfully (all six phases, correct
order), the feedback will call out specifically what went well, plus one
refinement to focus on next rep. If something's off, it'll identify the
issue and give one concrete correction.

## How it works

1. **YOLOv8** (`best.pt`) processes the video frame by frame, detecting
   which lift phase is happening in each frame, plus tracking the
   barbell's position.
2. Phase detections are smoothed (ignoring brief misclassification blips)
   and collapsed into clean segments with timestamps.
3. If `Release` isn't confidently detected by the vision model, it's
   inferred instead from a sustained downward drop in barbell position
   after `Complete` -- a hybrid approach that turned out to be more
   reliable than relying on the vision model alone for that specific
   phase.
4. The segment list is converted into a plain-text summary and sent to a
   local LLM (via Ollama) with a coaching system prompt, which generates
   the final feedback.

## Retraining

If you want to retrain the model on your own annotated dataset:

```bash
python train.py
```

This expects a `Data` folder (containing `images/train`, `images/val`,
`images/test` and matching `labels` folders) placed next to `data.yaml`
in this same directory -- **not included in this repo** due to its size
(~19,000 annotated frames). If you have your own annotated dataset in
YOLO format, drop it in and update `data.yaml` if your class list
differs.

Training is CPU-feasible but slow (potentially many hours to days). For
a dramatically faster free option, train in
[Google Colab](https://colab.research.google.com) with a free GPU
instead -- upload your `Data` folder to Google Drive and point
`data.yaml`'s `path` at it there.

## Notes / known limitations

- The `Release` phase is inherently harder to detect than the others --
  it's brief and visually can resemble earlier phases. The barbell-motion
  fallback described above significantly improves this, but a longer
  training dataset specifically for `Release` frames would likely help
  further.
- Debug mode (`python analyze_video.py your_video.mp4 --debug`) prints
  every raw detection with its confidence, useful for diagnosing why a
  particular phase isn't showing up.
