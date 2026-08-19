# Tennis Match Prediction

A neural network that predicts pre-match Bet365 betting odds for WTA Grand
Slam matches, with a desktop app for testing predictions interactively.

Trained on WTA Grand Slam match data (2016-2023) combined with Bet365 odds,
sourced from [Jeff Sackmann's WTA match data](https://github.com/JeffSackmann/tennis_wta)
and [tennis-data.co.uk](http://www.tennis-data.co.uk/).

## Demo

<!-- TODO: paste your recorded video link here, e.g.:
https://github.com/user-attachments/assets/your-video-id-here
-->

A short walkthrough selecting players, running a prediction, and using
the live match-duration slider to see odds update in real time.

## What's in this repo

- `Full_Dataset.xlsx` — raw merged match + odds data
- `clean_data.py` — cleans the raw data into model-ready features, and
  builds the player lookup (names, heights, hands, current rank/points)
  used by the app's dropdowns
- `tennis_cv.py` — trains the neural network (5-fold cross-validation,
  batch normalization, L2 regularization)
- `tennis_app.py` — desktop app (Tkinter) for testing predictions, with
  dropdowns for players/country/surface and a live match-duration slider
- `active_players.csv` — current WTA rankings snapshot used to filter the
  player dropdown to active players only (retired players stay in the
  training data, just hidden from the dropdown)
- `tennis_model.h5`, `scalers.pkl`, `encoders.pkl`, `features.pkl` —
  pre-trained model artifacts, included so you can run the app immediately
  without training anything yourself
- `requirements.txt` — pinned dependency versions

## Setup

Requires Python 3.10-3.13 (TensorFlow doesn't yet support 3.14+).

**If you're also trying my other projects (like CoachCV) on the same
machine, use an isolated virtual environment for each one** -- this repo
and CoachCV have some overlapping-but-different dependency requirements
(notably around numpy), and installing everything into one shared global
Python environment can cause one project's setup to silently break
another's. A virtual environment keeps each project's packages completely
separate.

```bash
git clone <this-repo-url>
cd tennis-neural-network

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

## Quick start (use the pre-trained model)

The trained model is already included in this repo, so you can go straight
to the app:

```bash
python tennis_app.py
```

## Retraining from scratch

If you want to retrain the model yourself (e.g. after updating the data or
tweaking the architecture):

```bash
python clean_data.py     # rebuilds the training data + player lookup
python tennis_cv.py      # retrains the model (takes a while -- 300 epochs x 5 folds)
python tennis_app.py     # opens the prediction app with your newly trained model
```

This will overwrite `tennis_model.h5`, `scalers.pkl`, `encoders.pkl`, and
`features.pkl` with your own freshly trained versions.

## Updating the active player list

`active_players.csv` is a snapshot of WTA rankings (name, rank, points) at
the time it was created. To refresh it: copy an updated rankings table
(e.g. from a live rankings page) into the same `name / current_rank /
current_pts` CSV format, replace `active_players.csv`, and re-run
`clean_data.py`. No filter file at all means every player in the dataset
shows up in the dropdown, active or not.

## Notes

- The player dropdown auto-fills height, dominant hand, age (projected to
  the 2025 Grand Slam you select), rank, and points for any recognized
  player -- pick "Unknown Player" to enter those manually instead.
- A handful of players have no recorded height in the source data; those
  fall back to the dataset median rather than being left blank.
- Court Surface auto-fills based on the Tournament Country, since each
  Grand Slam is always played on the same surface.
