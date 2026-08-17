# Tennis Match Prediction

A neural network that predicts pre-match Bet365 betting odds for WTA Grand
Slam matches, with a desktop app for testing predictions interactively.

Trained on WTA Grand Slam match data (2016-2023) combined with Bet365 odds,
sourced from [Jeff Sackmann's WTA match data](https://github.com/JeffSackmann/tennis_wta)
and [tennis-data.co.uk](http://www.tennis-data.co.uk/).

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
- `requirements.txt` — pinned dependency versions

## Setup

Requires Python 3.10-3.13 (TensorFlow doesn't yet support 3.14+).

```bash
pip install -r requirements.txt
```

## Running it

Run these in order from the project folder:

```bash
python clean_data.py     # builds the training data + player lookup
python tennis_cv.py      # trains the model (takes a while -- 300 epochs x 5 folds)
python tennis_app.py     # opens the prediction app
```

Each step only needs to be re-run if you change the underlying data --
`tennis_app.py` can be re-run on its own once the model exists.

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
