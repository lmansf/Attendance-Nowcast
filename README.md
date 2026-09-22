# Attendance Nowcast — a data assimilation learning series

Estimate a theme park's **final daily attendance** and revise that estimate every
hour as cumulative gate counts arrive, using a Kalman filter written by hand in
NumPy. The pre-open daily forecast is the prior; each hour's gate count tightens it.

This is a learning project: every notebook explains the concept in plain English
before any code, and the filter math lives in small, commented functions.

## Status

| Notebook | Topic | Status |
| --- | --- | --- |
| `01_blending_two_guesses` | Toy example: one constant, a noisy sensor, the gain | ✅ built |
| `02_attendance_nowcast` | 1D filter on hourly gate counts, tightening ±2σ band | ⏳ waiting on data decision |
| `03_tuning_and_scoring` | `Q` and `R_t` tuning; error by hour vs forecast-only | ⏳ |
| `04_hidden_state` | 2D filter: final total + arrival pace | after sign-off on 01–03 |
| `05_ensemble_kalman` | Ensemble Kalman filter | after sign-off on 01–03 |

## Data source

Daily totals come from the Kaggle dataset
[`ayushtankha/hackathon`](https://www.kaggle.com/datasets/ayushtankha/hackathon)
(PortAventura World / Tivoli Gardens), the same source as
[Attendance-Projections](https://github.com/lmansf/Attendance-Projections).

**Simulated-data disclosure:** the dataset has daily attendance, not hourly gate
entries. Hourly arrivals used from notebook 02 onward will be **simulated** from
the daily totals with a configurable arrival curve, and labelled as simulated in
code, plots and here.

## How to run

```bash
pip install -r requirements.txt   # also installs src/nowcast in editable mode
python -m pytest                  # unit tests
jupyter lab notebooks/            # run any notebook top to bottom
```

Python 3.11+. No filter libraries and no AI/LLM APIs are used anywhere.

## Layout

```
src/nowcast/kalman.py   predict(), update(), run_filter() — 1D
src/nowcast/style.py    shared plot colours/style
notebooks/              the learning series
tests/                  pytest suite
```
