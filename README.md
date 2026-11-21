# Forecasting Lahore's Smog

**One-day-ahead PM2.5, AQI category and hazardous-air warnings from air-quality, weather and crop-burning data**

DSTT Term Project, FAST-NUCES Lahore · Instructor: Dr. Asma Ahmad · TA: Burhan Hadi Butt
**Team:** Umar Farooq (26L-8032) · Muhammad Umer (26L-8023)

---

## 1. Project overview

Every winter, Lahore's air is among the most polluted in the world. On many days the 24-hour average of PM2.5 goes past **225.5 µg/m³**, the US EPA *Hazardous* level. The causes are weather that traps pollution near the ground, and smoke from crop-residue burning in Pakistani and Indian Punjab.

**Goal:** every evening, use what is already known (today's PM2.5, tomorrow's weather forecast, satellite fire detections) to predict:

| # | Task | Type | Target |
|---|---|---|---|
| 1 | Tomorrow's average PM2.5 | Regression | `pm25_tomorrow` |
| 2 | Tomorrow's AQI category | Multi-class classification | `aqi_category_tomorrow` |
| 3 | Will tomorrow be Hazardous (≥ 225.5 µg/m³)? | Binary early warning | `hazardous_tomorrow` |
| 4 | Do crop-fire features add predictive power? | Research question | fire features |

The project has two notebooks:

| Notebook | Content |
|---|---|
| **`Lahore_Smog_Phase1_EDA.ipynb`** | Data acquisition (APIs) → loading and description → cleaning (missing values, duplicates, outliers) → preprocessing (calibration, vector-averaged wind, fire geometry) → feature engineering and selection → EDA (19 charts) → insights |
| **`Lahore_Smog_Phase2_Modelling.ipynb`** | Train / validation / test split (time-based) → baselines → model selection (Ridge, Random Forest, Gradient Boosting) → hyperparameter tuning (`RandomizedSearchCV` + `TimeSeriesSplit`) → optimisation (target choice, ensemble) → warning threshold → final evaluation on two held-out winters → error analysis and permutation importance → fire-feature ablation study → **final saved pipeline and `forecast()` function** |

Both notebooks are saved **with all outputs**, so the results can be read on GitHub without running anything.

**Open in Google Colab:**
[Phase 1 notebook](https://colab.research.google.com/github/umfar46/lahore-smog-forecasting/blob/main/Lahore_Smog_Phase1_EDA.ipynb) · [Phase 2 notebook](https://colab.research.google.com/github/umfar46/lahore-smog-forecasting/blob/main/Lahore_Smog_Phase2_Modelling.ipynb). Click **Runtime → Run all**; the data downloads automatically.

## 2. Results

Final model: an **ensemble** (average) of tuned Ridge, Random Forest and tuned Histogram Gradient Boosting. It predicts the *change* from today's PM2.5, using 15 features. It was evaluated on two smog winters that were never used for training or tuning:

| Test winter | Persistence MAE ("tomorrow = today") | **Final model MAE** | Improvement | R² | Hazardous warning recall / precision | AQI accuracy (persistence → model) |
|---|---|---|---|---|---|---|
| 2024–25 | 52.5 µg/m³ | **44.3 µg/m³** | **15.5 %** | 0.64 | 91 % / 75 % | 73 % → 77 % |
| 2025–26 | 33.8 µg/m³ | **26.9 µg/m³** | **20.4 %** | 0.81 | 96 % / 84 % | 75 % → 82 % |

- **Research question:** the fire features made no difference on the earlier validation winters. They gave a **statistically significant improvement in the October–November 2025 burning season** (MAE −2.3 µg/m³, 95 % CI 0.9 to 4.2). Weather and persistence explain most of tomorrow's PM2.5.
- **The weather drives the day-to-day changes:** today's PM2.5, tomorrow's wind speed, boundary-layer height and rain are the most important inputs.

## 3. Repository structure

```
lahore-smog-forecasting/
├── Lahore_Smog_Phase1_EDA.ipynb          ← Part 1: data acquisition → EDA
├── Lahore_Smog_Phase2_Modelling.ipynb    ← Part 2: model training → final pipeline
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                  ← the 5 original API downloads (input of Phase 1, ~42 MB, .csv.gz)
│   ├── processed/            ← created by Phase 1, input of Phase 2
│   │   ├── lahore_smog_model_ready.csv   (1,910 evenings × 27 columns, 0 empty cells)
│   │   ├── selected_features.json        (the 15 selected features and the targets)
│   │   ├── lahore_daily_full.csv         (full daily table, 2,710 days)
│   │   └── cleaning_log.csv
│   └── figures/              ← all charts (Phase 1: 01–19, Phase 2: p2_01–p2_11)
├── models/
│   ├── smog_forecaster.joblib            ← the final trained pipeline (scikit-learn)
│   └── smog_forecaster_config.json       ← features, warning threshold, AQI bins, test scores
└── results/
    ├── test_results.csv                  ← all models × both test winters × all metrics
    ├── fire_ablation.csv                 ← research question: with / without fire features
    └── final_summary.csv
```

## 4. Data sources

We built our own dataset: no ready-made dataset combines these sources. All of them are open and free.

| Source | Data | Access | Period |
|---|---|---|---|
| [OpenAQ](https://openaq.org) AWS archive | Hourly PM2.5, US Diplomatic Post Lahore (reference-grade monitor) | Public S3 bucket (no key) | May 2019 – Mar 2025 |
| [OpenAQ API v3](https://docs.openaq.org) | Daily PM2.5 from 80+ low-cost sensors in Lahore | REST API (free key) | Nov 2023 – Oct 2026 |
| [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov) | 3.27 million VIIRS fire detections over both Punjabs | Country files + area API (free key) | 2012 – Oct 2026 |
| [Open-Meteo](https://open-meteo.com) | Hourly ERA5 weather (incl. boundary-layer height) for Lahore, Faisalabad, Ludhiana | REST API (no key) | 2012 – Oct 2026 |

The data was collected on 22–24 Oct 2026 and is stored unchanged in `data/raw/`. A copy of the full project data is also on Google Drive:
<https://drive.google.com/file/d/1qN9e0QUoiOoJhBz53r3HgtQCrzQPZj9z/view?usp=sharing>

If the data folder is missing (for example when a notebook is opened directly in Colab), **both notebooks download it automatically from this link**.

Licences: NASA FIRMS (NASA open data); OpenAQ (cite OpenAQ and the original providers); Open-Meteo / ERA5 (CC BY 4.0).

## 5. Requirements

Python 3.10+ and these libraries, all of which are pre-installed on Google Colab:

```
pandas  numpy  matplotlib  seaborn  scikit-learn  scipy  joblib  requests
```

Install them locally with `pip install -r requirements.txt`.

## 6. How to reproduce the results

### Option A: Google Colab (no installation)
1. Click one of the **Open in Google Colab** links above, or in Colab choose **File → Open notebook → GitHub** and paste `umfar46/lahore-smog-forecasting`.
2. Click **Runtime → Run all**. The data is downloaded automatically (Section 1 of each notebook).
3. Run time: about 1–2 minutes for Phase 1 and 3–5 minutes for Phase 2.

*Or* clone the whole repository in a Colab cell, so that the notebooks use the data from the repository:
```python
!git clone https://github.com/umfar46/lahore-smog-forecasting.git
%cd lahore-smog-forecasting
```
Then open the notebook from the Files panel (`lahore-smog-forecasting/`) and choose Run all.

### Option B: on your own computer
```bash
git clone https://github.com/umfar46/lahore-smog-forecasting.git
cd lahore-smog-forecasting
python -m venv .venv
.venv\Scripts\activate            # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
jupyter notebook
```
Open the notebooks in this order and run all cells:
1. **`Lahore_Smog_Phase1_EDA.ipynb`**: rebuilds `data/processed/` from `data/raw/` and creates the EDA figures.
2. **`Lahore_Smog_Phase2_Modelling.ipynb`**: trains, tunes and evaluates the models, then saves `models/` and `results/`.

Phase 2 can also be run on its own, because `data/processed/` is already in the repository. All random seeds are fixed (`SEED = 42`), so the numbers in the notebooks are reproduced exactly.

### Using the trained model
```python
import joblib, json, pandas as pd
model = joblib.load("models/smog_forecaster.joblib")
cfg = json.load(open("models/smog_forecaster_config.json"))
X = pd.read_csv("data/processed/lahore_smog_model_ready.csv").tail(3)     # any rows with the 15 features
pm25_forecast = model.predict(X) + X["pm25_today"]                         # the model predicts the change
warning = pm25_forecast >= cfg["warning_threshold_ug_m3"]
```
Phase 2, Section 12 has a complete `forecast()` function that also returns the AQI category.

*Note:* the saved model was trained with scikit-learn 1.9. If your scikit-learn version cannot load it, re-run the Phase-2 notebook (about 3 minutes) and it will be re-created.

### (Optional) Re-download the raw data from the APIs
In Phase 1, Section 2, set `DOWNLOAD_FRESH = True`. You need free API keys for NASA FIRMS and OpenAQ, which are read from the environment variables `FIRMS_MAP_KEY` and `OPENAQ_API_KEY` or typed in when asked, and never stored. It takes 20–40 minutes. Fresh data includes newer days, so the numbers will differ from the notebooks.

## 7. Method in brief
- **No data leakage:** every feature uses only information available on the evening of the forecast, which is checked with assertions in Phase 1. All splits are by time. Hyperparameters are tuned only on the training period, and the test winters are used once, at the end.
- **Split:**
  - training: May 2019 – Sep 2023;
  - validation: winter 2023–24, plus walk-forward validation over the 2020–21, 2021–22 and 2023–24 winters;
  - test 1: winter 2024–25;
  - test 2: winter 2025–26 (calibrated low-cost sensors).
- **Metrics:**
  - regression: MAE (main), RMSE, R²;
  - hazardous warning: recall (priority), precision, F1, PR-AUC;
  - AQI category: accuracy, macro-F1, within-one-category accuracy.
- **One model, three answers:** the PM2.5 forecast gives the AQI category (EPA breakpoints) and the warning (forecast ≥ 215 µg/m³, a threshold chosen on validation data for ≥ 85 % recall). This approach beat dedicated classifiers on validation and test data.
