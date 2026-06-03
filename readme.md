# ☀️ Munich Solar Panel Adoption — Forecasting & Analysis

A comprehensive toolkit for **analyzing** and **forecasting** photovoltaic (PV) solar panel adoption across Munich, Germany. It combines geospatial, demographic, economic, and infrastructure data with machine learning models to predict future rooftop solar deployment at the district and tile level.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dashboards](#dashboards)
- [Dataset Description](#dataset-description)
- [Model Training](#model-training)
- [Project Structure](#project-structure)
- [Data Sources](#data-sources)
- [Notes & Caveats](#notes--caveats)
- [Citation](#citation)

---

## Overview

This project provides:

1. **A curated, tile-based dataset** covering Munich from 2003–2024, enriched with solar potential, demographics, EV infrastructure, and PV pricing data.
2. **A two-stage ML pipeline** (classification → regression) using five base models plus a stacking ensemble to forecast future panel area.
3. **Two interactive Streamlit dashboards** for exploring historical trends and running what-if forecasting scenarios.

Each tile represents a **1000 × 1000 m area** in Munich. The pipeline predicts which tiles will adopt solar panels (Stage 1) and how much panel area will be installed (Stage 2).

---

## Features

- 📊 **Interactive choropleth map** of Munich districts with selectable metrics and years
- 🔮 **What-if forecasting** — adjust unemployment, population age, EV infrastructure, PV prices, and more to see how they affect solar adoption
- 🤖 **5 ML models** (LightGBM, XGBoost, CatBoost, Random Forest, HistGBR) with live model selection in the dashboard
- 📍 **District-level analysis** with top-10 rankings and geographic distribution
- 📐 **Sensitivity analysis** to understand which factors drive adoption most
- 📈 **Historical trends** for solar adoption, demographics, EV charging, and PV pricing
- 🌍 **Economic scenario presets** (Good Economy / Recession) grounded in real-world data

---

## Prerequisites

- **Python 3.9+** (tested with 3.10, 3.11)
- **pip** (or conda)
- ~1 GB free disk space (for model files)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/AhmedElsherif04/ViewPython.git
cd ViewPython
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

For the **dashboards** (map viewer + forecasting):

```bash
pip install -r requirements_streamlit.txt
```

Additional packages needed depending on what you want to run:

| Component | Extra packages |
|-----------|---------------|
| Map dashboard (`app.py`) | `folium`, `geopandas`, `streamlit-folium` |
| Forecasting dashboard (`forecast_app.py`) | Already covered by `requirements_streamlit.txt` |
| Model training | `xgboost`, `catboost`, `lightgbm`, `scikit-learn`, `joblib` |
| Data download (WOMS) | `requests` |

Install all extras at once:

```bash
pip install folium geopandas streamlit-folium xgboost catboost requests
```

---

## Quick Start

After installation, launch either dashboard:

```bash
# Forecasting dashboard (full-featured)
streamlit run forecast_app.py

# Map viewer (choropleth by district/year)
streamlit run app.py
```

The app will open at **http://localhost:8501** in your default browser.

> **Tip:** If port 8501 is busy, specify another:
> ```bash
> streamlit run forecast_app.py --server.port 8502
> ```

---

## Dashboards

### 1. Forecasting Dashboard (`forecast_app.py`)

The main application with five modes accessible via the sidebar:

| Mode | What it does |
|------|-------------|
| **📂 View Raw Data** | Browse, filter, and download the full dataset |
| **📊 Overview & Historical Data** | Key metrics, historical trends for solar adoption, demographics, EV charging, and PV prices |
| **🔮 Interactive Forecasting** | Set a target year (up to 10 years ahead), pick an economic scenario or custom parameters, and generate baseline vs. adjusted forecasts |
| **📍 District Analysis** | Top-10 district rankings, geographic distributions, district-level forecasts |
| **📐 Sensitivity Analysis** | Understand which demographic and economic levers most affect solar adoption |

**Stage 2 model selector:** Use the sidebar dropdown to switch between LightGBM (default), XGBoost, CatBoost, Random Forest, or HistGBR at any time.

#### Economic Scenario Presets

| Scenario | Description |
|----------|-------------|
| **📈 Good Economy** | Lower unemployment, population growth, cheaper PV, more EV charging — based on Munich's 2015–2019 growth trajectory |
| **📉 Recession** | Higher unemployment, population decline, reduced infrastructure investment — based on the 2008–2009 financial crisis |
| **⚙️ Custom** | Set each parameter manually |

### 2. Map Dashboard (`app.py`)

A lightweight choropleth map viewer. Select a year and metric to visualize district-level data on an interactive Folium map.

---

## Dataset Description

### Primary Dataset (`data/CleanupDataSet/final_model_ev_updated.csv`)

The full feature-engineered dataset used by the forecasting dashboard and model training.

| Variable | Description |
|----------|-------------|
| `tile` | Unique tile identifier (e.g., `tile_r0_c0`) |
| `year` | Year of observation (2003–2024) |
| `total_rooftops` | Total number of rooftops in the tile |
| `rooftops_without_solar` | Number of rooftops without solar panels |
| `square_meters_with_solar_m2` | Total rooftop area (m²) containing solar panels |
| `panel_area_m2` | Actual area (m²) of detected PV panels |
| `district_number` | Munich district number |
| `Unemployment_Rate` | District-level unemployment rate (%) |
| `Average_Age` | Average population age in the district (years) |
| `Elderly_Population` | Number of residents aged 65+ |
| `Young_Population` | Number of residents aged 0–18 |
| `Total_Population` | Total number of residents in the district |
| `employed` | Number of employed residents |
| `pv_price` | PV module price (€/kWp) |
| `ev_points_164m` | Number of EV charging points within 164 m radius |
| `panel_area_lag1` | Panel area from the previous observation year |
| `tile_encoded` | Encoded tile identifier (numerical) |
| `tile_centroid_lat` | Latitude of tile centroid |
| `tile_centroid_lon` | Longitude of tile centroid |

### Legacy Dataset (`data/Train Data/rooftop.csv`)

An earlier version of the dataset without EV/pricing features. See the [Dataset Description (legacy)](#legacy-dataset-details) section below.

---

## Model Training

### Two-Stage Pipeline

| Stage | Model | Task |
|-------|-------|------|
| **Stage 1** | LGBMClassifier | Binary classification — does a tile have solar panels? |
| **Stage 2** | 5 regressors + stacking | Predict `panel_area_log` for tiles with solar panels |

### Stage 2 Models

| Model | Description |
|-------|-------------|
| Random Forest | 1000 trees, `max_features=sqrt` |
| XGBoost | 1000 rounds, `max_depth=6`, early stopping |
| LightGBM | 1000 rounds, 63 leaves, early stopping |
| HistGBR | 1000 iterations, 63 leaf nodes |
| CatBoost | 1000 iterations, `depth=6`, early stopping |
| **Stacking** | Ridge meta-learner on out-of-fold predictions of the 5 base models |

### Retraining Models

If you modify the dataset or want to retrain:

```bash
python Training/train_all_models_and_stack.py
```

This will:
1. Train the Stage 1 classifier
2. Train all 5 Stage 2 regressors
3. Build the stacking ensemble
4. Save everything to `Training/saved_models/`
5. Print a full comparison table with R², MAE, and RMSE

> **Note:** Training requires `xgboost`, `catboost`, and `lightgbm` to be installed. The Random Forest model file is ~800 MB.

### Pre-trained Models

The model files are too large for Git, so they are hosted on Google Drive:

📥 **[Download pre-trained models from Google Drive](https://drive.google.com/drive/folders/1Tc2ycjn7wMHNZbapjqIXrmrbmktumziG?usp=sharing)**

After downloading, place all files into `Training/saved_models/`:

```bash
mkdir -p Training/saved_models
# Move the downloaded files into the directory
mv ~/Downloads/*.joblib Training/saved_models/
mv ~/Downloads/model_metadata.json Training/saved_models/
```

Model files included:

| File | Size | Description |
|------|------|-------------|
| `stage1_classifier.joblib` | ~0.6 MB | LGBMClassifier |
| `stage2_lightgbm.joblib` | ~6 MB | LightGBM regressor (default) |
| `stage2_xgboost.joblib` | ~4 MB | XGBoost regressor |
| `stage2_catboost.joblib` | ~0.2 MB | CatBoost regressor |
| `stage2_histgbr.joblib` | ~1 MB | HistGBR regressor |
| `stage2_randomforest.joblib` | ~800 MB | Random Forest regressor |
| `stage2_stacking.joblib` | <1 KB | Stacking meta-learner |
| `model_metadata.json` | ~4 KB | Feature columns, metrics, weights |

---

## Project Structure

```
ViewPython/
├── app.py                          # Choropleth map dashboard (Streamlit + Folium)
├── forecast_app.py                 # Forecasting dashboard (Streamlit + Plotly)
├── requirements_streamlit.txt      # Python dependencies for dashboards
├── readme.md                       # This file
├── README_DASHBOARD.md             # Additional dashboard documentation
│
├── data/
│   ├── CleanupDataSet/
│   │   ├── final_model_ev_updated.csv  # Primary dataset (with EV + pricing)
│   │   └── final_model.csv             # Dataset without EV/pricing features
│   ├── Train Data/
│   │   └── rooftop.csv                 # Legacy dataset
│   ├── Rooftop_Data/                   # Previous dataset versions
│   ├── CSV_data/                       # Raw CSV exports
│   ├── EV_installations/               # EV charging point data
│   ├── Woms Data/
│   │   └── downloadwoms.py             # Script to download solar/land-use layers
│   ├── munich_districts_4326.geojson   # District boundaries (EPSG:4326)
│   └── price_pv.csv                    # Historical PV module prices
│
├── Training/
│   ├── train_all_models_and_stack.py   # Full training pipeline
│   ├── saved_models/                   # Pre-trained model files (.joblib)
│   ├── 2stage_improved.ipynb           # Model development notebook
│   ├── forecast_improved.ipynb         # Forecasting experiments
│   ├── improved_pipeline_noleak.ipynb  # Leak-free pipeline
│   ├── lstm.ipynb                      # LSTM experiments
│   ├── model_deep_analysis.ipynb       # In-depth model analysis
│   ├── model_results_analysis.ipynb    # Results visualization
│   ├── per_district.ipynb              # Per-district analysis
│   └── generate_deep_analysis.py       # Auto-generate analysis reports
│
├── GeoJsontoZip/
│   ├── GeojsontoZip.py                 # GeoJSON/Shapefile conversion utility
│   └── areasMunich.json                # Munich district boundaries (GeoJSON)
│
├── geometry.py                     # GML building geometry parser (LoD2)
├── viewer.py                       # GML building attribute inspector
├── merge.ipynb                     # Data merging notebook
└── .gitignore
```

---

## Data Sources

### 1. Munich Open Data Portal
[Geoportal München Open Data](https://geoportal.muenchen.de/portal/opendata/#LayerInfoDataDownload)

- **Stadtbezirke der Landeshauptstadt München** — District boundaries
- **Solarpotenzial_Globalstrahlung** — Solar potential (downloaded via `data/Woms Data/downloadwoms.py`)
- **Digitaler Flächennutzungsplan** — Digital land use plan (downloaded via `data/Woms Data/downloadwoms.py`)

### 2. Munich Statistical Office
[München Indikatorenatlas](https://mstatistik.muenchen.de/indikatorenatlas/export/export.php)

District-level indicators: population, age distribution, unemployment rate, housing.

### 3. Additional Sources

- **EV Charging Points** — Bundesnetzagentur (German Federal Network Agency)
- **PV Module Prices** — Fraunhofer ISE Photovoltaics Report
- **Building Geometry (LoD2)** — Bavarian State Office for Digitisation

---

## Notes & Caveats

- **WOMS layers** (solar potential & land use) cannot be downloaded directly from the portal — use `data/Woms Data/downloadwoms.py`.
- **Decimal separators**: Some raw fields use comma as decimal (e.g., `"3,8"` → 3.8%). The cleaned dataset already handles this.
- **Large model files**: `stage2_randomforest.joblib` is ~800 MB. Consider using LightGBM (default, ~6 MB) if storage is a concern.
- **Tile size**: Each tile represents a **1000 × 1000 m** area in Munich.
- **Forecast horizon**: The dashboard supports forecasting up to 10 years beyond the latest data year. Predictions further out carry more uncertainty.

---

## Legacy Dataset Details

The previous version of the dataset (`data/Train Data/rooftop.csv`) included additional geospatial information derived from house and solar panel segmentation models:

| Variable | Description |
|----------|-------------|
| `square_meters_without_solar_m2` | Rooftop area without solar panels |
| `tile_centroid_lat` | Latitude of tile centroid |
| `tile_centroid_lon` | Longitude of tile centroid |
| `district_name` | Name of the district |
| `match_type` | Tile–district association method (`exact` or `nearest`) |
| `distance_to_district_m` | Distance from tile centroid to district center (meters) |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **Port already in use** | `streamlit run forecast_app.py --server.port 8502` |
| **Model file not found** | Check that `Training/saved_models/` contains the `.joblib` files |
| **Missing columns in CSV** | Ensure you're using `final_model_ev_updated.csv`, not the legacy dataset |
| **`ModuleNotFoundError`** | Run `pip install -r requirements_streamlit.txt` and install extras (see [Installation](#installation)) |
| **Slow forecasting** | Use LightGBM (default) instead of Random Forest |

---

## Citation

If you use this dataset or code in your research, please reference this repository:

```
@misc{munich-solar-forecasting,
  title  = {Munich Solar Panel Adoption Forecasting},
  author = {Ahmed Elsherif},
  year   = {2026},
  url    = {https://github.com/AhmedElsherif04/ViewPython}
}
```

---

## License

This project is part of a master's thesis on solar panel adoption forecasting at [fortiss](https://www.fortiss.org/) / Technical University of Munich.
