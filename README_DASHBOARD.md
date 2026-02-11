# ☀️ Solar Panel Adoption Forecasting Dashboard

Interactive web application for forecasting solar panel adoption in Munich using machine learning models.

## Features

### 📊 Overview & Historical Data
- Key metrics and statistics
- Historical trends visualization
- Demographic trends analysis
- Solar adoption rates over time

### 🔮 Interactive Forecasting
- Adjust demographic parameters in real-time
- See instant forecast updates
- Compare baseline vs adjusted scenarios
- Interactive sliders for:
  - Unemployment rate
  - Average age
  - Population changes
  - Youth and elderly population

### 📍 District Analysis
- Top 10 districts ranking
- Geographic distribution visualization
- Detailed district-level forecasts
- Correlation analysis between demographics and adoption

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements_streamlit.txt
   ```

2. **Ensure you have the required model files:**
   - `Training/random_forest_model_lag.joblib`
   - `data/CleanupDataSet/final_model.csv`

## Running the Application

```bash
streamlit run forecast_app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage Guide

### Overview Mode
1. View overall historical trends
2. Analyze demographic changes over time
3. Understand solar adoption patterns

### Interactive Forecasting Mode
1. Select a forecast year (2025-2035)
2. Adjust demographic sliders:
   - **Unemployment**: +5% means 5 percentage points higher
   - **Age**: +3 means 3 years older on average
   - **Population**: +10% means 10% population growth
3. View results instantly:
   - Total panel area forecast
   - Adoption rate changes
   - Demographic comparison table

### District Analysis Mode
1. Select forecast year
2. View top 10 districts
3. Explore detailed district statistics
4. Analyze geographic patterns

## Technical Details

### Models Used
- **Stage 1**: LightGBM Classifier (binary solar adoption prediction)
- **Stage 2**: Random Forest Regressor (panel area prediction)
- **Combined**: Two-stage prediction system

### Features
9 input features:
- Total rooftops
- Unemployment rate
- Average age
- Elderly population
- Young population
- Total population
- Tile encoding
- Employed population
- Panel area lag (previous year)

### Forecasting Method
- Linear trend extrapolation for demographic features
- Time-series aware data splitting
- Bounded predictions (realistic ranges)

## Tips for Best Results

1. **Start with small adjustments** (±2-5%) to see realistic impacts
2. **Combine related adjustments** (e.g., if unemployment increases, population might decrease)
3. **Use district analysis** to identify high-potential areas
4. **Compare multiple scenarios** using different adjustment combinations

## Troubleshooting

**Port already in use:**
```bash
streamlit run forecast_app.py --server.port 8502
```

**Model file not found:**
- Check that `random_forest_model_lag.joblib` exists in `Training/` folder
- Verify the file path in the code matches your directory structure

**Data file errors:**
- Ensure `final_model.csv` is in `data/CleanupDataSet/` folder
- Check that all required columns exist in the CSV

## Future Enhancements

- [ ] Add map visualization with actual coordinates
- [ ] Export forecast results to CSV
- [ ] Multi-year comparison charts
- [ ] Confidence intervals for predictions
- [ ] Historical vs forecast comparison plots
- [ ] Policy scenario templates

## License

This project is part of a thesis on solar panel adoption forecasting.
