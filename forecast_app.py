from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import lightgbm as lgb
import joblib
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "CleanupDataSet" / "final_model_ev_updated.csv"
SAVED_MODELS_DIR = BASE_DIR / "Training" / "saved_models"
STAGE1_MODEL_PATH = SAVED_MODELS_DIR / "stage1_classifier.joblib"

# Available Stage 2 models (display name → filename)
# LightGBM is listed first so it becomes the default (index=0).
AVAILABLE_MODELS = {
    "LightGBM": "stage2_lightgbm.joblib",
    "CatBoost": "stage2_catboost.joblib",
    "HistGBR": "stage2_histgbr.joblib",
    "XGBoost": "stage2_xgboost.joblib",
    "RandomForest": "stage2_randomforest.joblib",
}
FEATURE_COLS = [
    'year',
    'total_rooftops',
    'Unemployment_Rate',
    'Average_Age',
    'Elderly_Population',
    'Young_Population',
    'Total_Population',
    'employed',
    'pv_price',
    'panel_area_lag1',
    'ev_points_164m',
    'tile_encoded',
    'tile_centroid_lat',
    'tile_centroid_lon',
]

# Page configuration
st.set_page_config(
    page_title="Solar Panel Adoption Forecast",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Cache data and models
@st.cache_data
def load_data():
    """Load and prepare the dataset"""
    df = pd.read_csv(DATA_FILE)
    # Create binary indicator: 1 if has solar panels, 0 otherwise
    df['has_solar'] = (df['panel_area_m2'] > 0).astype(int)
    # Create log-transformed panel area for Stage 2 regression
    df['panel_area_log'] = np.log1p(df['panel_area_m2'])
    return df

@st.cache_resource
def load_stage1():
    """Load Stage 1 classifier (LGBMClassifier)."""
    return joblib.load(STAGE1_MODEL_PATH)

@st.cache_resource
def load_stage2(model_name):
    """Load a Stage 2 regression model by name."""
    filename = AVAILABLE_MODELS[model_name]
    return joblib.load(SAVED_MODELS_DIR / filename)


def _get_model_name_for_stage2(model_obj):
    """Return a human-readable model class name (used for HistGBR floor logic)."""
    cls = type(model_obj).__name__
    return cls

def create_forecast_features(df, forecast_year, feature_cols, adjustments=None):
    """Create features for forecasting with optional adjustments.

    Uses vectorized linear extrapolation per tile for demographic features.
    """
    latest_year = df['year'].max()
    forecast_data = df[df['year'] == latest_year].copy()
    forecast_data['year'] = forecast_year

    demo_cols = [
        'total_rooftops', 'Average_Age', 'Elderly_Population',
        'Young_Population', 'Total_Population', 'Unemployment_Rate',
        'employed', 'pv_price', 'ev_points_164m',
    ]

    # Vectorized linear extrapolation per tile
    for col in demo_cols:
        if col not in df.columns:
            continue

        def extrapolate(group):
            years = group['year'].values.astype(float)
            values = group[col].values.astype(float)
            valid = ~np.isnan(values)
            if valid.sum() > 1:
                slope, intercept, _, _, _ = stats.linregress(years[valid], values[valid])
                return slope * forecast_year + intercept
            return values[valid][-1] if valid.any() else np.nan

        tile_forecasts = df.groupby('tile').apply(extrapolate)
        forecast_data[col] = forecast_data['tile'].map(tile_forecasts)

        # Apply adjustments if provided
        if adjustments and col in adjustments:
            adj = adjustments[col]
            if adj['type'] == 'percent':
                forecast_data[col] *= (1 + adj['value'] / 100)
            elif adj['type'] == 'absolute':
                forecast_data[col] += adj['value']

    # Apply bounds
    forecast_data['Unemployment_Rate'] = forecast_data['Unemployment_Rate'].clip(0, 20)
    forecast_data['Average_Age'] = forecast_data['Average_Age'].clip(20, 60)
    for col in demo_cols:
        if col not in ['Unemployment_Rate', 'Average_Age'] and col in forecast_data.columns:
            forecast_data[col] = forecast_data[col].clip(lower=0)

    # Lag = last known panel area
    last_panel = df.sort_values('year').groupby('tile')['panel_area_m2'].last()
    forecast_data['panel_area_lag1'] = forecast_data['tile'].map(last_panel)

    return forecast_data


def generate_sequential_forecast(df, clf, model_stage2, feature_cols, target_year, adjustments=None):
    """Forecast sequentially so each year reuses the previous year's predicted lag.

    The *raw* (un-weighted) prediction is used as ``panel_area_lag1`` for
    the next step.  Previously, the probability-weighted value
    (``predicted_panel_area_m2``) was fed back, which introduced a
    compounding decay because probability < 1 systematically shrinks
    the lag each year.
    """
    base_year = int(df['year'].max())
    if target_year <= base_year:
        raise ValueError("Forecast year must be greater than the last historical year.")

    df_current = df.copy()
    forecasts_all = []
    prev_panel_area = None

    for year in range(base_year + 1, target_year + 1):
        year_adjustments = adjustments if (adjustments and year == target_year) else None
        forecast_data = create_forecast_features(df_current, year, feature_cols, year_adjustments)

        if prev_panel_area is not None:
            mapped_lag = forecast_data['tile'].map(prev_panel_area)
            forecast_data['panel_area_lag1'] = mapped_lag.fillna(forecast_data['panel_area_lag1'])

        forecast_pred = predict_future(clf, model_stage2, forecast_data, feature_cols)
        forecasts_all.append(forecast_pred)

        # Use the RAW (un-weighted) prediction as the lag for the next year.
        prev_panel_area = dict(zip(
            forecast_pred['tile'],
            forecast_pred['predicted_panel_area_raw'].values,
        ))

        new_rows = forecast_pred.copy()
        new_rows = new_rows.rename(columns={'predicted_panel_area_m2': 'panel_area_m2'})

        df_current = df_current.loc[:, ~df_current.columns.duplicated()]
        new_rows = new_rows.loc[:, ~new_rows.columns.duplicated()]

        common_cols = [c for c in df_current.columns if c in new_rows.columns]
        new_rows = new_rows[common_cols]
        df_current = pd.concat([df_current, new_rows], ignore_index=True)

    all_forecasts = pd.concat(forecasts_all, ignore_index=True)
    latest_forecast = all_forecasts[all_forecasts['year'] == target_year].copy()
    return latest_forecast, all_forecasts


def predict_future(clf, model_stage2, forecast_data, feature_cols):
    """Make 2-stage predictions using LGBMClassifier + selected Stage 2 model.

    The probability weighting is **soft**: high-probability tiles
    (≥ 0.5) receive the full raw prediction, while medium-probability
    tiles are scaled by ``p_solar`` and low-probability tiles (< 0.2)
    are zeroed out.

    **Non-decrease floor**: The raw prediction is floored at
    ``panel_area_lag1`` so that predicted panel area never *decreases*
    compared to the previous year.  This reflects the physical reality
    that installed panels are not removed.
    """
    X_forecast = forecast_data[feature_cols].copy()

    # Stage 1: Adoption probability (LGBMClassifier → predict_proba)
    p_solar = clf.predict_proba(X_forecast)[:, 1]

    # Stage 2: Panel area prediction (log scale → expm1)
    y_pred_log = model_stage2.predict(X_forecast)
    y_pred_size = np.clip(np.expm1(y_pred_log), 0, None)

    # Floor: predictions must not decrease below the lag value
    lag_values = forecast_data['panel_area_lag1'].values
    y_pred_size = np.maximum(y_pred_size, lag_values)

    # Soft probability gating
    y_pred_final = np.where(
        p_solar >= 0.5,
        y_pred_size,                   # high-confidence → full prediction
        np.where(
            p_solar >= 0.2,
            p_solar * y_pred_size,      # mid-confidence → scaled
            0.0,                        # low-confidence → zero
        ),
    )

    forecast_out = forecast_data.copy()
    forecast_out['predicted_adoption_prob'] = p_solar
    forecast_out['predicted_adoption_binary'] = (p_solar >= 0.5).astype(int)
    forecast_out['predicted_panel_area_raw'] = y_pred_size
    forecast_out['predicted_panel_area_m2'] = y_pred_final

    return forecast_out

# Main App
def main():
    st.markdown('<h1 class="main-header">☀️ Solar Panel Adoption Forecasting Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data and models..."):
        df = load_data()
        clf = load_stage1()
        feature_cols = FEATURE_COLS
    first_forecast_year = int(df['year'].max() + 1)
    max_forecast_year = first_forecast_year + 10
    default_forecast_year = min(first_forecast_year + 2, max_forecast_year)
    
    # Sidebar
    st.sidebar.title("⚙️ Controls")
    
    # Model selector
    st.sidebar.markdown("### 🤖 Stage 2 Model")
    selected_model_name = st.sidebar.selectbox(
        "Regression Model",
        list(AVAILABLE_MODELS.keys()),
        index=0,  # LightGBM as default (first in dict)
        help="Choose which model predicts panel area. LightGBM is the default."
    )
    model_stage2 = load_stage2(selected_model_name)
    st.sidebar.caption(f"Active: **{selected_model_name}**")
    
    st.sidebar.divider()
    
    # Select mode
    mode = st.sidebar.radio(
        "Select Mode",
        ["📂 View Raw Data", "📊 Overview & Historical Data", "🔮 Interactive Forecasting", "📍 District Analysis", "📐 Sensitivity Analysis"]
    )
    
    # ==================== DATA VIEWER ====================
    if mode == "📂 View Raw Data":
        st.header("📂 Raw Dataset Viewer")
        
        st.info("View and explore all your data from the beginning")
        
        # Dataset Overview
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Years Covered", f"{df['year'].min()}-{df['year'].max()}")
        with col4:
            st.metric("Unique Tiles", f"{df['tile'].nunique():,}")
        
        # Filter options
        st.subheader("🔍 Filter Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            year_filter = st.multiselect(
                "Select Year(s)",
                sorted(df['year'].unique()),
                default=sorted(df['year'].unique())[-3:]
            )
        
        with col2:
            district_filter = st.multiselect(
                "Select District(s)",
                sorted(df['district_number'].unique()),
                default=sorted(df['district_number'].unique())[:5]
            )
        
        with col3:
            has_solar_filter = st.selectbox(
                "Solar Panel Status",
                ["All", "With Solar", "Without Solar"]
            )
        
        # Apply filters
        filtered_df = df[df['year'].isin(year_filter) & df['district_number'].isin(district_filter)]
        
        if has_solar_filter == "With Solar":
            filtered_df = filtered_df[filtered_df['has_solar'] == 1]
        elif has_solar_filter == "Without Solar":
            filtered_df = filtered_df[filtered_df['has_solar'] == 0]
        
        # Display data
        st.subheader(f"Displaying {len(filtered_df):,} records")
        
        st.dataframe(filtered_df, width='stretch', height=400)
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name="solar_data_filtered.csv",
            mime="text/csv"
        )
        
        # Statistics
        st.subheader("📈 Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Panel Area Statistics**")
            st.dataframe(filtered_df[['panel_area_m2']].describe())
        
        with col2:
            st.write("**Population Statistics**")
            st.dataframe(filtered_df[['Total_Population', 'Young_Population', 'Elderly_Population']].describe())
        
        with col3:
            st.write("**EV Infrastructure Stats**")
            if 'ev_points_164m' in filtered_df.columns:
                st.dataframe(filtered_df[['ev_points_164m']].describe())
            else:
                st.info("EV data unavailable in current selection")
    
    # ==================== OVERVIEW MODE ====================
    elif mode == "📊 Overview & Historical Data":
        st.header("📊 Historical Solar Panel Adoption")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        latest_year = df['year'].max()
        latest_data = df[df['year'] == latest_year]
        
        with col1:
            st.metric(
                "Total Panel Area (Latest)",
                f"{latest_data['panel_area_m2'].sum():,.0f} m²"
            )
        
        with col2:
            st.metric(
                "Adoption Rate",
                f"{(latest_data['has_solar'].mean() * 100):.1f}%"
            )
        
        with col3:
            st.metric(
                "Total Tiles",
                f"{len(latest_data):,}"
            )
        
        with col4:
            st.metric(
                "Years of Data",
                f"{df['year'].nunique()} years"
            )
        
        with col5:
            ev_mean = latest_data.get('ev_points_164m')
            if ev_mean is not None:
                st.metric(
                    "Average EV Points",
                    f"{ev_mean.mean():.1f}"
                )
            else:
                st.metric("Average EV Points", "N/A")
        
        # Historical Trends
        st.subheader("📈 Historical Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            yearly_stats = df.groupby('year').agg({
                'panel_area_m2': 'sum',
                'has_solar': 'mean'
            }).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=yearly_stats['year'],
                y=yearly_stats['panel_area_m2'],
                mode='lines+markers',
                name='Total Panel Area',
                line=dict(color='#FF6B35', width=3),
                marker=dict(size=8)
            ))
            fig.update_layout(
                title="Total Solar Panel Area Over Time",
                xaxis_title="Year",
                yaxis_title="Total Panel Area (m²)",
                hovermode='x unified'
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=yearly_stats['year'],
                y=yearly_stats['has_solar'] * 100,
                mode='lines+markers',
                name='Adoption Rate',
                line=dict(color='#4ECDC4', width=3),
                marker=dict(size=8),
                fill='tozeroy'
            ))
            fig.update_layout(
                title="Solar Adoption Rate Over Time",
                xaxis_title="Year",
                yaxis_title="Adoption Rate (%)",
                hovermode='x unified'
            )
            st.plotly_chart(fig, width='stretch')
        
        # Demographic trends
        st.subheader("👥 Demographic Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            demo_trends = df.groupby('year').agg({
                'Average_Age': 'mean',
                'Unemployment_Rate': 'mean'
            }).reset_index()
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Scatter(x=demo_trends['year'], y=demo_trends['Average_Age'],
                          name="Average Age", line=dict(color='#667eea')),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(x=demo_trends['year'], y=demo_trends['Unemployment_Rate'],
                          name="Unemployment %", line=dict(color='#f093fb')),
                secondary_y=True
            )
            
            fig.update_layout(title="Age & Unemployment Trends", hovermode='x unified')
            fig.update_xaxes(title_text="Year")
            fig.update_yaxes(title_text="Average Age", secondary_y=False)
            fig.update_yaxes(title_text="Unemployment Rate (%)", secondary_y=True)
            
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            pop_trends = df.groupby('year').agg({
                'Young_Population': 'mean',
                'Elderly_Population': 'mean'
            }).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pop_trends['year'], y=pop_trends['Young_Population'],
                name='Young Population', line=dict(color='#06A77D'),
                mode='lines+markers'
            ))
            fig.add_trace(go.Scatter(
                x=pop_trends['year'], y=pop_trends['Elderly_Population'],
                name='Elderly Population', line=dict(color='#F18F01'),
                mode='lines+markers'
            ))
            
            fig.update_layout(
                title="Population Age Distribution Trends",
                xaxis_title="Year",
                yaxis_title="Population Count",
                hovermode='x unified'
            )
            st.plotly_chart(fig, width='stretch')

        st.subheader("🔌 EV Infrastructure Trends")
        if 'ev_points_164m' in df.columns:
            ev_trends = df.groupby('year')['ev_points_164m'].agg(['sum', 'mean']).reset_index()
            fig_ev = go.Figure()
            fig_ev.add_trace(go.Scatter(
                x=ev_trends['year'],
                y=ev_trends['sum'],
                mode='lines+markers',
                name='Total EV Points',
                line=dict(color='#2E86AB', width=3)
            ))
            fig_ev.add_trace(go.Scatter(
                x=ev_trends['year'],
                y=ev_trends['mean'],
                mode='lines+markers',
                name='Average EV Points per Tile',
                line=dict(color='#8BF4D6', width=3, dash='dash')
            ))
            fig_ev.update_layout(
                title="EV Charging Points Over Time",
                xaxis_title="Year",
                yaxis_title="EV Points",
                hovermode='x unified'
            )
            st.plotly_chart(fig_ev, width='stretch')

        st.subheader("💰 PV Price Trends")
        if 'pv_price' in df.columns:
            pv_trends = df.groupby('year')['pv_price'].mean().reset_index()
            pv_trends.columns = ['year', 'avg_pv_price']
            fig_pv = go.Figure()
            fig_pv.add_trace(go.Scatter(
                x=pv_trends['year'],
                y=pv_trends['avg_pv_price'],
                mode='lines+markers',
                name='Average PV Price (€/kWp)',
                line=dict(color='#FF6B35', width=3),
                fill='tozeroy',
                fillcolor='rgba(255, 107, 53, 0.15)'
            ))
            fig_pv.update_layout(
                title="PV Module Price Over Time (€/kWp)",
                xaxis_title="Year",
                yaxis_title="Price per kWp (€)",
                hovermode='x unified'
            )
            st.plotly_chart(fig_pv, width='stretch')
    
    # ==================== FORECASTING MODE ====================
    elif mode == "🔮 Interactive Forecasting":
        st.header("🔮 Interactive Solar Panel Forecasting")
        
        st.success("✨ Adjust the demographic parameters below to forecast solar adoption!")
        
        # Create two columns for controls and results
        col_controls, col_results = st.columns([1, 1])
        
        with col_controls:
            st.subheader("⚙️ Forecast Settings")
            
            forecast_year = st.slider(
                "Forecast Year",
                first_forecast_year,
                max_forecast_year,
                default_forecast_year,
            )

            # ── Economic Scenario Presets ──
            st.markdown("### 🌍 Economic Scenario")

            # Define scenario parameters
            SCENARIOS = {
                "📈 Good Economy": {
                    'Unemployment_Rate': -1.0,    # Munich currently ~3.2%, drop to ~2.2%
                    'Average_Age': 0.0,
                    'Total_Population': 5.0,       # Strong migration to Munich
                    'Young_Population': 8.0,       # More young workers move in
                    'Elderly_Population': 2.0,     # Slight increase
                    'ev_points_164m': 40.0,        # Germany Masterplan 2.0 on track
                    'pv_price': -20.0,             # Continued price decline
                    'total_rooftops': 5.0,         # New construction boom
                },
                "📉 Recession": {
                    'Unemployment_Rate': 2.0,      # 3.2% → 5.2% (2009 crisis: 3.5→4.5%)
                    'Average_Age': 0.5,            # Aging accelerates (fewer young migrants)
                    'Total_Population': -1.0,      # Net outmigration from Munich
                    'Young_Population': -5.0,      # Young workers leave
                    'Elderly_Population': 3.0,     # Elderly population still grows
                    'ev_points_164m': -20.0,       # Infrastructure investment cut
                    'pv_price': 10.0,              # Subsidies reduced, demand drops
                    'total_rooftops': -2.0,        # Construction slowdown
                },
                "⚙️ Custom": None,
            }

            scenario = st.radio(
                "Select a scenario",
                list(SCENARIOS.keys()),
                index=2,  # Default to Custom
                horizontal=True,
                key='scenario_selector'
            )

            # Get preset values or defaults
            preset = SCENARIOS.get(scenario)
            if preset is None:
                preset = {k: 0.0 for k in ['Unemployment_Rate', 'Average_Age',
                    'Total_Population', 'Young_Population', 'Elderly_Population',
                    'ev_points_164m', 'pv_price', 'total_rooftops']}

            # Show scenario info
            with st.expander("📚 Scenario Details & Sources", expanded=False):
                st.markdown("""
**📈 Good Economy** — Based on Munich's pre-pandemic growth trajectory (2015-2019):
| Parameter | Change | Rationale |
|-----------|--------|-----------|
| Unemployment | −1.0 pp | Munich hit 2.2% in 2019 (Bundesagentur für Arbeit) |
| Population | +5% | Munich grew ~1.4%/yr avg; good economy attracts more workers (Statistisches Amt München) |
| Youth Pop. | +8% | Young professionals (25-35) are the primary migration driver |
| PV Price | −20% | Fraunhofer ISE: module prices fell ~15%/yr during 2020-2023; economies of scale continue |
| EV Points | +40% | Germany's Masterplan Ladeinfrastruktur 2.0 targets 1M points by 2030 (BMDV) |
| Rooftops | +5% | New construction boom in outer districts (Aubing, Feldmoching) |

**📉 Recession** — Based on the 2008-2009 financial crisis & 2023 Germany recession:
| Parameter | Change | Rationale |
|-----------|--------|-----------|
| Unemployment | +2.0 pp | Munich rose from 3.5% to 4.5% in 2009 (BA Statistik); modeled slightly worse |
| Population | −1% | Net outmigration during downturns as jobs contract (Destatis) |
| Youth Pop. | −5% | Young workers most mobile; leave for other opportunities |
| PV Price | +10% | Reduced subsidies (EEG cuts), fewer installations → less scale (BSW Solar) |
| EV Points | −20% | Public infrastructure investment first to be cut (KfW reports) |
| Rooftops | −2% | Construction permits fell 26% YoY in Germany during 2023 recession (Destatis) |

**Sources:** Bundesagentur für Arbeit, Statistisches Amt München, Fraunhofer ISE Photovoltaics Report 2024, BMDV Masterplan Ladeinfrastruktur 2.0, Destatis Baustatistik, BSW Solar Marktdaten.
                """)

            st.markdown("### 📊 Demographic Adjustments")
            if scenario != "⚙️ Custom":
                st.caption(f"Values pre-filled from **{scenario}** — adjust if needed.")
            else:
                st.caption("Enter changes from baseline")

            unemployment_change = st.number_input(
                "Unemployment Rate Change (pp)",
                -5.0, 5.0, preset['Unemployment_Rate'], 0.5
            )
            
            age_change = st.number_input(
                "Average Age Change (years)",
                -5.0, 5.0, preset['Average_Age'], 0.5
            )
            
            population_change = st.number_input(
                "Population Change (%)",
                -20.0, 20.0, preset['Total_Population'], 1.0
            )
            
            youth_change = st.number_input(
                "Youth Population Change (%)",
                -20.0, 20.0, preset['Young_Population'], 1.0
            )
            
            elderly_change = st.number_input(
                "Elderly Population Change (%)",
                -20.0, 20.0, preset['Elderly_Population'], 1.0
            )
            
            ev_change = st.number_input(
                "EV Charging Points Change (%)",
                -50.0, 100.0, preset['ev_points_164m'], 5.0
            )
            
            pv_price_change = st.number_input(
                "PV Price Change (%)",
                -50.0, 50.0, preset['pv_price'], 5.0,
                help="Adjust the forecasted PV module price (€/kWp). Negative = cheaper panels, Positive = more expensive."
            )
            
            rooftops_change = st.number_input(
                "Total Rooftops Change (%)",
                -50.0, 100.0, preset['total_rooftops'], 5.0,
                help="Adjust the forecasted total number of rooftops. Positive = more rooftops (new construction), Negative = fewer."
            )
            
            # Predict button
            predict_button = st.button("🚀 Generate Forecast", use_container_width=True, type='primary')

        # ---- Run forecasts OUTSIDE the column block to avoid delta-path crash ----
        interactive_cache_key = 'interactive_forecast_cache'
        cached = st.session_state.get(interactive_cache_key)
        use_cached = (
            cached is not None
            and cached.get('year') == forecast_year
            and not predict_button
        )

        pred_baseline = None
        pred_adjusted = None
        baseline_all = None
        adjusted_all = None
        forecast_error = None

        if predict_button:
            with st.spinner("Generating baseline forecast..."):
                try:
                    adjustments = {
                        'Unemployment_Rate': {'type': 'absolute', 'value': unemployment_change},
                        'Average_Age': {'type': 'absolute', 'value': age_change},
                        'Total_Population': {'type': 'percent', 'value': population_change},
                        'Young_Population': {'type': 'percent', 'value': youth_change},
                        'Elderly_Population': {'type': 'percent', 'value': elderly_change},
                        'ev_points_164m': {'type': 'percent', 'value': ev_change},
                        'pv_price': {'type': 'percent', 'value': pv_price_change},
                        'total_rooftops': {'type': 'percent', 'value': rooftops_change},
                    }

                    pred_baseline, baseline_all = generate_sequential_forecast(
                        df, clf, model_stage2, feature_cols, forecast_year
                    )
                    pred_adjusted, adjusted_all = generate_sequential_forecast(
                        df, clf, model_stage2, feature_cols, forecast_year,
                        adjustments=adjustments,
                    )

                    # Store in session state so results survive reruns
                    st.session_state[interactive_cache_key] = {
                        'year': forecast_year,
                        'pred_baseline': pred_baseline.copy(),
                        'pred_adjusted': pred_adjusted.copy(),
                        'baseline_all': baseline_all.copy(),
                        'adjusted_all': adjusted_all.copy(),
                    }
                except Exception as e:
                    forecast_error = str(e)

        elif use_cached:
            pred_baseline = cached['pred_baseline'].copy()
            pred_adjusted = cached['pred_adjusted'].copy()
            baseline_all = cached['baseline_all'].copy()
            adjusted_all = cached['adjusted_all'].copy()

        # ---- Render results in the right column ----
        with col_results:
            st.subheader("📈 Forecast Results")

            if forecast_error is not None:
                st.error(f"Error generating forecast: {forecast_error}")
                st.write("Please check your inputs and try again.")

            elif pred_baseline is not None and pred_adjusted is not None:
                baseline_area = pred_baseline['predicted_panel_area_m2'].sum()
                adjusted_area = pred_adjusted['predicted_panel_area_m2'].sum()
                change = 0 if baseline_area == 0 else ((adjusted_area / baseline_area) - 1) * 100
                ev_baseline = pred_baseline['ev_points_164m'].sum()
                ev_adjusted = pred_adjusted['ev_points_164m'].sum()
                ev_change_pct = 0 if ev_baseline == 0 else ((ev_adjusted / ev_baseline) - 1) * 100
                
                # Show metrics
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric(
                        "Baseline Forecast",
                        f"{baseline_area:,.0f} m²",
                        "No changes"
                    )
                
                with col_b:
                    st.metric(
                        "Adjusted Forecast",
                        f"{adjusted_area:,.0f} m²",
                        f"{change:+.1f}%",
                        delta_color="normal" if change > 0 else "inverse"
                    )
                
                with col_c:
                    st.metric(
                        "Total EV Points",
                        f"{ev_adjusted:,.0f}",
                        f"{ev_change_pct:+.1f}% vs baseline"
                    )
                
                st.divider()
                
                # Comparison chart
                comparison_df = pd.DataFrame({
                    'Scenario': ['Baseline', 'Adjusted'],
                    'Total Panel Area (m²)': [baseline_area, adjusted_area],
                })

                colors = ['#667eea', '#f093fb']

                fig_area = go.Figure(
                    go.Bar(
                        x=comparison_df['Scenario'],
                        y=comparison_df['Total Panel Area (m²)'],
                        marker_color=colors,
                        text=comparison_df['Total Panel Area (m²)'].apply(lambda x: f'{x:,.0f}'),
                        textposition='outside'
                    )
                )
                fig_area.update_layout(
                    title="Total Panel Area Comparison",
                    yaxis_title="Panel Area (m²)",
                    height=350
                )

                st.plotly_chart(fig_area, width='stretch')

                # Timeline chart from first historical year to forecast year
                base_year = first_forecast_year - 1
                min_year = int(df['year'].min())
                historical_totals = df.groupby('year')['panel_area_m2'].sum().to_dict()
                baseline_totals = baseline_all.groupby('year')['predicted_panel_area_m2'].sum().to_dict()
                adjusted_totals = adjusted_all.groupby('year')['predicted_panel_area_m2'].sum().to_dict()

                timeline_years = list(range(min_year, forecast_year + 1))
                timeline_df = pd.DataFrame({'year': timeline_years})

                def map_series(year, forecast_map):
                    if year <= base_year:
                        return historical_totals.get(year, np.nan)
                    return forecast_map.get(year, np.nan)

                timeline_df['Baseline'] = timeline_df['year'].apply(lambda y: map_series(y, baseline_totals))
                timeline_df['Adjusted'] = timeline_df['year'].apply(lambda y: map_series(y, adjusted_totals))

                fig_timeline = go.Figure()
                fig_timeline.add_trace(
                    go.Scatter(
                        x=timeline_df['year'],
                        y=timeline_df['Baseline'],
                        mode='lines+markers',
                        name='Baseline',
                        line=dict(color='#667eea', width=3)
                    )
                )
                fig_timeline.add_trace(
                    go.Scatter(
                        x=timeline_df['year'],
                        y=timeline_df['Adjusted'],
                        mode='lines+markers',
                        name='Adjusted',
                        line=dict(color='#f093fb', width=3, dash='dash')
                    )
                )
                fig_timeline.update_layout(
                    title="Panel Area Trajectory (2004 → Forecast Year)",
                    xaxis_title="Year",
                    yaxis_title="Panel Area (m²)",
                    hovermode='x unified',
                    height=400
                )

                st.plotly_chart(fig_timeline, width='stretch')
                
                # Demographic comparison table
                st.divider()
                st.subheader("📋 Demographic Details")
                
                demo_comparison = pd.DataFrame({
                    'Metric': [
                        'Total Rooftops',
                        'Unemployment Rate (%)',
                        'Average Age (years)',
                        'Total Population',
                        'Young Population',
                        'Elderly Population',
                        'EV Charging Points',
                        'PV Price (€/kWp)',
                    ],
                    'Baseline': [
                        pred_baseline['total_rooftops'].mean(),
                        pred_baseline['Unemployment_Rate'].mean(),
                        pred_baseline['Average_Age'].mean(),
                        pred_baseline['Total_Population'].mean(),
                        pred_baseline['Young_Population'].mean(),
                        pred_baseline['Elderly_Population'].mean(),
                        pred_baseline['ev_points_164m'].mean(),
                        pred_baseline['pv_price'].mean(),
                    ],
                    'Adjusted': [
                        pred_adjusted['total_rooftops'].mean(),
                        pred_adjusted['Unemployment_Rate'].mean(),
                        pred_adjusted['Average_Age'].mean(),
                        pred_adjusted['Total_Population'].mean(),
                        pred_adjusted['Young_Population'].mean(),
                        pred_adjusted['Elderly_Population'].mean(),
                        pred_adjusted['ev_points_164m'].mean(),
                        pred_adjusted['pv_price'].mean(),
                    ],
                })
                
                demo_comparison['Change (%)'] = ((demo_comparison['Adjusted'] / demo_comparison['Baseline']) - 1) * 100
                
                st.dataframe(
                    demo_comparison.style.format({
                        'Baseline': '{:.2f}',
                        'Adjusted': '{:.2f}',
                        'Change (%)': '{:+.2f}%'
                    }).background_gradient(subset=['Change (%)'], cmap='RdYlGn'),
                    width='stretch'
                )
                
                st.subheader("🔌 EV Hotspots (Top Tiles)")
                ev_tiles = pred_adjusted[['tile', 'district_number', 'ev_points_164m', 'predicted_panel_area_m2']].copy()
                ev_tiles = ev_tiles.sort_values(
                    ['ev_points_164m', 'predicted_panel_area_m2'], ascending=[False, False]
                ).head(10)
                st.dataframe(
                    ev_tiles.rename(columns={
                        'tile': 'Tile',
                        'district_number': 'District',
                        'ev_points_164m': 'EV Points',
                        'predicted_panel_area_m2': 'Predicted Panel Area (m²)'
                    }).style.format({
                        'EV Points': '{:,.1f}',
                        'Predicted Panel Area (m²)': '{:,.0f}'
                    }),
                    use_container_width=True
                )
                full_ev_tiles = pred_adjusted[
                    ['year', 'tile', 'district_number', 'ev_points_164m', 'predicted_panel_area_m2', 'predicted_adoption_prob']
                ].copy()
                full_ev_tiles = full_ev_tiles.sort_values(
                    ['ev_points_164m', 'predicted_panel_area_m2'], ascending=[False, False]
                )
                ev_csv = full_ev_tiles.to_csv(index=False)
                st.download_button(
                    "📥 Download Tile-Level EV Stats",
                    data=ev_csv,
                    file_name=f"ev_tile_stats_{forecast_year}.csv",
                    mime="text/csv"
                )
                with st.expander("View EV stats for all tiles", expanded=False):
                    st.dataframe(
                        full_ev_tiles.rename(columns={
                            'district_number': 'District',
                            'ev_points_164m': 'EV Points',
                            'predicted_panel_area_m2': 'Predicted Panel Area (m²)',
                            'predicted_adoption_prob': 'Adoption Probability'
                        }).style.format({
                            'EV Points': '{:,.1f}',
                            'Predicted Panel Area (m²)': '{:,.0f}',
                            'Adoption Probability': '{:.2f}'
                        }),
                        use_container_width=True,
                        height=400
                    )

            else:
                st.info("👈 Click the 'Generate Forecast' button to see predictions")
    
    # ==================== DISTRICT ANALYSIS MODE ====================
    elif mode == "📍 District Analysis":
        st.header("📍 District-Level Analysis")

        # Load GeoJSON for Munich districts
        GEOJSON_PATH = BASE_DIR / "data" / "munich_districts_4326.geojson"

        @st.cache_data
        def load_geojson():
            import json as _json
            with open(GEOJSON_PATH) as f:
                return _json.load(f)

        @st.cache_data
        def get_district_names():
            geo = load_geojson()
            return {int(f['properties']['sb_nummer']): f['properties']['name'] for f in geo['features']}

        district_name_map = get_district_names()

        forecast_year = st.slider(
            "Select Forecast Year",
            first_forecast_year,
            max_forecast_year,
            default_forecast_year,
            key='district_year'
        )

        # ---- Per-district demographic adjustments ----
        st.subheader("🎛️ Per-District Demographic Adjustments")
        st.caption("Select a district and adjust its demographics before running the forecast.")

        all_district_nums = sorted(df['district_number'].unique())
        district_labels = [f"{d} — {district_name_map.get(d, 'Unknown')}" for d in all_district_nums]

        adj_district_sel = st.selectbox(
            "District to adjust",
            options=all_district_nums,
            format_func=lambda d: f"{d} — {district_name_map.get(d, 'Unknown')}",
            key='adj_district_sel'
        )

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            adj_unemp = st.number_input(
                "Unemployment Change (%)",
                -5.0, 5.0, 0.0, 0.5,
                key=f'adj_unemp_{adj_district_sel}'
            )
        with col_b:
            adj_pop = st.number_input(
                "Population Change (%)",
                -20.0, 20.0, 0.0, 1.0,
                key=f'adj_pop_{adj_district_sel}'
            )
        with col_c:
            adj_pv = st.number_input(
                "PV Price Change (%)",
                -50.0, 50.0, 0.0, 5.0,
                key=f'adj_pv_{adj_district_sel}'
            )
        with col_d:
            adj_ev = st.number_input(
                "EV Points Change (%)",
                -50.0, 100.0, 0.0, 5.0,
                key=f'adj_ev_{adj_district_sel}'
            )

        # Save / clear adjustments
        if 'district_adjustments' not in st.session_state:
            st.session_state['district_adjustments'] = {}

        col_save, col_clear = st.columns(2)
        with col_save:
            if st.button("💾 Save Adjustment", use_container_width=True):
                if any(v != 0 for v in [adj_unemp, adj_pop, adj_pv, adj_ev]):
                    st.session_state['district_adjustments'][adj_district_sel] = {
                        'Unemployment_Rate': adj_unemp,
                        'Total_Population': adj_pop,
                        'Young_Population': adj_pop,
                        'Elderly_Population': adj_pop,
                        'pv_price': adj_pv,
                        'ev_points_164m': adj_ev,
                    }
                    st.success(f"Saved adjustments for District {adj_district_sel}")
                else:
                    st.warning("All adjustments are 0 — nothing to save.")
        with col_clear:
            if st.button("🗑️ Clear All Adjustments", use_container_width=True):
                st.session_state['district_adjustments'] = {}
                st.info("All adjustments cleared.")

        # Show current adjustments
        if st.session_state.get('district_adjustments'):
            with st.expander("📋 Current adjustments", expanded=False):
                for d_num, adjs in sorted(st.session_state['district_adjustments'].items()):
                    name = district_name_map.get(d_num, 'Unknown')
                    parts = []
                    if adjs.get('Unemployment_Rate', 0) != 0:
                        parts.append(f"Unemp: {adjs['Unemployment_Rate']:+.1f}%")
                    if adjs.get('Total_Population', 0) != 0:
                        parts.append(f"Pop: {adjs['Total_Population']:+.1f}%")
                    if adjs.get('pv_price', 0) != 0:
                        parts.append(f"PV Price: {adjs['pv_price']:+.1f}%")
                    if adjs.get('ev_points_164m', 0) != 0:
                        parts.append(f"EV: {adjs['ev_points_164m']:+.1f}%")
                    st.write(f"**District {d_num} ({name}):** {', '.join(parts)}")

        st.divider()
        district_predict_btn = st.button("🚀 Generate District Forecast", use_container_width=True, type='primary')

        cache_key = 'district_forecast_cache'
        cached_forecast = st.session_state.get(cache_key)
        use_cached = (
            cached_forecast is not None
            and cached_forecast.get('year') == forecast_year
            and not district_predict_btn
        )

        predictions = None
        all_forecasts = None

        if district_predict_btn:
            with st.spinner("Generating district forecasts..."):
                # Run baseline forecast
                predictions_base, all_forecasts_base = generate_sequential_forecast(
                    df, clf, model_stage2, feature_cols, forecast_year
                )

                # Apply per-district adjustments if any
                district_adjs = st.session_state.get('district_adjustments', {})
                if district_adjs:
                    adj_df = df.copy()
                    for d_num, adjs in district_adjs.items():
                        mask = adj_df['district_number'] == d_num
                        for col_name, pct_change in adjs.items():
                            if col_name in adj_df.columns and pct_change != 0:
                                if col_name == 'Unemployment_Rate':
                                    adj_df.loc[mask, col_name] += pct_change
                                else:
                                    adj_df.loc[mask, col_name] *= (1 + pct_change / 100)
                    predictions, all_forecasts = generate_sequential_forecast(
                        adj_df, clf, model_stage2, feature_cols, forecast_year
                    )
                else:
                    predictions = predictions_base
                    all_forecasts = all_forecasts_base

            st.session_state[cache_key] = {
                'year': forecast_year,
                'predictions': predictions.copy(),
                'all_forecasts': all_forecasts.copy(),
                'predictions_base': predictions_base.copy(),
            }
        elif use_cached:
            predictions = cached_forecast['predictions'].copy()
            all_forecasts = cached_forecast['all_forecasts'].copy()

        if predictions is not None and all_forecasts is not None:
            # Compute district-level stats
            district_stats = predictions.groupby('district_number').agg({
                'predicted_panel_area_m2': 'sum',
                'Total_Population': 'mean',
                'Average_Age': 'mean',
                'Unemployment_Rate': 'mean',
                'ev_points_164m': 'mean'
            }).reset_index()

            # Compute historical baseline for growth
            latest_year_hist = int(df['year'].max())
            hist_area = (
                df[df['year'] == latest_year_hist]
                .groupby('district_number')['panel_area_m2']
                .sum()
                .rename('historical_area')
            )
            district_stats = district_stats.merge(hist_area, on='district_number', how='left')
            district_stats['historical_area'] = district_stats['historical_area'].fillna(0)
            district_stats['growth_m2'] = district_stats['predicted_panel_area_m2'] - district_stats['historical_area']
            district_stats['growth_pct'] = np.where(
                district_stats['historical_area'] > 0,
                (district_stats['growth_m2'] / district_stats['historical_area']) * 100,
                0
            )
            district_stats['district_name'] = district_stats['district_number'].map(district_name_map).fillna('Unknown')
            district_stats = district_stats.sort_values('predicted_panel_area_m2', ascending=False)

            # ── Summary metrics ──
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Predicted Area", f"{district_stats['predicted_panel_area_m2'].sum():,.0f} m²")
            with col2:
                st.metric("Historical Area", f"{district_stats['historical_area'].sum():,.0f} m²")
            with col3:
                total_growth = district_stats['growth_m2'].sum()
                st.metric("Total Growth", f"{total_growth:+,.0f} m²")
            with col4:
                total_hist = district_stats['historical_area'].sum()
                total_growth_pct = (total_growth / total_hist * 100) if total_hist > 0 else 0
                st.metric("Growth %", f"{total_growth_pct:+.1f}%")

            # ── Choropleth Map ──
            st.subheader("🗺️ Munich District Map")

            map_metric = st.radio(
                "Color map by:",
                ["Predicted Panel Area (m²)", "Growth (m²)", "Growth (%)"],
                horizontal=True,
                key='map_metric'
            )
            metric_col_map = {
                "Predicted Panel Area (m²)": "predicted_panel_area_m2",
                "Growth (m²)": "growth_m2",
                "Growth (%)": "growth_pct",
            }
            chosen_metric = metric_col_map[map_metric]

            try:
                geojson_data = load_geojson()

                # Build mapping: geojson sb_nummer (str) → data district_number (int)
                map_df = district_stats.copy()
                map_df['district_str'] = map_df['district_number'].astype(int).astype(str).str.zfill(2)

                fig_map = px.choropleth_mapbox(
                    map_df,
                    geojson=geojson_data,
                    locations='district_str',
                    featureidkey='properties.sb_nummer',
                    color=chosen_metric,
                    color_continuous_scale='YlOrRd' if 'Growth' not in map_metric else 'RdYlGn',
                    hover_name='district_name',
                    hover_data={
                        'district_str': False,
                        'predicted_panel_area_m2': ':.0f',
                        'growth_m2': ':+.0f',
                        'growth_pct': ':+.1f',
                        'Total_Population': ':.0f',
                        'Unemployment_Rate': ':.2f',
                    },
                    labels={
                        'predicted_panel_area_m2': 'Panel Area (m²)',
                        'growth_m2': 'Growth (m²)',
                        'growth_pct': 'Growth (%)',
                        'Total_Population': 'Population',
                        'Unemployment_Rate': 'Unemployment %',
                    },
                    mapbox_style='carto-positron',
                    center={'lat': 48.137, 'lon': 11.575},
                    zoom=10.5,
                    opacity=0.7,
                )
                fig_map.update_layout(
                    margin=dict(l=0, r=0, t=30, b=0),
                    height=550,
                    title=f"Munich Districts — {map_metric} ({forecast_year})",
                )
                st.plotly_chart(fig_map, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render map: {e}")
                st.info("Falling back to bar chart view.")

            # ── Bar chart ──
            st.subheader(f"🏆 District Rankings ({forecast_year})")
            col1, col2 = st.columns([3, 1])
            with col2:
                show_all = st.checkbox("Show All Districts", value=False)

            display_districts = district_stats if show_all else district_stats.head(10)
            chart_height = max(500, len(display_districts) * 28)

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                y=display_districts['district_name'],
                x=display_districts['predicted_panel_area_m2'],
                orientation='h',
                name='Predicted',
                marker=dict(
                    color=display_districts['growth_pct'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Growth %")
                ),
                text=display_districts.apply(
                    lambda r: f"{r['predicted_panel_area_m2']:,.0f} m² ({r['growth_pct']:+.1f}%)", axis=1
                ),
                textposition='outside',
            ))
            fig_bar.update_layout(
                title=f"{'All' if show_all else 'Top 10'} Districts — Predicted Area & Growth ({forecast_year})",
                xaxis_title="Panel Area (m²)",
                height=chart_height,
            )
            fig_bar.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_bar, use_container_width=True)

            # ── Timeline ──
            st.subheader("📈 Installed Area Timeline")
            base_year = int(df['year'].max())
            min_year = int(df['year'].min())
            historical_totals = df.groupby('year')['panel_area_m2'].sum().to_dict()
            forecast_totals = all_forecasts.groupby('year')['predicted_panel_area_m2'].sum().to_dict()
            timeline_years = list(range(min_year, forecast_year + 1))
            actual_years = [y for y in timeline_years if y <= base_year]
            forecast_years_seq = [y for y in timeline_years if y > base_year]

            fig_timeline = go.Figure()
            if actual_years:
                fig_timeline.add_trace(go.Scatter(
                    x=actual_years,
                    y=[historical_totals.get(y, np.nan) for y in actual_years],
                    mode='lines+markers', name='Historical',
                    line=dict(color='#667eea', width=3)
                ))
            if forecast_years_seq:
                fig_timeline.add_trace(go.Scatter(
                    x=forecast_years_seq,
                    y=[forecast_totals.get(y, np.nan) for y in forecast_years_seq],
                    mode='lines+markers', name='Forecast',
                    line=dict(color='#f093fb', width=3, dash='dash')
                ))
            fig_timeline.update_layout(
                title="Installed Panel Area (All Districts)",
                xaxis_title="Year", yaxis_title="Panel Area (m²)",
                hovermode='x unified', height=400
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

            # ── District trajectory explorer ──
            st.subheader("🎯 District Trajectory Explorer")
            district_options = sorted(district_stats['district_number'].unique())
            selected_district = st.selectbox(
                "Select a district",
                district_options,
                format_func=lambda d: f"{d} — {district_name_map.get(d, 'Unknown')}",
                key='district_trend_selector'
            )

            district_history = (
                df[df['district_number'] == selected_district]
                .groupby('year')['panel_area_m2'].sum()
            )
            district_forecast = (
                all_forecasts[all_forecasts['district_number'] == selected_district]
                .groupby('year')['predicted_panel_area_m2'].sum()
            )

            fig_district = go.Figure()
            if not district_history.empty:
                fig_district.add_trace(go.Scatter(
                    x=sorted(district_history.index.tolist()),
                    y=district_history.sort_index().values,
                    mode='lines+markers', name='Historical',
                    line=dict(color='#06A77D', width=3)
                ))
            if not district_forecast.empty:
                fig_district.add_trace(go.Scatter(
                    x=sorted(district_forecast.index.tolist()),
                    y=district_forecast.sort_index().values,
                    mode='lines+markers', name='Forecast',
                    line=dict(color='#F18F01', width=3, dash='dash')
                ))
            if fig_district.data:
                fig_district.update_layout(
                    title=f"District {selected_district} — {district_name_map.get(selected_district, '')}",
                    xaxis_title="Year", yaxis_title="Panel Area (m²)",
                    hovermode='x unified', height=400
                )
                st.plotly_chart(fig_district, use_container_width=True)

            # ── Detailed table ──
            st.subheader(f"📊 All Districts — Detailed View")
            table_df = district_stats[[
                'district_number', 'district_name', 'predicted_panel_area_m2',
                'historical_area', 'growth_m2', 'growth_pct',
                'Total_Population', 'Average_Age', 'Unemployment_Rate', 'ev_points_164m'
            ]].copy()
            table_df.columns = [
                'District #', 'Name', 'Predicted Area (m²)', 'Historical Area (m²)',
                'Growth (m²)', 'Growth (%)', 'Population', 'Avg Age', 'Unemployment %', 'EV Points'
            ]
            st.dataframe(
                table_df.style.format({
                    'Predicted Area (m²)': '{:,.0f}',
                    'Historical Area (m²)': '{:,.0f}',
                    'Growth (m²)': '{:+,.0f}',
                    'Growth (%)': '{:+.1f}',
                    'Population': '{:,.0f}',
                    'Avg Age': '{:.1f}',
                    'Unemployment %': '{:.2f}',
                    'EV Points': '{:,.1f}',
                }).background_gradient(subset=['Growth (%)'], cmap='RdYlGn'),
                use_container_width=True,
                height=600,
            )

        else:
            latest_year = int(df['year'].max())
            st.info(f"Showing actual observations for {latest_year}. Adjust demographics above and click 'Generate District Forecast'.")

            latest_actual = df[df['year'] == latest_year]
            actual_stats = latest_actual.groupby('district_number').agg({
                'panel_area_m2': 'sum',
                'Total_Population': 'mean',
                'Average_Age': 'mean',
                'Unemployment_Rate': 'mean',
                'ev_points_164m': 'mean'
            }).reset_index()
            actual_stats['district_name'] = actual_stats['district_number'].map(district_name_map).fillna('Unknown')
            actual_stats = actual_stats.sort_values('panel_area_m2', ascending=False)

            # Map of current state
            try:
                geojson_data = load_geojson()
                map_df_actual = actual_stats.copy()
                map_df_actual['district_str'] = map_df_actual['district_number'].astype(int).astype(str).str.zfill(2)

                fig_map_actual = px.choropleth_mapbox(
                    map_df_actual,
                    geojson=geojson_data,
                    locations='district_str',
                    featureidkey='properties.sb_nummer',
                    color='panel_area_m2',
                    color_continuous_scale='YlOrRd',
                    hover_name='district_name',
                    hover_data={
                        'district_str': False,
                        'panel_area_m2': ':.0f',
                        'Total_Population': ':.0f',
                        'Unemployment_Rate': ':.2f',
                    },
                    labels={'panel_area_m2': 'Panel Area (m²)', 'Total_Population': 'Population', 'Unemployment_Rate': 'Unemployment %'},
                    mapbox_style='carto-positron',
                    center={'lat': 48.137, 'lon': 11.575},
                    zoom=10.5,
                    opacity=0.7,
                )
                fig_map_actual.update_layout(
                    margin=dict(l=0, r=0, t=30, b=0),
                    height=550,
                    title=f"Munich Districts — Installed Panel Area ({latest_year})",
                )
                st.plotly_chart(fig_map_actual, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render map: {e}")

            # Bar chart
            col_chart, col_toggle = st.columns([3, 1])
            with col_chart:
                st.subheader(f"🏛️ Actual District Performance ({latest_year})")
            with col_toggle:
                show_all_actual = st.checkbox("Show All Districts", value=False, key='show_all_actual')

            top_actual = actual_stats if show_all_actual else actual_stats.head(10)
            chart_height_actual = max(500, len(top_actual) * 28)

            fig_actual = go.Figure(go.Bar(
                y=top_actual['district_name'],
                x=top_actual['panel_area_m2'],
                orientation='h',
                marker=dict(color='#667eea'),
                text=top_actual['panel_area_m2'].apply(lambda x: f'{x:,.0f} m²'),
                textposition='outside'
            ))
            fig_actual.update_layout(
                title=f"{'All' if show_all_actual else 'Top 10'} Districts ({latest_year})",
                xaxis_title="Panel Area (m²)",
                height=chart_height_actual
            )
            fig_actual.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_actual, use_container_width=True)

    # ==================== SENSITIVITY ANALYSIS MODE ====================
    elif mode == "📐 Sensitivity Analysis":
        st.header("📐 Sensitivity Analysis")
        st.info(
            "This tool sweeps one parameter at a time across a range while holding "
            "all others at baseline.  The resulting **tornado chart** reveals which "
            "factors have the biggest impact on total predicted panel area."
        )

        # ── Controls ──
        col_ctrl, col_chart = st.columns([1, 2])

        with col_ctrl:
            sa_forecast_year = st.slider(
                "Forecast Year",
                first_forecast_year,
                max_forecast_year,
                default_forecast_year,
                key='sa_year'
            )

            st.markdown("### Parameters to Sweep")

            # Each entry: (label, internal key, adjustment type, default range)
            SA_PARAMS = [
                ("PV Price (%)",           "pv_price",           "percent",  (-40, 40, 10)),
                ("Unemployment Rate (pp)", "Unemployment_Rate",  "absolute", (-3, 3, 1)),
                ("Total Population (%)",   "Total_Population",   "percent",  (-15, 15, 5)),
                ("Youth Population (%)",   "Young_Population",   "percent",  (-15, 15, 5)),
                ("Elderly Population (%)", "Elderly_Population", "percent",  (-15, 15, 5)),
                ("EV Points (%)",          "ev_points_164m",     "percent",  (-40, 40, 10)),
                ("Total Rooftops (%)",     "total_rooftops",     "percent",  (-20, 20, 5)),
            ]

            selected_params = st.multiselect(
                "Select parameters to sweep",
                [p[0] for p in SA_PARAMS],
                default=[p[0] for p in SA_PARAMS[:4]],
                help="Choose which parameters to include in the tornado chart."
            )

            sa_steps = st.slider("Steps per parameter", 3, 11, 5, 2, key='sa_steps')

            sa_run = st.button("🚀 Run Sensitivity Analysis", use_container_width=True, type='primary')

        # ── Run ──
        sa_cache_key = 'sensitivity_cache'
        cached_sa = st.session_state.get(sa_cache_key)
        use_cached_sa = (
            cached_sa is not None
            and cached_sa.get('year') == sa_forecast_year
            and not sa_run
        )

        sa_results = None
        sa_baseline_total = None
        sa_error = None

        if sa_run:
            with st.spinner("Running baseline forecast..."):
                try:
                    baseline_pred, _ = generate_sequential_forecast(
                        df, clf, model_stage2, feature_cols, sa_forecast_year
                    )
                    sa_baseline_total = baseline_pred['predicted_panel_area_m2'].sum()
                except Exception as e:
                    sa_error = str(e)

            if sa_error is None:
                sa_results = []
                param_lookup = {p[0]: p for p in SA_PARAMS}

                progress_bar = st.progress(0)
                total_runs = 0
                for pname in selected_params:
                    total_runs += sa_steps
                progress_counter = 0

                for pname in selected_params:
                    label, key, adj_type, (lo, hi, _step) = param_lookup[pname]
                    sweep_values = np.linspace(lo, hi, sa_steps)
                    param_totals = []

                    for val in sweep_values:
                        progress_counter += 1
                        progress_bar.progress(progress_counter / total_runs, text=f"Sweeping {label}: {val:+.1f}")

                        adj = {key: {'type': adj_type, 'value': float(val)}}
                        try:
                            pred, _ = generate_sequential_forecast(
                                df, clf, model_stage2, feature_cols, sa_forecast_year,
                                adjustments=adj,
                            )
                            total_area = pred['predicted_panel_area_m2'].sum()
                        except Exception:
                            total_area = sa_baseline_total  # fallback
                        param_totals.append(total_area)

                    sa_results.append({
                        'label': label,
                        'key': key,
                        'sweep': sweep_values.tolist(),
                        'totals': param_totals,
                        'min_total': min(param_totals),
                        'max_total': max(param_totals),
                    })

                progress_bar.empty()

                st.session_state[sa_cache_key] = {
                    'year': sa_forecast_year,
                    'results': sa_results,
                    'baseline': sa_baseline_total,
                }

        elif use_cached_sa:
            sa_results = cached_sa['results']
            sa_baseline_total = cached_sa['baseline']

        # ── Render ──
        with col_chart:
            if sa_error:
                st.error(f"Error: {sa_error}")

            elif sa_results is not None and sa_baseline_total is not None:
                st.subheader(f"🌪️ Tornado Chart — {sa_forecast_year}")
                st.caption(f"Baseline total: **{sa_baseline_total:,.0f} m²**  |  Model: **{selected_model_name}**")

                # Sort by impact range (largest spread first)
                sa_results_sorted = sorted(
                    sa_results,
                    key=lambda r: r['max_total'] - r['min_total'],
                    reverse=True
                )

                labels = [r['label'] for r in sa_results_sorted]
                low_deltas = [(r['min_total'] - sa_baseline_total) for r in sa_results_sorted]
                high_deltas = [(r['max_total'] - sa_baseline_total) for r in sa_results_sorted]

                fig_tornado = go.Figure()

                # Negative side (red)
                fig_tornado.add_trace(go.Bar(
                    y=labels,
                    x=low_deltas,
                    orientation='h',
                    name='Decrease',
                    marker_color='#e74c3c',
                    text=[f"{d:+,.0f}" for d in low_deltas],
                    textposition='outside',
                ))

                # Positive side (green)
                fig_tornado.add_trace(go.Bar(
                    y=labels,
                    x=high_deltas,
                    orientation='h',
                    name='Increase',
                    marker_color='#2ecc71',
                    text=[f"{d:+,.0f}" for d in high_deltas],
                    textposition='outside',
                ))

                fig_tornado.update_layout(
                    title="Impact on Total Predicted Panel Area (m²)",
                    xaxis_title="Change from Baseline (m²)",
                    barmode='overlay',
                    height=max(400, len(labels) * 60),
                    yaxis=dict(autorange='reversed'),
                    legend=dict(orientation='h', y=-0.15),
                )

                # Add baseline line
                fig_tornado.add_vline(x=0, line_dash='dash', line_color='gray', line_width=1)

                st.plotly_chart(fig_tornado, use_container_width=True)

                # ── Per-parameter sweep charts ──
                st.subheader("📈 Parameter Sweep Curves")
                st.caption("Each chart shows how total panel area changes as one parameter is varied.")

                n_params = len(sa_results_sorted)
                cols_per_row = 2
                for i in range(0, n_params, cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx >= n_params:
                            break
                        r = sa_results_sorted[idx]
                        with col:
                            fig_sweep = go.Figure()
                            fig_sweep.add_trace(go.Scatter(
                                x=r['sweep'],
                                y=r['totals'],
                                mode='lines+markers',
                                name=r['label'],
                                line=dict(color='#667eea', width=3),
                                marker=dict(size=8),
                            ))
                            # Baseline reference
                            fig_sweep.add_hline(
                                y=sa_baseline_total,
                                line_dash='dash', line_color='gray',
                                annotation_text='Baseline',
                                annotation_position='bottom right'
                            )
                            fig_sweep.update_layout(
                                title=r['label'],
                                xaxis_title="Adjustment",
                                yaxis_title="Total Panel Area (m²)",
                                height=300,
                                margin=dict(t=40, b=30),
                            )
                            st.plotly_chart(fig_sweep, use_container_width=True)

                # ── Summary table ──
                st.subheader("📋 Sensitivity Summary")
                summary_rows = []
                for r in sa_results_sorted:
                    spread = r['max_total'] - r['min_total']
                    pct_spread = (spread / sa_baseline_total) * 100 if sa_baseline_total else 0
                    summary_rows.append({
                        'Parameter': r['label'],
                        'Min Total (m²)': r['min_total'],
                        'Max Total (m²)': r['max_total'],
                        'Spread (m²)': spread,
                        'Spread (%)': pct_spread,
                    })
                summary_df = pd.DataFrame(summary_rows)
                st.dataframe(
                    summary_df.style.format({
                        'Min Total (m²)': '{:,.0f}',
                        'Max Total (m²)': '{:,.0f}',
                        'Spread (m²)': '{:,.0f}',
                        'Spread (%)': '{:.1f}%',
                    }).background_gradient(subset=['Spread (%)'], cmap='YlOrRd'),
                    use_container_width=True
                )

            else:
                st.info("👈 Select parameters and click 'Run Sensitivity Analysis' to see results.")


if __name__ == "__main__":
    main()
