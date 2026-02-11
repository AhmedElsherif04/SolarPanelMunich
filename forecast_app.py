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
STAGE1_MODEL_PATH = BASE_DIR / "Training" / "lgb_model_1Stage_lag_ev.txt"
STAGE2_MODEL_PATH = BASE_DIR / "Training" / "random_forest_model_lag_ev.joblib"
FEATURE_COLS = [
    'year',
    'total_rooftops',
    'Unemployment_Rate',
    'Average_Age',
    'Elderly_Population',
    'Young_Population',
    'Total_Population',
    'tile_encoded',
    'employed',
    'pv_price',
    'panel_area_lag1',
    'ev_points_164m',
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
def load_models():
    """Load trained models saved from the notebook pipeline"""
    clf = lgb.Booster(model_file=str(STAGE1_MODEL_PATH))
    model_stage2 = joblib.load(STAGE2_MODEL_PATH)
    return clf, model_stage2, FEATURE_COLS

def create_forecast_features(df, forecast_year, feature_cols, adjustments=None):
    """Create features for forecasting with optional adjustments"""
    latest_year = df['year'].max()
    latest_data = df[df['year'] == latest_year].copy()

    forecast_data = latest_data.copy()
    forecast_data['year'] = forecast_year

    for tile in latest_data['tile'].unique():
        tile_history = df[df['tile'] == tile].sort_values('year')

        if len(tile_history) < 2:
            continue

        years = tile_history['year'].values.astype(float)

        # Forecast demographic features using linear trends
        demo_cols = [
            'Average_Age',
            'Elderly_Population',
            'Young_Population',
            'Total_Population',
            'Unemployment_Rate',
            'employed',
            'pv_price',
            'ev_points_164m',
        ]

        for col in demo_cols:
            if col not in tile_history.columns:
                continue

            values = tile_history[col].values.astype(float)
            valid_mask = ~np.isnan(values)
            valid_years = years[valid_mask]
            valid_values = values[valid_mask]

            if len(valid_years) > 1:
                slope, intercept, _, _, _ = stats.linregress(valid_years, valid_values)
                forecast_value = slope * forecast_year + intercept

                # Apply adjustments if provided
                if adjustments and col in adjustments:
                    adj = adjustments[col]
                    if adj['type'] == 'percent':
                        forecast_value *= (1 + adj['value'] / 100)
                    elif adj['type'] == 'absolute':
                        forecast_value += adj['value']

                # Apply bounds
                if col == 'Unemployment_Rate':
                    forecast_value = np.clip(forecast_value, 0, 20)
                elif col == 'Average_Age':
                    forecast_value = np.clip(forecast_value, 20, 60)
                else:
                    forecast_value = max(0, forecast_value)

                forecast_data.loc[forecast_data['tile'] == tile, col] = forecast_value

        # Use latest panel_area_m2 as lag1
        latest_panel_area = tile_history.iloc[-1]['panel_area_m2']
        forecast_data.loc[forecast_data['tile'] == tile, 'panel_area_lag1'] = latest_panel_area

    return forecast_data


def generate_sequential_forecast(df, clf, model_stage2, feature_cols, target_year, adjustments=None):
    """Forecast sequentially so each year reuses the previous year's predicted lag."""
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

        prev_panel_area = dict(zip(forecast_pred['tile'], forecast_pred['predicted_panel_area_m2']))

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
    """Make 2-stage predictions using the saved models."""
    X_forecast = forecast_data[feature_cols].copy()

    if hasattr(clf, 'best_iteration') and clf.best_iteration is not None:
        p_solar = clf.predict(X_forecast, num_iteration=clf.best_iteration)
    else:
        p_solar = clf.predict(X_forecast)

    y_pred_log = model_stage2.predict(X_forecast)
    y_pred_size = np.expm1(y_pred_log)

    y_pred_final = np.where(
        p_solar > 0.5,
        (p_solar + 0.5) * y_pred_size,
        np.where(p_solar < 0.3, 0.0, p_solar * y_pred_size)
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
    
    # Load data and models
    with st.spinner("Loading data and models..."):
        df = load_data()
        clf, model_stage2, feature_cols = load_models()
    first_forecast_year = int(df['year'].max() + 1)
    max_forecast_year = first_forecast_year + 10
    default_forecast_year = min(first_forecast_year + 2, max_forecast_year)
    
    # Sidebar
    st.sidebar.title("⚙️ Controls")
    
    # Select mode
    mode = st.sidebar.radio(
        "Select Mode",
        ["📂 View Raw Data", "📊 Overview & Historical Data", "🔮 Interactive Forecasting", "📍 District Analysis"]
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
            
            st.markdown("### 📊 Demographic Adjustments")
            st.caption("Enter changes from baseline")
            
            unemployment_change = st.number_input(
                "Unemployment Rate Change (%)",
                -5.0, 5.0, 0.0, 0.5
            )
            
            age_change = st.number_input(
                "Average Age Change (years)",
                -5.0, 5.0, 0.0, 0.5
            )
            
            population_change = st.number_input(
                "Population Change (%)",
                -20.0, 20.0, 0.0, 1.0
            )
            
            youth_change = st.number_input(
                "Youth Population Change (%)",
                -20.0, 20.0, 0.0, 1.0
            )
            
            elderly_change = st.number_input(
                "Elderly Population Change (%)",
                -20.0, 20.0, 0.0, 1.0
            )
            
            ev_change = st.number_input(
                "EV Charging Points Change (%)",
                -50.0, 100.0, 0.0, 5.0
            )
            
            # Predict button
            predict_button = st.button("🚀 Generate Forecast", use_container_width=True, type='primary')
        
        with col_results:
            st.subheader("📈 Forecast Results")
            
            if predict_button:
                with st.spinner("Generating forecast..."):
                    try:
                        adjustments = {
                            'Unemployment_Rate': {'type': 'absolute', 'value': unemployment_change},
                            'Average_Age': {'type': 'absolute', 'value': age_change},
                            'Total_Population': {'type': 'percent', 'value': population_change},
                            'Young_Population': {'type': 'percent', 'value': youth_change},
                            'Elderly_Population': {'type': 'percent', 'value': elderly_change},
                            'ev_points_164m': {'type': 'percent', 'value': ev_change}
                        }
                        
                        pred_baseline, baseline_all = generate_sequential_forecast(
                            df, clf, model_stage2, feature_cols, forecast_year
                        )
                        pred_adjusted, adjusted_all = generate_sequential_forecast(
                            df,
                            clf,
                            model_stage2,
                            feature_cols,
                            forecast_year,
                            adjustments=adjustments,
                        )

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
                                'Unemployment Rate (%)',
                                'Average Age (years)',
                                'Total Population',
                                'Young Population',
                                'Elderly Population',
                                'EV Charging Points',
                            ],
                            'Baseline': [
                                pred_baseline['Unemployment_Rate'].mean(),
                                pred_baseline['Average_Age'].mean(),
                                pred_baseline['Total_Population'].mean(),
                                pred_baseline['Young_Population'].mean(),
                                pred_baseline['Elderly_Population'].mean(),
                                pred_baseline['ev_points_164m'].mean(),
                            ],
                            'Adjusted': [
                                pred_adjusted['Unemployment_Rate'].mean(),
                                pred_adjusted['Average_Age'].mean(),
                                pred_adjusted['Total_Population'].mean(),
                                pred_adjusted['Young_Population'].mean(),
                                pred_adjusted['Elderly_Population'].mean(),
                                pred_adjusted['ev_points_164m'].mean(),
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
                        
                    except Exception as e:
                        st.error(f"Error generating forecast: {str(e)}")
                        st.write("Please check your inputs and try again.")
            else:
                st.info("👈 Click the 'Generate Forecast' button to see predictions")
    
    # ==================== DISTRICT ANALYSIS MODE ====================
    else:  # District Analysis
        st.header("📍 District-Level Analysis")

        forecast_year = st.slider(
            "Select Forecast Year",
            first_forecast_year,
            max_forecast_year,
            default_forecast_year,
            key='district_year'
        )

        st.caption("Review the latest observed data below, then click the button to generate a forward-looking district forecast.")
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
                predictions, all_forecasts = generate_sequential_forecast(
                    df, clf, model_stage2, feature_cols, forecast_year
                )
            st.session_state[cache_key] = {
                'year': forecast_year,
                'predictions': predictions.copy(),
                'all_forecasts': all_forecasts.copy()
            }
        elif use_cached:
            predictions = cached_forecast['predictions'].copy()
            all_forecasts = cached_forecast['all_forecasts'].copy()

        if predictions is not None and all_forecasts is not None:
            district_stats = predictions.groupby('district_number').agg({
                'predicted_panel_area_m2': 'sum',
                'Total_Population': 'mean',
                'Average_Age': 'mean',
                'Unemployment_Rate': 'mean',
                'ev_points_164m': 'mean'
            }).reset_index()

            district_stats = district_stats.sort_values('predicted_panel_area_m2', ascending=False)
            
            # District visualization options
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader(f"🏆 Districts Solar Adoption in {forecast_year}")
            with col2:
                show_all = st.checkbox("Show All Districts", value=False)
            
            # Select districts to display
            if show_all:
                display_districts = district_stats
                chart_title = f"All {len(district_stats)} Districts by Predicted Solar Panel Area ({forecast_year})"
                chart_height = max(600, len(district_stats) * 25)
            else:
                display_districts = district_stats.head(10)
                chart_title = f"Top 10 Districts by Predicted Solar Panel Area ({forecast_year})"
                chart_height = 500
            
            fig = go.Figure(go.Bar(
                y=display_districts['district_number'].astype(str),
                x=display_districts['predicted_panel_area_m2'],
                orientation='h',
                marker=dict(
                    color=display_districts['predicted_panel_area_m2'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Panel Area (m²)")
                ),
                text=display_districts['predicted_panel_area_m2'].apply(lambda x: f'{x:,.0f} m²'),
                textposition='outside'
            ))
            
            fig.update_layout(
                title=chart_title,
                xaxis_title="Predicted Panel Area (m²)",
                yaxis_title="District Number",
                height=chart_height
            )
            fig.update_yaxes(autorange="reversed")
            
            st.plotly_chart(fig, width='stretch')

            # Installed area timeline across the city
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
                actual_values = [historical_totals.get(y, np.nan) for y in actual_years]
                fig_timeline.add_trace(
                    go.Scatter(
                        x=actual_years,
                        y=actual_values,
                        mode='lines+markers',
                        name='Historical',
                        line=dict(color='#667eea', width=3)
                    )
                )
            if forecast_years_seq:
                forecast_values = [forecast_totals.get(y, np.nan) for y in forecast_years_seq]
                fig_timeline.add_trace(
                    go.Scatter(
                        x=forecast_years_seq,
                        y=forecast_values,
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='#f093fb', width=3, dash='dash')
                    )
                )

            fig_timeline.update_layout(
                title="Installed Panel Area (All Districts)",
                xaxis_title="Year",
                yaxis_title="Panel Area (m²)",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig_timeline, width='stretch')

            # District-specific trajectory explorer
            st.subheader("🎯 District Trajectory Explorer")
            district_options = sorted(district_stats['district_number'].unique())
            default_index = 0
            if display_districts.shape[0] > 0:
                top_district = display_districts.iloc[0]['district_number']
                if top_district in district_options:
                    default_index = district_options.index(top_district)
            selected_district = st.selectbox(
                "Select a district to inspect",
                district_options,
                index=default_index,
                key='district_trend_selector'
            )

            district_history = (
                df[df['district_number'] == selected_district]
                .groupby('year')['panel_area_m2']
                .sum()
            )
            district_forecast = (
                all_forecasts[all_forecasts['district_number'] == selected_district]
                .groupby('year')['predicted_panel_area_m2']
                .sum()
            )

            fig_district = go.Figure()
            if not district_history.empty:
                hist_years = sorted(district_history.index.tolist())
                fig_district.add_trace(
                    go.Scatter(
                        x=hist_years,
                        y=district_history.reindex(hist_years).values,
                        mode='lines+markers',
                        name='Historical',
                        line=dict(color='#06A77D', width=3)
                    )
                )
            if not district_forecast.empty:
                forecast_years_district = sorted(district_forecast.index.tolist())
                fig_district.add_trace(
                    go.Scatter(
                        x=forecast_years_district,
                        y=district_forecast.reindex(forecast_years_district).values,
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='#F18F01', width=3, dash='dash')
                    )
                )

            if fig_district.data:
                fig_district.update_layout(
                    title=f"District {selected_district} Installed Panel Area", 
                    xaxis_title="Year",
                    yaxis_title="Panel Area (m²)",
                    hovermode='x unified',
                    height=400
                )
                st.plotly_chart(fig_district, width='stretch')
            else:
                st.info("No historical or forecast data available for this district.")
            
            # District details table
            st.subheader(f"📊 All {len(district_stats)} Districts - Detailed View")
            
            district_display = district_stats.copy()
            district_display.columns = ['District', 'Panel Area (m²)',
                                         'Population', 'Avg Age', 'Unemployment %', 'EV Points']
            
            # Show summary stats
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Districts", len(district_stats))
            with col2:
                st.metric("Total Predicted Area", f"{district_stats['predicted_panel_area_m2'].sum():,.0f} m²")
            
                st.dataframe(
                    district_display.style.format({
                        'Panel Area (m²)': '{:,.0f}',
                        'Population': '{:,.0f}',
                        'Avg Age': '{:.1f}',
                        'Unemployment %': '{:.2f}',
                        'EV Points': '{:,.1f}'
                    }).background_gradient(subset=['Panel Area (m²)'], cmap='YlGn'),
                    width='stretch',
                    height=600
                )
        else:
            latest_year = int(df['year'].max())
            st.info(f"Showing actual observations for {latest_year}. Adjust the year above and run the forecast when you're ready.")

            latest_actual = df[df['year'] == latest_year]
            actual_stats = latest_actual.groupby('district_number').agg({
                'panel_area_m2': 'sum',
                'Total_Population': 'mean',
                'Average_Age': 'mean',
                'Unemployment_Rate': 'mean',
                'ev_points_164m': 'mean'
            }).reset_index()
            actual_stats = actual_stats.sort_values('panel_area_m2', ascending=False)

            col_chart, col_toggle = st.columns([3, 1])
            with col_chart:
                st.subheader(f"🏛️ Actual District Performance ({latest_year})")
            with col_toggle:
                show_all_actual = st.checkbox("Show All Districts", value=False, key='show_all_actual')

            if show_all_actual:
                top_actual = actual_stats
                chart_title_actual = f"All {len(actual_stats)} Districts by Installed Panel Area ({latest_year})"
                chart_height_actual = max(600, len(actual_stats) * 25)
            else:
                top_actual = actual_stats.head(10)
                chart_title_actual = f"Top 10 Districts by Installed Panel Area ({latest_year})"
                chart_height_actual = 500

            fig_actual = go.Figure(go.Bar(
                y=top_actual['district_number'].astype(str),
                x=top_actual['panel_area_m2'],
                orientation='h',
                marker=dict(color='#667eea'),
                text=top_actual['panel_area_m2'].apply(lambda x: f'{x:,.0f} m²'),
                textposition='outside'
            ))

            fig_actual.update_layout(
                title=chart_title_actual,
                xaxis_title="Actual Panel Area (m²)",
                yaxis_title="District Number",
                height=chart_height_actual
            )
            fig_actual.update_yaxes(autorange="reversed")

            st.plotly_chart(fig_actual, width='stretch')

            st.subheader("📊 Detailed Actuals")
            actual_display = actual_stats.copy()
            actual_display.columns = ['District', 'Panel Area (m²)',
                                      'Population', 'Avg Age', 'Unemployment %', 'EV Points']

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Districts", len(actual_stats))
            with col2:
                st.metric("Total Installed Area", f"{actual_stats['panel_area_m2'].sum():,.0f} m²")

            st.dataframe(
                actual_display.style.format({
                    'Panel Area (m²)': '{:,.0f}',
                    'Population': '{:,.0f}',
                    'Avg Age': '{:.1f}',
                    'Unemployment %': '{:.2f}',
                    'EV Points': '{:,.1f}'
                }).background_gradient(subset=['Panel Area (m²)'], cmap='Blues'),
                width='stretch',
                height=600
            )

if __name__ == "__main__":
    main()
