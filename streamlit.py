import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64

import numpy as np
from io import BytesIO
from openai import OpenAI
from dataset_creator import DatasetCreator 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from dotenv import load_dotenv
import os
load_dotenv()

def calculate_smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def prepare_ml_data(data_instance, brand_filter=None, category_filter=None, business_level_filter=None):
    """
    1. Analyzes optimal lag for each channel (highest correlation).
    2. Pivots data to wide format.
    3. Creates dynamic Lag features based on step 1.
    4. Handles NaNs.
    """
    # --- STEP 1: Determine Optimal Lags ---
    lag_df, _ = data_instance.analyze_spending_lag_impact(
        brand_name=brand_filter, 
        category_name=category_filter, 
        business_level=business_level_filter,
        max_lag=4 
    )
    
    best_lags = {}
    
    if lag_df is not None and not lag_df.empty:
        for channel in lag_df['Channel'].unique():
            chan_data = lag_df[lag_df['Channel'] == channel]
            
            # --- FIX: Check for Valid Correlations ---
            # Drop NaNs before finding max. If all are NaN, valid_corr will be empty.
            valid_corr = chan_data['Correlation'].dropna()
            
            if not valid_corr.empty:
                # Get the index of the max valid correlation
                best_idx = valid_corr.idxmax()
                best_row = chan_data.loc[best_idx]
                best_lags[channel.lower()] = int(best_row['Lag_Months'])
            else:
                # If correlation is NaN (e.g., constant sales/spend), default to Lag 1
                best_lags[channel.lower()] = 1
    
    # --- STEP 2: Filter & Pivot Data ---
    df_raw = data_instance.df_non_dup.copy()
    
    if business_level_filter: df_raw = df_raw[df_raw['business_level'] == business_level_filter]
    if category_filter: df_raw = df_raw[df_raw['category'] == category_filter]
    if brand_filter: df_raw = df_raw[df_raw['brand'] == brand_filter]

    # Pivot to Wide Format
    df_pivot = df_raw.pivot_table(index=['monthyear', 'brand'], columns='variable', values='Amount', aggfunc='sum').reset_index()
    
    # Clean Column Names
    rename_map = {c: c.replace('Spend on ', '').lower() for c in df_pivot.columns if 'Spend' in c}
    df_pivot.rename(columns=rename_map, inplace=True)
    df_pivot = df_pivot.fillna(0)
    
    if 'Sales' not in df_pivot.columns:
        return None, None

    # --- STEP 3: Feature Engineering ---
    df_pivot = df_pivot.sort_values(['brand', 'monthyear'])
    
    # Always add Sales Lag 1
    df_pivot['Sales_Lag1'] = df_pivot.groupby('brand')['Sales'].shift(1)
    
    # Add Dynamic Spend Lags
    dataset_channels = [c for c in df_pivot.columns if c in best_lags.keys() or c in [k.replace('spend on ', '') for k in rename_map.values()]]
    
    used_lags_info = {} 
    
    for col in dataset_channels:
        if col == 'Sales': continue
        
        # Default to 1 if analysis didn't find this channel
        optimal_lag = best_lags.get(col, 1)
        used_lags_info[col] = optimal_lag
        
        feature_name = f'{col}_Lag{optimal_lag}'
        
        if optimal_lag == 0:
            df_pivot[feature_name] = df_pivot[col]
        else:
            df_pivot[feature_name] = df_pivot.groupby('brand')[col].shift(optimal_lag)

    # --- STEP 4: Final Cleanup ---
    df_model = df_pivot.dropna()
    
    return df_model, used_lags_info

def run_ml_experiment(df_model):
    """
    Runs LR, XGB, LGBM on the prepared data.
    Returns: Results DF, Predictions DF, Feature Importance, Test Dates
    """
    # 1. Separate Features, Target, and Dates
    drop_cols = ['monthyear', 'brand', 'Sales']
    
    # Ensure monthyear is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_model['monthyear']):
        df_model['monthyear'] = pd.to_datetime(df_model['monthyear'])

    X = df_model[[c for c in df_model.columns if c not in drop_cols]]
    y = df_model['Sales']
    dates = df_model['monthyear'] # Capture dates
    
    # 2. Time-based Split
    split_idx = int(len(X) * 0.8)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_test = dates.iloc[split_idx:] # Split dates exactly the same way
    
    results = []
    models = {
        "Linear Regression": LinearRegression(),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "LightGBM": LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1)
    }
    
    pred_dict = {'Actual': y_test.values}
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            preds = np.maximum(preds, 0) 
            pred_dict[name] = preds
            
            r2 = r2_score(y_test, preds)
            smape = calculate_smape(y_test, preds)
            
            results.append({
                "Model": name,
                "R2 Score": r2,
                "SMAPE (%)": smape
            })
        except Exception as e:
            st.error(f"Error training {name}: {e}")
        
    results_df = pd.DataFrame(results).set_index("Model")
    pred_df = pd.DataFrame(pred_dict) # Don't set index yet, we return dates separately
    
    # Reliability Check
    best_model = results_df.sort_values('R2 Score', ascending=False).iloc[0] if not results_df.empty else None
    is_reliable = False
    if best_model is not None:
        is_reliable = (best_model['R2 Score'] > 0.6) and (best_model['SMAPE (%)'] < 25)
    
    return results_df, pred_df, is_reliable, dates_test

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def get_ai_analysis(api_key, context_text, analysis_type, figure=None):
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API Key in the sidebar to use AI features.")
        return None
    
    client = OpenAI(api_key=api_key)
    
    system_instruction = (
        "You are a Senior Marketing Data Analyst. "
        "Analyze the provided chart image(s) and data context. "
        "Provide a concise, professional strategic assessment focusing on "
        "ROI, growth opportunities, and spending efficiency."
    )

    user_content = [
        {
            "type": "text", 
            "text": f"Analyze this {analysis_type} visualization.\n\nContext: {context_text}\n\n"
                    "Please provide:\n"
                    "1. **Key Observation**: What is the most important pattern here?\n"
                    "2. **Strategic Recommendation**: What specific action should the marketing director take?"
        }
    ]

    if figure:
        # Handle both single figure and list of figures
        figures_list = figure if isinstance(figure, list) else [figure]
        
        for fig in figures_list:
            image_base64 = fig_to_base64(fig)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}"}
            })

    try:
        with st.spinner(f"🤖 AI is analyzing the {analysis_type} chart..."):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=350
            )
            return response.choices[0].message.content
    except Exception as e:
        st.error(f"AI Error: {e}")
        return None

# ==========================================
# 3. PAGE CONFIG & DATA LOADING
# ==========================================
st.set_page_config(layout="wide", page_title="Marketing Analytics Dashboard")

@st.cache_resource
def load_data():
    # Update path if necessary
    data = DatasetCreator("interviewexercisebriefrisqi/modelling_exercise_raw_data.xlsx") 
    return data

data_instance = load_data()

# ==========================================
# 4. SIDEBAR SETTINGS & FILTERS
# ==========================================
st.sidebar.header("⚙️ AI Settings")
api_key = os.getenv("api_key", "")

st.sidebar.header("🔍 Filter Data")

# Business Level
all_levels = ['All'] + list(data_instance.df_non_dup['business_level'].unique())
selected_level = st.sidebar.selectbox("Business Level", all_levels)
level_filter = None if selected_level == 'All' else selected_level

# Category
if level_filter:
    cat_options = data_instance.df_non_dup[data_instance.df_non_dup['business_level'] == level_filter]['category'].unique()
else:
    cat_options = data_instance.df_non_dup['category'].unique()

all_cats = ['All'] + list(cat_options)
selected_cat = st.sidebar.selectbox("Category", all_cats)
cat_filter = None if selected_cat == 'All' else selected_cat

# Brand
if cat_filter:
    brand_options = data_instance.df_non_dup[data_instance.df_non_dup['category'] == cat_filter]['brand'].unique()
else:
    brand_options = data_instance.df_non_dup['brand'].unique()

all_brands = ['All'] + list(brand_options)
selected_brand = st.sidebar.selectbox("Brand", all_brands)
brand_filter = None if selected_brand == 'All' else selected_brand

# ==========================================
# 5. MAIN LAYOUT
# ==========================================
st.title("📊 Marketing Mix & Causality Dashboard")
st.markdown(f"**Current View:** Level: `{selected_level}` | Category: `{selected_cat}` | Brand: `{selected_brand}`")

# TAB STRUCTURE
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Performance & Portfolio",   
    "Temporal Dynamics",        
    "Causality",
    "Brand Deep Dive & Seasonality",
    "ML Forecasting Experiment"
])

# -------------------------------------------------------------------
# TAB 1: PERFORMANCE & PORTFOLIO
# -------------------------------------------------------------------
with tab1:
    st.header("🏆 Performance Extremes & Portfolio Mix")

    # --- LOGIC START: Aggregate Data for Leaders & Counters ---
    # 1. Filter raw data based on selections
    df_filtered = data_instance.df_non_dup.copy()
    if level_filter: df_filtered = df_filtered[df_filtered['business_level'] == level_filter]
    if cat_filter: df_filtered = df_filtered[df_filtered['category'] == cat_filter]
    
    # 2. Separate Sales vs Spend
    sales_df = df_filtered[df_filtered['variable'] == 'Sales']
    spend_df = df_filtered[df_filtered['variable'].str.contains('Spend', case=False, na=False)]

    # 3. Aggregate totals by Brand
    brand_sales_agg = sales_df.groupby('brand')['Amount'].sum().rename('total_sales')
    brand_spend_agg = spend_df.groupby('brand')['Amount'].sum().rename('total_spend')
    brand_perf = pd.concat([brand_sales_agg, brand_spend_agg], axis=1).fillna(0)

    # 4. Identify specific brand lists
    all_brands_in_view = brand_perf.index.tolist()
    no_revenue_brands = brand_perf[brand_perf['total_sales'] == 0].index.tolist()
    
    # --- SECTION A: Metrics Row ---
    # --- SECTION A: Metrics Display ---
    
    # 1. Calculate Extremes (safe check)
    if not brand_perf.empty:
        top_spender = brand_perf['total_spend'].idxmax()
        top_achiever = brand_perf['total_sales'].idxmax()
        ts_val = brand_perf.loc[top_spender, 'total_spend']
        ta_val = brand_perf.loc[top_achiever, 'total_sales']
    else:
        top_spender, top_achiever = "-", "-"
        ts_val, ta_val = 0, 0

    # ==========================================
    # ROW 1: Performance Extremes & Risk
    # ==========================================
    st.markdown("##### 1. Key Performance Indicators")
    r1_col1, r1_col2, r1_col3 = st.columns(3)

    r1_col1.metric("💸 Top Spender", top_spender, f"IDR {ts_val:,.0f}")
    r1_col2.metric("💰 Top Sales", top_achiever, f"IDR {ta_val:,.0f}")
    r1_col3.metric("⚠️ Zero Revenue Brands", f"{len(no_revenue_brands)}", help="Count of brands with spend activity but 0 recorded sales.")

    # ==========================================
    # ROW 2: Portfolio Composition (Tiers)
    # ==========================================
    st.markdown("##### 2. Brand Tier Composition")
    r2_col1, r2_col2, r2_col3 = st.columns(3)

    if brand_filter:
        # CASE 1: Single Brand Selected -> Show specific tier in first column
        cluster_tier = data_instance.get_cluster_of_brand(brand_filter)
        r2_col1.metric("🏷️ Current Brand Tier", cluster_tier, help="Calculated via K-Means Clustering on Total Sales.")
        # Leave col2 and col3 empty or use for other info if needed
    else:
        # CASE 2: Overview/Category View -> Show breakdown across 3 columns
        tier_counts = {"Top Whale": 0, "Mid Level": 0, "Low Level": 0}
        
        # Iterate and Count
        for b_name in all_brands_in_view:
            t = data_instance.get_cluster_of_brand(b_name)
            if t in tier_counts:
                tier_counts[t] += 1
            
        # Display Counts in separate columns
        r2_col1.metric("🐳 Whales (High Rev)", f"{tier_counts['Top Whale']}", help="Top performing brands by volume.")
        r2_col2.metric("🦈 Sharks (Mid Rev)", f"{tier_counts['Mid Level']}", help="Average performing brands.")
        r2_col3.metric("🐟 Fish (Low Rev)", f"{tier_counts['Low Level']}", help="Low volume or niche brands.")

    # --- EXPANDER FOR NO REVENUE BRANDS ---
    if len(no_revenue_brands) > 0:
        with st.expander(f"See list of {len(no_revenue_brands)} brands with Zero Revenue"):
            st.dataframe(pd.DataFrame(no_revenue_brands, columns=["Brand Name"]), hide_index=True)

    st.divider()

    # --- SECTION B: Spend vs Sales Mix ---
    st.subheader("2. Efficiency: Spend vs. Sales Mix")
    fig_mix = data_instance.plot_spend_vs_sales(
        brand_name=brand_filter, category_name=cat_filter, business_level=level_filter
    )
    if fig_mix:
        st.pyplot(fig_mix)
    
    st.divider()

    # --- SECTION C: Single Channel Reliance ---
    st.subheader("3. Risk: Single Channel Reliance")
    favorites_df, _ = data_instance.get_is_brand_rely_on_single_channel(
        threshold=50, brand_name=brand_filter, category_name=cat_filter, business_level=level_filter
    )
    
    col_risk_1, col_risk_2 = st.columns([2, 1])
    with col_risk_1:
        if favorites_df is not None and not favorites_df.empty:
            st.markdown("**Systemic Risk: Count of Brands Reliant on Specific Channels (>50%)**")
            risk_counts = favorites_df['channel'].value_counts().reset_index()
            risk_counts.columns = ['Channel', 'Brand Count']
            
            fig_risk, ax_risk = plt.subplots(figsize=(8, 4))
            sns.barplot(data=risk_counts, x='Channel', y='Brand Count', palette='Reds_r', ax=ax_risk)
            ax_risk.set_title("Number of Brands Heavily Reliant on Each Platform")
            st.pyplot(fig_risk)
        else:
            st.success("✅ No brands detected with dangerous single-channel reliance (>50%).")

    with col_risk_2:
        if favorites_df is not None and not favorites_df.empty:
            risky_brands_count = favorites_df['brand'].nunique()
            total_brands_count = len(all_brands_in_view)
            risk_pct = (risky_brands_count / total_brands_count) * 100 if total_brands_count > 0 else 0
            
            st.metric("Portfolio Vulnerability", f"{risk_pct:.1f}%")
            st.dataframe(favorites_df[['brand', 'channel', 'Share_Pct']].style.format({'Share_Pct': '{:.1f}%'}), height=200)

# -------------------------------------------------------------------
# TAB 2: TEMPORAL DYNAMICS
# -------------------------------------------------------------------
with tab2:
    st.header("📈 Temporal Dynamics & Correlations")

    st.subheader("1. Sales vs. Media Spend Trends")
    fig_trend = data_instance.plot_multi_channel_trends(
        brand_name=brand_filter, category_name=cat_filter, business_level=level_filter
    )
    if fig_trend:
        st.pyplot(fig_trend)
    else:
        st.warning("Not enough data to generate trend stack.")
    
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("2. Correlation Matrix")
        corr_df, corr_fig = data_instance.plot_sales_correlation_heatmap(
            brand_name=brand_filter, category_name=cat_filter, business_level=level_filter
        )
        if corr_fig: st.pyplot(corr_fig)

    with col2:
        st.subheader("3. Time-Lag Analysis")
        lag_df, lag_fig = data_instance.analyze_spending_lag_impact(
            brand_name=brand_filter, category_name=cat_filter, business_level=level_filter
        )
        if lag_fig: st.pyplot(lag_fig)
    
    if st.button("Analyze Trends AI", key="btn_trend_ai"):
        insight = get_ai_analysis(api_key, "Reviewing alignment between spend spikes and sales spikes. sales_correlation_heatmap {corr_df}", "Trend Chart", figure=[fig_trend, corr_fig, lag_fig])
        if insight: st.success(insight)

# -------------------------------------------------------------------
# TAB 3: CAUSALITY
# -------------------------------------------------------------------
with tab3:
    st.header("🔗 Granger Causality")
    st.markdown("Test if knowing past spend predicts future sales.")
    
    if st.button("Run Causality Test"):
        with st.spinner("Running statistical tests..."):
            fig_causality = data_instance.test_granger_causality(
                brand_name=brand_filter, category_name=cat_filter
            )
            if fig_causality:
                st.pyplot(fig_causality)
                insight = get_ai_analysis(api_key, "Granger Causality P-Values.", "Causality Matrix", figure=fig_causality)
                if insight: st.success(insight)
            else:
                st.warning("Insufficient data variation for causality test.")

# -------------------------------------------------------------------
# TAB 4: STRATEGY & SEASONALITY DEEP DIVE (Updated for All Filters)
# -------------------------------------------------------------------
with tab4:
    st.header("🔍 Strategy & Seasonality Deep Dive")
    
    # 1. Determine Display Name for Titles
    if brand_filter:
        view_name = f"Brand: {brand_filter}"
    elif cat_filter:
        view_name = f"Category: {cat_filter}"
    elif level_filter:
        view_name = f"Level: {level_filter}"
    else:
        view_name = "All Brands (Portfolio View)"

    # --- A. Media Strategy Mix ---
    st.subheader(f"A. Media Strategy Mix: {view_name}")
    
    # Now works for ALL filters (Brand, Category, or Global)
    fig_deep_dive = data_instance.get_spending_chart_by_brand(
        brand_name=brand_filter, category_name=cat_filter, business_level=level_filter
    )
    
    if fig_deep_dive:
        st.pyplot(fig_deep_dive)
        
        # Dynamic Button Text
        audit_btn_key = f"btn_audit_{brand_filter or 'all'}" 
        if st.button(f"Audit Strategy ({view_name})", key=audit_btn_key):
            insight = get_ai_analysis(
                api_key, 
                f"Deep dive analysis for {view_name}.", 
                "Strategy Dashboard", 
                figure=fig_deep_dive
            )
            if insight: st.success(insight)
    else:
        st.warning("No spending data available for this selection.")
    
    st.divider()

    # --- B. Spend Seasonality Patterns ---
    st.subheader(f"B. Spend Seasonality Patterns: {view_name}")
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.markdown("#### Total Spend Seasonality")
        fig_season_total = data_instance.print_heatmap_time_of_spending_brand(
            brand_name=brand_filter, category_name=cat_filter, business_level=level_filter
        )
        if fig_season_total: st.pyplot(fig_season_total)
    
    with col_s2:
        st.markdown("#### Sales Seasonality")
        fig_sales_season = data_instance.print_heatmap_sales_seasonality(
        brand_name=brand_filter, category_name=cat_filter, business_level=level_filter
        )
        if fig_sales_season: st.pyplot(fig_sales_season)

    st.divider()

    # --- C. Sales Seasonality ---
    st.subheader(f"C. Specific Channel Seasonality: {view_name}")
    fig_season_channel = data_instance.plot_channel_seasonality_heatmap(
            brand_name=brand_filter, category_name=cat_filter, business_level=level_filter
        )
    if fig_season_channel: st.pyplot(fig_season_channel)   
        
    season_btn_key = f"btn_sales_season_ai_{brand_filter or 'all'}"
    if st.button("Analyze Sales Seasonality (AI)", key=season_btn_key):
        summary_text = f"Analyzing monthly sales revenue heatmaps for {view_name} to find peak buying months."
        insight = get_ai_analysis(api_key, summary_text, "Sales Seasonality Heatmap", figure=fig_sales_season)
        if insight: st.success(insight)
    else:
        st.info("No Sales data available for seasonality analysis.")

with tab5:
    st.header("🧪 Predictive Model Experiment")
    st.markdown("""
    **Experiment Setup:**
    1. **Features:** Historical Lags ($Sales_{t-1}$, $Spend_{t-1}$).
    2. **Models:** Linear Regression, XGBoost, LightGBM.
    3. **Split:** Time-based (Training on past data, Testing on recent data).
    """)
    
    # 1. Prepare Data
    st.write("---")
    st.subheader("1. Data Preparation")
    
    # Determine scope: If Brand selected, run for Brand. Else run Global Model (All Brands).
    if brand_filter:
        scope_text = f"Running Experiment for Single Brand: **{brand_filter}**"
    else:
        scope_text = "Running Global Experiment (All Brands pooled together)"
        
    st.info(scope_text)
    
    if st.button("🚀 Run Experiment"):
        with st.spinner("Analyzing Optimal Lags & Training Models..."):
            
            df_ml, lag_info = prepare_ml_data(
                data_instance, 
                brand_filter=brand_filter, 
                category_filter=cat_filter, 
                business_level_filter=level_filter
            )
            
            if df_ml is not None and len(df_ml) > 10: # Lowered threshold slightly
                st.success("✅ Data Prepared with Dynamic Lags")
                with st.expander("View Selected Lag Features"):
                    st.json(lag_info)
                
                # Unpack 4 return values now (added dates_test)
                results_df, pred_df, is_reliable, dates_test = run_ml_experiment(df_ml)
                
                # --- A. Evaluation ---
                st.subheader("2. Model Evaluation")
                col_m1, col_m2 = st.columns([1, 2])
                
                with col_m1:
                    st.dataframe(results_df.style.highlight_max(axis=0, subset=['R2 Score'], color='#90EE90')
                                           .highlight_min(axis=0, subset=['SMAPE (%)'], color='#90EE90'))
                    
                    if is_reliable:
                        st.success("✅ **Reliable Model:** R2 > 0.6 & SMAPE < 25%.")
                    else:
                        st.error("❌ **Unreliable:** High error or low R2.")

                with col_m2:
                    fig_eval, ax_eval = plt.subplots(figsize=(6, 3))
                    sns.barplot(data=results_df.reset_index(), x='Model', y='R2 Score', palette='viridis', ax=ax_eval)
                    ax_eval.set_ylim(0, 1)
                    st.pyplot(fig_eval)

                # --- B. Prediction Visualization (FIXED X-AXIS) ---
                st.subheader("3. Actual vs Predicted (Test Set)")
                
                # Import date formatter
                import matplotlib.dates as mdates

                fig_pred, ax_pred = plt.subplots(figsize=(12, 5))
                
                # Plot using dates_test as x-axis
                # Ensure dates_test aligns with pred_df rows
                ax_pred.plot(dates_test, pred_df['Actual'], label='Actual Sales', color='black', linewidth=2, linestyle='--')
                
                if 'Linear Regression' in pred_df.columns:
                    ax_pred.plot(dates_test, pred_df['Linear Regression'], label='Linear Regression', alpha=0.7)
                if 'XGBoost' in pred_df.columns:
                    ax_pred.plot(dates_test, pred_df['XGBoost'], label='XGBoost', alpha=0.7)
                if 'LightGBM' in pred_df.columns:
                    ax_pred.plot(dates_test, pred_df['LightGBM'], label='LightGBM', alpha=0.7)
                
                ax_pred.set_title("Forecasting Performance on Unseen Data")
                ax_pred.set_ylabel("Sales Volume")
                ax_pred.set_xlabel("Date")
                
                # Format X-Axis Ticks to Month-Year
                ax_pred.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
                ax_pred.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) # Show tick every month (adjust interval if crowded)
                plt.setp(ax_pred.get_xticklabels(), rotation=45, ha="right") # Rotate labels
                
                ax_pred.legend()
                ax_pred.grid(True, alpha=0.3)
                st.pyplot(fig_pred)

                # AI Analysis
                if st.button("Analyze Model Reliability (AI)", key="btn_ml_ai"):
                    summary = f"R2 Scores: {results_df['R2 Score'].to_dict()}. SMAPE: {results_df['SMAPE (%)'].to_dict()}."
                    insight = get_ai_analysis(api_key, summary, "ML Experiment Results", figure=fig_pred)
                    if insight: st.success(insight)
                    
            else:
                st.warning("⚠️ Insufficient data points. Try selecting 'All' brands or check data quality.")