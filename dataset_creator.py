import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.ticker as mtick
from sklearn.cluster import KMeans
from statsmodels.tsa.stattools import grangercausalitytests
import warnings

class DatasetCreator:
    def __init__(self, filename):
        self.filename = filename
        self.df = pd.read_excel(self.filename)
        self.df['monthyear'] = pd.to_datetime(self.df['monthyear'])
        self.df_non_dup = self.df.drop_duplicates()
        self.df_non_dup['Quarter'] = self.df_non_dup['monthyear'].dt.to_period('Q').astype(str)
        
        self.spend_df = self.df_non_dup[self.df_non_dup['variable'].str.contains('Spend')]
        self.spend_df['channel'] = self.spend_df['variable'].str.replace('Spend on ', '')
        
        self.df_model = self.df_non_dup.pivot_table(
            index=['business_level',  'category', 'brand', 'monthyear'],
            columns='variable',
            values='Amount'
        ).reset_index()
        
        self.df_wide = self.df_non_dup.pivot_table(
            index=['monthyear', 'brand'], 
            columns='variable', 
            values='Amount'
        ).reset_index()
        
        self.df_wide = self.df_wide.fillna(0)
        self.df_wide.columns = [c.replace('Spend on ', '').title() if 'Spend' in c else c for c in self.df_wide.columns]
        
        # Calculate total sales per brand for clustering
        self.brand_sales = self.df_wide.groupby('brand')['Sales'].sum().sort_values(ascending=False)
        
        # Initialize model attributes to None
        self.kmeans_model = None
        self.tier_map = None

    def get_duplicates(self):
        return self.df[self.df.duplicated(subset=['brand', 'monthyear', 'variable'], keep=False)]
    
    def get_non_duplicates(self):
        return self.df_non_dup
    
    def get_brands_with_sales(self):
        sales_df = self.df_non_dup[self.df_non_dup['variable'] == 'Sales']
        return sales_df['brand'].unique().tolist()
    
    def _apply_filters(self, df, brand_name=None, category_name=None, business_level=None):
        """Internal helper to apply filters to a dataframe."""
        if brand_name:
            df = df[df['brand'] == brand_name]
        if category_name:
            df = df[df['category'] == category_name]
        if business_level:
            df = df[df['business_level'] == business_level]
        return df

   

    # ==========================================
    # 2. CLUSTERING LOGIC (Returns DF, Figure)
    # ==========================================
    def _train_cluster_model(self):
        """Internal method: Trains the KMeans model once and saves it to self."""
        active_brands = self.brand_sales[self.brand_sales > 0]
        if active_brands.empty:
            return

        X = active_brands.values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X)
        self.kmeans_model = kmeans
        
        centers = kmeans.cluster_centers_.flatten()
        sorted_indices = np.argsort(centers)
        self.tier_map = {
            sorted_indices[0]: 'Low Level',
            sorted_indices[1]: 'Mid Level',
            sorted_indices[2]: 'Top Whale'
        }

    def get_cluster_of_brand(self, brand_name):
        if self.kmeans_model is None:
            self._train_cluster_model()
        
        brand_sales_value = self.brand_sales.get(brand_name, 0)
        if brand_sales_value <= 0:
            return "No Sales Data"
        
        X = np.array([[brand_sales_value]])
        label = self.kmeans_model.predict(X)[0]
        return self.tier_map[label]

    def get_spending_chart_by_brand(self, brand_name=None, category_name=None, business_level=None):
        """
        Generates a 3-part dashboard (Stack Mix, Volume Area, Heatmap).
        Now supports filtering by Brand, Category, or Business Level.
        """
        # 1. Apply Filters (Standardized)
        df = self.spend_df.copy()
        df = self._apply_filters(df, brand_name, category_name, business_level)
        
        if df.empty:
            return None

        # 2. Prepare Data for Stacked Bar (Strategy Mix)
        # We pivot by Quarter and Channel. 
        # If multiple brands are present (e.g. Category view), this sums them up.
        chart_data_billions = df.pivot_table(
            index='Quarter', 
            columns='channel', 
            values='Amount', 
            aggfunc='sum'
        ).fillna(0) / 1_000_000_000
        
        row_sums = chart_data_billions.sum(axis=1)
        chart_data_pct = chart_data_billions.div(row_sums.replace(0, np.nan), axis=0) * 100
        
        # 3. Prepare Data for Heatmap
        heatmap_data_billions = df.pivot_table(
            index='channel', 
            columns='Quarter', 
            values='Amount', 
            aggfunc='sum'
        ).fillna(0) / 1_000_000_000
        
        # 4. Plotting
        fig = plt.figure(figsize=(18, 12))
        
        # Dynamic Title Logic
        if brand_name:
            title_text = f'Media Spend Dashboard: {brand_name}'
        elif category_name:
            title_text = f'Media Spend Dashboard: {category_name} Category'
        elif business_level:
            title_text = f'Media Spend Dashboard: {business_level} Level'
        else:
            title_text = 'Media Spend Dashboard: All Data'
            
        fig.suptitle(title_text, fontsize=20, fontweight='bold', y=1.02)
        
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

        # Subplot 1: Share %
        ax1 = plt.subplot(gs[0, 0])
        chart_data_pct.plot(kind='bar', stacked=True, ax=ax1, colormap='viridis', width=0.8)
        ax1.set_title('Channel Share % (Strategy Mix)', fontsize=14, fontweight='bold')
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax1.set_xlabel('')
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
        ax1.grid(axis='y', linestyle='--', alpha=0.3)

        # Subplot 2: Total Volume
        ax2 = plt.subplot(gs[0, 1])
        chart_data_billions.plot(kind='area', stacked=True, ax=ax2, colormap='viridis', alpha=0.8)
        ax2.set_title('Total Spend Evolution (Volume)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('')
        ax2.set_ylabel('Amount (B IDR)')
        ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f'{x:,.0f}'))
        ax2.legend().set_visible(False)
        ax2.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Subplot 3: Intensity Heatmap
        ax3 = plt.subplot(gs[1, :])
        sns.heatmap(heatmap_data_billions, cmap='YlGnBu', ax=ax3, linewidths=.5, annot=True, fmt=',.0f')
            
        ax3.set_title('Spend Intensity Heatmap', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Quarter')
        ax3.set_ylabel('Media Channel')
        
        plt.tight_layout()
        return fig # <-- Returns Tuple

    # ==========================================
    # 3. SINGLE CHANNEL RELIANCE (Returns DF)
    # ==========================================
    def get_is_brand_rely_on_single_channel(self, threshold=50, brand_name=None, category_name=None, business_level=None):
        """
        Identifies brands that spend > threshold% of their budget on a single channel.
        Returns: (DataFrame, Figure) tuple.
        """
        # 1. Prepare Data & Apply Filters
        df = self.spend_df.copy()
        df = self._apply_filters(df, brand_name, category_name, business_level)
        
        if df.empty:
            return None, None
        
        # 2. Calculate Shares
        df['Amount_Billions'] = df['Amount'] / 1_000_000_000
        brand_channel_spend = df.groupby(['brand', 'channel'])['Amount_Billions'].sum().reset_index()
        brand_total_spend = df.groupby('brand')['Amount_Billions'].sum().reset_index().rename(columns={'Amount_Billions': 'Total'})
        
        merged = brand_channel_spend.merge(brand_total_spend, on='brand')
        merged['Share_Pct'] = (merged['Amount_Billions'] / merged['Total']) * 100

        # 3. Filter for Dominance
        # Find the max share channel for every brand
        dominant_channels = merged.loc[merged.groupby('brand')['Share_Pct'].idxmax()]
        
        # Filter where that max share exceeds the threshold
        heavy_favorites = dominant_channels[dominant_channels['Share_Pct'] > threshold].sort_values('Share_Pct', ascending=False)
        
        if heavy_favorites.empty:
            print(f"No brands found with >{threshold}% reliance on a single channel.")
            return heavy_favorites, None

        print(f"\nBrands heavily relying on single channel (>{threshold}%)")
        display_cols = ['brand', 'channel', 'Share_Pct', 'Total']
        print(heavy_favorites[display_cols].round(1))

        # 4. Generate Visualization (Bar Chart)
        # We plot Brand vs Reliance %, colored by Channel to see the mix
        fig = plt.figure(figsize=(10, max(6, len(heavy_favorites) * 0.4))) # Dynamic height based on row count
        
        sns.barplot(
            data=heavy_favorites,
            y='brand',
            x='Share_Pct',
            hue='channel',
            dodge=False, # Stacked look since there's only 1 bar per brand
            palette='viridis'
        )
        
        plt.title(f'Brands with >{threshold}% Reliance on a Single Channel', fontsize=14, fontweight='bold')
        plt.xlabel('Share of Wallet (%)')
        plt.ylabel('Brand')
        plt.xlim(0, 100)
        plt.axvline(threshold, color='red', linestyle='--', alpha=0.5, label=f'Threshold ({threshold}%)')
        plt.legend(title='Dominant Channel', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        # 5. Return Tuple
        return heavy_favorites, fig

    # ==========================================
    # 4. SEASONALITY (Returns Figure)
    # ==========================================
    def print_heatmap_time_of_spending_brand(self, brand_name=None, category_name=None, business_level=None):
        df = self.spend_df.copy()
        df = self._apply_filters(df, brand_name, category_name, business_level)

        if df.empty:
            return None

        df['Amount_Billions'] = df['Amount'] / 1_000_000_000
        
        seasonal_data = df.groupby([
            df['monthyear'].dt.year.rename('Year'), 
            df['monthyear'].dt.month.rename('Month')
        ])['Amount_Billions'].sum().reset_index()
        
        seasonal_data.columns = ['Year', 'Month', 'Amount']
        seasonality_pivot = seasonal_data.pivot(index='Month', columns='Year', values='Amount')

        fig = plt.figure(figsize=(10, 8)) # <-- Captured figure
        
        title_suffix = ""
        if brand_name: title_suffix += f" ({brand_name})"
        elif category_name: title_suffix += f" ({category_name})"
        
        sns.heatmap(seasonality_pivot, cmap='Oranges', annot=True, fmt='.1f', linewidths=.5)
        plt.title(f'Seasonality Heatmap: Spend (Billions){title_suffix}')
        plt.ylabel('Month (1=Jan, 12=Dec)')
        plt.tight_layout()
        # plt.show() <-- REMOVED

        return fig # <-- ADDED

    def plot_channel_seasonality_heatmap(self, brand_name=None, category_name=None, business_level=None):
        df = self.spend_df.copy()
        df = self._apply_filters(df, brand_name, category_name, business_level)

        if df.empty:
            return None

        df['Amount_Billions'] = df['Amount'] / 1_000_000_000
        unique_channels = df['channel'].unique()
        n_channels = len(unique_channels)
        
        if n_channels == 0:
            return None

        fig, axes = plt.subplots(1, n_channels, figsize=(5 * n_channels, 6), sharey=True) # <-- Captured Figure
        if n_channels == 1: axes = [axes] 

        for i, channel in enumerate(unique_channels):
            channel_data = df[df['channel'] == channel]
            seasonal_data = channel_data.groupby([
                channel_data['monthyear'].dt.year.rename('Year'), 
                channel_data['monthyear'].dt.month.rename('Month')
            ])['Amount_Billions'].sum().reset_index()
            
            heatmap_matrix = seasonal_data.pivot(index='Month', columns='Year', values='Amount_Billions')
            
            sns.heatmap(heatmap_matrix, ax=axes[i], cmap='Oranges', cbar=False, annot=True, fmt='.1f', linewidths=0.5)
            
            axes[i].set_title(f'{channel}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Year')
            if i == 0: axes[i].set_ylabel('Month (1=Jan, 12=Dec)')
            else: axes[i].set_ylabel('')

        plt.suptitle(f'Seasonality by Channel (Billions) | Filter: {brand_name or "All"}', fontsize=16, y=1.05)
        plt.tight_layout()
        # plt.show() <-- REMOVED
        return fig # <-- ADDED

    # ==========================================
    # 5. CORRELATION (Returns DF, Figure)
    # ==========================================
    def plot_sales_correlation_heatmap(self, brand_name=None, category_name=None, business_level=None):
        df = self.df_non_dup.copy()
        df = self._apply_filters(df, brand_name, category_name, business_level)
        
        if df.empty: return None, None

        df_pivot = df.pivot_table(index=['monthyear', 'brand', 'business_level', 'category'], 
                                  columns='variable', values='Amount', aggfunc='sum').reset_index()
        df_pivot = df_pivot.fillna(0)
        sales_col = 'Sales'
        
        if sales_col not in df_pivot.columns: return None, None
        channel_cols = [c for c in df_pivot.columns if 'Spend on' in c]
        if not channel_cols: return None, None

        rename_map = {c: c.replace('Spend on ', '').capitalize() for c in channel_cols}
        df_pivot.rename(columns=rename_map, inplace=True)
        clean_channel_cols = list(rename_map.values())

        correlation_matrix = df_pivot[clean_channel_cols + [sales_col]].corr()
        correlation_series = correlation_matrix[sales_col].drop(sales_col)
        correlation_df = correlation_series.to_frame(name='Correlation with Sales')
        correlation_df = correlation_df.sort_values('Correlation with Sales', ascending=False)
        
        if correlation_df.isnull().all().all(): return None, None

        fig = plt.figure(figsize=(6, 8)) # <-- Captured
        sns.heatmap(correlation_df, annot=True, fmt='.2f', cmap='RdBu_r', vmin=-1, vmax=1, 
                    cbar_kws={'label': 'Pearson Correlation Coefficient'})

        title_suffix = ""
        if brand_name: title_suffix = f"\n({brand_name})"
        
        plt.title(f'Impact of Channel Spend on Sales{title_suffix}', fontsize=12, fontweight='bold')
        plt.ylabel('Marketing Channel')
        plt.xlabel('Target (Sales)')
        plt.tight_layout()
        # plt.show() <-- REMOVED
        return correlation_df, fig # <-- Returns Tuple

    # ==========================================
    # 6. SPEND VS SALES (Returns Figure)
    # ==========================================
    def plot_spend_vs_sales(self, brand_name=None, category_name=None, business_level=None):
        df = self.df_non_dup.copy()
        df = self._apply_filters(df, brand_name, category_name, business_level)
        
        if df.empty: return None

        sales_df = df[df['variable'] == 'Sales'].groupby('monthyear')['Amount'].sum() / 1_000_000_000
        spend_df = df[df['variable'].str.contains('Spend')].copy()
        spend_df['channel'] = spend_df['variable'].str.replace('Spend on ', '')
        spend_df['Amount_Billions'] = spend_df['Amount'] / 1_000_000_000
        
        spend_pivot = spend_df.pivot_table(index='monthyear', columns='channel', values='Amount_Billions', aggfunc='sum').fillna(0)

        common_index = spend_pivot.index.union(sales_df.index).sort_values()
        spend_pivot = spend_pivot.reindex(common_index).fillna(0)
        sales_df = sales_df.reindex(common_index).fillna(0)

        fig, ax1 = plt.subplots(figsize=(14, 7)) # <-- Captured

        spend_pivot.plot(kind='bar', stacked=True, ax=ax1, colormap='viridis', width=0.8, alpha=1)
        ax1.set_ylabel('Media Spend (Billions)', fontsize=12, fontweight='bold', color='gray')
        ax1.set_xlabel('Month-Year', fontsize=12)
        
        tick_labels = [item.strftime('%Y-%m') for item in common_index]
        ax1.set_xticklabels(tick_labels, rotation=45, ha='right')

        ax2 = ax1.twinx()
        ax2.plot(range(len(common_index)), sales_df.values, color='crimson', linewidth=3, marker='o', label='Total Sales')
        ax2.set_ylabel('Total Sales (Billions)', fontsize=12, fontweight='bold', color='crimson')
        ax2.grid(False)

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', bbox_to_anchor=(0, 1.15), ncol=4)

        title_suffix = ""
        if brand_name: title_suffix = f": {brand_name}"
        plt.title(f'Marketing Spend Mix vs. Sales Performance{title_suffix}', fontsize=16, y=1.2)
        plt.tight_layout()
        # plt.show() <-- REMOVED
        return fig # <-- ADDED

    # ==========================================
    # 7. LAG ANALYSIS (Returns DF, Figure)
    # ==========================================
    def analyze_spending_lag_impact(self, brand_name=None, category_name=None, business_level=None, max_lag=6):
        df = self.df_non_dup.copy()
        df = self._apply_filters(df, brand_name, category_name, business_level)
        if df.empty: return None, None

        df_pivot = df.pivot_table(index='monthyear', columns='variable', values='Amount', aggfunc='sum').fillna(0)
        rename_map = {c: c.replace('Spend on ', '') for c in df_pivot.columns}
        df_pivot.rename(columns=rename_map, inplace=True)

        if 'Sales' not in df_pivot.columns: return None, None

        df_log = np.log1p(df_pivot)
        channel_cols = [c for c in df_log.columns if c != 'Sales']
        lag_results = []

        for lag in range(0, max_lag + 1):
            for channel in channel_cols:
                corr = df_log['Sales'].corr(df_log[channel].shift(lag))
                lag_results.append({'Channel': channel.capitalize(), 'Lag_Months': lag, 'Correlation': corr})

        df_lags = pd.DataFrame(lag_results)

        fig = plt.figure(figsize=(12, 6)) # <-- Captured
        sns.lineplot(data=df_lags, x='Lag_Months', y='Correlation', hue='Channel', 
                     style='Channel', markers=True, dashes=False, linewidth=2, palette='viridis')

        title_suffix = ""
        if brand_name: title_suffix = f" ({brand_name})"
        plt.title(f'Time-Lag Analysis: How Long for Spend to Convert?{title_suffix}', fontsize=14, fontweight='bold')
        plt.xlabel('Lag (Months after Spend)', fontsize=12)
        plt.ylabel('Correlation with Sales (Log-Log)', fontsize=12)
        plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, max_lag + 1))
        plt.tight_layout()
        # plt.show() <-- REMOVED
        return df_lags, fig # <-- Returns Tuple

    # ==========================================
    # 8. GRANGER CAUSALITY (Returns Figure)
    # ==========================================
    def test_granger_causality(self, brand_name=None, category_name=None, business_level=None, max_lag=4):
        warnings.filterwarnings("ignore")
        df = self.df_non_dup.copy()
        df = self._apply_filters(df, brand_name, category_name, business_level)
        
        if df.empty: return None

        df_pivot = df.pivot_table(index='monthyear', columns='variable', values='Amount', aggfunc='sum').fillna(0)
        rename_map = {c: c.replace('Spend on ', '') for c in df_pivot.columns}
        df_pivot.rename(columns=rename_map, inplace=True)
        
        if 'Sales' not in df_pivot.columns: return None

        df_diff = df_pivot.diff().dropna()
        channel_cols = [c for c in df_diff.columns if c != 'Sales']
        results_list = []

        for channel in channel_cols:
            if df_diff[channel].std() == 0 or df_diff['Sales'].std() == 0: continue
            data_for_test = df_diff[['Sales', channel]]
            try:
                test_result = grangercausalitytests(data_for_test, maxlag=max_lag, verbose=False)
                for lag in range(1, max_lag + 1):
                    p_val = test_result[lag][0]['ssr_chi2test'][1]
                    results_list.append({'Channel': channel, 'Lag (Months)': lag, 'P_Value': p_val})
            except Exception: pass

        if not results_list: return None

        df_results = pd.DataFrame(results_list)
        heatmap_data = df_results.pivot(index='Channel', columns='Lag (Months)', values='P_Value')

        fig = plt.figure(figsize=(10, 6)) # <-- Captured
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn_r', vmin=0, vmax=0.1, 
                    cbar_kws={'label': 'P-Value (Lower is Better)'})
        
        plt.title(f'Granger Causality Heatmap: Do Channels Predict Sales?\n(P-Value < 0.05 implies Causality)', fontsize=14)
        plt.tight_layout()
        # plt.show() <-- REMOVED
        return fig # <-- ADDED

    # ==========================================
    # 9. MULTI-CHANNEL TRENDS (Returns Figure)
    # ==========================================
    def plot_multi_channel_trends(self, brand_name=None, category_name=None, business_level=None):
        df = self.df_non_dup.copy()
        df = self._apply_filters(df, brand_name, category_name, business_level)
        
        if df.empty: return None

        df_pivot = df.pivot_table(index='monthyear', columns='variable', values='Amount', aggfunc='sum').fillna(0)
        
        requested_metrics = [
            ('Sales', 'Sales'),
            ('TV Spend', 'Spend on tv'),
            ('Meta Spend', 'Spend on meta'),
            ('YouTube Spend', 'Spend on youtube'),
            ('UContent Spend', 'Spend on ucontent'),
            ('TikTok Spend', 'Spend on tiktok')
        ]
        
        available_metrics = []
        for label, col in requested_metrics:
            if col in df_pivot.columns: available_metrics.append((label, col))
        
        if not available_metrics: return None

        n_rows = len(available_metrics)
        fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2 * n_rows), sharex=True) # <-- Captured
        if n_rows == 1: axes = [axes]

        colors = plt.cm.tab10(np.linspace(0, 1, n_rows))

        for i, (label, col) in enumerate(available_metrics):
            ax = axes[i]
            data = df_pivot[col] / 1_000_000_000 # Convert to Billions
            ylabel = f"{label}\n(Bn)"

            ax.plot(df_pivot.index, data, color=colors[i], linewidth=1.5)
            ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
            ax.grid(True, linestyle=':', alpha=0.5)
            if i < n_rows - 1: plt.setp(ax.get_xticklabels(), visible=False)
            else: ax.set_xlabel('Timeline', fontsize=12)

        plt.subplots_adjust(hspace=0.05)
        title_suffix = f": {brand_name}" if brand_name else ""
        fig.suptitle(f'Sales vs Media Spend Trends{title_suffix}', y=0.92, fontsize=16)
        # plt.show() <-- REMOVED
        return fig # <-- ADDED

    # ==========================================
    # NEW: SALES SEASONALITY HEATMAP
    # ==========================================
    def print_heatmap_sales_seasonality(self, brand_name=None, category_name=None, business_level=None):
        # 1. Use the main dataframe (non-duplicates)
        df = self.df_non_dup.copy()
        
        # 2. Apply Filters (Brand/Category/Level)
        df = self._apply_filters(df, brand_name, category_name, business_level)
        
        # 3. Filter specifically for 'Sales' variable
        df = df[df['variable'] == 'Sales']

        if df.empty:
            return None

        # 4. Convert to Billions
        df['Amount_Billions'] = df['Amount'] / 1_000_000_000
        
        # 5. Group by Year-Month
        seasonal_data = df.groupby([
            df['monthyear'].dt.year.rename('Year'), 
            df['monthyear'].dt.month.rename('Month')
        ])['Amount_Billions'].sum().reset_index()
        
        seasonal_data.columns = ['Year', 'Month', 'Amount']
        
        # 6. Pivot for Heatmap (Rows=Month, Cols=Year)
        seasonality_pivot = seasonal_data.pivot(index='Month', columns='Year', values='Amount')

        # 7. Plot
        fig = plt.figure(figsize=(10, 8)) 
        
        title_suffix = ""
        if brand_name: title_suffix += f" ({brand_name})"
        elif category_name: title_suffix += f" ({category_name})"
        
        # Using 'Greens' cmap for Revenue/Sales
        sns.heatmap(seasonality_pivot, cmap='Greens', annot=True, fmt='.1f', linewidths=.5)
        plt.title(f'Seasonality Heatmap: Sales Revenue (Billions){title_suffix}')
        plt.ylabel('Month (1=Jan, 12=Dec)')
        plt.tight_layout()

        return fig