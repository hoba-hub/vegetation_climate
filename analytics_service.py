import pandas as pd
import numpy as np
from scipy import stats
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

sns.set_theme()

SEASON_MONTHS = {
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "autumn": [9, 10, 11],
    "fall": [9, 10, 11]
}

class AnalyticsService:
    def __init__(self):
        self.data = None

    # ---------- Loaders ----------
    def load_from_gee_list(self, gee_data_list):
        if not gee_data_list or len(gee_data_list) == 0:
            raise ValueError("Empty gee_data_list")

        df = pd.DataFrame(gee_data_list)
        if 'system:time_start' in df.columns:
            df['Date'] = pd.to_datetime(df['system:time_start'], unit='ms')
        elif 'date' in df.columns:
            df['Date'] = pd.to_datetime(df['date'])
        else:
            raise ValueError("Data missing 'system:time_start' or 'date' column")

        df.set_index('Date', inplace=True)
        cols_to_interpolate = ['NDVI', 'Temperature', 'Wind_Speed', 'Pressure', 'Precipitation']
        for col in cols_to_interpolate:
            if col in df.columns:
                df[col] = df[col].interpolate(method='time')

        self.data = df.sort_index()
        return True

    def load_cmip6_csv(self, csv_path_or_buffer, area_id: int, target_year: int, season: str):
        """
        Loads CMIP6 CSV and filters by area_id, target_year, and season.
        Accepts a file-path string or a buffer that pd.read_csv can read.
        Handles area_id values that might be 'N/A' in the CSV by allowing those rows.
        """
        df = pd.read_csv(csv_path_or_buffer)

        # ---- Standardize date column ----
        if 'date' not in df.columns and 'Date' in df.columns:
            df.rename(columns={'Date': 'date'}, inplace=True)

        if 'date' not in df.columns:
            raise ValueError("CMIP6 CSV must contain a 'date' column")

        df['ds'] = pd.to_datetime(df['date'])
        df['year'] = df['ds'].dt.year
        df['month'] = df['ds'].dt.month

        # FILTER BY AREA (allow "N/A" values)
        if 'area_id' in df.columns:
            # Normalize to string for safe comparison (handles numeric or textual 'N/A')
            df['area_id'] = df['area_id'].astype(str)
            df = df[df['area_id'].isin([str(area_id), "N/A"])]
        else:
            raise ValueError("CMIP6 CSV must contain 'area_id' column")

        # FILTER BY YEAR
        df = df[df['year'] == target_year]

        # FILTER BY SEASON
        if 'season' in df.columns:
            df = df[df['season'].str.lower() == season.lower()]
        else:
            # If CSV does not have explicit season, fall back to month-based filtering using SEASON_MONTHS
            months = SEASON_MONTHS.get(season.lower(), SEASON_MONTHS['summer'])
            df = df[df['month'].isin(months)]

        # Keep ONLY needed columns
        keep_cols = ['ds']
        for c in ['Temperature', 'Precipitation', 'Pressure']:
            if c in df.columns:
                keep_cols.append(c)

        return df[keep_cols].sort_values('ds').reset_index(drop=True)


# ---------- helpers ----------
    def _save_figure_to_base64(self):
        img = io.BytesIO()
        plt.tight_layout()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode()

    # correlation heatmap
    def generate_correlation_heatmap(self, df=None):
        if df is None:
            df = self.data
        if df is None:
            return None
        available_cols = [col for col in ['NDVI', 'Temperature', 'Wind_Speed', 'Pressure', 'Precipitation'] if col in df.columns]
        if len(available_cols) < 2:
            return None
        corr = df[available_cols].corr()
        plt.figure(figsize=(8,7))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, cbar_kws={'label':'Pearson Correlation Coefficient'})
        plt.title('Correlation Heatmap: Vegetation vs. Environmental Variables')
        return self._save_figure_to_base64()

    # NDVI time series (optionally include future_df with columns ds,yhat)
    def generate_ndvi_time_series_plot(self, df=None, future_df=None):
        if df is None:
            df = self.data
        if df is None or 'NDVI' not in df.columns:
            return None
        plt.figure(figsize=(10,6))
        plt.plot(df.index, df['NDVI'], label='Historical NDVI')
        if future_df is not None and 'ds' in future_df.columns and 'yhat' in future_df.columns:
            try:
                plt.plot(future_df['ds'], future_df['yhat'], linestyle='--', label='Forecast NDVI')
            except Exception:
                pass
        plt.title('NDVI Vegetation Time Series')
        plt.xlabel('Date')
        plt.ylabel('NDVI')
        plt.legend()
        return self._save_figure_to_base64()

    def generate_ndvi_distribution_plot(self, df=None):
        if df is None:
            df = self.data
        if df is None or 'NDVI' not in df.columns:
            return None
        plt.figure(figsize=(8,5))
        sns.histplot(df['NDVI'].dropna(), kde=True, bins=20)
        plt.title("NDVI Distribution (Histogram + KDE)")
        plt.xlabel("NDVI")
        plt.ylabel("Frequency")
        return self._save_figure_to_base64()

    def generate_climate_distribution_plots(self, df=None):
        """
        Generate histogram+KDE plots for available climate vars.
        Safe: only plots variables that exist and have data.
        """
        if df is None:
            df = self.data
        if df is None:
            return {}
        plots = {}
        # Only plot climate variables that are relevant in CMIP6 (and present in df)
        climate_vars = ['Temperature', 'Pressure', 'Precipitation', 'Wind_Speed']
        for var in climate_vars:
            if var in df.columns and df[var].dropna().shape[0] > 0:
                plt.figure(figsize=(8,5))
                sns.histplot(df[var].dropna(), kde=True, bins=20)
                plt.title(f"{var} Distribution (Histogram + KDE)")
                plt.xlabel(var)
                plt.ylabel("Frequency")
                plots[var] = self._save_figure_to_base64()
        return plots

    # ---------- analysis helpers ----------
    def filter_by_year_season(self, year:int, season:str):
        if self.data is None:
            raise ValueError("No data loaded")
        season = season.lower()
        if season not in SEASON_MONTHS:
            raise ValueError(f"Unknown season: {season}")
        months = SEASON_MONTHS[season]
        def in_season(idx):
            m = idx.month
            y = idx.year
            if months == [12,1,2]:
                return (m == 12 and y == year-1) or (m in [1,2] and y == year)
            else:
                return (m in months and y == year)
        mask = [in_season(idx) for idx in self.data.index]
        return self.data.loc[mask]

    def summarize_period(self, df_subset):
        if df_subset is None or len(df_subset) == 0:
            return {}
        summary = {}
        if 'NDVI' in df_subset.columns:
            summary['ndvi_mean'] = float(df_subset['NDVI'].mean())
            summary['ndvi_median'] = float(df_subset['NDVI'].median())
        if 'Temperature' in df_subset.columns:
            summary['temperature_mean'] = float(df_subset['Temperature'].mean())
        if 'Pressure' in df_subset.columns:
            summary['pressure_mean'] = float(df_subset['Pressure'].mean())
        if 'Precipitation' in df_subset.columns:
            summary['precipitation_mean'] = float(df_subset['Precipitation'].mean())
        return summary

    # ---------- public analyses ----------
    def perform_full_analysis_past(self, gee_data_list, year:int, season:str):
        self.load_from_gee_list(gee_data_list)
        df_period = self.filter_by_year_season(year, season)
        if df_period is None or len(df_period) == 0:
            return {"error": "No data for requested year/season"}

        response = {}
        response['data_values'] = self.summarize_period(df_period)
        response['correlation_heatmap_base64'] = self.generate_correlation_heatmap(df=df_period)
        response['ndvi_distribution_base64'] = self.generate_ndvi_distribution_plot(df=df_period)
        response['climate_distributions'] = self.generate_climate_distribution_plots(df=df_period)

        full_year_mask = (self.data.index.year == year)
        df_year = self.data.loc[full_year_mask]
        if len(df_year) == 0:
            df_year = df_period
        response['plot_image_base64'] = self.generate_ndvi_time_series_plot(df=df_year, future_df=None)
        response['raw_records'] = df_period.reset_index().to_dict(orient='records')
        return response

    #new added
    def perform_full_analysis_future(self, historical_gee_data_list, cmip6_csv_buffer_or_path, area_id:int, target_year:int, season="summer", forecast_years=5, freq='M'):

        self.load_from_gee_list(historical_gee_data_list)
        if 'NDVI' not in self.data.columns or len(self.data) < 12:
            return {"error": "Not enough historical NDVI data to train Prophet (need at least 12 samples)"}

        df_prophet = self.data.reset_index()[['Date','NDVI']].rename(columns={'Date':'ds','NDVI':'y'})
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(df_prophet)

        periods = forecast_years * 12 if freq == 'M' else forecast_years
        future = m.make_future_dataframe(periods=periods, freq=freq)
        forecast = m.predict(future)

        # load CMIP6 (filtered by area, year, season)
        cmip6_df = self.load_cmip6_csv(cmip6_csv_buffer_or_path, area_id=area_id, target_year=target_year, season=season)

        # Align monthly timestamps
        cmip6_df['ds'] = pd.to_datetime(cmip6_df['ds']).dt.to_period('M').dt.to_timestamp()
        forecast['ds'] = pd.to_datetime(forecast['ds']).dt.to_period('M').dt.to_timestamp()

        last_hist = df_prophet['ds'].max()
        future_forecast = forecast[(forecast['ds'] > last_hist) & (forecast['ds'].dt.year == target_year)].copy()

        # safety check: ensure we actually have forecasted months for that year
        if future_forecast.empty:
            return {"error": f"No forecast data available for year {target_year}. Increase forecast_years or check training data."}

        merged = pd.merge(future_forecast[['ds','yhat','yhat_lower','yhat_upper']], cmip6_df, on='ds', how='left')

        # compute mean summary (safe: handle empty / NaN)
        summary_future = {}
        if 'yhat' in merged.columns and merged['yhat'].notna().any():
            summary_future['ndvi_mean'] = float(merged['yhat'].mean())
        else:
            summary_future['ndvi_mean'] = None

        if 'Temperature' in merged.columns and merged['Temperature'].notna().any():
            summary_future['temperature_mean'] = float(merged['Temperature'].mean())
        else:
            summary_future['temperature_mean'] = None

        if 'Pressure' in merged.columns and merged['Pressure'].notna().any():
            summary_future['pressure_mean'] = float(merged['Pressure'].mean())
        else:
            summary_future['pressure_mean'] = None

        if 'Precipitation' in merged.columns and merged['Precipitation'].notna().any():
            summary_future['precipitation_mean'] = float(merged['Precipitation'].mean())
        else:
            summary_future['precipitation_mean'] = None

        response = {}
        response['data_values'] = summary_future
        response['forecast'] = merged.to_dict(orient='records')
        plot_future_df = merged[['ds','yhat']].copy()
        response['plot_image_base64'] = self.generate_ndvi_time_series_plot(df=self.data, future_df=plot_future_df)
        response['correlation_heatmap_base64'] = self.generate_correlation_heatmap(df=self.data)
        response['ndvi_distribution_base64'] = self.generate_ndvi_distribution_plot(df=self.data)
        response['climate_distributions'] = self.generate_climate_distribution_plots(df=cmip6_df)
        response['raw_forecast_table'] = merged.head(500).to_dict(orient='records')
        return response
