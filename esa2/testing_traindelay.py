import os
import logging
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import pandas as pd
import numpy as np
from glob import glob
from datetime import datetime
import json
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class OptimizedTrainDelayAnalyzer:
    """Handles combining multiple parquet files and training models with improved features and preprocessing."""
    
    def __init__(self, input_path: str, n_workers: int = 4, memory_per_worker: int = 8):
        self.input_path = input_path
        self.n_workers = n_workers
        self.memory_per_worker = memory_per_worker
        self.cluster = None
        self.client = None
        
        self.model_params = {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
            'n_jobs': -1,
            'random_state': 42,
            'bootstrap': True,
            'oob_score': True,
            'warm_start': False
        }
        
        self.feature_cols = None
        self.weather_cols = [
            'temperature_2m', 'dewpoint_2m',  
            'wind_speed', 'wind_direction', 'surface_pressure',
            'total_precipitation', 'snow_cover', 'solar_radiation'
        ]
        self.initial_columns = None

    def initialize_cluster(self) -> None:
        """Initialize Dask cluster for distributed processing."""
        print(f"\n=== Initializing Analysis Environment ===")
        print(f"Workers: {self.n_workers}")
        print(f"Memory per worker: {self.memory_per_worker}GB")
        
        self.cluster = LocalCluster(
            n_workers=self.n_workers,
            threads_per_worker=2,
            memory_limit=f"{self.memory_per_worker}GB",
            silence_logs=logging.WARNING
        )
        self.client = Client(self.cluster)
        print(f"Dashboard: {self.client.dashboard_link}")

    def engineer_features(self, df):
        """Add engineered features with improved performance."""
        print("\nEngineering additional features...")
        
        # Store initial columns
        self.initial_columns = df.columns.tolist()
        
        # Time-based features from scheduled times
        df['SCHEDULED_HOUR'] = df['ABFAHRTSZEIT_MINUTES'] // 60
        df['RUSH_HOUR'] = ((df['SCHEDULED_HOUR'] >= 7) & (df['SCHEDULED_HOUR'] <= 9) | 
                          (df['SCHEDULED_HOUR'] >= 16) & (df['SCHEDULED_HOUR'] <= 18)).astype(int)
        
        # Create a temporary journey identifier
        df['DATE'] = pd.to_datetime(df['datetime']).dt.date
        df['TEMP_JOURNEY_ID'] = df.apply(
            lambda x: f"{x['DATE']}_{x['LINIEN_ID_encoded']}_{x['SCHEDULED_HOUR']}", 
            axis=1
        )
        
        # Calculate distances first using LAT/LON
        df = df.sort_values(['TEMP_JOURNEY_ID', 'ABFAHRTSZEIT_MINUTES'])
        df['NEXT_LAT'] = df.groupby('TEMP_JOURNEY_ID')['STATION_LAT'].shift(-1)
        df['NEXT_LON'] = df.groupby('TEMP_JOURNEY_ID')['STATION_LON'].shift(-1)
        
        df['ROUTE_DISTANCE'] = np.sqrt(
            np.square(df['NEXT_LAT'] - df['STATION_LAT']) + 
            np.square(df['NEXT_LON'] - df['STATION_LON'])
        )
        df['ROUTE_DISTANCE'] = df['ROUTE_DISTANCE'].fillna(0)
        
        # Calculate journey metrics
        df['CUMULATIVE_DISTANCE'] = df.groupby('TEMP_JOURNEY_ID')['ROUTE_DISTANCE'].cumsum()
        
        max_distances = df.groupby('TEMP_JOURNEY_ID')['CUMULATIVE_DISTANCE'].transform('max')
        df['JOURNEY_PROGRESS'] = df['CUMULATIVE_DISTANCE'] / max_distances
        df['JOURNEY_PROGRESS'] = df['JOURNEY_PROGRESS'].fillna(0)
        
        # Weather interaction features
        df['SEVERE_WEATHER'] = ((df['wind_speed'] > df['wind_speed'].quantile(0.75)) | 
                              (df['total_precipitation'] > df['total_precipitation'].quantile(0.75))).astype(int)
        df['TEMP_DEWPOINT_DIFF'] = df['temperature_2m'] - df['dewpoint_2m']
        
        # Previous station delay impact
        df['PREV_STATION_DELAY'] = df.groupby('TEMP_JOURNEY_ID')['DEPARTURE_TIME_DIFF_SECONDS'].shift(1)
        df['PREV_STATION_DELAY'] = df['PREV_STATION_DELAY'].fillna(0)
        
        # Calculate delay trend
        df['DELAY_TREND'] = df.groupby('TEMP_JOURNEY_ID')['DEPARTURE_TIME_DIFF_SECONDS'].diff()
        df['DELAY_TREND'] = df['DELAY_TREND'].fillna(0)
        
        # Create time windows and historical patterns
        df['TIME_WINDOW'] = pd.to_datetime(df['datetime']).dt.floor('3H')
        
        df['HISTORICAL_DELAY_PATTERN'] = df.groupby(
            ['LINIEN_ID_encoded', 'HALTESTELLEN_NAME_encoded', 'TIME_WINDOW']
        )['DEPARTURE_TIME_DIFF_SECONDS'].transform('mean').fillna(0)
        
        # Station sequence and remaining stops
        df['STATION_SEQUENCE'] = df.groupby('TEMP_JOURNEY_ID').cumcount()
        df['TOTAL_STOPS'] = df.groupby('TEMP_JOURNEY_ID')['STATION_SEQUENCE'].transform('max')
        df['REMAINING_STOPS'] = df['TOTAL_STOPS'] - df['STATION_SEQUENCE']
        
        # Drop temporary and redundant columns
        columns_to_drop = [
            'DATE', 'TEMP_JOURNEY_ID', 'NEXT_LAT', 'NEXT_LON', 'TOTAL_STOPS',
            'STATION_LAT', 'STATION_LON',  # Remove coordinate columns
            'LINIEN_TEXT_encoded',  # Remove redundant line text
            'FAELLT_AUS_TF_encoded'  # Remove constant cancelled trains feature
        ]
        df = df.drop(columns_to_drop, axis=1)
        
        # Convert float columns to float32 for memory efficiency
        float_cols = df.select_dtypes(include=['float64']).columns
        df[float_cols] = df[float_cols].astype('float32')
        
        print("\nNew features created:")
        new_features = [col for col in df.columns if col not in self.initial_columns]
        print(new_features)
        
        return df
    
    def prepare_features(self, df):
        """Prepare and preprocess features for modeling."""
        # Define feature groups
        time_features = [
            'SCHEDULED_HOUR',
            'ABFAHRTSZEIT_DAY_OF_WEEK',
            'ABFAHRTSZEIT_MONTH',
            'ABFAHRTSZEIT_IS_WEEKEND'
        ]
        
        geo_features = [
            'ROUTE_DISTANCE', 
            'CUMULATIVE_DISTANCE', 
            'JOURNEY_PROGRESS', 
            'REMAINING_STOPS'
        ]
        
        weather_features = self.weather_cols + ['TEMP_DEWPOINT_DIFF']
        
        categorical_features = [
            'ZUSATZFAHRT_TF_encoded',
            'DURCHFAHRT_TF_encoded',
            'LINIEN_ID_encoded',
            'HALTESTELLEN_NAME_encoded'
        ]
        
        engineered_features = [
            'RUSH_HOUR', 'SEVERE_WEATHER', 'HISTORICAL_DELAY_PATTERN',
            'PREV_STATION_DELAY', 'DELAY_TREND'
        ]
        
        # Combine all features
        self.feature_cols = (time_features + geo_features + weather_features + 
                           categorical_features + engineered_features)
        
        # Get feature matrix and print info
        X = df[self.feature_cols]
        print("\nFeature groups:")
        print(f"Time features: {time_features}")
        print(f"Geo features: {geo_features}")
        print(f"Weather features: {weather_features}")
        print(f"Categorical features: {categorical_features}")
        print(f"Engineered features: {engineered_features}")
        
        return X

    def create_preprocessing_pipeline(self):
        """Create preprocessing pipeline with different scalers for different feature types."""
        # Define feature groups for different preprocessing
        numeric_features = [col for col in self.feature_cols 
                          if not col.endswith('_encoded') 
                          and col not in ['SEVERE_WEATHER', 'RUSH_HOUR']
                          and col != 'PREV_STATION_DELAY']
        
        categorical_features = [col for col in self.feature_cols if col.endswith('_encoded')]
        binary_features = ['SEVERE_WEATHER', 'RUSH_HOUR']
        delay_features = ['PREV_STATION_DELAY']
        
        print("\nPreprocessing feature groups:")
        print(f"Numeric features: {numeric_features}")
        print(f"Categorical features: {categorical_features}")
        print(f"Binary features: {binary_features}")
        print(f"Delay features: {delay_features}")
        
        # Create preprocessing steps
        numeric_transformer = Pipeline(steps=[
            ('scaler', RobustScaler(copy=False))
        ])
        
        # Create column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', 'passthrough', categorical_features),
                ('bin', 'passthrough', binary_features),
                ('delay', 'passthrough', delay_features)
            ],
            sparse_threshold=0,
            n_jobs=4
        )
        
        return preprocessor

    def train_models(self, df):
        """Train improved delay prediction models with enhanced validation."""
        models = {}
        
        # Prepare feature matrix
        X = self.prepare_features(df)
        
        # Handle missing values with more robust strategy
        X_cleaned = X.copy()
        for col in X.columns:
            if col in self.weather_cols:
                # Use robust statistics for weather data
                median_val = X_cleaned[col].median()
                iqr = X_cleaned[col].quantile(0.75) - X_cleaned[col].quantile(0.25)
                lower_bound = median_val - 1.5 * iqr
                upper_bound = median_val + 1.5 * iqr
                
                # Replace outliers with bounds
                X_cleaned[col] = X_cleaned[col].clip(lower_bound, upper_bound)
                # Fill remaining missing values with median
                X_cleaned[col] = X_cleaned[col].fillna(median_val)
            else:
                X_cleaned[col] = X_cleaned[col].fillna(0)
        
        # Convert float64 to float32
        float_cols = X_cleaned.select_dtypes(include=['float64']).columns
        X_cleaned[float_cols] = X_cleaned[float_cols].astype('float32')
        
        # Train models for arrival and departure delays
        for target in ['arrival', 'departure']:
            print(f"\nTraining {target} model...")
            
            y = df[f'{target.upper()}_TIME_DIFF_SECONDS'].astype('float32')
            
            # Enhanced outlier removal using IQR
            q1, q3 = y.quantile([0.25, 0.75])
            iqr = q3 - q1
            valid_mask = (y >= q1 - 3*iqr) & (y <= q3 + 3*iqr)
            X_filtered = X_cleaned[valid_mask]
            y_filtered = y[valid_mask]
            
            # Take a sample if dataset is very large
            if len(X_filtered) > 100000:
                sample_idx = np.random.choice(len(X_filtered), 100000, replace=False)
                X_filtered = X_filtered.iloc[sample_idx]
                y_filtered = y_filtered.iloc[sample_idx]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_filtered, 
                test_size=0.2, 
                random_state=42
            )
            
            # Create pipeline with grid search
            pipeline = Pipeline([
                ('preprocessor', self.create_preprocessing_pipeline()),
                ('regressor', RandomForestRegressor(**self.model_params))
            ])
            
            # Simple parameter grid
            param_grid = {
                'regressor__min_samples_split': [10, 20],
                'regressor__min_samples_leaf': [5, 10]
            }
            
            # Perform grid search
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=TimeSeriesSplit(n_splits=3),
                scoring='neg_root_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get best model
            model = grid_search.best_estimator_
            
            # Evaluate
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # Calculate feature importance with permutation importance
            perm_importance = permutation_importance(
                model, X_test, y_test,
                n_repeats=5,
                random_state=42,
                n_jobs=-1
            )
            
            # Store results
            feature_names = X_filtered.columns.tolist()
            importances = pd.DataFrame({
                'feature': feature_names,
                'importance': perm_importance.importances_mean,
                'importance_std': perm_importance.importances_std
            }).sort_values('importance', ascending=False)
            models[target] = {
                'model': model,
                'metrics': {
                    'r2': r2,
                    'rmse': rmse,
                    'best_params': grid_search.best_params_,
                    'cv_results': grid_search.cv_results_
                },
                'feature_importance': importances
            }
            
            # Print results
            print(f"\n{target.title()} Model Performance:")
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"R² Score: {r2:.3f}")
            print(f"RMSE: {rmse/60:.2f} minutes")
            
            # Clear memory
            del grid_search
            gc.collect()
        
        return models

    def create_feature_importance_plot(self, models, plots_dir):
        """Create enhanced feature importance visualization."""
        plt.style.use('default')
        fig = plt.figure(figsize=(15, 20))
        gs = plt.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
        
        for idx, (target, model_info) in enumerate(models.items()):
            ax = fig.add_subplot(gs[idx])
            importances = model_info['feature_importance'].head(15)
            
            bars = ax.barh(range(len(importances)), importances['importance'],
                        xerr=importances['importance_std'],
                        alpha=0.8, capsize=5,
                        color='royalblue')
            
            ax.set_yticks(range(len(importances)))
            ax.set_yticklabels(importances['feature'])
            ax.set_xlabel('Feature Importance Score')
            ax.set_title(f'{target.title()} Model - Top 15 Features\n'
                        f'R² Score: {model_info["metrics"]["r2"]:.3f}, '
                        f'RMSE: {model_info["metrics"]["rmse"]/60:.2f} minutes')
            
            ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        plt.suptitle('Feature Importance Analysis', fontsize=16, y=0.95)
        plt.savefig(os.path.join(plots_dir, 'feature_importance.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    def create_delay_distribution_plot(self, df, plots_dir):
        """Create enhanced delay distribution visualization."""
        plt.figure(figsize=(15, 10))
        
        # Main subplot for KDE
        plt.subplot(2, 1, 1)
        
        departure_delays = df['DEPARTURE_TIME_DIFF_SECONDS'] / 60
        arrival_delays = df['ARRIVAL_TIME_DIFF_SECONDS'] / 60
        
        sns.kdeplot(data=departure_delays, label='Departure', alpha=0.6)
        sns.kdeplot(data=arrival_delays, label='Arrival', alpha=0.6)
        
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel('Delay (minutes)')
        plt.ylabel('Density')
        plt.title('Distribution of Train Delays')
        plt.legend()
        
        stats_text = (
            f"Departure Delays:\n"
            f"Mean: {departure_delays.mean():.1f} min\n"
            f"Median: {departure_delays.median():.1f} min\n"
            f"Std Dev: {departure_delays.std():.1f} min\n\n"
            f"Arrival Delays:\n"
            f"Mean: {arrival_delays.mean():.1f} min\n"
            f"Median: {arrival_delays.median():.1f} min\n"
            f"Std Dev: {arrival_delays.std():.1f} min"
        )
        
        plt.text(0.95, 0.95, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.savefig(os.path.join(plots_dir, 'delay_distribution.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    def create_weather_impact_plot(self, df, plots_dir):
        """Create visualization of weather impacts on delays."""
        plt.figure(figsize=(20, 15))
        
        weather_impacts = [
            ('temperature_2m', 'Temperature (°C)'),
            ('wind_speed', 'Wind Speed (m/s)'),
            ('total_precipitation', 'Precipitation (mm)'),
            ('snow_cover', 'Snow Cover (%)'),
            ('surface_pressure', 'Surface Pressure (hPa)'),
            ('solar_radiation', 'Solar Radiation (W/m²)')
        ]
        
        for i, (feature, label) in enumerate(weather_impacts, 1):
            plt.subplot(2, 3, i)
            
            plt.scatter(df[feature], 
                    df['DEPARTURE_TIME_DIFF_SECONDS'] / 60,
                    alpha=0.1, s=1, color='royalblue')
            
            z = np.polyfit(df[feature], 
                        df['DEPARTURE_TIME_DIFF_SECONDS'] / 60, 1)
            p = np.poly1d(z)
            plt.plot(df[feature], p(df[feature]), 
                    color='crimson', linestyle='--', alpha=0.8)
            
            plt.xlabel(label)
            plt.ylabel('Delay (minutes)')
            plt.title(f'Impact of {label} on Delays')
            
            corr = df[feature].corr(df['DEPARTURE_TIME_DIFF_SECONDS'])
            plt.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'weather_impact.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    def create_model_performance_plot(self, models, plots_dir):
        """Create visualization comparing model performances."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        model_names = list(models.keys())
        r2_scores = [info['metrics']['r2'] for info in models.values()]
        rmse_scores = [info['metrics']['rmse']/60 for info in models.values()]
        
        ax1.bar(model_names, r2_scores, color='royalblue', alpha=0.7)
        ax1.set_ylabel('R² Score')
        ax1.set_title('R² Score by Model')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        ax2.bar(model_names, rmse_scores, color='crimson', alpha=0.7)
        ax2.set_ylabel('RMSE (minutes)')
        ax2.set_title('RMSE by Model')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.suptitle('Model Performance Comparison', y=1.05)
        plt.tight_layout()
        
        plt.savefig(os.path.join(plots_dir, 'model_performance.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    def create_visualizations(self, df, models):
        """Create and save analysis visualizations."""
        plots_dir = os.path.join(self.input_path, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        self.create_feature_importance_plot(models, plots_dir)
        self.create_delay_distribution_plot(df, plots_dir)
        self.create_weather_impact_plot(df, plots_dir)
        self.create_model_performance_plot(models, plots_dir)

    def save_results(self, df, models):
        """Save models and analysis results efficiently."""
        models_dir = os.path.join(self.input_path, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        for target, model_info in models.items():
            model_path = os.path.join(models_dir, f'{target}_model.joblib')
            joblib.dump(model_info['model'], model_path)
            
            importance_path = os.path.join(models_dir, f'{target}_feature_importance.csv')
            model_info['feature_importance'].to_csv(importance_path, index=False)
            
            cv_results_path = os.path.join(models_dir, f'{target}_cv_results.json')
            cv_results = {
                'best_params': model_info['metrics']['best_params'],
                'mean_cv_score': float(-model_info['metrics']['cv_results']['mean_test_score'][0]),
                'std_cv_score': float(model_info['metrics']['cv_results']['std_test_score'][0])
            }
            with open(cv_results_path, 'w') as f:
                json.dump(cv_results, f, indent=2)
        
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'data_shape': df.shape,
            'features_used': self.feature_cols,
            'model_parameters': self.model_params,
            'performance': {
                target: {
                    'r2_score': float(model_info['metrics']['r2']),
                    'rmse_minutes': float(model_info['metrics']['rmse']/60),
                    'best_parameters': model_info['metrics']['best_params']
                }
                for target, model_info in models.items()
            }
        }
        
        with open(os.path.join(models_dir, 'analysis_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

    def combine_and_analyze(self):
        """Combine parquet files and perform analysis."""
        try:
            self.initialize_cluster()
            
            parquet_files = glob(os.path.join(self.input_path, "*.parquet"))
            if not parquet_files:
                raise ValueError(f"No parquet files found in {self.input_path}")
            
            print(f"\nFound {len(parquet_files)} parquet files")
            
            print("\nReading and combining parquet files...")
            ddf = dd.read_parquet(parquet_files)
            
            print("\nConverting to pandas DataFrame...")
            df = ddf.compute()
            
            df = self.engineer_features(df)
            
            print("\nTraining models...")
            models = self.train_models(df)
            
            print("\nGenerating visualizations...")
            self.create_visualizations(df, models)
            
            print("\nSaving results...")
            self.save_results(df, models)
            
            return models
            
        except Exception as e:
            logging.error(f"Analysis error: {str(e)}", exc_info=True)
            raise
        finally:
            if self.client:
                self.client.close()
            if self.cluster:
                self.cluster.close()

def main():
    """Main execution function."""
    try:
        input_path = "/home/vmk0863/dat503/data/processed/transport_weather_combined"
        
        analyzer = OptimizedTrainDelayAnalyzer(
            input_path=input_path,
            n_workers=4,
            memory_per_worker=8
        )
        
        analyzer.combine_and_analyze()
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
