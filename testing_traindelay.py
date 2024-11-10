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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ImprovedCombinedDataAnalyzer:
    """Handles combining multiple parquet files and training models with improved features and preprocessing."""
    
    def __init__(self, input_path: str, n_workers: int = 4, memory_per_worker: int = 8):
        self.input_path = input_path
        self.n_workers = n_workers
        self.memory_per_worker = memory_per_worker
        self.cluster = None
        self.client = None
        
        # Improved model parameters
        self.model_params = {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
            'n_jobs': -1,
            'random_state': 42
        }
        
        self.feature_cols = None
        self.geo_cols = ['STATION_LAT', 'STATION_LON']
        self.weather_cols = [
            'temperature_2m', 'dewpoint_2m',  
            'wind_speed', 'wind_direction', 'surface_pressure',
            'total_precipitation', 'snow_cover', 'solar_radiation'
        ]

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
        """Add engineered features to improve model performance."""
        print("\nEngineering additional features...")
        
        # Time-based features
        df['HOUR'] = df['ABFAHRTSZEIT_MINUTES'] // 60
        df['RUSH_HOUR'] = ((df['HOUR'] >= 7) & (df['HOUR'] <= 9) | 
                          (df['HOUR'] >= 16) & (df['HOUR'] <= 18)).astype(int)
        
        # Create a temporary journey identifier using date and line information
        df['DATE'] = pd.to_datetime(df['datetime']).dt.date
        df['TEMP_JOURNEY_ID'] = df.apply(
            lambda x: f"{x['DATE']}_{x['LINIEN_ID_encoded']}_{x['ABFAHRTSZEIT_MINUTES'] // (24*60)}", 
            axis=1
        )
        
        # Sort by our temporary journey ID and time
        df = df.sort_values(['TEMP_JOURNEY_ID', 'ABFAHRTSZEIT_MINUTES'])
        
        # Calculate distances without using apply
        df['NEXT_LAT'] = df.groupby('TEMP_JOURNEY_ID')['STATION_LAT'].shift(-1)
        df['NEXT_LON'] = df.groupby('TEMP_JOURNEY_ID')['STATION_LON'].shift(-1)
        
        # Calculate route distance using the current and next station
        df['ROUTE_DISTANCE'] = np.sqrt(
            np.square(df['NEXT_LAT'] - df['STATION_LAT']) + 
            np.square(df['NEXT_LON'] - df['STATION_LON'])
        )
        
        # Fill NaN values for last station in each journey
        df['ROUTE_DISTANCE'] = df['ROUTE_DISTANCE'].fillna(0)
        
        # Calculate cumulative distance
        df['CUMULATIVE_DISTANCE'] = df.groupby('TEMP_JOURNEY_ID')['ROUTE_DISTANCE'].cumsum()
        
        # Calculate max distance for each journey
        max_distances = df.groupby('TEMP_JOURNEY_ID')['CUMULATIVE_DISTANCE'].transform('max')
        
        # Calculate journey progress (0-1)
        df['JOURNEY_PROGRESS'] = df['CUMULATIVE_DISTANCE'] / max_distances
        df['JOURNEY_PROGRESS'] = df['JOURNEY_PROGRESS'].fillna(0)
        
        # Weather interaction features
        df['SEVERE_WEATHER'] = ((df['wind_speed'] > df['wind_speed'].quantile(0.75)) | 
                              (df['total_precipitation'] > df['total_precipitation'].quantile(0.75))).astype(int)
        df['TEMP_DEWPOINT_DIFF'] = df['temperature_2m'] - df['dewpoint_2m']
        
        # Previous station delay impact - now by journey
        df['PREV_STATION_DELAY'] = df.groupby('TEMP_JOURNEY_ID')['DEPARTURE_TIME_DIFF_SECONDS'].shift(1)
        df['PREV_STATION_DELAY'] = df['PREV_STATION_DELAY'].fillna(0)
        
        # Calculate delay trend
        df['DELAY_TREND'] = df.groupby('TEMP_JOURNEY_ID')['DEPARTURE_TIME_DIFF_SECONDS'].diff()
        df['DELAY_TREND'] = df['DELAY_TREND'].fillna(0)
        
        # Create time windows for aggregating historical delays
        df['TIME_WINDOW'] = pd.to_datetime(df['datetime']).dt.floor('3H')
        
        # Calculate historical delay patterns - by line and station
        df['HISTORICAL_DELAY_PATTERN'] = df.groupby(
            ['LINIEN_ID_encoded', 'HALTESTELLEN_NAME_encoded', 'TIME_WINDOW']
        )['DEPARTURE_TIME_DIFF_SECONDS'].transform('mean').fillna(0)
        
        # Calculate station sequence and remaining stops
        df['STATION_SEQUENCE'] = df.groupby('TEMP_JOURNEY_ID').cumcount()
        df['TOTAL_STOPS'] = df.groupby('TEMP_JOURNEY_ID')['STATION_SEQUENCE'].transform('max')
        df['REMAINING_STOPS'] = df['TOTAL_STOPS'] - df['STATION_SEQUENCE']
        
        # Clean up temporary columns
        df = df.drop(['DATE', 'TEMP_JOURNEY_ID', 'NEXT_LAT', 'NEXT_LON', 'TOTAL_STOPS'], axis=1)
        
        # Fill any remaining NaN values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        print("\nNew features created:")
        new_features = [col for col in df.columns if col not in self.initial_columns]
        print(new_features)
        
        return df

    

    def prepare_features(self, df):
        """Prepare and preprocess features for modeling."""
        # Define feature groups
        time_features = ['ABFAHRTSZEIT_MINUTES', 'ANKUNFTSZEIT_MINUTES', 'HOUR']
        
        geo_features = [
            'STATION_LAT', 'STATION_LON', 'ROUTE_DISTANCE', 
            'CUMULATIVE_DISTANCE', 'JOURNEY_PROGRESS', 'REMAINING_STOPS'
        ]
        
        weather_features = self.weather_cols + ['TEMP_DEWPOINT_DIFF']
        
        categorical_features = [col for col in df.columns if col.endswith('_encoded')]
        
        engineered_features = [
            'RUSH_HOUR', 'SEVERE_WEATHER', 'HISTORICAL_DELAY_PATTERN',
            'PREV_STATION_DELAY', 'DELAY_TREND'
        ]
        
        # Combine all features
        self.feature_cols = (time_features + geo_features + weather_features + 
                           categorical_features + engineered_features)
        
        # Get feature matrix and print column info
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
                          and col != 'PREV_STATION_DELAY']  # Exclude PREV_STATION_DELAY from scaling
        
        categorical_features = [col for col in self.feature_cols if col.endswith('_encoded')]
        binary_features = ['SEVERE_WEATHER', 'RUSH_HOUR']
        delay_features = ['PREV_STATION_DELAY']  # Handle separately
        
        print("\nPreprocessing feature groups:")
        print(f"Numeric features: {numeric_features}")
        print(f"Categorical features: {categorical_features}")
        print(f"Binary features: {binary_features}")
        print(f"Delay features: {delay_features}")
        
        # Create preprocessing steps
        numeric_transformer = Pipeline(steps=[
            ('scaler', RobustScaler())
        ])
        
        # Create column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', 'passthrough', categorical_features),
                ('bin', 'passthrough', binary_features),
                ('delay', 'passthrough', delay_features)  # Pass through delay features without transformation
            ],
            verbose_feature_names_out=True
        )
        
        return preprocessor
    
    def combine_and_analyze(self):
        """Combine parquet files and perform analysis."""
        try:
            self.initialize_cluster()
            
            # Find all parquet files
            parquet_files = glob(os.path.join(self.input_path, "*.parquet"))
            if not parquet_files:
                raise ValueError(f"No parquet files found in {self.input_path}")
            
            print(f"\nFound {len(parquet_files)} parquet files")
            
            # Read and combine all parquet files
            print("\nReading and combining parquet files...")
            ddf = dd.read_parquet(parquet_files)
            
            # Convert to pandas for modeling
            print("\nConverting to pandas DataFrame...")
            df = ddf.compute()
            
            # Store initial columns for reference
            self.initial_columns = df.columns.tolist()
            
            # Engineer features
            df = self.engineer_features(df)
            
            # Train models
            print("\nTraining models...")
            models = self.train_models(df)
            
            # Create and save visualizations
            print("\nGenerating visualizations...")
            self.create_visualizations(df, models)
            
            # Save models and analysis results
            self.save_results(df, models)
            
            return models
            
        except Exception as e:
            logging.error(f"Analysis error: {str(e)}", exc_info=True)
            raise
        finally:
            if self.cluster:
                self.cluster.close()

    def train_models(self, df):
        """Train improved delay prediction models."""
        models = {}
        
        # Prepare feature matrix
        print("\nPreparing features...")
        X = self.prepare_features(df)
        
        # Handle missing values
        X_cleaned = X.copy()
        for col in X.columns:
            if col in self.weather_cols:
                X_cleaned[col] = X_cleaned[col].fillna(X_cleaned[col].median())
            else:
                X_cleaned[col] = X_cleaned[col].fillna(0)
        
        # Print feature columns for debugging
        print("\nFeature columns:")
        for i, col in enumerate(X_cleaned.columns):
            print(f"{i}: {col}")
        
        # Create preprocessing pipeline
        preprocessor = self.create_preprocessing_pipeline()
        
        # Train models for arrival and departure delays
        for target in ['arrival', 'departure']:
            print(f"\nTraining {target} model...")
            
            y = df[f'{target.upper()}_TIME_DIFF_SECONDS']
            
            # Remove extreme outliers
            valid_mask = np.abs(y - y.mean()) <= (3 * y.std())
            X_filtered = X_cleaned[valid_mask]
            y_filtered = y[valid_mask]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_filtered, test_size=0.2, random_state=42
            )
            
            # Create full pipeline
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(**self.model_params))
            ])
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            print(f"{target.title()} Model Performance:")
            print(f"R² Score: {r2:.3f}")
            print(f"RMSE: {rmse/60:.2f} minutes")
            
            # Get feature names and importance
            feature_names = X_filtered.columns.tolist()
            feature_importance = model.named_steps['regressor'].feature_importances_
            
            print(f"\nFeature array shapes:")
            print(f"Number of feature names: {len(feature_names)}")
            print(f"Number of importance values: {len(feature_importance)}")
            
            # Find the extra feature
            if len(feature_names) > len(feature_importance):
                print("\nExtra feature found:")
                print(feature_names[-1])
                # Remove the extra feature
                feature_names = feature_names[:-1]
            
            # Create DataFrame safely
            importances = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 most important features for {target} delays:")
            for _, row in importances.head(10).iterrows():
                print(f"- {row['feature']}: {row['importance']:.4f}")
            
            models[target] = {
                'model': model,
                'metrics': {
                    'r2': r2,
                    'rmse': rmse
                },
                'feature_importance': importances
            }
        
        return models
    

    def create_visualizations(self, df, models):
        """Create and save analysis visualizations."""
        plots_dir = os.path.join(self.input_path, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Enhanced Delay Distribution Plot
        plt.figure(figsize=(15, 10))
        
        # Create main subplot for KDE
        plt.subplot(2, 1, 1)
        
        # Convert seconds to minutes for better readability
        departure_delays = df['DEPARTURE_TIME_DIFF_SECONDS'] / 60
        arrival_delays = df['ARRIVAL_TIME_DIFF_SECONDS'] / 60
        
        # Plot KDE with improved styling
        sns.kdeplot(data=departure_delays, label='Departure', alpha=0.6, color='blue')
        sns.kdeplot(data=arrival_delays, label='Arrival', alpha=0.6, color='red')
        
        # Add vertical line at 0 (on-time)
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='On Time')
        
        # Add descriptive labels
        plt.xlabel('Delay (minutes)')
        plt.ylabel('Density')
        plt.title('Distribution of Train Delays\nKernel Density Estimation')
        plt.legend()
        
        # Add text box with statistics
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
        
        # Add histogram subplot for more detail
        plt.subplot(2, 1, 2)
        
        # Create bins centered around zero
        max_delay = max(abs(departure_delays.min()), abs(departure_delays.max()),
                       abs(arrival_delays.min()), abs(arrival_delays.max()))
        bins = np.linspace(-max_delay, max_delay, 50)
        
        # Plot histograms
        plt.hist(departure_delays, bins=bins, alpha=0.5, label='Departure', color='blue')
        plt.hist(arrival_delays, bins=bins, alpha=0.5, label='Arrival', color='red')
        
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='On Time')
        plt.xlabel('Delay (minutes)')
        plt.ylabel('Frequency')
        plt.title('Histogram of Train Delays')
        plt.legend()
        
        # Add percentage annotations
        on_time_threshold = 1  # Define on-time as within 1 minute
        dep_on_time = (abs(departure_delays) <= on_time_threshold).mean() * 100
        arr_on_time = (abs(arrival_delays) <= on_time_threshold).mean() * 100
        
        plt.text(0.95, 0.95, 
                f"On-time performance (±{on_time_threshold} min):\n"
                f"Departures: {dep_on_time:.1f}%\n"
                f"Arrivals: {arr_on_time:.1f}%",
                transform=plt.gca().transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'delay_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Feature Importance Plot
        plt.figure(figsize=(15, 10))
        for i, (target, model_info) in enumerate(models.items(), 1):
            plt.subplot(2, 1, i)
            importances = model_info['feature_importance'].head(15)
            sns.barplot(data=importances, x='importance', y='feature')
            plt.title(f'Top 15 Features for {target.title()} Delay Prediction')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'feature_importance.png'))
        plt.close()
        
        # 3. Weather Impact Plot
        plt.figure(figsize=(15, 10))
        weather_cols_to_plot = self.weather_cols[:6] + ['TEMP_DEWPOINT_DIFF']
        for i, weather_col in enumerate(weather_cols_to_plot, 1):
            plt.subplot(3, 3, i)
            plt.scatter(df[weather_col], df['DEPARTURE_TIME_DIFF_SECONDS']/60, 
                       alpha=0.1, s=1)
            plt.xlabel(weather_col)
            plt.ylabel('Delay (minutes)')
            plt.title(f'Impact of {weather_col}')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'weather_impact.png'))
        plt.close()
        
        # 4. Rush Hour Impact
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='RUSH_HOUR', y=df['DEPARTURE_TIME_DIFF_SECONDS']/60)
        plt.xlabel('Rush Hour (0/1)')
        plt.ylabel('Delay (minutes)')
        plt.title('Impact of Rush Hour on Delays')
        plt.savefig(os.path.join(plots_dir, 'rush_hour_impact.png'))
        plt.close()
        
        # 3. Weather Impact Plot
        plt.figure(figsize=(15, 10))
        weather_cols_to_plot = self.weather_cols[:6] + ['TEMP_DEWPOINT_DIFF']
        for i, weather_col in enumerate(weather_cols_to_plot, 1):
            plt.subplot(3, 3, i)
            plt.scatter(df[weather_col], df['DEPARTURE_TIME_DIFF_SECONDS']/60, 
                       alpha=0.1, s=1)
            plt.xlabel(weather_col)
            plt.ylabel('Delay (minutes)')
            plt.title(f'Impact of {weather_col}')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'weather_impact.png'))
        plt.close()
        
        # 4. Rush Hour Impact
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='RUSH_HOUR', y=df['DEPARTURE_TIME_DIFF_SECONDS']/60)
        plt.xlabel('Rush Hour (0/1)')
        plt.ylabel('Delay (minutes)')
        plt.title('Impact of Rush Hour on Delays')
        plt.savefig(os.path.join(plots_dir, 'rush_hour_impact.png'))
        plt.close()

    def save_results(self, df, models):
        """Save models and analysis results."""
        # Create output directories
        models_dir = os.path.join(self.input_path, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Save models
        for target, model_info in models.items():
            model_path = os.path.join(models_dir, f'{target}_model.joblib')
            joblib.dump(model_info['model'], model_path)
            print(f"Saved {target} model to: {model_path}")
        
        # Save feature importances
        for target, model_info in models.items():
            importance_path = os.path.join(models_dir, f'{target}_feature_importance.csv')
            model_info['feature_importance'].to_csv(importance_path, index=False)
        
        # Save analysis summary
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'data_shape': df.shape,
            'features_used': self.feature_cols,
            'model_parameters': self.model_params,
            'performance': {
                target: {
                    'r2_score': float(model_info['metrics']['r2']),
                    'rmse_minutes': float(model_info['metrics']['rmse']/60)
                }
                for target, model_info in models.items()
            }
        }
        
        with open(os.path.join(models_dir, 'analysis_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

def main():
    """Main execution function."""
    try:
        input_path = "/home/vmk0863/dat503/data/processed/transport_weather_combined"
        
        analyzer = ImprovedCombinedDataAnalyzer(
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
