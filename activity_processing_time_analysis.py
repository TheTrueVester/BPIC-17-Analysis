"""
Activity Processing Time Analysis for BPIC 2017

This script analyzes processing times of activities in the BPI Challenge 2017 dataset
using two approaches:

1. Probability Distribution Fitting:
   - Fits probability distributions (normal, lognormal, exponential, gamma) on historical 
     activity processing times
   - Visualizes distributions with histograms and fitted curves
   - Reports distribution parameters and goodness-of-fit statistics

2. Machine Learning Point Estimation:
   - Trains regression models (Random Forest, Gradient Boosting, Linear Regression) 
     to predict processing times based on contextual features
   - Features include: activity type, hour of day, day of week, case attributes,
     activity position in trace, previous activity, etc.
   - Reports model performance metrics (MAE, RMSE, R²)
   - Provides feature importance analysis
"""

import os
import warnings
from typing import Dict, List, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import lognorm, expon, gamma, norm
import pm4py

# Machine learning imports
try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. ML models will be skipped.")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Configuration
DATA_PATH = os.path.join("data", "BPI_Challenge_2017.xes.gz")
OUTPUT_DIR = "results"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "processing_time_plots")
RANDOM_STATE = 42

# Distribution fitting configuration
DISTRIBUTIONS_TO_FIT = ['lognorm', 'gamma', 'expon', 'norm']

# ML model configuration
ML_MODELS = {
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_STATE),
    'Ridge Regression': Ridge(alpha=1.0, random_state=RANDOM_STATE),
}

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_and_preprocess_data(path: str) -> pd.DataFrame:
    """
    Load XES event log and convert to DataFrame with processing times.
    
    Processing time is calculated as the time between consecutive events
    in the same case (inter-event time).
    
    Returns:
        DataFrame with events and their processing times
    """
    print("\n" + "=" * 80)
    print("LOADING AND PREPROCESSING DATA")
    print("=" * 80)
    
    # Load XES file
    print(f"Loading event log from: {path}")
    log = pm4py.read_xes(path)
    print(f"Loaded {len(log)} cases")
    
    # Convert to DataFrame
    df = pm4py.convert_to_dataframe(log)
    print(f"Converted to DataFrame: {df.shape}")
    
    # Ensure timestamp is datetime
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], errors='coerce')
    
    # Drop rows with missing timestamps
    before = len(df)
    df = df.dropna(subset=['time:timestamp'])
    print(f"Dropped {before - len(df)} rows with missing timestamps")
    
    # Sort by case and timestamp
    df = df.sort_values(['case:concept:name', 'time:timestamp'])
    
    # Calculate processing time (time since previous event in same case)
    df['prev_timestamp'] = df.groupby('case:concept:name')['time:timestamp'].shift(1)
    df['processing_time_seconds'] = (df['time:timestamp'] - df['prev_timestamp']).dt.total_seconds()
    
    # Drop first event of each case (no processing time)
    df = df.dropna(subset=['processing_time_seconds'])
    
    # Filter out negative and zero processing times (data quality issues)
    before = len(df)
    df = df[df['processing_time_seconds'] > 0]
    print(f"Filtered out {before - len(df)} events with non-positive processing times")
    
    # Add time-based features
    df['hour_of_day'] = df['time:timestamp'].dt.hour
    df['day_of_week'] = df['time:timestamp'].dt.dayofweek
    df['month'] = df['time:timestamp'].dt.month
    
    # Add position in trace
    df['event_position'] = df.groupby('case:concept:name').cumcount() + 1
    
    # Add previous activity
    df['prev_activity'] = df.groupby('case:concept:name')['concept:name'].shift(1)
    df['prev_activity'] = df['prev_activity'].fillna('START')
    
    print(f"Final DataFrame shape: {df.shape}")
    print(f"Total events with processing times: {len(df)}")
    print(f"Unique activities: {df['concept:name'].nunique()}")
    
    return df


def compute_activity_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute basic statistics for each activity's processing time.
    
    Returns:
        DataFrame with statistics per activity
    """
    print("\n" + "=" * 80)
    print("ACTIVITY PROCESSING TIME STATISTICS")
    print("=" * 80)
    
    stats_list = []
    
    for activity in sorted(df['concept:name'].unique()):
        activity_df = df[df['concept:name'] == activity]
        times = activity_df['processing_time_seconds'].values
        
        if len(times) > 0:
            stats_dict = {
                'activity': activity,
                'count': len(times),
                'mean_seconds': np.mean(times),
                'median_seconds': np.median(times),
                'std_seconds': np.std(times),
                'min_seconds': np.min(times),
                'max_seconds': np.max(times),
                'q25_seconds': np.percentile(times, 25),
                'q75_seconds': np.percentile(times, 75),
            }
            stats_list.append(stats_dict)
    
    stats_df = pd.DataFrame(stats_list)
    
    # Sort by mean processing time
    stats_df = stats_df.sort_values('mean_seconds', ascending=False)
    
    print(f"\nTop 10 activities by mean processing time:")
    print("-" * 80)
    for idx, row in stats_df.head(10).iterrows():
        print(f"{row['activity'][:40]:40} | Count: {row['count']:6} | "
              f"Mean: {row['mean_seconds']/3600:7.2f}h | "
              f"Median: {row['median_seconds']/3600:7.2f}h")
    
    # Save to CSV
    output_path = os.path.join(OUTPUT_DIR, "activity_statistics.csv")
    stats_df.to_csv(output_path, index=False)
    print(f"\nStatistics saved to: {output_path}")
    
    return stats_df


def fit_distribution(data: np.ndarray, dist_name: str) -> Tuple[Any, float]:
    """
    Fit a distribution to data and compute goodness-of-fit.
    
    Args:
        data: Array of processing times
        dist_name: Name of distribution ('lognorm', 'gamma', 'expon', 'norm')
    
    Returns:
        Tuple of (fitted distribution parameters, goodness-of-fit score)
    """
    try:
        if dist_name == 'lognorm':
            # Fit lognormal distribution
            shape, loc, scale = lognorm.fit(data, floc=0)
            params = (shape, loc, scale)
            # Compute KS test
            ks_stat, p_value = stats.kstest(data, lambda x: lognorm.cdf(x, shape, loc, scale))
            
        elif dist_name == 'gamma':
            # Fit gamma distribution
            shape, loc, scale = gamma.fit(data, floc=0)
            params = (shape, loc, scale)
            ks_stat, p_value = stats.kstest(data, lambda x: gamma.cdf(x, shape, loc, scale))
            
        elif dist_name == 'expon':
            # Fit exponential distribution
            loc, scale = expon.fit(data, floc=0)
            params = (loc, scale)
            ks_stat, p_value = stats.kstest(data, lambda x: expon.cdf(x, loc, scale))
            
        elif dist_name == 'norm':
            # Fit normal distribution
            loc, scale = norm.fit(data)
            params = (loc, scale)
            ks_stat, p_value = stats.kstest(data, lambda x: norm.cdf(x, loc, scale))
            
        else:
            raise ValueError(f"Unknown distribution: {dist_name}")
        
        # Use negative KS statistic as goodness-of-fit (lower is better)
        return params, ks_stat
        
    except Exception as e:
        print(f"Error fitting {dist_name}: {e}")
        return None, float('inf')


def analyze_activity_distributions(df: pd.DataFrame, top_n: int = 10):
    """
    Fit probability distributions to processing times of top activities.
    
    Creates visualizations showing:
    - Histogram of actual processing times
    - Fitted probability distribution curves
    - Distribution parameters and goodness-of-fit
    
    Args:
        df: DataFrame with events and processing times
        top_n: Number of top activities to analyze
    """
    print("\n" + "=" * 80)
    print(f"PROBABILITY DISTRIBUTION FITTING (Top {top_n} activities)")
    print("=" * 80)
    
    # Get top activities by count
    activity_counts = df['concept:name'].value_counts().head(top_n)
    
    results = []
    
    for activity in activity_counts.index:
        print(f"\nAnalyzing: {activity}")
        print("-" * 80)
        
        activity_df = df[df['concept:name'] == activity]
        times = activity_df['processing_time_seconds'].values
        
        if len(times) < 30:
            print(f"Skipping {activity} - insufficient data ({len(times)} samples)")
            continue
        
        # Try fitting different distributions
        best_dist = None
        best_params = None
        best_score = float('inf')
        
        fit_results = {}
        
        for dist_name in DISTRIBUTIONS_TO_FIT:
            params, score = fit_distribution(times, dist_name)
            if params is not None:
                fit_results[dist_name] = (params, score)
                if score < best_score:
                    best_score = score
                    best_dist = dist_name
                    best_params = params
        
        print(f"Best fit: {best_dist} (KS statistic: {best_score:.4f})")
        
        # Print all distribution results
        for dist_name, (params, score) in fit_results.items():
            if dist_name == best_dist:
                print(f"  ✓ {dist_name:12} | KS: {score:.4f} | params: {params}")
            else:
                print(f"    {dist_name:12} | KS: {score:.4f} | params: {params}")
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot histogram
        times_hours = times / 3600  # Convert to hours for better readability
        ax.hist(times_hours, bins=50, density=True, alpha=0.6, color='skyblue', 
                edgecolor='black', label='Actual data')
        
        # Plot fitted distributions
        x = np.linspace(times_hours.min(), times_hours.max(), 1000)
        x_seconds = x * 3600  # Convert back to seconds for distribution
        
        colors = {'lognorm': 'red', 'gamma': 'green', 'expon': 'orange', 'norm': 'purple'}
        
        for dist_name, (params, score) in fit_results.items():
            if dist_name == 'lognorm':
                shape, loc, scale = params
                y = lognorm.pdf(x_seconds, shape, loc, scale) * 3600  # Scale for hours
                label = f'Lognormal (KS={score:.3f})'
                linestyle = '--' if dist_name == best_dist else ':'
                linewidth = 2.5 if dist_name == best_dist else 1.5
                ax.plot(x, y, color=colors[dist_name], linestyle=linestyle, 
                       linewidth=linewidth, label=label)
                
            elif dist_name == 'gamma':
                shape, loc, scale = params
                y = gamma.pdf(x_seconds, shape, loc, scale) * 3600
                label = f'Gamma (KS={score:.3f})'
                linestyle = '--' if dist_name == best_dist else ':'
                linewidth = 2.5 if dist_name == best_dist else 1.5
                ax.plot(x, y, color=colors[dist_name], linestyle=linestyle,
                       linewidth=linewidth, label=label)
                
            elif dist_name == 'expon':
                loc, scale = params
                y = expon.pdf(x_seconds, loc, scale) * 3600
                label = f'Exponential (KS={score:.3f})'
                linestyle = '--' if dist_name == best_dist else ':'
                linewidth = 2.5 if dist_name == best_dist else 1.5
                ax.plot(x, y, color=colors[dist_name], linestyle=linestyle,
                       linewidth=linewidth, label=label)
                
            elif dist_name == 'norm':
                loc, scale = params
                y = norm.pdf(x_seconds, loc, scale) * 3600
                label = f'Normal (KS={score:.3f})'
                linestyle = '--' if dist_name == best_dist else ':'
                linewidth = 2.5 if dist_name == best_dist else 1.5
                ax.plot(x, y, color=colors[dist_name], linestyle=linestyle,
                       linewidth=linewidth, label=label)
        
        ax.set_xlabel('Processing Time (hours)', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title(f'Processing Time Distribution: {activity}\n'
                    f'(n={len(times)}, Best fit: {best_dist})', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Save plot
        safe_filename = activity.replace('/', '_').replace(' ', '_')
        plot_path = os.path.join(PLOTS_DIR, f"dist_{safe_filename}.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to: {plot_path}")
        
        # Store results
        results.append({
            'activity': activity,
            'count': len(times),
            'best_distribution': best_dist,
            'best_params': str(best_params),
            'ks_statistic': best_score,
            'mean_hours': np.mean(times) / 3600,
            'median_hours': np.median(times) / 3600,
        })
    
    # Save distribution fitting results
    results_df = pd.DataFrame(results)
    output_path = os.path.join(OUTPUT_DIR, "distribution_fitting_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\n\nDistribution fitting results saved to: {output_path}")
    
    return results_df


def prepare_ml_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare features for machine learning models.
    
    Features include:
    - Activity name (encoded)
    - Previous activity (encoded)
    - Hour of day
    - Day of week
    - Month
    - Event position in trace
    - Case attributes (if available)
    
    Returns:
        Tuple of (features DataFrame, target Series, feature names)
    """
    print("\n" + "=" * 80)
    print("PREPARING MACHINE LEARNING FEATURES")
    print("=" * 80)
    
    feature_df = df.copy()
    
    # Target variable (log-transformed for better model performance)
    target = np.log1p(feature_df['processing_time_seconds'])
    
    # Select features
    features = []
    feature_names = []
    
    # Encode categorical features
    label_encoders = {}
    
    # Activity name
    le_activity = LabelEncoder()
    features.append(le_activity.fit_transform(feature_df['concept:name']))
    feature_names.append('activity_encoded')
    label_encoders['activity'] = le_activity
    
    # Previous activity
    le_prev_activity = LabelEncoder()
    features.append(le_prev_activity.fit_transform(feature_df['prev_activity']))
    feature_names.append('prev_activity_encoded')
    label_encoders['prev_activity'] = le_prev_activity
    
    # Temporal features
    features.append(feature_df['hour_of_day'].values)
    feature_names.append('hour_of_day')
    
    features.append(feature_df['day_of_week'].values)
    feature_names.append('day_of_week')
    
    features.append(feature_df['month'].values)
    feature_names.append('month')
    
    # Position in trace
    features.append(feature_df['event_position'].values)
    feature_names.append('event_position')
    
    # Create feature matrix
    X = np.column_stack(features)
    X_df = pd.DataFrame(X, columns=feature_names)
    
    print(f"Feature matrix shape: {X_df.shape}")
    print(f"Features: {feature_names}")
    print(f"Target (log-transformed processing time) shape: {target.shape}")
    
    return X_df, target, feature_names


def train_ml_models(X: pd.DataFrame, y: pd.Series, feature_names: List[str]) -> Dict[str, Any]:
    """
    Train and evaluate machine learning models for processing time prediction.
    
    Args:
        X: Feature matrix
        y: Target variable (log-transformed processing time)
        feature_names: List of feature names
    
    Returns:
        Dictionary with model results
    """
    if not SKLEARN_AVAILABLE:
        print("\n" + "=" * 80)
        print("SKIPPING ML MODELS - scikit-learn not available")
        print("=" * 80)
        return {}
    
    print("\n" + "=" * 80)
    print("TRAINING MACHINE LEARNING MODELS")
    print("=" * 80)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    results = {}
    
    for model_name, model in ML_MODELS.items():
        print(f"\n{'-'*80}")
        print(f"Training: {model_name}")
        print(f"{'-'*80}")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Transform back from log scale for interpretable metrics
        y_train_actual = np.expm1(y_train)
        y_train_pred_actual = np.expm1(y_train_pred)
        y_test_actual = np.expm1(y_test)
        y_test_pred_actual = np.expm1(y_test_pred)
        
        # Compute metrics on original scale
        train_mae = mean_absolute_error(y_train_actual, y_train_pred_actual)
        train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred_actual))
        train_r2 = r2_score(y_train, y_train_pred)  # R² on log scale
        
        test_mae = mean_absolute_error(y_test_actual, y_test_pred_actual)
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred_actual))
        test_r2 = r2_score(y_test, y_test_pred)
        
        print(f"\nTraining Performance:")
        print(f"  MAE:  {train_mae/3600:8.2f} hours ({train_mae:10.0f} seconds)")
        print(f"  RMSE: {train_rmse/3600:8.2f} hours ({train_rmse:10.0f} seconds)")
        print(f"  R²:   {train_r2:8.4f}")
        
        print(f"\nTest Performance:")
        print(f"  MAE:  {test_mae/3600:8.2f} hours ({test_mae:10.0f} seconds)")
        print(f"  RMSE: {test_rmse/3600:8.2f} hours ({test_rmse:10.0f} seconds)")
        print(f"  R²:   {test_r2:8.4f}")
        
        # Feature importance (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print(f"\nFeature Importances:")
            for i in range(min(10, len(feature_names))):
                idx = indices[i]
                print(f"  {i+1}. {feature_names[idx]:25} {importances[idx]:.4f}")
            
            # Plot feature importances
            fig, ax = plt.subplots(figsize=(10, 6))
            top_n = min(15, len(feature_names))
            top_indices = indices[:top_n]
            ax.barh(range(top_n), importances[top_indices])
            ax.set_yticks(range(top_n))
            ax.set_yticklabels([feature_names[i] for i in top_indices])
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_title(f'Feature Importances - {model_name}', fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            plt.tight_layout()
            
            plot_path = os.path.join(PLOTS_DIR, f"feature_importance_{model_name.replace(' ', '_')}.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\nFeature importance plot saved to: {plot_path}")
        
        results[model_name] = {
            'model': model,
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
        }
    
    # Create comparison visualization
    if results:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        model_names = list(results.keys())
        test_maes = [results[m]['test_mae']/3600 for m in model_names]
        test_rmses = [results[m]['test_rmse']/3600 for m in model_names]
        test_r2s = [results[m]['test_r2'] for m in model_names]
        
        # MAE comparison
        axes[0].bar(range(len(model_names)), test_maes, color='skyblue', edgecolor='black')
        axes[0].set_xticks(range(len(model_names)))
        axes[0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0].set_ylabel('MAE (hours)', fontsize=11)
        axes[0].set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # RMSE comparison
        axes[1].bar(range(len(model_names)), test_rmses, color='lightcoral', edgecolor='black')
        axes[1].set_xticks(range(len(model_names)))
        axes[1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1].set_ylabel('RMSE (hours)', fontsize=11)
        axes[1].set_title('Root Mean Squared Error', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # R² comparison
        axes[2].bar(range(len(model_names)), test_r2s, color='lightgreen', edgecolor='black')
        axes[2].set_xticks(range(len(model_names)))
        axes[2].set_xticklabels(model_names, rotation=45, ha='right')
        axes[2].set_ylabel('R²', fontsize=11)
        axes[2].set_title('R² Score', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3, axis='y')
        axes[2].set_ylim([0, 1])
        
        plt.suptitle('Machine Learning Model Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        plot_path = os.path.join(PLOTS_DIR, "ml_model_comparison.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n\nModel comparison plot saved to: {plot_path}")
    
    # Save results to CSV
    results_list = []
    for model_name, metrics in results.items():
        results_list.append({
            'model': model_name,
            'train_mae_hours': metrics['train_mae'] / 3600,
            'train_rmse_hours': metrics['train_rmse'] / 3600,
            'train_r2': metrics['train_r2'],
            'test_mae_hours': metrics['test_mae'] / 3600,
            'test_rmse_hours': metrics['test_rmse'] / 3600,
            'test_r2': metrics['test_r2'],
        })
    
    results_df = pd.DataFrame(results_list)
    output_path = os.path.join(OUTPUT_DIR, "ml_model_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nML model results saved to: {output_path}")
    
    return results


def generate_summary_report(stats_df: pd.DataFrame, dist_results: pd.DataFrame, 
                           ml_results: Dict[str, Any]):
    """
    Generate a comprehensive summary report.
    """
    print("\n" + "=" * 80)
    print("GENERATING SUMMARY REPORT")
    print("=" * 80)
    
    report_path = os.path.join(OUTPUT_DIR, "processing_time_analysis_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ACTIVITY PROCESSING TIME ANALYSIS REPORT\n")
        f.write("BPI Challenge 2017 Dataset\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Section 1: Overview
        f.write("-" * 80 + "\n")
        f.write("1. OVERVIEW\n")
        f.write("-" * 80 + "\n\n")
        f.write(f"Total activities analyzed: {len(stats_df)}\n")
        f.write(f"Total events with processing times: {stats_df['count'].sum()}\n\n")
        
        # Section 2: Top activities
        f.write("-" * 80 + "\n")
        f.write("2. TOP 10 ACTIVITIES BY MEAN PROCESSING TIME\n")
        f.write("-" * 80 + "\n\n")
        for idx, row in stats_df.head(10).iterrows():
            f.write(f"{row['activity']}\n")
            f.write(f"  Count:  {row['count']:6}\n")
            f.write(f"  Mean:   {row['mean_seconds']/3600:7.2f} hours\n")
            f.write(f"  Median: {row['median_seconds']/3600:7.2f} hours\n")
            f.write(f"  Std:    {row['std_seconds']/3600:7.2f} hours\n\n")
        
        # Section 3: Distribution fitting
        if not dist_results.empty:
            f.write("-" * 80 + "\n")
            f.write("3. PROBABILITY DISTRIBUTION FITTING RESULTS\n")
            f.write("-" * 80 + "\n\n")
            f.write("Best-fit distributions for top activities:\n\n")
            for idx, row in dist_results.iterrows():
                f.write(f"{row['activity']}\n")
                f.write(f"  Best distribution: {row['best_distribution']}\n")
                f.write(f"  Parameters: {row['best_params']}\n")
                f.write(f"  KS statistic: {row['ks_statistic']:.4f}\n")
                f.write(f"  Mean: {row['mean_hours']:.2f} hours\n")
                f.write(f"  Median: {row['median_hours']:.2f} hours\n\n")
        
        # Section 4: ML results
        if ml_results:
            f.write("-" * 80 + "\n")
            f.write("4. MACHINE LEARNING MODEL RESULTS\n")
            f.write("-" * 80 + "\n\n")
            f.write("Test set performance:\n\n")
            for model_name, metrics in ml_results.items():
                f.write(f"{model_name}:\n")
                f.write(f"  MAE:  {metrics['test_mae']/3600:7.2f} hours\n")
                f.write(f"  RMSE: {metrics['test_rmse']/3600:7.2f} hours\n")
                f.write(f"  R²:   {metrics['test_r2']:7.4f}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("-" * 80 + "\n")
    
    print(f"Summary report saved to: {report_path}")


def main():
    """
    Main execution function.
    """
    print("\n" + "=" * 80)
    print("ACTIVITY PROCESSING TIME ANALYSIS")
    print("BPI Challenge 2017")
    print("=" * 80)
    
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print(f"\nERROR: Data file not found: {DATA_PATH}")
        print("Please ensure the BPI Challenge 2017 XES file is in the data/ directory.")
        return
    
    # Load and preprocess data
    df = load_and_preprocess_data(DATA_PATH)
    
    # Compute basic statistics
    stats_df = compute_activity_statistics(df)
    
    # Analyze probability distributions
    dist_results = analyze_activity_distributions(df, top_n=10)
    
    # Prepare features and train ML models
    if SKLEARN_AVAILABLE:
        X, y, feature_names = prepare_ml_features(df)
        ml_results = train_ml_models(X, y, feature_names)
    else:
        ml_results = {}
        print("\n" + "=" * 80)
        print("NOTE: scikit-learn not installed. ML analysis skipped.")
        print("To enable ML models, install scikit-learn: pip install scikit-learn")
        print("=" * 80)
    
    # Generate summary report
    generate_summary_report(stats_df, dist_results, ml_results)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print(f"Plots saved to: {PLOTS_DIR}/")
    print("\nGenerated files:")
    print(f"  - activity_statistics.csv")
    print(f"  - distribution_fitting_results.csv")
    if ml_results:
        print(f"  - ml_model_results.csv")
    print(f"  - processing_time_analysis_report.txt")
    print(f"  - Various plots in {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
