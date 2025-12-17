# Activity Processing Time Analysis

This script provides comprehensive analysis of activity processing times in the BPI Challenge 2017 dataset using two complementary approaches:

## Overview

### 1. Probability Distribution Fitting
Fits multiple probability distributions to historical activity processing times to model uncertainty and variability:
- **Distributions tested**: Lognormal, Gamma, Exponential, Normal
- **Goodness-of-fit**: Kolmogorov-Smirnov (KS) test statistics
- **Visualizations**: Histograms with fitted distribution curves
- **Use case**: Simulation, process modeling, capacity planning

### 2. Machine Learning Point Estimation
Trains regression models to predict processing times based on contextual features:
- **Models**: Random Forest, Gradient Boosting, Ridge Regression
- **Features**: Activity type, previous activity, temporal context, event position
- **Metrics**: MAE, RMSE, R² score
- **Use case**: Dynamic predictions, process optimization, resource allocation

## Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- pm4py >= 2.7.0
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.0.0
- scipy (installed with pm4py)
- matplotlib (installed with pm4py)

## Usage

### Basic Usage

```bash
python activity_processing_time_analysis.py
```

The script will:
1. Load the BPI Challenge 2017 dataset from `data/BPI_Challenge_2017.xes.gz`
2. Compute processing times (inter-event times)
3. Fit probability distributions to top activities
4. Train and evaluate ML models
5. Generate comprehensive reports and visualizations

### Processing Time Definition

**Processing time** is defined as the time elapsed between consecutive events in the same case (inter-event time). This represents:
- Time spent on an activity
- Waiting time before the next activity
- Overall throughput time between process steps

## Output Files

### Directory Structure

```
results/
├── activity_statistics.csv              # Basic statistics per activity
├── distribution_fitting_results.csv     # Best-fit distributions
├── ml_model_results.csv                 # ML model performance metrics
├── processing_time_analysis_report.txt  # Comprehensive text report
└── processing_time_plots/               # Visualizations
    ├── dist_*.png                       # Distribution plots (one per activity)
    ├── feature_importance_*.png         # Feature importance plots
    └── ml_model_comparison.png          # Model comparison chart
```

### 1. Activity Statistics (`activity_statistics.csv`)

Basic descriptive statistics for each activity:
- Count: Number of occurrences
- Mean, Median, Std: Central tendency and spread
- Min, Max: Range
- Q25, Q75: Quartiles

**Sample output:**
```csv
activity,count,mean_seconds,median_seconds,std_seconds,...
W_Validate application,209496,21333.4,8.44,79668.3,...
W_Call after offers,191092,108480.9,31.5,236761.1,...
```

### 2. Distribution Fitting Results (`distribution_fitting_results.csv`)

Best-fit distributions for top activities:
- Best distribution type (lognorm, gamma, expon, norm)
- Distribution parameters
- KS statistic (goodness-of-fit, lower is better)
- Mean and median processing times

**Sample output:**
```csv
activity,count,best_distribution,best_params,ks_statistic,mean_hours,median_hours
W_Validate application,209496,gamma,"(0.093, 0, 228908.3)",0.1952,5.93,0.002
W_Call after offers,191092,lognorm,"(7.297, 0, 15.817)",0.1958,30.13,0.009
```

**Interpretation:**
- **KS statistic < 0.2**: Good fit
- **KS statistic 0.2-0.3**: Acceptable fit
- **KS statistic > 0.3**: Poor fit (consider alternative distributions)

### 3. ML Model Results (`ml_model_results.csv`)

Performance metrics for each ML model:
- Train/Test MAE: Mean Absolute Error (hours)
- Train/Test RMSE: Root Mean Squared Error (hours)
- Train/Test R²: Coefficient of determination (0-1, higher is better)

**Sample output:**
```csv
model,train_mae_hours,train_rmse_hours,train_r2,test_mae_hours,test_rmse_hours,test_r2
Random Forest,9.65,40.22,0.603,9.74,40.83,0.603
Gradient Boosting,10.06,42.40,0.594,10.18,43.21,0.594
Ridge Regression,14.15,67.06,0.018,14.15,66.97,0.019
```

**Interpretation:**
- **R² = 0.603**: Model explains ~60% of variance in processing times
- **MAE ≈ 9.7 hours**: Average prediction error
- **Lower MAE/RMSE**: Better predictions
- **Random Forest** typically performs best for this dataset

### 4. Summary Report (`processing_time_analysis_report.txt`)

Comprehensive text report including:
- Overview statistics
- Top 10 activities by mean processing time
- Distribution fitting results with parameters
- ML model performance summary

### 5. Visualizations (`processing_time_plots/`)

#### Distribution Plots (`dist_*.png`)
For each analyzed activity:
- Histogram of actual processing times
- Overlaid fitted distribution curves
- Best-fit distribution highlighted
- Includes sample size and KS statistics

**What to look for:**
- How well the fitted curve matches the histogram
- Skewness and tail behavior
- Multiple modes (if present)

#### Feature Importance (`feature_importance_*.png`)
For tree-based models (Random Forest, Gradient Boosting):
- Bar chart showing relative importance of each feature
- Helps understand which features drive predictions

**Typical importance ranking:**
1. **activity_encoded** (50-60%): Activity type is most predictive
2. **prev_activity_encoded** (20-30%): Previous activity matters
3. **event_position** (15-20%): Position in case affects timing
4. **hour_of_day** (2-5%): Time of day has minor effect
5. **day_of_week, month** (<1%): Minimal impact

#### Model Comparison (`ml_model_comparison.png`)
Side-by-side comparison of:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)  
- R² (Coefficient of Determination)

Helps select the best model for deployment.

## Understanding the Results

### Probability Distributions

Each distribution type models processing times differently:

**1. Lognormal Distribution**
- **Best for**: Activities with right-skewed distributions (long tail)
- **Parameters**: (shape, loc, scale)
- **Use case**: Most common for process times with occasional long delays
- **Example**: `W_Call after offers` - many fast calls, few very long ones

**2. Gamma Distribution**
- **Best for**: Activities with moderate skewness
- **Parameters**: (shape, loc, scale)
- **Use case**: Processes with waiting times, service times
- **Example**: `W_Validate application` - validation can take varying time

**3. Exponential Distribution**
- **Best for**: Memoryless processes (constant rate)
- **Parameters**: (loc, scale)
- **Use case**: Simple waiting times, arrival processes
- **Note**: Rarely best fit for complex process activities

**4. Normal Distribution**
- **Best for**: Symmetric distributions around mean
- **Parameters**: (mean, std)
- **Use case**: Highly controlled, predictable activities
- **Example**: `A_Validating` - standardized validation checks

### Machine Learning Models

**Random Forest**
- **Strengths**: Handles non-linear relationships, robust to outliers
- **Typical performance**: R² ≈ 0.60, MAE ≈ 9.7 hours
- **Best for**: General-purpose prediction with good accuracy

**Gradient Boosting**
- **Strengths**: Sequential improvement, captures complex patterns
- **Typical performance**: R² ≈ 0.59, MAE ≈ 10.2 hours
- **Best for**: When slight accuracy trade-off acceptable for faster training

**Ridge Regression**
- **Strengths**: Simple, interpretable, fast
- **Typical performance**: R² ≈ 0.02, MAE ≈ 14.1 hours
- **Best for**: Baseline comparison (typically underperforms)

### Feature Engineering

The script extracts these contextual features:

1. **activity_encoded**: The activity being performed
2. **prev_activity_encoded**: Previous activity in the case
3. **hour_of_day**: Hour when event occurred (0-23)
4. **day_of_week**: Day of week (0=Monday, 6=Sunday)
5. **month**: Month of year (1-12)
6. **event_position**: Position in the case (1, 2, 3, ...)

**Why these features matter:**
- Activities have inherent processing times
- Previous activities affect readiness for next step
- Time of day affects staff availability
- Event position captures case complexity (later events may take longer)

## Configuration

Edit the script to customize behavior:

```python
# Data path
DATA_PATH = os.path.join("data", "BPI_Challenge_2017.xes.gz")

# Output directories
OUTPUT_DIR = "results"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "processing_time_plots")

# Random seed for reproducibility
RANDOM_STATE = 42

# Distributions to fit
DISTRIBUTIONS_TO_FIT = ['lognorm', 'gamma', 'expon', 'norm']

# ML models (add/remove as needed)
ML_MODELS = {
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, ...),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, ...),
    'Ridge Regression': Ridge(alpha=1.0, ...),
}
```

## Advanced Usage

### Analyzing Specific Activities

Modify `analyze_activity_distributions()` call in `main()`:

```python
# Analyze top 20 activities instead of 10
dist_results = analyze_activity_distributions(df, top_n=20)
```

### Adding Custom Features

Extend `prepare_ml_features()` to include additional features:

```python
# Add case-level features
feature_df['case_length'] = feature_df.groupby('case:concept:name')['event_position'].transform('max')

# Add resource information (if available)
if 'org:resource' in feature_df.columns:
    le_resource = LabelEncoder()
    features.append(le_resource.fit_transform(feature_df['org:resource']))
    feature_names.append('resource_encoded')
```

### Custom ML Models

Add your own models to the `ML_MODELS` dictionary:

```python
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor

ML_MODELS = {
    'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500),
}
```

## Interpretation Guide

### For Simulation and Modeling

Use **probability distributions** to:
1. Generate realistic process times in simulation
2. Model uncertainty in process predictions
3. Perform Monte Carlo analysis
4. Calculate confidence intervals

**Example:**
```python
# Use fitted lognormal distribution for simulation
import scipy.stats as stats

# Get parameters from results
shape, loc, scale = 7.297, 0, 15.817  # W_Call after offers

# Generate 1000 random processing times
simulated_times = stats.lognorm.rvs(shape, loc, scale, size=1000)
```

### For Prediction and Optimization

Use **ML models** to:
1. Predict processing times for new cases
2. Identify bottlenecks in real-time
3. Optimize resource allocation
4. Trigger alerts for delays

**Example workflow:**
1. Train Random Forest model (best performance)
2. For new event, extract features (activity, prev_activity, time, position)
3. Predict processing time
4. If prediction > threshold, alert process owner

### Key Insights from Results

From the BPI Challenge 2017 analysis:

1. **High Variability**: 
   - Processing times vary significantly (high std relative to mean)
   - Lognormal/gamma distributions fit best (right-skewed)
   - Many activities have long tails (occasional extreme delays)

2. **Context Matters**:
   - R² ≈ 0.60 means context explains ~60% of variance
   - Activity type is most important (50-60% importance)
   - Previous activity also significant (20-30%)
   - Temporal features have minor impact (<5%)

3. **Top Time-Consuming Activities**:
   - `A_Cancelled` (median 640 hours): Long cancellation process
   - `W_Call after offers` (mean 30 hours): Follow-up delays
   - `O_Accepted` (mean 48 hours): Acceptance processing

4. **Prediction Accuracy**:
   - MAE ≈ 9.7 hours: Typical prediction error
   - For activities with 30-hour mean, ~30% error
   - Better for short activities, worse for highly variable ones

## Troubleshooting

### Issue: "Data file not found"
**Solution**: Ensure `BPI_Challenge_2017.xes.gz` is in the `data/` directory.

### Issue: "scikit-learn not available"
**Solution**: Install scikit-learn:
```bash
pip install scikit-learn>=1.0.0
```
Script will skip ML analysis if not available.

### Issue: Out of memory
**Solution**: The script processes ~1.2M events. For large datasets:
- Reduce sample size in `train_test_split()` (e.g., sample 10% of data)
- Use simpler models (e.g., only Ridge Regression)
- Analyze fewer activities (`top_n=5`)

### Issue: Plots not generated
**Solution**: Requires matplotlib (installed with pm4py). Check:
```python
import matplotlib
print(matplotlib.__version__)
```

### Issue: Poor distribution fits (high KS statistics)
**Solution**: Processing times may not follow standard distributions. Consider:
- Analyzing subgroups (by resource, time of day, etc.)
- Using mixture models
- Log-transforming times before fitting
- Filtering outliers

## Extending the Analysis

### 1. Add More Distributions

```python
from scipy.stats import weibull_min, burr

DISTRIBUTIONS_TO_FIT = ['lognorm', 'gamma', 'expon', 'norm', 'weibull_min']
```

### 2. Analyze by Subgroups

```python
# Analyze processing times by resource
for resource in df['org:resource'].unique():
    resource_df = df[df['org:resource'] == resource]
    analyze_activity_distributions(resource_df, top_n=5)
```

### 3. Time Series Analysis

```python
# Analyze how processing times change over time
df['date'] = df['time:timestamp'].dt.date
daily_means = df.groupby('date')['processing_time_seconds'].mean()
# Plot trend
```

### 4. Dependency Analysis

```python
# Analyze which activity pairs have longest combined times
df['activity_pair'] = df['prev_activity'] + ' -> ' + df['concept:name']
pair_times = df.groupby('activity_pair')['processing_time_seconds'].mean()
print(pair_times.nlargest(10))
```

## References

- **BPI Challenge 2017**: https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b
- **PM4Py Documentation**: https://pm4py.fit.fraunhofer.de/
- **Scikit-learn**: https://scikit-learn.org/
- **Scipy Statistics**: https://docs.scipy.org/doc/scipy/reference/stats.html

## Citation

If you use this analysis in academic work, please cite:
- The BPI Challenge 2017 dataset (see link above)
- PM4Py library
- Scikit-learn library

## Contact

For questions or issues, please open an issue in the repository.
