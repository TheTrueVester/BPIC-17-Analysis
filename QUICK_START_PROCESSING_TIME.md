# Quick Start Guide: Activity Processing Time Analysis

## Prerequisites

```bash
# Ensure you have Python 3.8+ and pip installed
python3 --version

# Install dependencies
pip install -r requirements.txt
```

## Running the Analysis

```bash
# Run the analysis script
python3 activity_processing_time_analysis.py
```

Expected runtime: 3-5 minutes on a modern machine

## What It Does

1. **Loads data** from `data/BPI_Challenge_2017.xes.gz`
2. **Calculates processing times** (inter-event times for each activity)
3. **Fits probability distributions** to top 10 activities:
   - Tests: Lognormal, Gamma, Exponential, Normal
   - Reports best-fit distribution with parameters
   - Generates histogram plots with fitted curves
4. **Trains ML models** to predict processing times:
   - Random Forest (typically best: R² ≈ 0.60)
   - Gradient Boosting
   - Ridge Regression
   - Uses features: activity type, previous activity, time, position
5. **Generates outputs**:
   - CSV files with statistics and results
   - PNG plots (distributions, feature importance, model comparison)
   - Comprehensive text report

## Viewing Results

All results are saved to `results/` directory:

```bash
# View summary report
cat results/processing_time_analysis_report.txt

# View distribution fitting results
cat results/distribution_fitting_results.csv

# View ML model performance
cat results/ml_model_results.csv

# View plots
ls results/processing_time_plots/
```

## Example Output

### Distribution Fitting
```
Activity: W_Call after offers
Best fit: lognorm (KS statistic: 0.1958)
Mean: 30.13 hours
Median: 0.01 hours
```

### ML Model Performance
```
Random Forest:
  MAE:  9.74 hours
  RMSE: 40.83 hours
  R²:   0.6032
```

## Interpreting Results

### Probability Distributions
- **KS statistic < 0.2**: Good fit
- **Lognormal**: Common for activities with occasional long delays
- **Gamma**: Good for activities with moderate variability
- Use fitted distributions for simulation and capacity planning

### ML Models
- **R² ≈ 0.60**: Model explains ~60% of variance
- **MAE ≈ 9.7 hours**: Average prediction error
- **Top features**: Activity type (58%), previous activity (23%)
- Use for real-time predictions and process optimization

## Customization

Edit `activity_processing_time_analysis.py` to:
- Analyze more/fewer activities (`top_n` parameter)
- Add custom ML models
- Include additional features
- Change output directories

See [ACTIVITY_PROCESSING_TIME_README.md](ACTIVITY_PROCESSING_TIME_README.md) for detailed documentation.

## Troubleshooting

**Issue**: "Data file not found"
- Ensure `data/BPI_Challenge_2017.xes.gz` exists

**Issue**: "scikit-learn not available"
- Run: `pip install scikit-learn>=1.0.0`

**Issue**: Out of memory
- Reduce sample size in the script
- Analyze fewer activities

## Next Steps

1. Review the generated report and plots
2. Identify activities with high processing times
3. Use distributions for process simulation
4. Deploy ML model for real-time prediction
5. Extend analysis with custom features

For questions, see the full documentation or open an issue.
