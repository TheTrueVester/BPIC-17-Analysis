# BPIC 2017 Process Mining Analysis

Comprehensive process discovery, conformance checking, and model evaluation tool for the BPI Challenge 2017 dataset.

## Features

- **Process Discovery Algorithms**: Inductive Miner (with variants), Heuristics Miner, Alpha Miner
- **Model Formats**: Petri net and BPMN (automatic conversion)
- **Preprocessing**: Activity clustering, outcome filtering, outlier removal, noise filtering
- **Conformance Checking**: Token-based replay (fitness & precision), optional alignment-based metrics
- **Model Evaluation**: Flexibility, simplicity, and structural metrics
- **Parallel Processing**: True CPU parallelism for multiple coverage experiments
- **Visualization**: Automatic Petri net and BPMN generation with unique filenames

## Requirements

### Software
- Python 3.8 or higher
- Graphviz (for Petri net visualizations)

### Python Packages
See `requirements.txt` for complete list. Key dependencies:
- pm4py
- pandas
- numpy
- matplotlib
- graphviz

## Installation

### 1. Install Python Dependencies

```bash
# Using virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 2. Install Graphviz

**Windows:**
1. Download from: https://graphviz.org/download/
2. Install and add to PATH
3. Verify: `dot -V`

**macOS:**
```bash
brew install graphviz
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install graphviz
```

### 3. Prepare Data

Place your event log file in the `data/` folder:
```
data/BPI_Challenge_2017.xes.gz
```

## Configuration

Edit the configuration section at the top of `bpic17_reworked.py` (lines 54-120):

### Discovery Algorithm
```python
DISCOVERY_ALGO = "inductive"  # Options: "inductive", "heuristics", "alpha"
```

### Inductive Miner Variant (only for inductive)
```python
INDUCTIVE_MINER_VARIANT = "IMf"  # Options: "IM", "IMf", "IMd"
# "IM"  - Standard (balanced)
# "IMf" - Infrequent (highest precision, filters noise)
# "IMd" - Directly-follows (good precision)
```

### Noise Filtering
```python
NOISE_THRESHOLD = 0.05  # 0.0 to 1.0
# 0.0  - Disabled (keep all activities)
# 0.05 - Remove activities in < 5% of cases (recommended)
# 0.10 - Aggressive filtering
```

### Outcome Filtering
```python
PRECISION_LOG_MODE = "all"  # Options: "all", "accepted", "non_cancelled"
# "all"           - All complete cases
# "accepted"      - Only successfully completed cases
# "non_cancelled" - Exclude cancelled/denied cases
```

### Conformance Options
```python
ALIGNMENTS_ENABLED = False  # True = compute alignment metrics (slow)
EVAL_ON_DISCOVERY_LOG = False  # True = evaluate on filtered log only
```

### Output
```python
OUTPUT_DIR = "petri_nets"  # Folder for Petri net visualizations
BPMN_DIR = "bpmn_models"   # Folder for BPMN visualizations
RESULTS_DIR = "results"    # Folder for text result summaries
```

## Running the Analysis

### Basic Usage

```bash
python bpic17_reworked.py
```

### Example Configurations

**1. Maximum Precision (recommended for BPIC 2017)**
```python
DISCOVERY_ALGO = "inductive"
INDUCTIVE_MINER_VARIANT = "IMf"
NOISE_THRESHOLD = 0.05
PRECISION_LOG_MODE = "accepted"
ALIGNMENTS_ENABLED = False
```

**2. Balanced Fitness/Precision**
```python
DISCOVERY_ALGO = "inductive"
INDUCTIVE_MINER_VARIANT = "IM"
NOISE_THRESHOLD = 0.0
PRECISION_LOG_MODE = "all"
```

**3. Fast Exploration (Heuristics)**
```python
DISCOVERY_ALGO = "heuristics"
NOISE_THRESHOLD = 0.0
PRECISION_LOG_MODE = "all"
ALIGNMENTS_ENABLED = False
```

## Output

### Console Output

The script provides detailed progress information:
- Preprocessing statistics (cases dropped, filtering results)
- Basic statistics (cases, events, variants, durations)
- Service times per activity
- For each coverage experiment:
  - Discovery algorithm and variant
  - Token-based fitness & precision
  - Alignment metrics (if enabled)
  - Simplicity and flexibility scores
- Summary table comparing all experiments
- Recommended model selection

### Petri Net Visualizations

Generated in `petri_nets/<algorithm>/` folder with filenames including:
- Algorithm name (and variant for Inductive Miner)
- Noise threshold (percentage)
- Coverage percentage
- Number of traces used for discovery
- Timestamp

Example: `petri_nets/inductive_IMf/petri_inductive_IMf_noise5_cov60_traces1234_20250118_151513.png`

This means: Inductive Miner variant IMf, with 5% noise filtering, 60% variant coverage, discovered from 1234 traces, generated at 15:15:13 on 2025-01-18.

### BPMN Visualizations

Generated in `bpmn_models/<algorithm>/` folder with same naming convention:
- Automatically converted from Petri nets
- Same filename pattern as Petri nets but prefixed with "bpmn_"
- BPMN provides alternative business-friendly notation

Example: `bpmn_models/inductive_IMf/bpmn_inductive_IMf_noise5_cov60_traces1234_20250118_151513.png`

### Text Result Files

Generated in `results/<algorithm>/` folder with complete configuration and results:
- Full configuration used for the experiment
- All coverage experiment results
- Recommended model selection
- Timestamp for tracking

Example: `results/inductive_IMf/results_inductive_IMf_noise5_20250118_151513.txt`

Contains:
- Discovery algorithm and all parameters
- Token-based and alignment-based metrics (if enabled)
- Simplicity and flexibility scores
- Recommended model with justification

### Metrics Explained

**Token-Based Replay (TBR)**
- **Fitness**: % of traces that can replay without errors (0-1)
- **Precision**: How much behavior in model is actually in log (0-1)

**Alignment-Based** (if enabled)
- More accurate but computationally expensive
- Same interpretation as TBR

**Simplicity**
- **Nodes**: Total places + transitions (lower = simpler)
- **Density**: Arcs / nodes (lower = simpler)

**Flexibility**
- Score = density × (1 + invisible_ratio)
- Higher = more flexible/complex model

## Troubleshooting

### "Graphviz not found"
- Ensure Graphviz is installed and in PATH
- Restart terminal after installation

### "No cases remaining after preprocessing"
- Loosen filtering criteria (set `PRECISION_LOG_MODE = "all"`)
- Reduce `NOISE_THRESHOLD`
- Check if data file is correct

### Out of Memory
- Reduce variant coverage values (use smaller list like `[0.30, 0.10]`)
- Disable alignments: `ALIGNMENTS_ENABLED = False`
- Reduce `MAX_PRECISION_SAMPLE_SIZE`

### Process hangs/takes too long
- Disable alignments (most expensive operation)
- Use fewer coverage values
- Reduce parallel workers by modifying `max_workers` in code

## Understanding the Workflow

1. **Load Data**: Reads XES log from `data/` folder
2. **Preprocess**:
   - Drop missing timestamps and duplicates
   - Filter to complete cases
   - Cluster activities into macro-activities
   - Apply outcome filtering
   - Remove outliers and short cases
3. **Statistics**: Compute basic and extended statistics
4. **Discovery Loop** (parallel for each coverage):
   - Filter log by variant coverage
   - Apply noise filter
   - Discover process model
   - Evaluate conformance (fitness & precision)
   - Compute structural metrics
   - Save Petri net visualization
5. **Recommendation**: Select best model balancing fitness (~80%) and precision

## Advanced Usage

### Modify Coverage Values

In `main()` function (line ~968):
```python
coverages = [1.0, 0.60, 0.30, 0.10, 0.01]  # Customize this list
```

### Change Recommended Model Criteria

Modify `FITTING_TOLERANCE_MIN` and `FITTING_TOLERANCE_MAX` (lines 100-102):
```python
FITTING_TOLERANCE_MIN = 70.0  # Lower = prioritize precision over fitness
FITTING_TOLERANCE_MAX = 90.0  # Higher = allow more fitness variance
```

### Disable Service Time Calculation

Comment out line ~954 in `main()`:
```python
# compute_service_times(clean_log)
```

## Performance Tips

- **Fast iteration**: Disable alignments, use small coverage list
- **High precision**: Use IMf variant, increase noise threshold, use "accepted" mode
- **High fitness**: Use IM variant, no noise filtering, use "all" mode
- **Production run**: Enable alignments for publication-quality metrics

## File Structure

```
BPPSO/
├── bpic17_reworked.py      # Main analysis script
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── data/
│   └── BPI_Challenge_2017.xes.gz  # Event log
├── petri_nets/             # Petri net output folder (created automatically)
│   ├── inductive_IMf/      # Inductive Miner IMf variant
│   ├── inductive_IMd/      # Inductive Miner IMd variant
│   ├── alpha/              # Alpha Miner
│   └── heuristics/         # Heuristics Miner
├── bpmn_models/            # BPMN output folder (created automatically)
│   ├── inductive_IMf/      # BPMN models from IMf
│   ├── inductive_IMd/      # BPMN models from IMd
│   ├── alpha/              # BPMN models from Alpha
│   └── heuristics/         # BPMN models from Heuristics
├── results/                # Results folder (created automatically)
│   ├── inductive_IMf/      # Text summaries for IMf
│   ├── inductive_IMd/      # Text summaries for IMd
│   ├── alpha/              # Text summaries for Alpha
│   └── heuristics/         # Text summaries for Heuristics
└── venv/                   # Virtual environment (optional)
```

## Citation

If using this code for academic work, please cite:
- BPI Challenge 2017: https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b
- PM4Py: https://pm4py.fit.fraunhofer.de/

## License

This project is for academic/research purposes.

## Contact

For issues or questions, please create an issue in the repository.
