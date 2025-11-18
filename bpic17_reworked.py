"""

BPIC 2017 – Process discovery + conformance + parallel evaluation + visualization

Features:
    Preprocessing:
        Drop missing timestamps / duplicates
        Keep only complete cases that reach a final state
        Activity clustering into CL_* macro-activities (optional, controlled by USE_CLUSTERING)
        Outcome-based filtering on CL_* labels (only when clustering is enabled):
            - "all"            -> all complete cases
            - "accepted"       -> traces ending in CL_W_Complete_application or CL_W_Validate_application
            - "non_cancelled"  -> drop traces ending in CL_A_Cancelled or CL_A_Denied
            - "frequent_only"  -> only keep traces whose variant appears more than once
        Remove duration outliers (above 99th percentile)
        Remove cases with < 2 events
    Stats:
        Cases, events, variants, start/end activities
        Case length (amount of events mean and std)
        Case duration (mean and std in seconds and days)
        Case attributes, Event attributes, Categorical event attributes
    Process discovery algorithms:
        Inductive Miner (default)
        Heuristics Miner
        Alpha Miner
    Conformance:
        Token-based fitness (full or restricted log)
        Token-based precision (ETConformance-style) on a sample (max 200 traces (default))
        Optional alignment-based fitness & precision (same sample)
    Structure metrics:
        Simplicity_Nodes   = places + transitions
        Simplicity_Density = arcs / (places + transitions)
        Flexibility score  = density * (1 + invisible_ratio)
    Variant coverage experiments (cov=1.0, 0.60, 0.30, 0.10, 0.01)
    True CPU parallelism for each coverage via ProcessPoolExecutor
    Petri net visualizations saved as PNG per coverage & algorithm
"""

import os
import random
from typing import Any, Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pm4py
from pm4py import stats
from pm4py.util import constants

from pm4py.statistics.service_time.log import get as service_time_log_get
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.objects.log.obj import EventLog


# Config

# Discovery algorithm:
#   "inductive"  -> Inductive Miner (default)
#   "heuristics" -> Heuristics Miner
#   "alpha"      -> Alpha Miner
DISCOVERY_ALGO = "alpha"

# Inductive Miner variant (only used when DISCOVERY_ALGO = "inductive"):
#   "IM"   -> Standard Inductive Miner (balanced fitness/precision)
#   "IMf"  -> Inductive Miner - infrequent (filters noise)
#   "IMd"  -> Inductive Miner - directly-follows
INDUCTIVE_MINER_VARIANT = "IMd"

# Noise filtering threshold (0.0 to 1.0):
#   0.0  -> No filtering (keep all activities) (default)
#   0.05 -> Filter activities appearing in < 5% of cases 
#   0.10 -> More aggressive filtering
NOISE_THRESHOLD = 0.1

# Output directories
OUTPUT_DIR = "petri_nets"   # For Petri net visualizations
BPMN_DIR = "bpmn_models"    # For BPMN visualizations
RESULTS_DIR = "results"     # For text result summaries

# Option to disable alignment-based metrics to save time
ALIGNMENTS_ENABLED = True

# Enable activity clustering:
#   True  -> cluster activities into CL_* macro-activities (default)
#   False -> use original activity labels without clustering
USE_CLUSTERING = False

# Outcome-based filtering mode on labels:
#   "all"              -> all complete cases (default)
#   "accepted"         -> only traces whose last clustered activity is in either:
#                         CL_W_Complete_application or CL_W_Validate_application
#   "non_cancelled"    -> drop traces whose last clustered activity is in either:
#                         CL_A_Cancelled or CL_A_Denied
#   "frequent_only"    -> only keep traces whose variant appears more than once
PRECISION_LOG_MODE = "all"

# Evaluation log choice for conformance:
#   False -> evaluate using full unprocessed log (default)
#   True  -> evaluate using preprocessed log
EVAL_ON_DISCOVERY_LOG = False

# Sampling and threshold constants
MAX_PRECISION_SAMPLE_SIZE = 200
TARGET_FITTING_PERCENTAGE = 80.0
FITTING_TOLERANCE_MIN = 75.0
FITTING_TOLERANCE_MAX = 100.0

# Preprocessing constants
MIN_EVENTS_PER_CASE = 2
DURATION_OUTLIER_PERCENTILE = 0.99

# Column name constants
CASE_ID_COL = "case:concept:name"
ACTIVITY_COL = "concept:name"
TIMESTAMP_COL = "time:timestamp"

# Global seed for reproducibility of random behaviour
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
print(f"[SEED] Global seed set to {GLOBAL_SEED}")

# Validate configuration
if DISCOVERY_ALGO not in ["inductive", "heuristics", "alpha"]:
    raise ValueError(f"Invalid DISCOVERY_ALGO: {DISCOVERY_ALGO}. Must be 'inductive', 'heuristics', or 'alpha'")
if PRECISION_LOG_MODE not in ["all", "accepted", "non_cancelled", "frequent_only"]:
    raise ValueError(f"Invalid PRECISION_LOG_MODE: {PRECISION_LOG_MODE}. Must be 'all', 'accepted', 'non_cancelled', or 'frequent_only'")
if INDUCTIVE_MINER_VARIANT not in ["IM", "IMf", "IMd"]:
    raise ValueError(f"Invalid INDUCTIVE_MINER_VARIANT: {INDUCTIVE_MINER_VARIANT}. Must be 'IM', 'IMf', or 'IMd'")
if not 0.0 <= NOISE_THRESHOLD <= 1.0:
    raise ValueError(f"Invalid NOISE_THRESHOLD: {NOISE_THRESHOLD}. Must be between 0.0 and 1.0")



# 1. Activity clustering inspired by BPI 2017 academic winner

CLUSTERS = {
    "CL_A_Create_Application": [
        "A_Create Application", "A_Submitted", "A_Concept"
    ],
    "CL_W_Complete_application": [
        "W_Complete application", "A_Accepted", "O_Create Offer",
        "O_Created", "O_Sent (mail and online)", "W_Call after offers",
        "A_Complete"
    ],
    "CL_W_Call_incomplete_files": [
        "W_Call incomplete files", "A_Incomplete"
    ],
    "CL_W_Validate_application": [
        "W_Validate application", "A_Validating", "O_Returned"
    ],
    "CL_A_Denied": ["A_Denied", "O_Refused"],
    "CL_A_Cancelled": ["A_Cancelled", "O_Cancelled"],
    "CL_A_Pending": ["O_Accepted", "A_Pending"],
}

def cluster_activities_in_df(df: pd.DataFrame, activity_col: str = "concept:name") -> pd.DataFrame:
    """
    Cluster activities into macro-activities using predefined CLUSTERS mapping.
    Activities not in any cluster remain unchanged.
    """
    mapping = {}
    for cluster_label, acts in CLUSTERS.items():
        for a in acts:
            mapping[a] = cluster_label

    before = df[activity_col].nunique()
    df = df.copy()
    df[activity_col] = df[activity_col].map(mapping).fillna(df[activity_col])
    after = df[activity_col].nunique()

    print(f"[Clustering] Reduced activities {before} → {after}")
    return df

def filter_to_complete_cases(df: pd.DataFrame,
                             case_col: str = "case:concept:name",
                             activity_col: str = "concept:name") -> pd.DataFrame:
    """
    Keep only traces that contain a final application state
    (A_Pending, A_Cancelled, A_Denied or their clustered versions).
    This is applied before clustering (original labels).
    """
    final_states = {"A_Pending", "A_Cancelled", "A_Denied"}
    final_clustered = {"CL_A_Pending", "CL_A_Cancelled", "CL_A_Denied"}

    def is_complete(acts):
        s = set(acts)
        return bool((s & final_states) or (s & final_clustered))

    comp = df.groupby(case_col)[activity_col].agg(is_complete)
    keep = set(comp[comp].index)

    before = df[case_col].nunique()
    df2 = df[df[case_col].isin(keep)]
    after = df2[case_col].nunique()

    if before > 0:
        dropped = before - after
        print(f"[Complete cases filter] Kept {after}/{before} cases ({after/before:.1%}), dropped {dropped} incomplete cases")
    else:
        print(f"[Complete cases filter] Kept {after}/0 (N/A)")
    return df2


def filter_by_outcome_clustered(
    df: pd.DataFrame,
    mode: str = "all",
    case_col: str = "case:concept:name",
    activity_col: str = "concept:name"
) -> pd.DataFrame:
    """
    Outcome-based filtering using CL_* labels (after clustering).

    mode:
            "all"           -> all complete cases (default)
            "accepted"      -> only traces whose last clustered activity is in either:
                               CL_W_Complete_application or CL_W_Validate_application
            "non_cancelled" -> drop traces whose last clustered activity is in either:
                               CL_A_Cancelled or CL_A_Denied
            "frequent_only" -> only keep traces whose variant appears more than once
    """
    if mode == "all":
        return df

    df = df.copy()
    df = df.sort_values([case_col, "time:timestamp"])

    if mode == "frequent_only":
        # Get variant for each case
        from pm4py.statistics.variants.pandas import get as variants_get
        variants_dict = variants_get.get_variants_count(df, parameters={
            constants.PARAMETER_CONSTANT_CASEID_KEY: case_col,
            constants.PARAMETER_CONSTANT_ACTIVITY_KEY: activity_col
        })
        
        # Keep only variants with count > 1
        frequent_variants = {v for v, count in variants_dict.items() if count > 1}
        
        # Map each case to its variant
        case_to_variant = df.groupby(case_col)[activity_col].apply(lambda x: tuple(x))
        keep_cases = set(case_to_variant[case_to_variant.isin(frequent_variants)].index)
        
        before = df[case_col].nunique()
        df = df[df[case_col].isin(keep_cases)]
        after = df[case_col].nunique()
        dropped = before - after
        
        print(f"[Outcome filter: frequent_only] Kept {after}/{before} cases ({after/before:.1%}), dropped {dropped} cases with unique variants")
        return df

    last_evt = df.groupby(case_col).tail(1)[[case_col, activity_col]]

    if mode == "accepted":
        accepted_ends = {
            "CL_W_Complete_application",
            "CL_W_Validate_application",
        }
        keep_cases = set(
            last_evt.loc[last_evt[activity_col].isin(accepted_ends), case_col]
        )
        desc = "only successfully completed / validated applications"
    elif mode == "non_cancelled":
        negative_ends = {
            "CL_A_Cancelled",
            "CL_A_Denied",
        }
        keep_cases = set(
            last_evt.loc[~last_evt[activity_col].isin(negative_ends), case_col]
        )
        desc = "non-cancelled, non-denied clustered cases"
    else:
        raise ValueError(f"Unknown PRECISION_LOG_MODE for clustered filter: {mode}")

    before = df[case_col].nunique()
    df = df[df[case_col].isin(keep_cases)]
    after = df[case_col].nunique()
    dropped = before - after

    print(f"[Outcome filter: {mode}] {desc}: kept {after}/{before} cases ({after/before:.1%}), dropped {dropped} cases")
    return df


# 2. Load & convert


# Load XES event log
def load_log_xes(path: str) -> EventLog:
    """Load an XES event log from file."""
    log = pm4py.read_xes(path)
    print(f"[Load] Loaded log with {len(log)} cases")
    return log


def log_to_dataframe(log: EventLog) -> pd.DataFrame:
    """Convert PM4Py EventLog to pandas DataFrame."""
    df = pm4py.convert_to_dataframe(log)
    print(f"[DataFrame] {df.shape}")
    return df


# Converts a pandas DataFrame to a PM4Py EventLog object
def dataframe_to_event_log(df: pd.DataFrame) -> EventLog:
    """Convert pandas DataFrame to PM4Py EventLog."""
    log = pm4py.convert_to_event_log(df)
    print(f"[EventLog] Reconstructed log with {len(log)} traces")
    return log


# 3. Preprocess (returns df + case_durations)


def preprocess_log_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Winner-inspired preprocessing:
    - drop missing timestamps
    - drop duplicates
    - keep complete cases (original labels)
    - cluster activities into CL_* labels (if USE_CLUSTERING=True)
    - outcome-based filtering on clustered labels (PRECISION_LOG_MODE)
    - compute case durations, remove >99th percentile
    - drop cases with <2 events

    Returns:
        cleaned_df, case_durations (in seconds, AFTER outlier and short case removal)
    """
    print("\n=== PREPROCESSING ===")

    df = df.copy()
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["time:timestamp"])
    dropped_timestamps = before - len(df)
    print(f"[Timestamp filter] Dropped {dropped_timestamps} events with missing timestamps")

    before = len(df)
    df = df.drop_duplicates()
    dropped_duplicates = before - len(df)
    print(f"[Duplicate filter] Dropped {dropped_duplicates} duplicate events")

    # 1) keep only complete cases (original labels)
    df = filter_to_complete_cases(df)

    # 2) cluster into CL_* activities (conditional)
    if USE_CLUSTERING:
        df = cluster_activities_in_df(df)
        # 3) apply outcome filter on clustered labels
        df = filter_by_outcome_clustered(df, mode=PRECISION_LOG_MODE)
    else:
        print("[Clustering] Skipped (USE_CLUSTERING=False)")
        # Apply frequent_only filter even without clustering if specified
        if PRECISION_LOG_MODE == "frequent_only":
            df = filter_by_outcome_clustered(df, mode=PRECISION_LOG_MODE)
        elif PRECISION_LOG_MODE != "all":
            print(f"[Warning] PRECISION_LOG_MODE='{PRECISION_LOG_MODE}' ignored when USE_CLUSTERING=False (only 'all' and 'frequent_only' supported)")

    # 4) compute durations on clustered df
    df = df.sort_values(["case:concept:name", "time:timestamp"])
    case_durations = df.groupby("case:concept:name")["time:timestamp"].apply(
        lambda x: (x.max() - x.min()).total_seconds()
    )

    if len(case_durations) > 0:
        p99 = case_durations.quantile(DURATION_OUTLIER_PERCENTILE)
        long_cases = case_durations[case_durations > p99].index
    else:
        p99 = np.nan
        long_cases = []

    df = df[~df["case:concept:name"].isin(long_cases)]
    print(f"[Duration filter] Dropped {len(long_cases)} cases exceeding {DURATION_OUTLIER_PERCENTILE:.0%} percentile duration")

    counts = df["case:concept:name"].value_counts()
    short = counts[counts < MIN_EVENTS_PER_CASE].index
    df = df[~df["case:concept:name"].isin(short)]
    print(f"[Event count filter] Dropped {len(short)} cases with <{MIN_EVENTS_PER_CASE} events")

    # Filter case_durations to match remaining cases in df
    remaining_cases = df["case:concept:name"].unique()
    case_durations = case_durations[case_durations.index.isin(remaining_cases)]

    final_cases = df["case:concept:name"].nunique()
    if final_cases == 0:
        raise ValueError("No cases remaining after preprocessing. Check your filtering criteria.")
    
    print(f"[Remaining] Events: {len(df)}, Cases: {final_cases}")
    return df, case_durations


# 4. Basic + extended stats


def compute_basic_stats(log: EventLog,
                        df: pd.DataFrame,
                        case_durations: pd.Series = None) -> Dict[str, Any]:
    """
    Compute basic + extended statistics required for the report:
    - cases, events
    - variants
    - start/end activities
    - mean/std of case length (events per case)
    - mean/std of case duration (seconds and days)
    - case attributes, event attributes
    - categorical event attributes
    """
    
    print("\n=== BASIC & EXTENDED STATS ===")

    num_cases = len(log)
    num_events = len(df)
    print(f"Cases: {num_cases}")
    print(f"Events: {num_events}")

    variants = stats.get_variants(log)
    print(f"Variants: {len(variants)}")

    start = stats.get_start_activities(log)
    end = stats.get_end_activities(log)
    print("Start activities:", start)
    print("End activities:", end)

    # --- case length stats ---
    case_lengths = df.groupby("case:concept:name").size()
    mean_len = float(case_lengths.mean()) if len(case_lengths) > 0 else float("nan")
    std_len = float(case_lengths.std()) if len(case_lengths) > 0 else float("nan")
    print(f"Case length (events) – mean: {mean_len:.2f}, std: {std_len:.2f}")

    # --- case duration stats (always recompute in seconds) ---
    ts = pd.to_datetime(df["time:timestamp"], errors="coerce")
    tmp = df.copy()
    tmp["time:timestamp"] = ts

    cd = tmp.groupby("case:concept:name")["time:timestamp"].apply(
        lambda x: (x.max() - x.min()).total_seconds()
    )
    cd = cd.dropna()

    if len(cd) == 0:
        mean_dur = float("nan")
        std_dur = float("nan")
    else:
        mean_dur = float(cd.mean())
        std_dur = float(cd.std())

    print(f"Case duration (sec) – mean: {mean_dur:.2f}, std: {std_dur:.2f}")
    print(f"Case duration (days) – mean: {mean_dur / 86400:.2f}, std: {std_dur / 86400:.2f}")

    # --- attribute stats ---
    case_attr_cols = [c for c in df.columns if c.startswith("case:")]
    event_attr_cols = [c for c in df.columns if not c.startswith("case:")]

    num_case_attrs = len(case_attr_cols)
    num_event_attrs = len(event_attr_cols)
    print(f"Case attributes: {num_case_attrs} ({case_attr_cols})")
    print(f"Event attributes: {num_event_attrs} ({event_attr_cols})")

    categorical_event_attrs = []
    for c in event_attr_cols:
        if c == "time:timestamp":
            continue
        if not np.issubdtype(df[c].dtype, np.number):
            categorical_event_attrs.append(c)

    print(f"Categorical event attributes: {len(categorical_event_attrs)} ({categorical_event_attrs})")

    return {
        "num_cases": num_cases,
        "num_events": num_events,
        "variants": variants,
        "start": start,
        "end": end,
        "case_lengths": case_lengths,
        "case_durations": cd,  # numeric seconds
        "mean_case_length": mean_len,
        "std_case_length": std_len,
        "mean_case_duration_sec": mean_dur,
        "std_case_duration_sec": std_dur,
        "num_case_attributes": num_case_attrs,
        "num_event_attributes": num_event_attrs,
        "categorical_event_attributes": categorical_event_attrs,
    }


# 5. Service Time (sojourn)


def compute_service_times(log: EventLog) -> Dict[str, float]:
    """
    Compute average service time (sojourn time) for each activity (usually 0.00, mainly for debugging)
    Returns a dictionary mapping activity names to their average durations in seconds.
    """
    print("\n=== SERVICE TIME ===")
    params = {
        constants.PARAMETER_CONSTANT_ACTIVITY_KEY: "concept:name",
        constants.PARAMETER_CONSTANT_TIMESTAMP_KEY: "time:timestamp",
        constants.PARAMETER_CONSTANT_CASEID_KEY: "case:concept:name",
    }
    soj = service_time_log_get.apply(log, parameters=params)
    for k, v in soj.items():
        print(f"{k}: {v:.2f}")
    return soj


def show_variant_coverage(log: EventLog, target_coverage: float = 0.90):
    """
    Display variant statistics:
    - Top 5 variants with their case counts
    - Last variant that has more than 1 case
    - Total variants and coverage up to target_coverage
    """
    print(f"\n=== VARIANT COVERAGE ANALYSIS (Target: {target_coverage:.0%}) ===")
    
    variants = stats.get_variants(log)
    
    # Calculate total cases from variant counts (more reliable than len(log))
    total_cases = sum(variants.values())
    
    if total_cases == 0:
        print("No cases in log.")
        return
    
    # Sort variants by frequency (descending)
    sorted_variants = sorted(variants.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Total cases: {total_cases}")
    print(f"Total variants: {len(variants)}")
    print()
    
    # Show top 5 variants
    print("TOP 5 VARIANTS:")
    print(f"{'Rank':<6} {'Cases':<8} {'% of Total':<12} {'Variant Preview'}")
    print("-" * 80)
    
    for idx, (variant, count) in enumerate(sorted_variants[:5], 1):
        coverage = count / total_cases
        variant_preview = ','.join(list(variant)[:3])
        if len(variant) > 3:
            variant_preview += "..."
        print(f"{idx:<6} {count:<8} {coverage:<12.2%} {variant_preview}")
    
    # Find last variant with more than 1 case
    last_multi_case_idx = None
    last_multi_case_variant = None
    last_multi_case_count = None
    total_multi_case_count = 0
    
    for idx, (variant, count) in enumerate(sorted_variants, 1):
        if count > 1:
            last_multi_case_idx = idx
            last_multi_case_variant = variant
            last_multi_case_count = count
            total_multi_case_count += count
        else:
            break
    
    if last_multi_case_idx:
        print()
        print("LAST VARIANT WITH >1 CASE:")
        print(f"{'Rank':<6} {'Cases':<8} {'% of Total':<12} {'Variant Preview'}")
        print("-" * 80)
        coverage = last_multi_case_count / total_cases
        variant_preview = ','.join(list(last_multi_case_variant)[:3])
        if len(last_multi_case_variant) > 3:
            variant_preview += "..."
        print(f"{last_multi_case_idx:<6} {last_multi_case_count:<8} {coverage:<12.2%} {variant_preview}")
        
        # Show total coverage of all variants with >1 case
        multi_case_coverage = total_multi_case_count / total_cases
        print()
        print(f"All variants with >1 case: {last_multi_case_idx} variants covering {total_multi_case_count}/{total_cases} cases ({multi_case_coverage:.2%})")
    
    # Calculate coverage to target
    cumulative_cases = 0
    for idx, (variant, count) in enumerate(sorted_variants, 1):
        cumulative_cases += count
        cumulative_coverage = cumulative_cases / total_cases
        if cumulative_coverage >= target_coverage:
            print()
            print(f"To reach {target_coverage:.0%} coverage: {idx} variants needed (covering {cumulative_cases}/{total_cases} cases)")
            break




# 6. Variant filtering


def filter_log_by_variant_coverage(log: EventLog, coverage=0.60) -> EventLog:
    print(f"\n=== FILTER VARIANTS (cov={coverage:.0%}) ===")

    total = len(log)
    if total == 0:
        print("[Filtering] Log is empty, nothing to filter – returning empty log.")
        return log

    variants = stats.get_variants(log)
    # sort by frequency (desc), then by key for deterministic order
    variants_sorted = sorted(
        variants.items(),
        key=lambda kv: (-len(kv[1]), str(kv[0]))
    )

    admitted = []
    acc = 0

    for key, cases in variants_sorted:
        if acc / total < coverage:
            admitted.append(key)
            acc += len(cases)
        else:
            # Stop once coverage target is reached or exceeded
            break

    filtered = variants_filter.apply(log, admitted)
    kept = len(filtered)
    frac = acc / total if total > 0 else 0.0
    print(f"Kept {kept} traces ({frac:.1%})")
    return filtered


def apply_noise_filter(log: EventLog, threshold: float = 0.0) -> EventLog:
    """
    Filter out infrequent activities based on threshold.
    
    Args:
        log: Input event log
        threshold: Minimum relative frequency (0.0 to 1.0). Activities appearing 
                   in fewer than threshold% of cases are removed.
                   0.0 = no filtering (default)
    
    Returns:
        Filtered event log with infrequent activities removed
    """
    if threshold <= 0.0:
        print("[Noise Filter] Disabled (threshold = 0.0)")
        return log
    
    print(f"\n=== NOISE FILTERING (threshold={threshold:.1%}) ===")
    
    total_cases = len(log)
    if total_cases == 0:
        return log
    
    # Get activity frequencies
    from collections import Counter
    activity_counts = Counter()
    for trace in log:
        activities_in_trace = set(event["concept:name"] for event in trace)
        for activity in activities_in_trace:
            activity_counts[activity] += 1
    
    # Determine which activities to keep
    min_cases = int(threshold * total_cases)
    activities_to_keep = {
        activity for activity, count in activity_counts.items()
        if count >= min_cases
    }
    
    filtered_activities = set(activity_counts.keys()) - activities_to_keep
    
    if filtered_activities:
        print(f"Filtering {len(filtered_activities)} infrequent activities: {sorted(filtered_activities)}")
        
        # Filter the log
        from pm4py.objects.log.obj import Trace
        filtered_log = EventLog()
        for trace in log:
            new_trace = [event for event in trace if event["concept:name"] in activities_to_keep]
            if len(new_trace) > 0:  # Only keep non-empty traces
                t = Trace()
                # Copy attributes properly
                for key, value in trace.attributes.items():
                    t.attributes[key] = value
                # Add filtered events
                for event in new_trace:
                    t.append(event)
                filtered_log.append(t)
        
        print(f"Kept {len(activities_to_keep)} activities, removed {len(filtered_activities)}")
        print(f"Traces: {total_cases} → {len(filtered_log)}")
        return filtered_log
    else:
        print("No activities filtered (all meet threshold)")
        return log



# 7. Process discovery (configurable algorithm)


def discover_model(log: EventLog):
    """
    Dispatch to the selected process discovery algorithm and return
    a Petri net + initial/final markings.
    """
    print(f"\n[DISCOVERY] Algorithm: {DISCOVERY_ALGO}")
    
    if len(log) == 0:
        raise ValueError("Cannot discover model from empty log.")

    if DISCOVERY_ALGO == "inductive":
        # Use configured Inductive Miner variant
        print(f"[DISCOVERY] Inductive Miner variant: {INDUCTIVE_MINER_VARIANT}")
        
        if INDUCTIVE_MINER_VARIANT == "IM":
            variant = inductive_miner.Variants.IM
        elif INDUCTIVE_MINER_VARIANT == "IMf":
            variant = inductive_miner.Variants.IMf
        elif INDUCTIVE_MINER_VARIANT == "IMd":
            variant = inductive_miner.Variants.IMd
        else:
            # Fallback (should never reach here due to validation)
            variant = inductive_miner.Variants.IM
        
        # Inductive miner returns a process tree, convert to Petri net
        process_tree = inductive_miner.apply(log, variant=variant)
        net, im, fm = pt_converter.apply(process_tree)
        
    elif DISCOVERY_ALGO == "heuristics":
        net, im, fm = pm4py.discover_petri_net_heuristics(log)
    elif DISCOVERY_ALGO == "alpha":
        net, im, fm = pm4py.discover_petri_net_alpha(log)
    else:
        raise ValueError(f"Unsupported DISCOVERY_ALGO: {DISCOVERY_ALGO}")

    print("[DISCOVERY] Model discovered.")
    return net, im, fm



# 8. Flexibility + Simplicity metrics


def compute_flexibility(net) -> Dict[str, Any]:
    """
    Structural metrics:
      - Places, transitions, arcs
      - Invisible transitions
      - Simplicity_Nodes   = places + transitions  (fewer = simpler)
      - Simplicity_Density = arcs / (places+transitions) (lower = simpler)
      - Flexibility score  = density * (1 + invisible_ratio)
    """
    places = len(net.places)
    trans = len(net.transitions)
    arcs = len(net.arcs)
    nodes = places + trans

    invisible = len([t for t in net.transitions if t.label is None])

    if nodes > 0:
        density = arcs / nodes
    else:
        density = float("nan")

    inv_ratio = invisible / trans if trans > 0 else float("nan")

    if np.isnan(density) or np.isnan(inv_ratio):
        flex_score = float("nan")
    else:
        flex_score = density * (1 + inv_ratio)

    print("\n=== FLEXIBILITY & SIMPLICITY ===")
    print(f"Places: {places} | Transitions: {trans} | Arcs: {arcs}")
    print(f"Invisible transitions: {invisible}")
    print(f"Simplicity_Nodes    (places+transitions): {nodes}")
    print(f"Simplicity_Density  (arcs/nodes): {density:.4f}")
    print(f"Flexibility Score (higher = more flexible/complex): {flex_score:.4f}")

    return {
        "places": places,
        "transitions": trans,
        "arcs": arcs,
        "invisible": invisible,
        "simplicity_nodes": nodes,
        "simplicity_density": density,
        "flexibility_score": flex_score,
    }



# 9. Visualization


def visualize_petri_net(net, im, fm, filename: str) -> None:
    """
    Save Petri net visualization to PNG file.
    Requires Graphviz to be installed on the system.
    """
    try:
        print(f"[VIS] Saving Petri net → {filename}")
        pm4py.save_vis_petri_net(net, im, fm, filename)
    except Exception as e:
        print(f"[VIS ERROR] Failed to save visualization: {e}")
        print("Hint: Make sure Graphviz is installed and in PATH")


def convert_and_visualize_bpmn(net, im, fm, png_filename: str, bpmn_filename: str = None) -> None:
    """
    Convert Petri net to BPMN and save:
    1. Visualization to PNG file
    2. BPMN XML to .bpmn file (if bpmn_filename provided)
    Requires Graphviz for visualization
    """
    try:
        print(f"[BPMN] Converting Petri net to BPMN → {png_filename}")
        # Convert Petri net to BPMN using pm4py high-level API
        bpmn_graph = pm4py.convert_to_bpmn(net, im, fm)
        
        # Visualize and save BPMN as PNG
        pm4py.save_vis_bpmn(bpmn_graph, png_filename)
        
        # Save BPMN as .bpmn XML file if requested
        if bpmn_filename:
            print(f"[BPMN] Saving BPMN XML → {bpmn_filename}")
            pm4py.write_bpmn(bpmn_graph, bpmn_filename)
            
    except Exception as e:
        print(f"[BPMN ERROR] Failed to convert/save BPMN: {e}")
        print("Graphviz needs to be installed and in PATH")



# 10. Conformance (Token-based + optional alignments)


def evaluate_model_token_based(log: EventLog, net, im, fm):
    """
    Token-based conformance:
    - fitness on eval_log (full or restricted cleaned log)
    - precision on a sampled eval_log (max 200 traces)

    pm4py.precision_token_based_replay internally uses an
    ETConformance-style token approach.
    """
    print("\n=== TOKEN-BASED CONFORMANCE (ETC-style) ===")

    if len(log) == 0:
        print("[Conformance] Log is empty – returning trivial metrics.")
        fit = {
            "perc_fit_traces": 0.0,
            "average_trace_fitness": 0.0,
            "log_fitness": 0.0,
            "percentage_of_fitting_traces": 0.0,
        }
        prec = 1.0  # empty log / empty behaviour = trivially precise
        sample = log
        print("Fitness (TBR):", fit)
        print("Precision (TBR / ETConformance-style):", prec)
        return fit, prec, sample

    fit = pm4py.fitness_token_based_replay(log, net, im, fm)
    print("Fitness (TBR):", fit)

    if len(log) > MAX_PRECISION_SAMPLE_SIZE:
        # Use a fixed seed for reproducibility in sampling
        rng = random.Random(GLOBAL_SEED)
        indices = rng.sample(range(len(log)), MAX_PRECISION_SAMPLE_SIZE)
        sample = EventLog([log[i] for i in indices])
    else:
        sample = log

    print(f"Computing TBR precision on {len(sample)} traces")
    prec = pm4py.precision_token_based_replay(sample, net, im, fm)
    print("Precision (TBR / ETConformance-style):", prec)

    return fit, prec, sample


def evaluate_model_alignments(sample_log: EventLog, net, im, fm):
    """
    Alignment-based conformance on the same sample:
    - alignment-based fitness
    - alignment-based precision

    This is much more expensive than token-based, so we only compute it
    on a small sample (up to 200 traces).
    """
    print("\n=== ALIGNMENT-BASED CONFORMANCE ===")
    fit = pm4py.fitness_alignments(sample_log, net, im, fm)
    prec = pm4py.precision_alignments(sample_log, net, im, fm)

    print("Alignment fitness:", fit)
    print("Alignment precision:", prec)

    return fit, prec



# 11. Worker for one coverage (single-process logic)


def analyze_single_coverage(clean_log: EventLog, cov: float) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"### Coverage Experiment: {cov:.2f} | Algorithm: {DISCOVERY_ALGO} ###")
    if DISCOVERY_ALGO == "inductive":
        print(f"### Inductive Miner Variant: {INDUCTIVE_MINER_VARIANT} ###")
    print(f"### Noise Threshold: {NOISE_THRESHOLD:.1%} ###")
    print("=" * 80)

    if cov < 1.0:
        disc_log = filter_log_by_variant_coverage(clean_log, cov)
    else:
        disc_log = clean_log
        print("[Filtering] Using full cleaned log (no variant filtering).")
    
    # Apply noise filtering if enabled
    disc_log = apply_noise_filter(disc_log, NOISE_THRESHOLD)

    net, im, fm = discover_model(disc_log)
    flex = compute_flexibility(net)

    # Decide on which log to evaluate conformance
    eval_log = disc_log if EVAL_ON_DISCOVERY_LOG else clean_log

    fit_tbr, prec_tbr, sample = evaluate_model_token_based(eval_log, net, im, fm)

    if ALIGNMENTS_ENABLED:
        fit_align, prec_align = evaluate_model_alignments(sample, net, im, fm)
        align_fit_log = fit_align.get("log_fitness") if fit_align else None
        align_fit_traces = fit_align.get("percentage_of_fitting_traces") if fit_align else None
    else:
        fit_align = None
        prec_align = None
        align_fit_log = None
        align_fit_traces = None

    # Add timestamp and variant info to make filenames unique
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create algorithm-specific subdirectory
    if DISCOVERY_ALGO == "inductive":
        algo_subdir = f"{DISCOVERY_ALGO}_{INDUCTIVE_MINER_VARIANT}"
    else:
        algo_subdir = DISCOVERY_ALGO
    
    # Create subdirectories for Petri nets and BPMN
    petri_subdir = os.path.join(OUTPUT_DIR, algo_subdir)
    bpmn_subdir = os.path.join(BPMN_DIR, algo_subdir)
    os.makedirs(petri_subdir, exist_ok=True)
    os.makedirs(bpmn_subdir, exist_ok=True)
    
    # Build filename with clustering, mode, algorithm, variant (if inductive), noise threshold, coverage, and trace count
    traces_count = len(disc_log)
    clustering_str = "clustered" if USE_CLUSTERING else "original"
    mode_str = f"mode_{PRECISION_LOG_MODE}" if USE_CLUSTERING else "mode_all"
    
    if DISCOVERY_ALGO == "inductive":
        base_filename = f"{clustering_str}_{mode_str}_{DISCOVERY_ALGO}_{INDUCTIVE_MINER_VARIANT}_noise{int(NOISE_THRESHOLD*100)}_cov{int(cov * 100)}_traces{traces_count}_{timestamp}"
    else:
        base_filename = f"{clustering_str}_{mode_str}_{DISCOVERY_ALGO}_noise{int(NOISE_THRESHOLD*100)}_cov{int(cov * 100)}_traces{traces_count}_{timestamp}"
    
    # Save Petri net
    petri_filename = f"petri_{base_filename}.png"
    petri_filepath = os.path.join(petri_subdir, petri_filename)
    visualize_petri_net(net, im, fm, petri_filepath)
    
    # Save BPMN (both PNG visualization and .bpmn XML file)
    bpmn_png_filename = f"bpmn_{base_filename}.png"
    bpmn_xml_filename = f"bpmn_{base_filename}.bpmn"
    bpmn_png_filepath = os.path.join(bpmn_subdir, bpmn_png_filename)
    bpmn_xml_filepath = os.path.join(bpmn_subdir, bpmn_xml_filename)
    convert_and_visualize_bpmn(net, im, fm, bpmn_png_filepath, bpmn_xml_filepath)

    return {
        "coverage": cov,
        "disc_traces": len(disc_log),
        "tbr_fit": fit_tbr.get("log_fitness"),
        "tbr_fitting": fit_tbr.get("perc_fit_traces"),
        "tbr_prec": prec_tbr,
        "align_fit": align_fit_log,
        "align_fitting": align_fit_traces,
        "align_prec": prec_align,
        "simplicity_nodes": flex["simplicity_nodes"],
        "simplicity_density": flex["simplicity_density"],
        "flexibility": flex["flexibility_score"],
    }



# 12. Process-based worker wrapper (true parallelism)


def analyze_single_coverage_worker(args):
    """
    Worker wrapper for ProcessPoolExecutor.

    We pass a cleaned DataFrame + coverage to the subprocess,
    reconstruct the EventLog there, and then call analyze_single_coverage().
    """
    # Set seed in worker process for reproducibility
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    
    clean_df, cov = args
    clean_log = dataframe_to_event_log(clean_df)
    return analyze_single_coverage(clean_log, cov)


def run_experiments(clean_df: pd.DataFrame, coverages: List[float]) -> List[Dict[str, Any]]:
    results = []
    print(f"\n[PROCESSES] Running {len(coverages)} experiments in parallel (true CPU parallelism)...")
    print(f"[CONFIG] DISCOVERY_ALGO = {DISCOVERY_ALGO}")
    if DISCOVERY_ALGO == "inductive":
        print(f"[CONFIG] INDUCTIVE_MINER_VARIANT = {INDUCTIVE_MINER_VARIANT}")
    print(f"[CONFIG] USE_CLUSTERING = {USE_CLUSTERING}")
    print(f"[CONFIG] NOISE_THRESHOLD = {NOISE_THRESHOLD:.1%}")
    print(f"[CONFIG] ALIGNMENTS_ENABLED = {ALIGNMENTS_ENABLED}")
    print(f"[CONFIG] PRECISION_LOG_MODE = {PRECISION_LOG_MODE}")
    print(f"[CONFIG] EVAL_ON_DISCOVERY_LOG = {EVAL_ON_DISCOVERY_LOG}")

    tasks = [(clean_df, c) for c in coverages]

    max_workers = min(len(tasks), os.cpu_count() or 1)
    print(f"[PROCESSES] Using up to {max_workers} worker processes.")

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(analyze_single_coverage_worker, t): t[1]
            for t in tasks
        }
        for f in as_completed(futs):
            res = f.result()
            results.append(res)

    results.sort(key=lambda x: x["coverage"])
    
    # Save results to text file
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create algorithm-specific subdirectory for results
    if DISCOVERY_ALGO == "inductive":
        algo_subdir = f"{DISCOVERY_ALGO}_{INDUCTIVE_MINER_VARIANT}"
    else:
        algo_subdir = DISCOVERY_ALGO
    
    results_subdir = os.path.join(RESULTS_DIR, algo_subdir)
    os.makedirs(results_subdir, exist_ok=True)
    
    # Build results filename with configuration info
    clustering_str = "clustered" if USE_CLUSTERING else "original"
    mode_str = f"mode_{PRECISION_LOG_MODE}" if USE_CLUSTERING else "mode_all"
    
    if DISCOVERY_ALGO == "inductive":
        results_filename = f"results_{clustering_str}_{mode_str}_{DISCOVERY_ALGO}_{INDUCTIVE_MINER_VARIANT}_noise{int(NOISE_THRESHOLD*100)}_{timestamp}.txt"
    else:
        results_filename = f"results_{clustering_str}_{mode_str}_{DISCOVERY_ALGO}_noise{int(NOISE_THRESHOLD*100)}_{timestamp}.txt"
    
    results_filepath = os.path.join(results_subdir, results_filename)
    
    # Write results to file
    with open(results_filepath, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PROCESS MINING ANALYSIS RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        # Configuration section
        f.write("CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Timestamp:                 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Discovery Algorithm:       {DISCOVERY_ALGO}\n")
        if DISCOVERY_ALGO == "inductive":
            f.write(f"Inductive Miner Variant:   {INDUCTIVE_MINER_VARIANT}\n")
        f.write(f"Use Clustering:            {USE_CLUSTERING}\n")
        f.write(f"Noise Threshold:           {NOISE_THRESHOLD:.1%}\n")
        f.write(f"Precision Log Mode:        {PRECISION_LOG_MODE}\n")
        f.write(f"Alignments Enabled:        {ALIGNMENTS_ENABLED}\n")
        f.write(f"Eval on Discovery Log:     {EVAL_ON_DISCOVERY_LOG}\n")
        f.write(f"Fitting Tolerance:         {FITTING_TOLERANCE_MIN:.1f}% - {FITTING_TOLERANCE_MAX:.1f}%\n")
        f.write(f"Target Fitting Percentage: {TARGET_FITTING_PERCENTAGE:.1f}%\n")
        f.write(f"Coverage Values:           {coverages}\n")
        f.write("\n")
        
        # Results section
        f.write("RESULTS SUMMARY\n")
        f.write("-" * 80 + "\n\n")

    print("\n=== SUMMARY (per coverage) ===")
    
    # Open results file for appending
    with open(results_filepath, 'a') as f:
        for r in results:
            tbr_fit_str = f"{r['tbr_fit']:.4f}" if r['tbr_fit'] is not None else "N/A"
            tbr_fitting_str = f"{r['tbr_fitting']:.1f}" if r['tbr_fitting'] is not None else "N/A"
            tbr_prec_str = f"{r['tbr_prec']:.3f}" if r['tbr_prec'] is not None else "N/A"
            
            base_str = (
                f"cov={r['coverage']:.2f} | "
                f"disc={r['disc_traces']} | "
                f"TBR_fit={tbr_fit_str} ({tbr_fitting_str}%) | "
                f"TBR_prec={tbr_prec_str} | "
                f"Simpl_nodes={r['simplicity_nodes']} | "
                f"Simpl_dens={r['simplicity_density']:.3f} | "
                f"Flex={r['flexibility']:.3f}"
            )
            if ALIGNMENTS_ENABLED:
                align_fit_str = f"{r['align_fit']:.4f}" if r['align_fit'] is not None else "N/A"
                align_fitting_str = f"{r['align_fitting']:.1f}" if r['align_fitting'] is not None else "N/A"
                align_prec_str = f"{r['align_prec']:.3f}" if r['align_prec'] is not None else "N/A"
                base_str += (
                    f" | ALIGN_fit={align_fit_str} ({align_fitting_str}%) | "
                    f"ALIGN_prec={align_prec_str}"
                )
            print(base_str)
            f.write(base_str + "\n")

    # Select recommended model: maximize TBR precision with ~80% fitting traces
    recommended = None
    best_prec = -1.0
    for r in results:
        ft = r["tbr_fitting"]
        if ft is None:
            continue
        if FITTING_TOLERANCE_MIN <= ft <= FITTING_TOLERANCE_MAX and r["tbr_prec"] is not None and r["tbr_prec"] > best_prec:
            best_prec = r["tbr_prec"]
            recommended = r

    if recommended is None and results:
        # fallback: closest to 80% fitting traces, then highest precision
        def score(x):
            ft = x["tbr_fitting"]
            prec = x["tbr_prec"] if x["tbr_prec"] is not None else 0.0
            if ft is None:
                return (1e9, -prec)
            return (abs(ft - TARGET_FITTING_PERCENTAGE), -prec)
        recommended = sorted(results, key=score)[0]

    if recommended is not None:
        print("\n=== RECOMMENDED MODEL (≈80% TBR fitting traces, max TBR precision) ===")
        
        rec_tbr_fit_str = f"{recommended['tbr_fit']:.4f}" if recommended['tbr_fit'] is not None else "N/A"
        rec_tbr_fitting_str = f"{recommended['tbr_fitting']:.1f}" if recommended['tbr_fitting'] is not None else "N/A"
        rec_tbr_prec_str = f"{recommended['tbr_prec']:.3f}" if recommended['tbr_prec'] is not None else "N/A"
        
        base_str = (
            f"cov={recommended['coverage']:.2f} | "
            f"disc={recommended['disc_traces']} | "
            f"TBR_fit={rec_tbr_fit_str} ({rec_tbr_fitting_str}%) | "
            f"TBR_prec={rec_tbr_prec_str} | "
            f"Simpl_nodes={recommended['simplicity_nodes']} | "
            f"Simpl_dens={recommended['simplicity_density']:.3f} | "
            f"Flex={recommended['flexibility']:.3f}"
        )
        if ALIGNMENTS_ENABLED:
            rec_align_fit_str = f"{recommended['align_fit']:.4f}" if recommended['align_fit'] is not None else "N/A"
            rec_align_fitting_str = f"{recommended['align_fitting']:.1f}" if recommended['align_fitting'] is not None else "N/A"
            rec_align_prec_str = f"{recommended['align_prec']:.3f}" if recommended['align_prec'] is not None else "N/A"
            base_str += (
                f" | ALIGN_fit={rec_align_fit_str} ({rec_align_fitting_str}%) | "
                f"ALIGN_prec={rec_align_prec_str}"
            )
        print(base_str)
        print(f"Corresponding Petri net was saved to '{OUTPUT_DIR}/{algo_subdir}/' folder.")
        
        # Write recommendation to file
        with open(results_filepath, 'a') as f:
            f.write("\n")
            f.write("RECOMMENDED MODEL\n")
            f.write("-" * 80 + "\n")
            f.write("Selection criteria: ~80% TBR fitting traces, maximize TBR precision\n\n")
            f.write(base_str + "\n")
            f.write(f"\nCorresponding Petri net was saved to '{OUTPUT_DIR}/{algo_subdir}/' folder.\n")
    
    # Write footer to file
    with open(results_filepath, 'a') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n>>> Results saved to: {results_filepath}")

    return results



# 13. main()


def main():
    print(">>> Starting BPIC17 Analysis Script")

    # XES file is located in ./data/
    log_path = os.path.join("data", "BPI_Challenge_2017.xes.gz")
    
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    raw = load_log_xes(log_path)
    show_variant_coverage(raw, target_coverage=0.90)
    
    df = log_to_dataframe(raw)
    clean_df, case_durations = preprocess_log_df(df)
    clean_log = dataframe_to_event_log(clean_df)

    _stats = compute_basic_stats(clean_log, clean_df, case_durations)
    compute_service_times(clean_log)

    if ALIGNMENTS_ENABLED:
        coverages = [0.60, 0.30, 0.20, 0.10, 0.01]
    else:
        coverages = [1.0, 0.60, 0.30, 0.10, 0.01]
    
    # Validate coverage values
    for cov in coverages:
        if not 0 < cov <= 1.0:
            raise ValueError(f"Invalid coverage value: {cov}. Must be between 0 and 1.0")
        
    run_experiments(clean_df, coverages)

    print(f"\n>>> DONE. Check generated Petri net visualizations in '{OUTPUT_DIR}/' folder.")
    print(f">>> Results summary saved in '{RESULTS_DIR}/' folder.")


if __name__ == "__main__":
    main()

