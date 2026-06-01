# =============================================================================
# data.py — Data Loading and Feature Metadata
# =============================================================================
# Handles everything data-related for the Algorithmic Recourse Tool.
#
# DATASET: UCI Adult Income (Census Income)
#   48,842 people from the 1994 U.S. Census.
#   Binary target: does this person earn >$50K/year?
#   14 features: mix of continuous and categorical.
#   Source: https://archive.ics.uci.edu/ml/datasets/adult
#
# WHY THIS DATASET?
#   It is the canonical dataset for algorithmic fairness research.
#   The recourse question is immediately relatable: "What do I need to
#   change to be predicted as a high earner?" It has clear immutable
#   features (sex, race, native-country) and clearly actionable ones
#   (education, hours-per-week, occupation), making it ideal for
#   demonstrating both the mechanics and the ethics of recourse.
#
# PIPELINE IN THIS FILE:
#   1. Download raw .data and .test files from UCI
#   2. Clean: strip whitespace, drop rows with '?' missing values
#   3. Encode: ordinal-encode categorical features to integers
#   4. Define FEATURE_META: actionability, valid ranges, step sizes
#   5. Utility functions used by the recourse algorithms
#
# ENCODING STRATEGY:
#   We use ordinal integer encoding (not one-hot) for all categorical
#   features. This is intentional:
#     - Decision trees and random forests work well with ordinal encoding
#     - It lets the recourse algorithms perturb categorical features by
#       step=1 (e.g., move education from "HS-grad" to "Some-college")
#     - It keeps the feature space compact (14 features, not 100+)
#   The mapping dictionaries below are the ground truth for encoding.
# =============================================================================

import os
import urllib.request
import pandas as pd
import numpy as np


# =============================================================================
# ENCODING MAPS
# =============================================================================
# Each categorical feature is mapped to integers in a meaningful order.
# The order matters for recourse:
#   - EDUCATION_ORDER goes from least to most education so that
#     "increase education" means "increase the integer value"
#   - WORKCLASS_ORDER groups similar employment types together
#
# These maps are also used by the UI to show human-readable labels.

EDUCATION_ORDER = {
    'Preschool':       0,
    '1st-4th':         1,
    '5th-6th':         2,
    '7th-8th':         3,
    '9th':             4,
    '10th':            5,
    '11th':            6,
    '12th':            7,
    'HS-grad':         8,
    'Some-college':    9,
    'Assoc-voc':       10,
    'Assoc-acdm':      11,
    'Bachelors':       12,
    'Masters':         13,
    'Prof-school':     14,
    'Doctorate':       15,
}

WORKCLASS_ORDER = {
    'Never-worked':    0,
    'Without-pay':     1,
    'Private':         2,
    'Self-emp-not-inc':3,
    'Self-emp-inc':    4,
    'Local-gov':       5,
    'State-gov':       6,
    'Federal-gov':     7,
}

MARITAL_ORDER = {
    'Never-married':           0,
    'Separated':               1,
    'Divorced':                2,
    'Widowed':                 3,
    'Married-spouse-absent':   4,
    'Married-AF-spouse':       5,
    'Married-civ-spouse':      6,
}

OCCUPATION_ORDER = {
    'Other-service':       0,
    'Priv-house-serv':     1,
    'Handlers-cleaners':   2,
    'Farming-fishing':     3,
    'Machine-op-inspct':   4,
    'Transport-moving':    5,
    'Adm-clerical':        6,
    'Armed-Forces':        7,
    'Craft-repair':        8,
    'Sales':               9,
    'Tech-support':        10,
    'Protective-serv':     11,
    'Exec-managerial':     12,
    'Prof-specialty':      13,
}

RELATIONSHIP_ORDER = {
    'Other-relative':  0,
    'Unmarried':       1,
    'Own-child':       2,
    'Not-in-family':   3,
    'Wife':            4,
    'Husband':         5,
}

RACE_ORDER = {
    'Other':                  0,
    'Amer-Indian-Eskimo':     1,
    'Black':                  2,
    'Asian-Pac-Islander':     3,
    'White':                  4,
}

SEX_ORDER = {
    'Female': 0,
    'Male':   1,
}

# Reverse maps: integer → human-readable label (used in the UI)
EDU_REVERSE     = {v: k for k, v in EDUCATION_ORDER.items()}
WORK_REVERSE    = {v: k for k, v in WORKCLASS_ORDER.items()}
MAR_REVERSE     = {v: k for k, v in MARITAL_ORDER.items()}
OCC_REVERSE     = {v: k for k, v in OCCUPATION_ORDER.items()}
REL_REVERSE     = {v: k for k, v in RELATIONSHIP_ORDER.items()}
RACE_REVERSE    = {v: k for k, v in RACE_ORDER.items()}
SEX_REVERSE     = {0: 'Female', 1: 'Male'}

# Master label-lookup: feature_name → {int_value: display_string}
CATEGORY_LABELS = {
    'education':      EDU_REVERSE,
    'workclass':      WORK_REVERSE,
    'marital_status': MAR_REVERSE,
    'occupation':     OCC_REVERSE,
    'relationship':   REL_REVERSE,
    'race':           RACE_REVERSE,
    'sex':            SEX_REVERSE,
}


# =============================================================================
# FEATURE METADATA
# =============================================================================
# This dict is the contract between the data and the recourse algorithms.
#
# Each entry:
#   label       — human-readable name for the UI
#   description — plain-English explanation
#   actionable  — True = algorithm may change this feature for recourse
#                 False = immutable, never modified by recourse algorithms
#   min / max   — valid integer range after encoding
#   step        — how much to change the feature per greedy iteration
#   unit        — display string for charts
#   is_cat      — True = categorical (show labels), False = continuous
#
# ACTIONABILITY REASONING:
#   Immutable: sex and race are protected demographic attributes — recourse
#     that says "change your sex" is meaningless and harmful.
#     Native-country is immutable (you cannot change where you were born).
#   Partially constrained: age can only increase (you cannot get younger),
#     so we allow upward-only perturbation in the algorithms.
#   Fully actionable: education, occupation, hours, workclass, marital
#     status — all can genuinely change in a person's life.

FEATURE_META = {
    'age': {
        'label':       'Age',
        'description': 'Person\'s age in years',
        'actionable':  True,        # only upward — handled separately in algorithms
        'increase_only': True,      # age can only increase, not decrease
        'min':         17,
        'max':         90,
        'step':        1,
        'unit':        'years',
        'is_cat':      False,
    },
    'workclass': {
        'label':       'Work Class',
        'description': 'Type of employer (private, government, self-employed)',
        'actionable':  True,
        'increase_only': False,
        'min':         0,
        'max':         7,
        'step':        1,
        'unit':        '',
        'is_cat':      True,
    },
    'education_num': {
        'label':       'Education Level',
        'description': 'Highest education level (1=preschool → 16=doctorate)',
        'actionable':  True,
        'increase_only': True,      # education can only increase
        'min':         1,
        'max':         16,
        'step':        1,
        'unit':        '(1–16)',
        'is_cat':      False,
    },
    'marital_status': {
        'label':       'Marital Status',
        'description': 'Current marital status',
        'actionable':  True,
        'increase_only': False,
        'min':         0,
        'max':         6,
        'step':        1,
        'unit':        '',
        'is_cat':      True,
    },
    'occupation': {
        'label':       'Occupation',
        'description': 'Type of work/occupation',
        'actionable':  True,
        'increase_only': False,
        'min':         0,
        'max':         13,
        'step':        1,
        'unit':        '',
        'is_cat':      True,
    },
    'relationship': {
        'label':       'Relationship',
        'description': 'Role in family unit',
        'actionable':  True,
        'increase_only': False,
        'min':         0,
        'max':         5,
        'step':        1,
        'unit':        '',
        'is_cat':      True,
    },
    'race': {
        'label':       'Race',
        'description': 'Race (protected attribute — immutable)',
        'actionable':  False,       # protected demographic — never changed
        'increase_only': False,
        'min':         0,
        'max':         4,
        'step':        1,
        'unit':        '',
        'is_cat':      True,
    },
    'sex': {
        'label':       'Sex',
        'description': 'Sex (protected attribute — immutable)',
        'actionable':  False,       # protected demographic — never changed
        'increase_only': False,
        'min':         0,
        'max':         1,
        'step':        1,
        'unit':        '(0=F, 1=M)',
        'is_cat':      True,
    },
    'capital_gain': {
        'label':       'Capital Gain',
        'description': 'Investment income in dollars',
        'actionable':  True,
        'increase_only': False,
        'min':         0,
        'max':         99999,
        'step':        1000,
        'unit':        '$',
        'is_cat':      False,
    },
    'capital_loss': {
        'label':       'Capital Loss',
        'description': 'Investment losses in dollars',
        'actionable':  True,
        'increase_only': False,
        'min':         0,
        'max':         4356,
        'step':        100,
        'unit':        '$',
        'is_cat':      False,
    },
    'hours_per_week': {
        'label':       'Hours per Week',
        'description': 'Hours worked per week',
        'actionable':  True,
        'increase_only': False,
        'min':         1,
        'max':         99,
        'step':        5,
        'unit':        'hrs',
        'is_cat':      False,
    },
    'native_country_us': {
        'label':       'Born in US',
        'description': '1 = born in United States, 0 = born abroad',
        'actionable':  False,       # you cannot change where you were born
        'increase_only': False,
        'min':         0,
        'max':         1,
        'step':        1,
        'unit':        '(0=abroad, 1=US)',
        'is_cat':      True,
    },
}

# Convenience lists
FEATURE_NAMES       = list(FEATURE_META.keys())
ACTIONABLE_FEATURES = [f for f, m in FEATURE_META.items() if m['actionable']]
IMMUTABLE_FEATURES  = [f for f, m in FEATURE_META.items() if not m['actionable']]


# =============================================================================
# DOWNLOAD AND CLEAN
# =============================================================================

# Raw column names as they appear in the UCI .data file
_UCI_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income'
]

_UCI_TRAIN_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
                  "/adult/adult.data")
_UCI_TEST_URL  = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
                  "/adult/adult.test")


def download_and_clean(data_dir='data', max_rows=10000, random_state=42):
    """
    Downloads the UCI Adult Income dataset and saves a cleaned version.

    Processing steps:
      1. Download adult.data (train) and adult.test from UCI
      2. Combine into one file
      3. Strip whitespace from string values
      4. Drop rows with '?' missing values
      5. Encode categorical features to integers using the maps above
      6. Simplify native_country → binary (US vs not-US)
      7. Drop columns we don't use (fnlwgt, raw education string)
      8. Sample max_rows rows (stratified) for performance
      9. Save as data/adult_clean.csv

    WHY WE SAMPLE:
      Full dataset is 48,842 rows. Computing recourse for every person
      predicted as low-income across 3 models would take very long.
      We sample 10,000 rows (stratified by income) to keep the app
      responsive while maintaining representative results.

    Args:
        data_dir    (str): directory to save the CSV
        max_rows    (int): number of rows to keep (stratified sample)
        random_state(int): for reproducibility

    Returns:
        str: path to saved CSV
    """
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, 'adult_clean.csv')

    if os.path.exists(path):
        print(f"Dataset already exists at {path}, skipping download.")
        return path

    print("Downloading UCI Adult Income dataset...")

    frames = []

    for url, skip_rows in [(_UCI_TRAIN_URL, 0), (_UCI_TEST_URL, 1)]:
        temp = path + f".{url.split('/')[-1]}.tmp"
        try:
            urllib.request.urlretrieve(url, temp)
            df_raw = pd.read_csv(
                temp, header=None, names=_UCI_COLUMNS,
                skiprows=skip_rows, na_values='?',
                skipinitialspace=True
            )
            frames.append(df_raw)
            os.remove(temp)
        except Exception as e:
            if os.path.exists(temp):
                os.remove(temp)
            raise RuntimeError(f"Download failed for {url}: {e}")

    # Combine train + test
    df = pd.concat(frames, ignore_index=True)
    print(f"  Combined rows before cleaning: {len(df)}")

    # Strip whitespace from string columns
    # Use explicit list to avoid pandas 3.x deprecation warning for 'object' dtype
    str_cols = [c for c in df.columns if df[c].dtype == object]
    for col in str_cols:
        df[col] = df[col].str.strip()

    # Remove the trailing '.' that the test file adds to income labels
    df['income'] = df['income'].str.rstrip('.')

    # Drop rows with any '?' missing values
    df = df.dropna().reset_index(drop=True)
    print(f"  Rows after dropping missing values: {len(df)}")

    # ── Encode target ────────────────────────────────────────────────────────
    # 0 = income <=50K (low income), 1 = income >50K (high income)
    df['target'] = (df['income'] == '>50K').astype(int)

    # ── Encode categorical features ──────────────────────────────────────────
    # Only encode rows where the value is in our known map.
    # Rows with unknown categories are dropped (very few in practice).

    def encode_col(series, mapping):
        """Replace string categories with integer codes. Drop unmapped rows."""
        mapped = series.map(mapping)
        return mapped

    df['workclass']      = encode_col(df['workclass'],      WORKCLASS_ORDER)
    df['marital_status'] = encode_col(df['marital_status'], MARITAL_ORDER)
    df['occupation']     = encode_col(df['occupation'],     OCCUPATION_ORDER)
    df['relationship']   = encode_col(df['relationship'],   RELATIONSHIP_ORDER)
    df['race']           = encode_col(df['race'],           RACE_ORDER)
    df['sex']            = encode_col(df['sex'],            SEX_ORDER)

    # Simplify native_country to binary: born in US (1) or abroad (0)
    # This captures the most relevant signal and avoids 40+ dummy variables
    df['native_country_us'] = (df['native_country'] == 'United-States').astype(int)

    # Drop columns we don't need
    # fnlwgt = sampling weight, not a real feature
    # education = string version of education_num (redundant)
    # native_country = replaced by native_country_us
    # income = replaced by target
    df = df.drop(columns=['fnlwgt', 'education', 'native_country', 'income'])

    # Drop any rows where encoding produced NaN (unknown categories)
    df = df.dropna().reset_index(drop=True)
    print(f"  Rows after encoding: {len(df)}")

    # ── Stratified sample ────────────────────────────────────────────────────
    # Sample max_rows rows, preserving the income class ratio.
    # We do this manually instead of groupby().apply() because pandas 3.x
    # changed how the grouping column is handled in apply() — the 'target'
    # column was getting silently dropped from the result.
    if len(df) > max_rows:
        low_income  = df[df['target'] == 0]
        high_income = df[df['target'] == 1]

        # How many to take from each class, proportional to their size
        n_low  = int(max_rows * len(low_income)  / len(df))
        n_high = int(max_rows * len(high_income) / len(df))

        # Ensure we don't ask for more rows than exist in each group
        n_low  = min(n_low,  len(low_income))
        n_high = min(n_high, len(high_income))

        df = pd.concat([
            low_income.sample(n_low,   random_state=random_state),
            high_income.sample(n_high, random_state=random_state),
        ]).reset_index(drop=True)

        print(f"  Sampled {len(df)} rows (stratified by income)")

    # Reorder columns so features come before target
    df = df[FEATURE_NAMES + ['target']]

    # Cast all feature columns to numeric (should already be, but be safe)
    for col in FEATURE_NAMES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().reset_index(drop=True)

    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} rows to {path}")
    print(f"  High income (>50K): {df['target'].sum()} | "
          f"Low income: {(df['target']==0).sum()}")

    return path


def load_data(data_dir='data'):
    """
    Loads the cleaned Adult Income dataset.

    Returns:
        X  (DataFrame): n × 12 feature matrix (FEATURE_NAMES columns)
        y  (Series):    n binary labels (0=<=50K, 1=>50K)
        df (DataFrame): full table including target
    """
    path = download_and_clean(data_dir)
    df   = pd.read_csv(path)

    # Enforce correct column order
    X = df[FEATURE_NAMES].astype(float)
    y = df['target'].astype(int)

    return X, y, df


# =============================================================================
# UTILITY FUNCTIONS (used by recourse algorithms)
# =============================================================================

def get_person(X, y, person_idx):
    """
    Returns one person's features as a dict and their true label.

    Args:
        X          (DataFrame): full feature matrix
        y          (Series):    full labels
        person_idx (int):       row index

    Returns:
        features   (dict):  {feature_name: value, ...}
        true_label (int):   0 or 1
    """
    features   = X.iloc[person_idx].to_dict()
    true_label = int(y.iloc[person_idx])
    return features, true_label


def clip_to_valid_range(features_dict):
    """
    Clips each feature value to its valid [min, max] range from FEATURE_META.
    Called after every perturbation step in the recourse algorithms.

    Args:
        features_dict (dict): {feature_name: value}

    Returns:
        dict: same keys, values clipped to valid ranges
    """
    clipped = {}
    for fname, value in features_dict.items():
        meta           = FEATURE_META[fname]
        clipped[fname] = float(np.clip(value, meta['min'], meta['max']))
    return clipped


def compute_distance(original, counterfactual, metric='l1'):
    """
    Normalized distance between original and counterfactual,
    considering only actionable features.

    Each feature's difference is divided by its range so all features
    are on a [0, 1] scale before summing. This ensures cholesterol
    (range ~500) is not penalized more than education (range 15).

    Args:
        original       (dict): original person features
        counterfactual (dict): modified features
        metric         (str):  'l1' or 'l2'

    Returns:
        float: normalized distance
    """
    diffs = []
    for fname in ACTIONABLE_FEATURES:
        meta          = FEATURE_META[fname]
        feature_range = meta['max'] - meta['min']
        if feature_range == 0:
            continue
        normalized = abs(float(counterfactual[fname]) -
                         float(original[fname])) / feature_range
        diffs.append(normalized)

    if metric == 'l1':
        return float(np.sum(diffs))
    elif metric == 'l2':
        return float(np.sqrt(np.sum(np.array(diffs) ** 2)))
    else:
        raise ValueError(f"Unknown metric '{metric}'. Use 'l1' or 'l2'.")


def count_changed_features(original, counterfactual, threshold=1e-6):
    """
    Counts how many actionable features differ between original and CF.
    This is the sparsity measure — fewer changes = better recourse.

    Returns:
        int: number of changed features
    """
    return sum(
        1 for fname in ACTIONABLE_FEATURES
        if abs(float(counterfactual[fname]) - float(original[fname])) > threshold
    )


def feature_value_label(fname, value):
    """
    Converts a numeric feature value to a human-readable label.
    For categorical features, looks up the display string.
    For continuous features, returns the formatted number.

    Args:
        fname (str):   feature name
        value (float): encoded value

    Returns:
        str: human-readable representation
    """
    if fname in CATEGORY_LABELS:
        return CATEGORY_LABELS[fname].get(int(value), str(int(value)))
    meta = FEATURE_META[fname]
    unit = meta['unit']
    if meta['is_cat']:
        return str(int(value))
    elif value == int(value):
        return f"{int(value)} {unit}".strip()
    else:
        return f"{value:.1f} {unit}".strip()


# =============================================================================
# MAIN — test this module
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("DATA MODULE TEST")
    print("=" * 60)

    X, y, df = load_data()

    print(f"\nDataset shape:  {df.shape}")
    print(f"High income:    {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"Low income:     {(y==0).sum()} ({(1-y.mean())*100:.1f}%)")

    print(f"\nActionable features ({len(ACTIONABLE_FEATURES)}):")
    for f in ACTIONABLE_FEATURES:
        m = FEATURE_META[f]
        print(f"  {f:<20} range [{m['min']}, {m['max']}]  "
              f"step={m['step']}  "
              f"{'increase-only' if m['increase_only'] else 'bidirectional'}")

    print(f"\nImmutable features ({len(IMMUTABLE_FEATURES)}):")
    for f in IMMUTABLE_FEATURES:
        print(f"  {f}")

    # Show a sample person
    features, label = get_person(X, y, 0)
    print(f"\nPerson #0 (label: {'>50K' if label else '<=50K'}):")
    for fname, val in features.items():
        label_str = feature_value_label(fname, val)
        action    = "(actionable)" if FEATURE_META[fname]['actionable'] else "(immutable)"
        print(f"  {fname:<22} = {label_str:<25} {action}")
