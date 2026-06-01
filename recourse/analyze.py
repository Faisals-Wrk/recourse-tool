# =============================================================================
# analyze.py — Recourse Analyses
# =============================================================================
# Takes the trained models and recourse algorithms from Phase 1+2 and runs
# four analyses that power the Streamlit app's most important pages.
#
# THE FOUR ANALYSES:
#
#   1. Effort Distribution
#      For a sample of people predicted as low-income (<=50K), compute
#      the recourse effort (normalized distance) each model requires to
#      flip the prediction to >50K.
#      Result: a DataFrame with one row per (person, model) pair.
#      Powers: Fairness page, Overview stats.
#
#   2. Fairness Analysis
#      Group the effort distribution by demographic attributes (sex, race).
#      Ask: does the model require more effort from women than men?
#      From non-white persons than white persons?
#      Result: aggregated stats per demographic group per model.
#      Powers: Fairness page charts.
#
#   3. Feature Frequency
#      Across all successful recourse paths, which features appear most
#      often? Which features are the "go-to" changes the model points to?
#      Result: {feature: count, fraction} per model.
#      Powers: Feature Frequency chart.
#
#   4. Algorithm Comparison (single person)
#      Given one person and one model, run all three algorithms and return
#      a side-by-side comparison of what each algorithm recommends.
#      Result: dict of RecourseResult objects.
#      Powers: Algorithm Comparison page.
#
# PERFORMANCE DESIGN:
#   Full effort distribution on all ~7,500 low-income persons × 3 models
#   would take ~30 minutes. We use a stratified sample of 500 persons
#   (preserving sex and race proportions) so the analysis runs in ~2 minutes
#   and still gives representative fairness results.
#
#   In the Streamlit app, this entire module is called once inside
#   @st.cache_data — so the user only waits on the first load.
# =============================================================================

import numpy as np
import pandas as pd
from collections import defaultdict

from recourse.data          import (
    FEATURE_META, FEATURE_NAMES,
    ACTIONABLE_FEATURES, IMMUTABLE_FEATURES,
    get_person, compute_distance, feature_value_label,
    RACE_REVERSE, SEX_REVERSE
)
from recourse.model         import predict_person
from recourse.counterfactual import (
    greedy_recourse, find_recourse, RecourseResult
)


# =============================================================================
# ANALYSIS 1 — EFFORT DISTRIBUTION
# =============================================================================

def compute_effort_distribution(trained_models, X, y,
                                 sample_size=500,
                                 max_iterations=150,
                                 random_state=42,
                                 verbose=True):
    """
    Computes recourse effort for a stratified sample of low-income persons.

    For each sampled person predicted as low-income (<=50K) by at least
    one model, we run the Greedy algorithm and record:
      - Whether recourse was found
      - The normalized L1 distance (effort required)
      - How many features changed (sparsity)
      - Which specific features changed and by how much
      - The person's demographic attributes (sex, race, age_group)

    WHY GREEDY ONLY?
      We are running this for hundreds of persons × 3 models.
      Greedy is ~10× faster than Proximity and produces valid results
      for most persons. Proximity is reserved for single-person
      interactive use on the "Find Recourse" page.

    WHY STRATIFIED SAMPLING?
      We want the fairness analysis to be reliable. If we sample randomly,
      we might get very few women or very few non-white persons (they are
      minorities in this 1994 Census dataset). Stratified sampling ensures
      we have enough of each group for meaningful group comparisons.

    Args:
        trained_models (dict):    output of train_all_models()
        X              (DataFrame): full feature matrix
        y              (Series):   true labels
        sample_size    (int):     number of persons to sample
        max_iterations (int):     greedy algorithm iteration cap
        random_state   (int):     for reproducibility
        verbose        (bool):    print progress

    Returns:
        DataFrame with columns:
          person_idx, model_key, model_name,
          found, distance, n_changed,
          true_label, predicted_label, pred_prob,
          sex, sex_label, race, race_label, age, age_group,
          changes_json  (JSON string of the changes dict for storage)
    """

    # ── Step 1: Find all persons who need recourse ─────────────────────────
    # A person "needs recourse" if any model predicts them as low-income.
    # We collect all such persons' indices first, then sample from them.

    if verbose:
        print("Identifying low-income predictions across all models...")

    # For efficiency: get all predictions upfront rather than calling
    # predict_person() in a loop (which would be very slow)
    low_income_indices = set()
    for key, record in trained_models.items():
        for idx, pred in enumerate(record['predictions']):
            if pred == 0:   # predicted low-income
                low_income_indices.add(idx)

    low_income_indices = sorted(low_income_indices)

    if verbose:
        print(f"  Persons with at least one low-income prediction: "
              f"{len(low_income_indices)}")

    # ── Step 2: Stratified sample by sex and race ─────────────────────────
    # Build a small DataFrame of candidate persons with their demographics,
    # then sample proportionally from each (sex × race) stratum.

    candidates = []
    for idx in low_income_indices:
        features, true_label = get_person(X, y, idx)
        candidates.append({
            'person_idx': idx,
            'sex':        int(features['sex']),
            'race':       int(features['race']),
            'age':        int(features['age']),
        })
    candidates_df = pd.DataFrame(candidates)

    # Create a combined stratum label: "sex_race"
    candidates_df['stratum'] = (candidates_df['sex'].astype(str) + '_' +
                                candidates_df['race'].astype(str))

    # Sample from each stratum proportionally
    sampled = _stratified_sample(candidates_df, sample_size, random_state)
    sampled_indices = sampled['person_idx'].tolist()

    if verbose:
        print(f"  Sampled {len(sampled_indices)} persons "
              f"(stratified by sex × race)")

    # ── Step 3: Run Greedy recourse for each (person, model) pair ─────────
    rows = []
    # Actual runs will be less than this because we skip persons already
    # predicted >50K by a given model. This is the upper bound for display.
    total_runs = len(sampled_indices) * len(trained_models)
    run_count  = 0

    for person_idx in sampled_indices:
        features, true_label = get_person(X, y, person_idx)

        # Demographic info for this person
        sex_val   = int(features['sex'])
        race_val  = int(features['race'])
        age_val   = int(features['age'])
        sex_label = SEX_REVERSE.get(sex_val, str(sex_val))
        race_label= RACE_REVERSE.get(race_val, str(race_val))

        for model_key, model_record in trained_models.items():
            run_count += 1
            if verbose and run_count % 200 == 0:
                pct = run_count / total_runs * 100
                print(f"  Progress: {run_count}/{total_runs} ({pct:.0f}%)...")

            model = model_record['model']
            pred, prob = predict_person(model, features)

            # Only run recourse if this model predicts low-income
            # (no point computing recourse for someone already predicted >50K)
            if pred != 0:
                continue

            result = greedy_recourse(
                model, features,
                target_class=1,
                max_iterations=max_iterations
            )

            # Serialize the changes dict so it can be stored in a DataFrame
            # Format: "feature:orig→new|feature:orig→new|..."
            changes_str = _serialize_changes(result.changes)

            rows.append({
                'person_idx':    person_idx,
                'model_key':     model_key,
                'model_name':    model_record['name'],
                'found':         result.found,
                'distance':      result.distance if result.found else np.nan,
                'n_changed':     result.n_changed if result.found else np.nan,
                'true_label':    true_label,
                'predicted_label': pred,
                'pred_prob':     round(prob, 4),
                'sex':           sex_val,
                'sex_label':     sex_label,
                'race':          race_val,
                'race_label':    race_label,
                'age':           age_val,
                'age_group':     _age_group(age_val),
                'changes_str':   changes_str,
            })

    effort_df = pd.DataFrame(rows)

    if verbose:
        found    = effort_df['found'].sum()
        total    = len(effort_df)
        print(f"\n  Recourse found: {found}/{total} "
              f"({found/total*100:.1f}%)")

    return effort_df


def _stratified_sample(df, n, random_state):
    """
    Samples n rows from df proportionally from each 'stratum' group.

    If n >= len(df), returns the full DataFrame.
    If a stratum has fewer rows than its proportional share, takes all of them.

    Args:
        df           (DataFrame): must have a 'stratum' column
        n            (int):       total rows to sample
        random_state (int):       for reproducibility

    Returns:
        DataFrame: sampled rows
    """
    if n >= len(df):
        return df.copy()

    rng    = np.random.default_rng(random_state)
    pieces = []

    for stratum_val, group in df.groupby('stratum'):
        # Proportional share for this stratum
        share = int(np.round(n * len(group) / len(df)))
        share = min(share, len(group))   # can't take more than exists
        share = max(share, 1)            # always take at least 1

        sampled_group = group.sample(
            n=min(share, len(group)),
            random_state=int(rng.integers(0, 10000))
        )
        pieces.append(sampled_group)

    result = pd.concat(pieces).reset_index(drop=True)

    # If rounding gave us slightly too many or too few, trim or keep as-is
    if len(result) > n:
        result = result.sample(n=n, random_state=random_state)

    return result.reset_index(drop=True)


def _age_group(age):
    """Maps a numeric age to a display-friendly age group string."""
    if age < 25:
        return 'Under 25'
    elif age < 35:
        return '25–34'
    elif age < 45:
        return '35–44'
    elif age < 55:
        return '45–54'
    elif age < 65:
        return '55–64'
    else:
        return '65+'


def _serialize_changes(changes_dict):
    """
    Converts a changes dict to a compact string for DataFrame storage.
    Format: "feature:orig_val→new_val|feature:orig_val→new_val"

    Example: "education_num:8→12|hours_per_week:40→50"

    This avoids storing nested dicts in a DataFrame column, which causes
    issues with CSV serialization and Streamlit caching.
    """
    if not changes_dict:
        return ''
    parts = []
    for fname, info in changes_dict.items():
        parts.append(f"{fname}:{info['original']:.2f}→{info['new']:.2f}")
    return '|'.join(parts)


def _deserialize_changes(changes_str):
    """
    Reverses _serialize_changes. Returns a list of (feature, orig, new) tuples.
    Used when the app needs to display individual changes from stored results.
    """
    if not changes_str or pd.isna(changes_str):
        return []
    result = []
    for part in changes_str.split('|'):
        if '→' in part and ':' in part:
            fname, vals = part.split(':', 1)
            orig_str, new_str = vals.split('→', 1)
            try:
                result.append((fname, float(orig_str), float(new_str)))
            except ValueError:
                continue
    return result


# =============================================================================
# ANALYSIS 2 — FAIRNESS ANALYSIS
# =============================================================================

def compute_fairness_analysis(effort_df):
    """
    Aggregates recourse effort by demographic group to detect disparities.

    WHAT WE ARE LOOKING FOR:
      Fairness in recourse means that people who differ only in protected
      attributes (sex, race) should require similar effort to receive a
      favorable prediction. If women systematically need to make larger
      changes than men to be predicted as >50K, that is a recourse
      disparity — even if the model has equal accuracy across sexes.

    We compute for each group:
      - mean and median effort (normalized distance)
      - success rate (fraction of persons who found recourse at all)
      - mean features changed (sparsity)
      - count of persons in the group (for statistical reliability)

    Results are computed separately for each model so we can see whether
    the disparity is model-specific or consistent across all three.

    Args:
        effort_df (DataFrame): output of compute_effort_distribution()

    Returns:
        dict with keys 'sex', 'race', 'age_group'
        Each value is a dict: {group_label → {model_key → stats_dict}}
    """

    # Only use rows where recourse was attempted (pred == 0)
    # and compute stats for found and not-found separately
    df = effort_df.copy()

    fairness = {
        'sex':       _aggregate_by_group(df, 'sex_label'),
        'race':      _aggregate_by_group(df, 'race_label'),
        'age_group': _aggregate_by_group(df, 'age_group'),
    }

    return fairness


def _aggregate_by_group(df, group_col):
    """
    For a given grouping column, computes per-group per-model statistics.

    Returns:
        dict: {group_label → {model_key → stats}}

        Stats per group-model pair:
          n_persons        — unique persons in this group
          n_attempted      — persons where recourse was attempted
          n_found          — persons where recourse was found
          success_rate     — n_found / n_attempted
          mean_distance    — mean effort (NaN excluded)
          median_distance  — median effort
          std_distance     — std dev of effort
          mean_n_changed   — mean features changed
    """
    result = {}

    for group_val, group_df in df.groupby(group_col):
        result[group_val] = {}

        for model_key, model_df in group_df.groupby('model_key'):
            n_attempted   = len(model_df)
            n_found       = int(model_df['found'].sum())
            found_df      = model_df[model_df['found'] == True]

            result[group_val][model_key] = {
                'n_persons':     int(model_df['person_idx'].nunique()),
                'n_attempted':   n_attempted,
                'n_found':       n_found,
                'success_rate':  round(n_found / n_attempted, 4)
                                 if n_attempted > 0 else 0.0,
                'mean_distance': round(float(found_df['distance'].mean()), 4)
                                 if not found_df.empty else np.nan,
                'median_distance': round(float(found_df['distance'].median()), 4)
                                   if not found_df.empty else np.nan,
                'std_distance':  round(float(found_df['distance'].std()), 4)
                                 if len(found_df) > 1 else np.nan,
                'mean_n_changed': round(float(found_df['n_changed'].mean()), 2)
                                  if not found_df.empty else np.nan,
            }

    return result


# =============================================================================
# ANALYSIS 3 — FEATURE FREQUENCY
# =============================================================================

def compute_feature_frequency(effort_df):
    """
    Counts how often each feature appears in successful recourse paths
    across all sampled persons and models.

    A high frequency for a feature means: "The model most often recommends
    changing this feature to go from low-income to high-income prediction."

    This tells us which features are the model's primary "levers":
      - High frequency: this feature is central to the model's decision
      - Low frequency: this feature rarely helps, even when changed

    Results are computed per model so we can compare which features
    each model relies on.

    Args:
        effort_df (DataFrame): output of compute_effort_distribution()
                               must have 'changes_str' and 'model_key' columns

    Returns:
        dict: {model_key → {feature_name → {'count': int, 'fraction': float}}}
    """
    result = {}

    for model_key, model_df in effort_df.groupby('model_key'):
        # Only look at rows where recourse was found
        found_df      = model_df[model_df['found'] == True]
        n_successful  = len(found_df)

        feature_counts = defaultdict(int)

        for _, row in found_df.iterrows():
            changes = _deserialize_changes(row['changes_str'])
            for fname, orig, new in changes:
                if fname in ACTIONABLE_FEATURES:
                    feature_counts[fname] += 1

        # Build frequency dict for all actionable features
        freq = {}
        for fname in ACTIONABLE_FEATURES:
            count    = feature_counts.get(fname, 0)
            fraction = count / n_successful if n_successful > 0 else 0.0
            freq[fname] = {
                'count':    count,
                'fraction': round(fraction, 4),
                'label':    FEATURE_META[fname]['label'],
            }

        result[model_key] = freq

    return result


# =============================================================================
# ANALYSIS 4 — ALGORITHM COMPARISON (SINGLE PERSON)
# =============================================================================

def compare_algorithms(model, original_features, target_class=1,
                        max_iterations=200):
    """
    Runs all three recourse algorithms on the same person and model,
    returning a side-by-side comparison.

    Used on the "Algorithm Comparison" page in the app. Lets the user see
    that the same model can suggest very different changes depending on
    which algorithm is used to find the recourse.

    Args:
        model             (sklearn model): fitted classifier
        original_features (dict):          person's features
        target_class      (int):           desired prediction (1 = >50K)
        max_iterations    (int):           cap for greedy algorithms

    Returns:
        dict: {'greedy': RecourseResult,
               'importance': RecourseResult,
               'proximity': RecourseResult}
    """
    results = {}

    for algo in ['greedy', 'importance', 'proximity']:
        # Proximity uses max_restarts, the others use max_iterations
        kwargs = ({'max_restarts': 8}
                  if algo == 'proximity'
                  else {'max_iterations': max_iterations})

        results[algo] = find_recourse(
            model, original_features,
            target_class=target_class,
            algorithm=algo,
            **kwargs
        )

    return results


# =============================================================================
# SUMMARY STATISTICS (for Home page)
# =============================================================================

def compute_summary_stats(effort_df, trained_models):
    """
    Computes high-level summary numbers shown on the Home page.

    These give the visitor an immediate sense of scale:
    "X% of low-income persons can receive actionable recourse"
    "The average effort required is Y"
    "Women need Z% more effort than men on average"

    Args:
        effort_df     (DataFrame): output of compute_effort_distribution()
        trained_models (dict):     output of train_all_models()

    Returns:
        dict with overall and per-model summary stats
    """
    overall = {
        'total_persons_sampled': int(effort_df['person_idx'].nunique()),
        'total_runs':            len(effort_df),
    }

    per_model = {}
    for model_key, model_df in effort_df.groupby('model_key'):
        found_df   = model_df[model_df['found'] == True]
        n_found    = len(found_df)
        n_total    = len(model_df)

        per_model[model_key] = {
            'model_name':      trained_models[model_key]['name'],
            'cv_accuracy':     trained_models[model_key]['cv_accuracy'],
            'n_attempted':     n_total,
            'n_found':         n_found,
            'success_rate':    round(n_found / n_total, 4) if n_total > 0 else 0.0,
            'mean_distance':   round(float(found_df['distance'].mean()), 4)
                               if not found_df.empty else np.nan,
            'median_distance': round(float(found_df['distance'].median()), 4)
                               if not found_df.empty else np.nan,
            'mean_n_changed':  round(float(found_df['n_changed'].mean()), 2)
                               if not found_df.empty else np.nan,
        }

    # Overall fairness gap: difference in mean distance between Male and Female
    # across all models combined
    sex_gap = _compute_sex_gap(effort_df)

    return {
        'overall':   overall,
        'per_model': per_model,
        'sex_gap':   sex_gap,
    }


def _compute_sex_gap(effort_df):
    """
    Computes the mean-distance gap between Male and Female persons.

    Returns:
        dict with male_mean, female_mean, gap, gap_pct (relative gap)
        Returns None if either group has no data.
    """
    found_df = effort_df[effort_df['found'] == True]
    if found_df.empty:
        return None

    male_df   = found_df[found_df['sex_label'] == 'Male']
    female_df = found_df[found_df['sex_label'] == 'Female']

    if male_df.empty or female_df.empty:
        return None

    male_mean   = float(male_df['distance'].mean())
    female_mean = float(female_df['distance'].mean())
    gap         = female_mean - male_mean
    gap_pct     = gap / male_mean * 100 if male_mean > 0 else 0.0

    return {
        'male_mean':   round(male_mean, 4),
        'female_mean': round(female_mean, 4),
        'gap':         round(gap, 4),
        'gap_pct':     round(gap_pct, 1),
    }


# =============================================================================
# MAIN — run all analyses and print a readable summary
# =============================================================================

if __name__ == "__main__":

    from recourse.data  import load_data
    from recourse.model import train_all_models

    print("=" * 65)
    print("ANALYZE.PY — FULL PIPELINE TEST")
    print("=" * 65)

    # ── Load data and train models ─────────────────────────────────────────
    print("\nStep 1: Loading data and training models...")
    X, y, df  = load_data()
    trained   = train_all_models(X, y, verbose=False)
    print(f"  {len(X)} persons, {y.sum()} high-income, "
          f"{(y==0).sum()} low-income")

    # ── Effort distribution ────────────────────────────────────────────────
    print("\nStep 2: Computing effort distribution "
          "(500 persons × 3 models)...")
    effort_df = compute_effort_distribution(
        trained, X, y,
        sample_size=500,
        max_iterations=150,
        verbose=True
    )

    print(f"\n  Effort DataFrame shape: {effort_df.shape}")
    print(f"  Columns: {list(effort_df.columns)}")

    # ── Summary stats ──────────────────────────────────────────────────────
    print("\nStep 3: Summary statistics...")
    stats = compute_summary_stats(effort_df, trained)

    print("\n  PER-MODEL SUMMARY:")
    print(f"  {'Model':<25} {'CV Acc':>7}  {'Success%':>9}  "
          f"{'Mean dist':>10}  {'Mean changed':>13}")
    print(f"  {'-'*70}")
    for key, s in stats['per_model'].items():
        print(f"  {s['model_name']:<25} "
              f"{s['cv_accuracy']:.4f}  "
              f"{s['success_rate']*100:>8.1f}%  "
              f"{s['mean_distance']:>10.4f}  "
              f"{s['mean_n_changed']:>12.2f}")

    if stats['sex_gap']:
        sg = stats['sex_gap']
        print(f"\n  SEX GAP (overall):")
        print(f"    Male mean distance:   {sg['male_mean']:.4f}")
        print(f"    Female mean distance: {sg['female_mean']:.4f}")
        print(f"    Gap:                  {sg['gap']:+.4f} "
              f"({sg['gap_pct']:+.1f}% relative)")

    # ── Fairness analysis ──────────────────────────────────────────────────
    print("\nStep 4: Fairness analysis by sex...")
    fairness = compute_fairness_analysis(effort_df)

    sex_fair = fairness['sex']
    print(f"\n  {'Group':<10} {'Model':<25} "
          f"{'Success%':>9}  {'Mean dist':>10}  {'N persons':>10}")
    print(f"  {'-'*70}")
    for group_label in sorted(sex_fair.keys()):
        for model_key, s in sex_fair[group_label].items():
            model_name = s.get('n_persons', '?')
            dist_str   = f"{s['mean_distance']:.4f}" \
                         if not (isinstance(s['mean_distance'], float) and
                                 np.isnan(s['mean_distance'])) else "  N/A"
            print(f"  {group_label:<10} "
                  f"{model_key:<25} "
                  f"{s['success_rate']*100:>8.1f}%  "
                  f"{dist_str:>10}  "
                  f"{s['n_persons']:>10}")

    # ── Feature frequency ──────────────────────────────────────────────────
    print("\nStep 5: Feature frequency across recourse paths...")
    feature_freq = compute_feature_frequency(effort_df)

    print("\n  TOP 5 FEATURES BY FREQUENCY (Decision Tree):")
    tree_freq = feature_freq.get('tree', {})
    sorted_feats = sorted(tree_freq.items(),
                          key=lambda x: x[1]['fraction'], reverse=True)
    for fname, info in sorted_feats[:5]:
        bar = '█' * int(info['fraction'] * 20)
        print(f"    {info['label']:<22} {info['fraction']*100:>5.1f}%  {bar}")

    # ── Algorithm comparison on one person ────────────────────────────────
    print("\nStep 6: Algorithm comparison on Person #0...")
    features_0, label_0 = get_person(X, y, 0)
    pred_0, prob_0 = predict_person(trained['tree']['model'], features_0)

    if pred_0 == 0:
        comparison = compare_algorithms(
            trained['tree']['model'], features_0
        )
        print(f"\n  Person #0 | P(>50K)={prob_0:.1%}")
        print(f"  {'Algorithm':<22} {'Found':>6}  {'Distance':>9}  "
              f"{'Changed':>8}")
        print(f"  {'-'*55}")
        for algo, result in comparison.items():
            found_str = "Yes" if result.found else "No"
            dist_str  = f"{result.distance:.4f}" if result.found else "  —"
            chng_str  = str(result.n_changed) if result.found else "—"
            print(f"  {algo:<22} {found_str:>6}  {dist_str:>9}  "
                  f"{chng_str:>8}")
    else:
        print("  Person #0 is already predicted >50K — no recourse needed.")

    print("\n" + "=" * 65)
    print("ALL ANALYSES COMPLETE")
    print("=" * 65)
