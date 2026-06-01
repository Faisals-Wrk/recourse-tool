# =============================================================================
# counterfactual.py — Algorithmic Recourse: Three Search Algorithms
# =============================================================================
# Answers the question: "What is the minimum change a person predicted as
# low-income (<=50K) needs to make to be predicted as high-income (>50K)?"
#
# All three algorithms share the same interface and constraints:
#
#   INPUT:  original person features (dict) + fitted model
#   OUTPUT: RecourseResult — the counterfactual + metadata
#
#   HARD CONSTRAINTS (all algorithms):
#     1. Immutable features (sex, race, native_country_us) are NEVER changed
#     2. Increase-only features (age, education_num) can only go UP
#     3. All feature values stay within their valid [min, max] ranges
#
# THE THREE ALGORITHMS:
#
#   A. Greedy Perturbation
#      Each step: try every valid ±perturbation across all actionable features.
#      Keep the one change that most increases P(>50K). Repeat until flip.
#      Fast, simple, works well for tree-based models.
#
#   B. Importance-Guided Perturbation
#      Like Greedy, but weights candidate moves by the model's feature
#      importances. High-importance features are preferred — the algorithm
#      aligns with how the model actually makes decisions.
#
#   C. Proximity Minimization (Wachter et al. 2017)
#      Treats recourse as an optimization problem. Uses scipy L-BFGS-B to
#      minimize normalized distance subject to prediction flip.
#      Theoretically cleanest, most computationally expensive.
#
# REFERENCE:
#   Wachter, S., Mittelstadt, B., & Russell, C. (2017).
#   "Counterfactual Explanations Without Opening the Black Box."
#   Harvard Journal of Law & Technology.
# =============================================================================

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses    import dataclass, field
from typing         import Optional, Dict

from recourse.data  import (
    FEATURE_META, FEATURE_NAMES,
    ACTIONABLE_FEATURES, IMMUTABLE_FEATURES,
    clip_to_valid_range, compute_distance, count_changed_features,
    feature_value_label
)
from recourse.model import predict_person


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class RecourseResult:
    """
    Standard output format returned by all three algorithms.

    Every field here is read by the Streamlit app to display the
    recourse recommendation. Using a dataclass makes the interface
    clean and self-documenting.

    Fields:
        found           — True if algorithm found a valid counterfactual
        original        — person's original feature values (dict)
        counterfactual  — modified feature values that flip the prediction
        changes         — {feature: {original, new, delta, label, unit}}
                          only includes features that actually changed
        n_changed       — number of features that changed (sparsity metric)
        distance        — normalized L1 distance (smaller = better recourse)
        original_pred   — model's prediction on original (should be 0 = <=50K)
        cf_pred         — model's prediction on CF (should be 1 = >50K)
        original_prob   — P(>50K) for the original person
        cf_prob         — P(>50K) for the counterfactual person
        n_iterations    — how many steps the algorithm took
        algorithm       — which algorithm produced this (for display)
        message         — human-readable summary shown in the UI
    """
    found:           bool
    original:        Dict
    counterfactual:  Optional[Dict]
    changes:         Dict
    n_changed:       int
    distance:        float
    original_pred:   int
    cf_pred:         Optional[int]
    original_prob:   float
    cf_prob:         Optional[float]
    n_iterations:    int
    algorithm:       str
    message:         str


# =============================================================================
# SHARED INTERNAL UTILITIES
# =============================================================================

def _changes_summary(original, counterfactual):
    """
    Computes a human-readable summary of what changed between the
    original person and the counterfactual.

    Only includes features that actually changed (delta != 0).
    Adds the human-readable label and unit from FEATURE_META.

    Returns:
        dict: {feature_name: {original, new, delta, label, unit,
                              original_label, new_label}}
    """
    changes = {}
    for fname in ACTIONABLE_FEATURES:
        orig_val = float(original[fname])
        new_val  = float(counterfactual[fname])
        
        # Use a meaningful threshold per feature type:
        # categorical/integer features: must change by at least 0.5
        # continuous features (capital_gain, hours): at least 0.1
        continuous = {'capital_gain', 'capital_loss', 'hours_per_week'}
        threshold  = 0.1 if fname in continuous else 0.5
        if abs(new_val - orig_val) > threshold:
            changes[fname] = {
                'original':      orig_val,
                'new':           new_val,
                'delta':         new_val - orig_val,
                'label':         FEATURE_META[fname]['label'],
                'unit':          FEATURE_META[fname]['unit'],
                # Human-readable versions (e.g. "Bachelors" instead of 12)
                'original_label': feature_value_label(fname, orig_val),
                'new_label':      feature_value_label(fname, new_val),
            }
    return changes


def _build_result(algorithm_name, original, counterfactual,
                  model, n_iterations, found, message):
    """
    Assembles a RecourseResult from the raw algorithm outputs.
    Called at the end of each algorithm function to produce a
    standardized result regardless of which algorithm ran.
    """
    orig_pred, orig_prob = predict_person(model, original)

    if found and counterfactual is not None:
        cf_pred, cf_prob = predict_person(model, counterfactual)
        changes          = _changes_summary(original, counterfactual)
        n_changed        = count_changed_features(original, counterfactual)
        distance         = compute_distance(original, counterfactual)
    else:
        # Not found — return empty/zero values
        cf_pred   = None
        cf_prob   = None
        changes   = {}
        n_changed = 0
        distance  = 0.0

    return RecourseResult(
        found          = found,
        original       = original,
        counterfactual = counterfactual,
        changes        = changes,
        n_changed      = n_changed,
        distance       = distance,
        original_pred  = orig_pred,
        cf_pred        = cf_pred,
        original_prob  = orig_prob,
        cf_prob        = cf_prob,
        n_iterations   = n_iterations,
        algorithm      = algorithm_name,
        message        = message,
    )


def _get_valid_directions(fname, current_value):
    """
    Returns which directions (+step, -step) are valid for a given feature
    at its current value.

    Respects two constraints:
      1. increase_only features can only go UP (direction = +1 only)
      2. Values at the boundary cannot move further in that direction

    Args:
        fname         (str):   feature name
        current_value (float): current value of the feature

    Returns:
        list of int: subset of [+1, -1] that are valid moves
    """
    meta       = FEATURE_META[fname]
    directions = []

    # Can we go up?
    if current_value < meta['max']:
        directions.append(+1)

    # Can we go down? (only if the feature is not increase-only)
    if not meta['increase_only'] and current_value > meta['min']:
        directions.append(-1)

    return directions


# =============================================================================
# ALGORITHM A — GREEDY PERTURBATION
# =============================================================================

def greedy_recourse(model, original_features, target_class=1,
                    max_iterations=300, verbose=False):
    """
    Finds recourse by greedily choosing the best single-feature
    perturbation at each step.

    ALGORITHM:
      1. Start from original features.
      2. Try every actionable feature in every valid direction.
         For each candidate, compute P(target_class | candidate).
      3. Keep the single change that most increases P(target_class).
      4. Apply it and repeat.
      5. Stop when prediction == target_class or max_iterations hit.

    WHY THIS WORKS:
      At each step we're making the locally optimal move — the feature
      change that most moves the probability toward the target. For
      decision trees, which have flat regions separated by sharp splits,
      this greedy approach efficiently finds the right splits to cross.

    WHY TARGET_CLASS=1 HERE:
      In the Adult Income context, we are always helping a person who is
      predicted as low-income (<=50K) to find a path to being predicted
      as high-income (>50K). So target_class=1 (>50K) is the default.

    Args:
        model             (sklearn model): fitted classifier or Pipeline
        original_features (dict):          person's original feature values
        target_class      (int):           desired prediction (1 = >50K)
        max_iterations    (int):           safety cap to prevent infinite loops
        verbose           (bool):          print step-by-step for debugging

    Returns:
        RecourseResult
    """
    # Work on a mutable copy — never modify the original
    current = {k: float(v) for k, v in original_features.items()}

    # Check if the person already has the target prediction
    # (recourse not needed — already earning >50K by model's judgment)
    current_pred, current_prob = predict_person(model, current)
    if current_pred == target_class:
        return _build_result(
            'Greedy', original_features, current, model, 0, True,
            "Already predicted as target class — no recourse needed."
        )

    for iteration in range(max_iterations):

        best_candidate = None
        best_prob      = -np.inf  # highest P(target_class) seen so far

        # ── Try every actionable feature in every valid direction ──────────
        for fname in ACTIONABLE_FEATURES:
            meta       = FEATURE_META[fname]
            directions = _get_valid_directions(fname, current[fname])

            for direction in directions:
                # Build candidate: change only this one feature by one step
                candidate        = current.copy()
                candidate[fname] = float(np.clip(
                    current[fname] + direction * meta['step'],
                    meta['min'],
                    meta['max']
                ))

                # Skip if value didn't actually change (at boundary)
                if abs(candidate[fname] - current[fname]) < 1e-9:
                    continue

                # How probable is the target class after this change?
                _, prob = predict_person(model, candidate)

                # For target_class=1, we want to maximize P(>50K)
                # For target_class=0, we want to minimize P(>50K) = maximize P(<=50K)
                target_prob = prob if target_class == 1 else (1.0 - prob)

                if target_prob > best_prob:
                    best_prob      = target_prob
                    best_candidate = candidate

        # If no valid perturbation exists, we are stuck
        if best_candidate is None:
            return _build_result(
                'Greedy', original_features, None, model,
                iteration, False,
                f"No valid perturbation found at iteration {iteration}. "
                "Person may be in a flat region of the model."
            )

        # Apply the best single-step change
        current = best_candidate
        pred, prob = predict_person(model, current)

        if verbose:
            print(f"  Step {iteration+1}: P(>50K)={prob:.3f} pred={pred}")

        # Check if prediction has flipped to the target
        if pred == target_class:
            return _build_result(
                'Greedy', original_features, current, model,
                iteration + 1, True,
                f"Recourse found in {iteration+1} steps."
            )

    # Exhausted all iterations without flipping
    return _build_result(
        'Greedy', original_features, None, model,
        max_iterations, False,
        f"Recourse not found within {max_iterations} iterations."
    )


# =============================================================================
# ALGORITHM B — IMPORTANCE-GUIDED PERTURBATION
# =============================================================================

def importance_guided_recourse(model, original_features, target_class=1,
                                max_iterations=300, verbose=False):
    """
    Like Greedy, but weights candidate moves by the model's feature
    importance scores.

    INTUITION:
      If the model relies heavily on 'education_num' to predict income,
      then changing education is the most direct path to flipping the
      prediction. Importance-Guided uses this knowledge to prefer
      high-importance features over low-importance ones.

      When two perturbations produce equally good probability gains,
      the one on a more-important feature is preferred. The algorithm
      aligns the recourse path with the model's actual decision logic.

    HOW IMPORTANCE IS EXTRACTED:
      - Random Forest / Decision Tree: model.feature_importances_
        (Gini importance — total impurity reduction per feature)
      - Logistic Regression (Pipeline): |coefficients| on scaled features
        (larger absolute coefficient → more influence on prediction)
      - Fallback (unknown model): uniform weights (same as plain Greedy)

    Args:
        model             (sklearn model): fitted classifier or Pipeline
        original_features (dict):          person's original features
        target_class      (int):           desired prediction
        max_iterations    (int):           safety cap
        verbose           (bool):          debug output

    Returns:
        RecourseResult
    """
    # Extract and normalize feature importances
    importances = _extract_feature_importances(model)
    total       = sum(importances.values())
    if total > 0:
        importances = {f: v / total for f, v in importances.items()}

    current = {k: float(v) for k, v in original_features.items()}

    # Check if already at target
    current_pred, _ = predict_person(model, current)
    if current_pred == target_class:
        return _build_result(
            'Importance-Guided', original_features, current, model, 0, True,
            "Already predicted as target class — no recourse needed."
        )

    for iteration in range(max_iterations):

        best_candidate = None
        best_score     = -np.inf  # combined score: prob_gain × importance_weight

        for fname in ACTIONABLE_FEATURES:
            meta       = FEATURE_META[fname]
            weight     = importances.get(fname, 1.0 / len(ACTIONABLE_FEATURES))
            directions = _get_valid_directions(fname, current[fname])

            for direction in directions:
                candidate        = current.copy()
                candidate[fname] = float(np.clip(
                    current[fname] + direction * meta['step'],
                    meta['min'], meta['max']
                ))

                if abs(candidate[fname] - current[fname]) < 1e-9:
                    continue

                _, prob     = predict_person(model, candidate)
                target_prob = prob if target_class == 1 else (1.0 - prob)

                # Score = probability gain weighted by feature importance.
                # The (1 + weight) ensures even zero-importance features
                # still get a base score of 1 × target_prob.
                score = target_prob * (1.0 + weight)

                if score > best_score:
                    best_score     = score
                    best_candidate = candidate

        if best_candidate is None:
            return _build_result(
                'Importance-Guided', original_features, None, model,
                iteration, False,
                f"No valid perturbation found at iteration {iteration}."
            )

        current    = best_candidate
        pred, prob = predict_person(model, current)

        if verbose:
            print(f"  Step {iteration+1}: P(>50K)={prob:.3f} pred={pred}")

        if pred == target_class:
            return _build_result(
                'Importance-Guided', original_features, current, model,
                iteration + 1, True,
                f"Recourse found in {iteration+1} steps (importance-guided)."
            )

    return _build_result(
        'Importance-Guided', original_features, None, model,
        max_iterations, False,
        f"Recourse not found within {max_iterations} iterations."
    )


def _extract_feature_importances(model):
    """
    Extracts feature importances from any of our three model types.

    Returns raw (unnormalized) importance scores as a dict.
    The calling function normalizes them.

    Returns:
        dict: {feature_name: importance_score (float)}
    """
    # Default: uniform — each feature equally weighted
    n            = len(FEATURE_NAMES)
    importances  = {f: 1.0 / n for f in FEATURE_NAMES}

    try:
        from sklearn.pipeline import Pipeline

        # Unwrap Pipeline to get the actual classifier
        clf = model.named_steps['clf'] if isinstance(model, Pipeline) else model

        if hasattr(clf, 'feature_importances_'):
            # Decision Tree and Random Forest: Gini-based importance
            for fname, imp in zip(FEATURE_NAMES, clf.feature_importances_):
                importances[fname] = float(imp)

        elif hasattr(clf, 'coef_'):
            # Logistic Regression: absolute value of coefficients
            # coef_ shape is (1, n_features) for binary classification
            for fname, coef in zip(FEATURE_NAMES, clf.coef_[0]):
                importances[fname] = float(abs(coef))

    except Exception:
        # If anything goes wrong, fall back to uniform weights silently
        pass

    return importances


# =============================================================================
# ALGORITHM C — PROXIMITY MINIMIZATION (WACHTER ET AL. 2017)
# =============================================================================

def proximity_recourse(model, original_features, target_class=1,
                       distance_metric='l1', lambda_weight=1.0,
                       max_restarts=5, verbose=False):
    """
    Finds recourse by solving a constrained optimization problem:

        minimize:   λ · normalized_distance(x, x')  +  flip_penalty(x')
        subject to: x'[immutable] = x[immutable]   (immutability)
                    x'[increase_only] >= x[...]     (one-directional)
                    x'[f] ∈ [min_f, max_f]          (valid ranges)

    The flip_penalty term is:
        max(0, 0.5 - P(target_class | x'))²
    This is 0 when the prediction already favors target_class, and
    grows quadratically the further the probability is from 0.5.

    WHY CONTINUOUS RELAXATION?
      scipy.optimize works in continuous space. We let all features take
      any real value during optimization, then round discrete (categorical
      and integer) features to the nearest valid integer at the end.
      This works because the optimization surface is smooth enough that
      continuous solutions land close to discrete optima.

    WHY MULTIPLE RESTARTS?
      The loss surface has local minima — a restart from a different
      starting point may find a shorter path. We run max_restarts times
      and return the one with the smallest distance.

    REFERENCE:
      Wachter et al. 2017: "Counterfactual Explanations Without
      Opening the Black Box: Automated Decisions and the GDPR"

    Args:
        model             (sklearn model): fitted classifier
        original_features (dict):          person's original features
        target_class      (int):           desired prediction (1 = >50K)
        distance_metric   (str):           'l1' or 'l2'
        lambda_weight     (float):         weight on the distance term
                                           (higher → prefer closer CF)
        max_restarts      (int):           number of random starting points
        verbose           (bool):          debug output

    Returns:
        RecourseResult
    """
    # Convert features dict to a numpy array for scipy.optimize
    x_orig = np.array([float(original_features[f]) for f in FEATURE_NAMES])

    # Ranges for normalization and clipping
    ranges = np.array([
        max(FEATURE_META[f]['max'] - FEATURE_META[f]['min'], 1.0)
        for f in FEATURE_NAMES
    ], dtype=float)

    # Build scipy bounds: (min, max) per feature
    # Enforce increase_only by setting lower bound to current value
    scipy_bounds = []
    for f in FEATURE_NAMES:
        meta = FEATURE_META[f]
        lo   = meta['min']
        hi   = meta['max']
        if meta.get('increase_only', False):
            # Feature can only go up from its current value
            lo = float(original_features[f])
        scipy_bounds.append((lo, hi))

    # Indices of immutable features — their contribution to the loss is zeroed
    immutable_idx = [FEATURE_NAMES.index(f) for f in IMMUTABLE_FEATURES]

    def loss(x_candidate):
        """
        Combined loss: distance + prediction penalty.
        Called repeatedly by scipy.optimize during minimization.
        """
        x_c = x_candidate.copy()

        # Hard-enforce immutability: reset immutable features to original
        for idx in immutable_idx:
            x_c[idx] = x_orig[idx]

        # ── Distance term ──────────────────────────────────────────────────
        # Only count actionable features in the distance.
        # Normalize each by its range so all features are on [0, 1] scale.
        action_idx = [FEATURE_NAMES.index(f) for f in ACTIONABLE_FEATURES]
        diffs      = np.abs(x_c[action_idx] - x_orig[action_idx]) / ranges[action_idx]

        if distance_metric == 'l1':
            dist = float(np.sum(diffs))
        else:
            dist = float(np.sqrt(np.sum(diffs ** 2)))

        # ── Prediction penalty ─────────────────────────────────────────────
        # We want P(target_class | x_c) > 0.5
        # Penalty = 0 if already flipped, grows quadratically otherwise
        features_c = {f: x_c[i] for i, f in enumerate(FEATURE_NAMES)}
        _, prob    = predict_person(model, features_c)
        target_prob = prob if target_class == 1 else (1.0 - prob)

        # Quadratic penalty kicks in when target_prob < 0.5
        penalty = max(0.0, 0.5 - target_prob) ** 2

        # The 10.0 weight on penalty means the optimizer strongly prioritizes
        # flipping the prediction over minimizing distance
        return lambda_weight * dist + 10.0 * penalty

    # ── Multiple random restarts ───────────────────────────────────────────
    best_result   = None
    best_distance = np.inf

    for restart in range(max_restarts):

        # Starting point:
        #   restart=0: start from the original features (closest to the person)
        #   restart>0: add small random perturbations to explore the space
        if restart == 0:
            x_start = x_orig.copy()
        else:
            rng     = np.random.default_rng(restart * 17)  # reproducible seed
            noise   = rng.uniform(-0.05, 0.05, size=len(x_orig))
            x_start = np.clip(
                x_orig + noise * ranges,
                [b[0] for b in scipy_bounds],
                [b[1] for b in scipy_bounds]
            )

        result = minimize(
            loss,
            x_start,
            method='L-BFGS-B',         # handles bounds natively
            bounds=scipy_bounds,
            options={'maxiter': 1000, 'ftol': 1e-12}
        )

        # ── Post-process the optimization result ───────────────────────────
        x_opt = result.x.copy()

        # Re-enforce immutability (optimizer may have drifted slightly)
        for idx in immutable_idx:
            x_opt[idx] = x_orig[idx]

        # Snap discrete features to nearest valid integer
        x_opt = _snap_discrete(x_opt, FEATURE_NAMES)

        # Clip everything to valid ranges
        features_opt = {f: float(np.clip(x_opt[i],
                                          FEATURE_META[f]['min'],
                                          FEATURE_META[f]['max']))
                        for i, f in enumerate(FEATURE_NAMES)}

        # Re-enforce increase_only after snapping
        for f in FEATURE_NAMES:
            if FEATURE_META[f].get('increase_only', False):
                features_opt[f] = max(features_opt[f],
                                      float(original_features[f]))

        # Check if this restart found a valid counterfactual
        pred, prob = predict_person(model, features_opt)

        if pred == target_class:
            dist = compute_distance(original_features, features_opt)
            if dist < best_distance:
                best_distance = dist
                best_result   = features_opt

        if verbose:
            print(f"  Restart {restart+1}: pred={pred} "
                  f"P(>50K)={prob:.3f} loss={result.fun:.4f}")

    # ── Return the best counterfactual found ───────────────────────────────
    if best_result is not None:
        return _build_result(
            'Proximity (Wachter)', original_features, best_result, model,
            max_restarts, True,
            f"Recourse found via proximity minimization "
            f"({max_restarts} restarts, distance={best_distance:.4f})."
        )
    else:
        return _build_result(
            'Proximity (Wachter)', original_features, None, model,
            max_restarts, False,
            "Proximity minimization could not find a valid counterfactual. "
            "Try increasing max_restarts or relaxing constraints."
        )


def _snap_discrete(x_array, feature_list):
    """
    After continuous optimization, rounds discrete-valued features to the
    nearest valid integer.

    Continuous features (capital_gain, capital_loss, hours_per_week) are
    left as-is since they can take any value in their range.

    All other features (age, education, occupation, etc.) are rounded
    because they represent integer or categorical values.

    Args:
        x_array      (np.ndarray): optimized feature array
        feature_list (list):       feature names in order

    Returns:
        np.ndarray: same array with discrete features rounded
    """
    # These features are continuous and should NOT be rounded
    continuous = {'capital_gain', 'capital_loss', 'hours_per_week'}

    x_snapped = x_array.copy()
    for i, fname in enumerate(feature_list):
        if fname not in continuous:
            meta             = FEATURE_META[fname]
            x_snapped[i]     = float(int(np.round(
                np.clip(x_snapped[i], meta['min'], meta['max'])
            )))
    return x_snapped


# =============================================================================
# UNIFIED ENTRY POINT
# =============================================================================

def find_recourse(model, original_features, target_class=1,
                  algorithm='greedy', **kwargs):
    """
    Unified dispatcher — runs the chosen algorithm and returns a
    RecourseResult.

    This is the function called by analyze.py and app.py.
    Having a single entry point means the rest of the codebase doesn't
    need to import the three individual algorithm functions.

    Args:
        model             (sklearn model): fitted classifier
        original_features (dict):          person's features
        target_class      (int):           desired prediction (1 = >50K)
        algorithm         (str):           'greedy', 'importance', 'proximity'
        **kwargs:                          forwarded to the algorithm function

    Returns:
        RecourseResult
    """
    if algorithm == 'greedy':
        return greedy_recourse(model, original_features,
                               target_class=target_class, **kwargs)
    elif algorithm == 'importance':
        return importance_guided_recourse(model, original_features,
                                          target_class=target_class, **kwargs)
    elif algorithm == 'proximity':
        return proximity_recourse(model, original_features,
                                  target_class=target_class, **kwargs)
    else:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            "Choose 'greedy', 'importance', or 'proximity'."
        )


# =============================================================================
# MAIN — test all three algorithms on three representative persons
# =============================================================================

if __name__ == "__main__":

    from recourse.data  import load_data, get_person
    from recourse.model import train_all_models

    print("=" * 65)
    print("COUNTERFACTUAL RECOURSE TEST")
    print("=" * 65)

    X, y, df   = load_data()
    trained    = train_all_models(X, y, verbose=False)

    # Find test persons: ones predicted as low-income by the decision tree
    tree_model = trained['tree']['model']

    # Get a few low-income predictions to test on
    test_indices = []
    for idx in range(len(X)):
        features, label = get_person(X, y, idx)
        pred, prob = predict_person(tree_model, features)
        if pred == 0:   # predicted low-income
            test_indices.append(idx)
        if len(test_indices) == 3:
            break

    for person_idx in test_indices:
        features, true_label = get_person(X, y, person_idx)
        pred, prob           = predict_person(tree_model, features)

        print(f"\n{'='*65}")
        print(f"Person #{person_idx}  "
              f"| True: {'>50K' if true_label else '<=50K'}"
              f"  | Predicted: {'>50K' if pred else '<=50K'}"
              f"  | P(>50K)={prob:.1%}")
        print(f"{'='*65}")

        for algo_name in ['greedy', 'importance', 'proximity']:
            # proximity uses max_restarts, not max_iterations
            kwargs = {'max_restarts': 5} if algo_name == 'proximity' \
                     else {'max_iterations': 200}
            result = find_recourse(
                tree_model, features,
                target_class=1,
                algorithm=algo_name,
                **kwargs
            )

            print(f"\n  [{algo_name.upper():12}]  "
                  f"Found: {result.found}  |  "
                  f"Distance: {result.distance:.4f}  |  "
                  f"Changed: {result.n_changed} features")

            if result.found:
                # Verify the counterfactual is actually valid
                cf_pred, cf_prob = predict_person(tree_model,
                                                  result.counterfactual)
                print(f"    CF prediction: {'>50K' if cf_pred else '<=50K'} "
                      f"(P={cf_prob:.1%})  "
                      f"[{'✓ VALID' if cf_pred == 1 else '✗ INVALID'}]")
                for fname, change in result.changes.items():
                    arrow = "↑" if change['delta'] > 0 else "↓"
                    print(f"    {FEATURE_META[fname]['label']:<22}  "
                          f"{change['original_label']} → "
                          f"{change['new_label']}  {arrow}")
            else:
                print(f"    {result.message}")

    print("\n" + "=" * 65)
    print("CONSTRAINT VERIFICATION")
    print("=" * 65)
    # Verify that immutable features were never changed
    # and increase_only features only went up
    features, _ = get_person(X, y, test_indices[0])
    result       = find_recourse(tree_model, features, algorithm='greedy')
    if result.found:
        print("\nImmutable features — should show NO change:")
        for fname in IMMUTABLE_FEATURES:
            orig = features[fname]
            cf   = result.counterfactual[fname]
            changed = "CHANGED ✗" if abs(orig - cf) > 1e-9 else "unchanged ✓"
            print(f"  {fname:<22} {orig} → {cf}  [{changed}]")
        print("\nIncrease-only features — should only go UP:")
        for fname in ACTIONABLE_FEATURES:
            if FEATURE_META[fname]['increase_only']:
                orig = features[fname]
                cf   = result.counterfactual[fname]
                direction = "DECREASED ✗" if cf < orig - 1e-9 else "unchanged or ↑ ✓"
                print(f"  {fname:<22} {orig:.0f} → {cf:.0f}  [{direction}]")
