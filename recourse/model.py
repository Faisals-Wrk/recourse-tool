# =============================================================================
# model.py — Classifier Training and Management
# =============================================================================
# Trains and stores three classifiers on the Adult Income dataset.
#
# THE THREE CLASSIFIERS AND WHY WE CHOSE THEM:
#
#   1. Logistic Regression
#      A linear model. The decision boundary is a hyperplane in feature
#      space. For this dataset, it captures the general trend (more education,
#      more hours → higher income) cleanly. Recourse is smooth and predictable
#      because small changes in any feature shift the probability smoothly.
#      Used in the original Wachter et al. 2017 recourse paper.
#
#   2. Decision Tree
#      Non-linear, axis-aligned splits. Recourse can be "all-or-nothing":
#      you might need to change education from level 8 to level 9 precisely
#      because that is where the tree splits, not level 8.5. This makes
#      recourse less smooth but more interpretable — you can trace the path.
#
#   3. Random Forest
#      Ensemble of 100 decision trees. The strongest predictor of the three.
#      The "black box" scenario — hard to explain why, but harder to fool.
#      Recourse is averaged over all trees, so it requires more substantial
#      real changes than a single tree might.
#
# WHY THESE THREE TOGETHER?
#   They represent the interpretability-accuracy tradeoff spectrum:
#     Logistic Regression  — most interpretable, moderate accuracy
#     Decision Tree        — interpretable, good accuracy
#     Random Forest        — least interpretable, best accuracy
#   Showing that recourse varies across them makes the point that "which
#   model was deployed" matters to the individual seeking recourse.
#
# PIPELINE NOTE:
#   Logistic Regression is wrapped in a sklearn Pipeline with StandardScaler.
#   The Adult dataset has features on very different scales:
#     capital_gain: 0-99999, hours_per_week: 1-99, education_num: 1-16
#   Scaling prevents large-range features from dominating the coefficients.
#   Decision Tree and Random Forest are scale-invariant, no scaling needed.
#
# CROSS VALIDATION:
#   Evaluated with Stratified 5-Fold CV.
#   "Stratified" preserves the class ratio (~24% high income) in each fold.
#   This gives us honest accuracy estimates comparable across models.
# =============================================================================

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline

from recourse.data import load_data, FEATURE_NAMES


# =============================================================================
# SUPPRESS SKLEARN FEATURE NAME WARNING — done ONCE at import time
# =============================================================================
# When sklearn models are trained on a named DataFrame but later receive a
# numpy array, they print:
#   "X does not have valid feature names, but <Model> was fitted with feature names"
#
# WHY WE SUPPRESS IT:
#   predict_person() is called in a tight loop millions of times during the
#   greedy recourse analysis. The prediction is IDENTICAL regardless — column
#   order is guaranteed correct by FEATURE_NAMES. The warning is purely cosmetic.
#
# WHY MODULE-LEVEL (not inside predict_person):
#   Using warnings.catch_warnings() as a context manager inside predict_person()
#   saves and restores the full warning filter state on EVERY call. At 1,000,000+
#   calls this context-manager overhead alone takes several minutes.
#   A single filterwarnings() call here costs nothing and silences it permanently.

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)


# =============================================================================
# CLASSIFIER CONFIGURATIONS
# =============================================================================

CLASSIFIER_CONFIGS = [
    {
        'name':        'Logistic Regression',
        'key':         'logistic',
        'description': 'Linear model · smooth decision surface · most interpretable',
        # StandardScaler is required because LR is sensitive to feature scale.
        # We scale inside a Pipeline so that predict() automatically applies
        # the same scaling that training used.
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    LogisticRegression(C=1.0, max_iter=2000, random_state=42))
        ]),
    },
    {
        'name':        'Decision Tree',
        'key':         'tree',
        'description': 'Axis-aligned splits · fully interpretable · traceable path',
        # max_depth=6 gives a deep enough tree to capture income patterns
        # without overfitting on 10,000 samples.
        # min_samples_leaf=20 prevents tiny leaf nodes (overfit risk).
        'model': DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=20, random_state=42
        ),
    },
    {
        'name':        'Random Forest',
        'key':         'forest',
        'description': 'Ensemble of 100 trees · highest accuracy · hardest to interpret',
        # 100 trees, each sees a random subset of features and data.
        # max_depth=8 per tree allows complex patterns.
        # min_samples_leaf=10 keeps individual trees from overfitting.
        'model': RandomForestClassifier(
            n_estimators=100, max_depth=8,
            min_samples_leaf=10, random_state=42
        ),
    },
]


# =============================================================================
# TRAINING
# =============================================================================

def train_all_models(X, y, cv_folds=5, verbose=True):
    """
    Trains all three classifiers on the full dataset and evaluates each
    with Stratified 5-Fold Cross Validation.

    We train on the FULL dataset (not a held-out split) because:
      - We want predictions on all rows for the Dataset page
      - Stratified CV gives an unbiased accuracy estimate despite this

    Predictions and probabilities on the full dataset are computed once
    here (batch call — fast) and stored in the returned dict. This means
    get_all_predictions() can just read those lists without calling
    predict_person() in a per-row loop.

    Args:
        X        (DataFrame): feature matrix
        y        (Series):    binary labels
        cv_folds (int):       number of CV folds
        verbose  (bool):      print progress

    Returns:
        dict: {model_key -> model_record}

        Each model_record contains:
          name          - display name
          key           - short identifier
          description   - one-line summary
          model         - fitted sklearn estimator
          cv_accuracy   - mean CV accuracy
          cv_std        - std across folds
          cv_scores     - list of individual fold scores
          predictions   - list[int] of predictions on full dataset (length n)
          probabilities - list[float] of P(>50K) on full dataset (length n)
    """
    cv      = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    trained = {}

    for config in CLASSIFIER_CONFIGS:
        name  = config['name']
        key   = config['key']
        model = config['model']  # unfitted at this point

        if verbose:
            print(f"  Training {name}...")

        # CV gives us unbiased accuracy before we fit on the full dataset
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

        # Fit on the full dataset for deployment
        model.fit(X, y)

        # Batch predict on full dataset — done once here, stored for reuse.
        # model.predict(X) is a vectorised numpy call, very fast.
        predictions   = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]   # P(>50K)

        trained[key] = {
            'name':          name,
            'key':           key,
            'description':   config['description'],
            'model':         model,
            'cv_accuracy':   float(cv_scores.mean()),
            'cv_std':        float(cv_scores.std()),
            'cv_scores':     cv_scores.tolist(),
            'predictions':   predictions.tolist(),
            'probabilities': probabilities.tolist(),
        }

        if verbose:
            print(f"    CV accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    if verbose:
        print("  All models trained.")

    return trained


# =============================================================================
# PREDICTION HELPERS
# =============================================================================

def predict_person(model, features_dict):
    """
    Runs the model on a single person's features (given as a dict).
    Returns both the binary prediction and the high-income probability.

    PERFORMANCE DESIGN:
      This function is called in a very tight inner loop by the greedy
      recourse algorithms — up to 1,000,000+ times during the fairness
      analysis (200 persons x 3 models x 100 iterations x ~18 candidates).

      Two things make it fast:

      1. Numpy array input (no DataFrame creation per call):
         sklearn models accept numpy arrays natively. Column order is
         guaranteed correct because we always use FEATURE_NAMES order.
         This is ~15x faster than pd.DataFrame([features_dict]) per call.

      2. Warning suppressed at module level (not per call):
         sklearn prints "X does not have valid feature names" when a numpy
         array is passed to a model trained on a DataFrame. Suppressing
         this with warnings.catch_warnings() INSIDE this function would
         save/restore the full warning state on every call — at 1M+ calls
         that overhead alone takes several minutes. Instead we call
         warnings.filterwarnings("ignore", ...) once at module import
         time (see top of this file). Free at call time, same effect.

    Args:
        model         (sklearn model): fitted classifier or Pipeline
        features_dict (dict): {feature_name: value, ...}

    Returns:
        prediction   (int):   0 = <=50K, 1 = >50K
        probability  (float): P(>50K) between 0 and 1
    """
    # Build a (1, n_features) numpy array in the fixed FEATURE_NAMES order.
    row         = np.array([[features_dict[f] for f in FEATURE_NAMES]],
                           dtype=float)
    prediction  = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0, 1])
    return prediction, probability


def get_prediction_summary(trained_models, features_dict):
    """
    Runs all three models on one person and returns a summary dict.
    Used to show consensus / disagreement on the Dataset page.

    Args:
        trained_models (dict): output of train_all_models()
        features_dict  (dict): one person's features

    Returns:
        dict: {model_key -> {'name', 'prediction', 'probability'}}
    """
    summary = {}
    for key, record in trained_models.items():
        pred, prob = predict_person(record['model'], features_dict)
        summary[key] = {
            'name':        record['name'],
            'prediction':  pred,
            'probability': round(prob, 4),
        }
    return summary


def get_all_predictions(trained_models, X):
    """
    Returns a DataFrame with predictions from all three models for every
    person. Used on the Dataset page for the feature distributions and
    model agreement section.

    PERFORMANCE FIX:
      Old version called predict_person() in a Python for-loop: 9,999 rows
      x 3 models = 29,997 individual model inference calls. Even with numpy
      this loop takes 1-2 minutes and prints 60,000+ warnings.

      New version reads from trained[key]['predictions'] which was computed
      by model.predict(X) (one fast vectorised call) inside train_all_models().
      This reduces 29,997 calls to 3 list reads — under 1 second.

    Returns:
        DataFrame with columns:
            person_idx, logistic_pred, tree_pred, forest_pred,
            logistic_prob, tree_prob, forest_prob, consensus
    """
    n    = len(X)
    data = {'person_idx': list(range(n))}

    # Read the pre-computed batch predictions stored during training.
    # No model inference needed here at all.
    preds_by_model = {}
    for key, record in trained_models.items():
        preds = record['predictions']          # list[int], length n
        probs = record['probabilities']        # list[float], length n
        data[f'{key}_pred'] = preds
        data[f'{key}_prob'] = [round(p, 4) for p in probs]
        preds_by_model[key] = preds

    # Consensus: do all three models agree for this person?
    model_keys = list(preds_by_model.keys())
    consensus  = [
        len({preds_by_model[k][i] for k in model_keys}) == 1
        for i in range(n)
    ]
    data['consensus'] = consensus

    return pd.DataFrame(data)


# =============================================================================
# MAIN -- test this module
# =============================================================================

if __name__ == "__main__":

    import time

    print("=" * 60)
    print("MODEL TRAINING TEST")
    print("=" * 60)

    X, y, df = load_data()
    print(f"\nDataset: {len(X)} rows, {len(X.columns)} features")
    print(f"High income: {y.sum()} ({y.mean()*100:.1f}%)")

    print("\nTraining all models...")
    trained = train_all_models(X, y)

    print("\n" + "=" * 60)
    print("ACCURACY SUMMARY")
    print("=" * 60)
    for key, record in trained.items():
        print(f"\n  {record['name']}")
        print(f"    CV Accuracy: {record['cv_accuracy']:.4f} +/- {record['cv_std']:.4f}")
        print(f"    Fold scores: {[round(s,4) for s in record['cv_scores']]}")

    # get_all_predictions should be instant
    t0        = time.time()
    all_preds = get_all_predictions(trained, X)
    elapsed   = time.time() - t0
    disagree  = all_preds[~all_preds['consensus']]
    print(f"\nget_all_predictions() runtime: {elapsed:.3f}s  "
          f"(should be <0.1s)")
    print(f"Persons where models disagree: "
          f"{len(disagree)} / {len(X)} "
          f"({len(disagree)/len(X)*100:.1f}%)")

    # predict_person: single call speed test
    from recourse.data import get_person
    features, label = get_person(X, y, 0)

    t0 = time.time()
    for _ in range(1000):
        predict_person(trained['tree']['model'], features)
    elapsed = (time.time() - t0) / 1000 * 1000  # ms per call
    print(f"\npredict_person() avg time: {elapsed:.3f}ms per call")
    print(f"  At 1M calls: {elapsed * 1000:.1f}s  (should be <60s)")

    summary = get_prediction_summary(trained, features)
    print(f"\nPerson #0 (true: {'>50K' if label else '<=50K'}):")
    for key, info in summary.items():
        pred_str = ">50K" if info['prediction'] == 1 else "<=50K"
        print(f"  {info['name']:<22} -> {pred_str}  "
              f"({info['probability']:.1%} probability)")
