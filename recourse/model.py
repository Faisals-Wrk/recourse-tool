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
#     capital_gain: 0–99999, hours_per_week: 1–99, education_num: 1–16
#   Scaling prevents large-range features from dominating the coefficients.
#   Decision Tree and Random Forest are scale-invariant — no scaling needed.
#
# CROSS VALIDATION:
#   Evaluated with Stratified 5-Fold CV.
#   "Stratified" preserves the class ratio (~24% high income) in each fold.
#   This gives us honest accuracy estimates comparable across models.
# =============================================================================

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
      - We want predictions on all rows for the Person Explorer page
      - Stratified CV gives an unbiased accuracy estimate despite this
      - This is the same strategy as Project 1 (Rashomon Set Visualizer)

    Args:
        X        (DataFrame): feature matrix
        y        (Series):    binary labels
        cv_folds (int):       number of CV folds
        verbose  (bool):      print progress

    Returns:
        dict: {model_key → model_record}

        Each model_record contains:
          name          — display name
          key           — short identifier
          description   — one-line summary
          model         — fitted sklearn estimator
          cv_accuracy   — mean CV accuracy
          cv_std        — std across folds
          cv_scores     — list of individual fold scores
          predictions   — list of predictions on full dataset
          probabilities — list of P(>50K) for each person
    """
    cv        = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    trained   = {}

    for config in CLASSIFIER_CONFIGS:
        name  = config['name']
        key   = config['key']
        model = config['model']  # unfitted at this point

        if verbose:
            print(f"  Training {name}...")

        # CV gives us unbiased accuracy before we fit on the full dataset
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

        # Fit on the full dataset for deployment (predictions on all persons)
        model.fit(X, y)

        predictions   = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]  # P(>50K)

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
            print(f"    CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

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

    This is called repeatedly by recourse algorithms after each
    perturbation to check if the prediction has flipped.

    Args:
        model         (sklearn model): fitted classifier or Pipeline
        features_dict (dict): {feature_name: value, ...}

    Returns:
        prediction   (int):   0 = <=50K, 1 = >50K
        probability  (float): P(>50K) between 0 and 1
    """
    row         = pd.DataFrame([features_dict])[FEATURE_NAMES]
    prediction  = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0, 1])
    return prediction, probability


def get_prediction_summary(trained_models, features_dict):
    """
    Runs all three models on one person and returns a summary.
    Used on the Person Explorer page to show consensus/disagreement.

    Args:
        trained_models (dict): output of train_all_models()
        features_dict  (dict): one person's features

    Returns:
        dict: {model_key → {'name', 'prediction', 'probability'}}
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
    Returns a DataFrame with predictions from all three models for
    every person. Used on the Dataset and Person Explorer pages.

    Columns:
        person_idx, logistic_pred, tree_pred, forest_pred,
        logistic_prob, tree_prob, forest_prob, consensus
    """
    rows = []
    for idx in range(len(X)):
        features = X.iloc[idx].to_dict()
        row      = {'person_idx': idx}
        preds    = []
        for key, record in trained_models.items():
            pred, prob           = predict_person(record['model'], features)
            row[f'{key}_pred']   = pred
            row[f'{key}_prob']   = round(prob, 4)
            preds.append(pred)
        # Consensus: all three models agree
        row['consensus'] = (len(set(preds)) == 1)
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# MAIN — test this module
# =============================================================================

if __name__ == "__main__":

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
        print(f"    CV Accuracy: {record['cv_accuracy']:.4f} ± {record['cv_std']:.4f}")
        print(f"    Fold scores: {[round(s,4) for s in record['cv_scores']]}")

    # Show model disagreement across the dataset
    all_preds    = get_all_predictions(trained, X)
    disagreements = all_preds[~all_preds['consensus']]
    print(f"\nPersons where models disagree: "
          f"{len(disagreements)} / {len(X)} "
          f"({len(disagreements)/len(X)*100:.1f}%)")

    # Show one person's predictions
    from recourse.data import get_person
    features, label = get_person(X, y, 0)
    summary = get_prediction_summary(trained, features)
    print(f"\nPerson #0 (true: {'>50K' if label else '<=50K'}):")
    for key, info in summary.items():
        pred_str = ">50K" if info['prediction'] == 1 else "<=50K"
        print(f"  {info['name']:<22} → {pred_str}  ({info['probability']:.1%} probability)")
