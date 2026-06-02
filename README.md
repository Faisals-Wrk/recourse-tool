# Algorithmic Recourse Tool

An interactive Streamlit application that explores **algorithmic recourse** — the minimum changes a person must make to receive a favourable prediction from a machine learning model — and the **fairness implications** of those requirements.

Built on the UCI Adult Income (1994 Census) dataset. Three classifiers, three search algorithms, six interactive pages.

---

## Live Demo

Deploy to Streamlit Cloud (see [Deployment](#deployment) below).  
First visitor triggers a 7–11 minute cache build. Every visitor after that is instant.

---

## Three Core Findings

| Finding | Result |
|---|---|
| **Architecture Matters** | Logistic Regression finds recourse for ~100% of people; Decision Tree for only ~14%. Model choice determines whether a person has any path forward — independent of accuracy. |
| **The Fairness Gap** | Women require ~33% more effort than men to flip the same prediction. The disparity is consistent across all three model architectures — it is a property of the 1994 Census data, not any single model. |
| **No Recourse Exists** | For some people, no change to any mutable feature can flip the Decision Tree's prediction. The model has made a permanent judgment about them. |

---

## Six Pages

| Page | What it shows |
|---|---|
| **Home** | Story, key findings, live metrics from the fairness analysis |
| **Dataset** | Feature dictionary, data browser, distributions by income group, model agreement |
| **Find Recourse** | Interactive: pick any person, model, and algorithm — see exactly what to change |
| **Compare Algorithms** | Same person, same model, all three algorithms side by side |
| **Fairness** | Mean effort by sex and race across all three models; feature frequency charts |
| **About** | Technical pipeline, research foundations, honest limitations |

---

## Three Algorithms

**A. Greedy Perturbation**  
At each step, tries every valid ±perturbation across all actionable features. Keeps the single change that most increases P(>50K). Fast and reliable for tree-based models. Includes PATIENCE=5 early termination: if probability doesn't improve for 5 consecutive steps, the person is in a flat Decision Tree region and exits immediately rather than wasting iterations.

**B. Importance-Guided Perturbation**  
Like Greedy, but weights candidate moves by the model's feature importances (Gini for trees, |coefficients| for logistic regression). Aligns the recourse path with the model's own decision logic.

**C. Proximity Minimization (Wachter et al. 2017)**  
Treats recourse as a constrained optimisation problem. Uses `scipy.optimize.minimize` (L-BFGS-B) to minimise normalised distance + quadratic flip penalty. Runs 5 random restarts and returns the shortest valid path. Theoretically grounded; occasionally finds shorter paths than Greedy by making small continuous changes across multiple features.

---

## Constraints (all algorithms)

- **Immutable features** (sex, race, native_country_us) are never changed
- **Increase-only features** (age, education_num) can only go up — you cannot get younger or un-educate yourself
- All values clipped to their valid `[min, max]` ranges after every step

---

## Project Structure

```
recourse-tool/
├── app.py                  # Streamlit application (6 pages)
├── requirements.txt        # Python dependencies
├── README.md
└── recourse/
    ├── __init__.py
    ├── data.py             # Dataset loading, feature metadata, utilities
    ├── model.py            # Three classifiers + training + prediction
    ├── counterfactual.py   # Three recourse algorithms + RecourseResult
    └── analyze.py          # Four analyses (effort, fairness, frequency, comparison)
```

---

## Installation

```bash
git clone https://github.com/your-username/recourse-tool.git
cd recourse-tool
pip install -r requirements.txt
```

The UCI Adult Income dataset is downloaded automatically on first run from the UCI ML Repository. It is cached as `data/adult_clean.csv` — subsequent starts skip the download.

---

## Running Locally

```bash
streamlit run app.py
```

**Important — local performance note:**  
The fairness analysis (`load_analyses`) runs 150 persons × 3 models × 80 greedy iterations of recourse. On a typical Windows laptop with ~18ms per Random Forest call this takes ~60 minutes. For local testing, temporarily change these values in `load_analyses()` inside `app.py`:

```python
# Local testing only — revert before pushing to GitHub
sample_size=20,    # was 150
max_iterations=10, # was 80
```

Revert to `sample_size=150, max_iterations=80` before deployment. At ~3ms per RF call on Streamlit Cloud's Linux servers, the analysis takes 7–11 minutes on the very first visit, then is cached permanently.

---

## Deployment

### Streamlit Cloud (recommended, free)

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repository → set main file to `app.py`
4. Click **Deploy**

The first visitor triggers the 7–11 minute cache build for the fairness analysis. Every visitor after that loads instantly — Streamlit Cloud persists the `@st.cache_data` result across sessions.

No secrets or environment variables are required.

---

## Dataset

**UCI Adult Income (Census Income)**  
48,842 people from the 1994 U.S. Census. Binary label: does this person earn more than $50,000/year?

We use a 9,999-row stratified sample (preserving the ~24.8% high-income class ratio) for performance. Seven categorical features are ordinal-encoded so the recourse algorithms can perturb them by integer steps. Three features are treated as immutable: sex, race, and native_country_us.

Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/adult)

---

## Key Parameters

| Parameter | Value | Why |
|---|---|---|
| `sample_size` | 150 | Gives ~48 female persons → ~31 with measured distance → reliable detection of the ~30% sex gap |
| `max_iterations` | 80 | Logistic Regression needs 15–30 greedy steps to converge on its smooth surface; cutting early distorts Finding 01 |
| `PATIENCE` | 5 | Exits Decision Tree flat regions after 5 consecutive no-improvement steps — 20× speedup with no effect on found-recourse results |
| `n_estimators` | 100 | Standard RF size; `max_depth=8`, `min_samples_leaf=10` prevent overfitting on ~10K rows |

---

## Research Foundation

- Wachter, Mittelstadt & Russell (2017). *Counterfactual Explanations Without Opening the Black Box.* Harvard JOLT. — The foundational paper; our Proximity algorithm implements this directly.
- Karimi et al. (2020). *Algorithmic Recourse Under Imperfect Causal Knowledge.* NeurIPS. — Motivates our increase-only constraints on age and education.
- Barocas, Hardt & Narayanan (2020). *The Hidden Assumptions Behind Counterfactual Explanations.* FAccT. — Motivates our immutable feature design.
- Gupta et al. (2019). *Fairness Implications of Recourse in Algorithmic Decision-Making.* AAAI AIES. — Shows recourse effort can be systematically higher for protected groups.

---

## Tech Stack

| Library | Role |
|---|---|
| Python 3.11 | Language |
| scikit-learn 1.2+ | ML models and pipelines |
| scipy 1.10+ | Proximity optimisation (L-BFGS-B) |
| Streamlit 1.30+ | Web application |
| Plotly 5.18+ | Interactive charts |
| Pandas + NumPy | Data handling |

---

## Limitations

**1994 data.** The Census data reflects historical income patterns from 30 years ago. The fairness gaps are real in the data but may not reflect today's labour market.

**Greedy for bulk analysis.** The 150-person fairness analysis uses the Greedy algorithm only. Proximity would find shorter paths in many cases, potentially reducing measured distances. Greedy distances are a conservative upper bound.

**Ordinal encoding.** Treating categorical features as ordered integers is an approximation. "Private" employment is not objectively between "Without-pay" and "Self-employed-inc" on a linear scale. This affects which recourse steps the greedy algorithms consider.
