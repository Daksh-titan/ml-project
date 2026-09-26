"""
mqtf_core.py

Core methodology for the MQTF (Multivariate Quality Trajectory Framework)
Research Analysis Tool.

SOURCE OF TRUTH
----------------
Sections marked "PORTED FROM NOTEBOOK" are transcribed, unmodified in logic,
from `MQTF_final_reproducible.ipynb` (the author's validated reproducible
analysis). They compute:
    - baseline-centered, early-period-standardized trajectories (Z)
    - Pareto-based feature selection (signal vs redundancy) per domain
    - STD (Spoilage/Domain-A Trajectory Displacement) and
      FTD (Functional/Domain-B Trajectory Displacement)
    - Monte Carlo FEATURE-SELECTION STABILITY (NOT a prediction-uncertainty
      model -- the notebook only uses Monte Carlo resampling to check how
      often the same feature subset is re-selected under measurement noise).

Sections marked "NEW: FORECASTING EXTENSION" are new. The notebook contains
NO future-trajectory forecasting of any kind. Per the project instructions,
this is explicitly flagged as a methodological extension, implemented as the
smallest change consistent with the existing MQTF machinery, and validated
with time-ordered backtesting rather than presented as pre-validated
methodology.
"""

import itertools
from collections import Counter

import numpy as np
import pandas as pd

# =====================================================================
# CONSTANTS -- PORTED FROM NOTEBOOK (verbatim)
# =====================================================================
TREATMENTS = ['Control', 'Foxtail', 'Little', 'Kodo', 'Proso', 'Barnyard']
DAYS = [0, 7, 14, 21, 28]
FEATURES = ['pH', 'Acidity', 'Viscosity', 'Fat', 'Antioxidants', 'Phenols',
            'Flavonoids', 'Phytate', 'Tannin', 'L*', 'a*', 'b*', 'Protein',
            'Fibre', 'Carbohydrate', 'Reducing_Sugars', 'Total_Sugars', 'TSS']
DOMAIN_A = ['pH', 'Acidity', 'Viscosity', 'Fat']
DOMAIN_B = ['Antioxidants', 'Phenols', 'Flavonoids', 'Phytate', 'Tannin']
N_REPLICATES = 3
EARLY_PERIOD_DAYS = [0, 7, 14]   # days used for normalization / feature selection

# =====================================================================
# VERIFIED DATASET -- PORTED FROM NOTEBOOK (verbatim, unmodified values)
# Each variable is stored as [day][treatment].
# =====================================================================
MEAN_DATA = {'Viscosity': [[49.73, 45.85, 46.35, 46.56, 47.18, 45.14], [47.34, 43.14, 42.12, 43.31, 45.27, 42.84], [44.04, 43.15, 40.17, 42.05, 43.49, 40.51], [41.38, 40.62, 38.25, 41.68, 42.18, 40.51], [41.38, 39.62, 37.08, 41.13, 40.54, 39.89]], 'L*': [[115.32, 117.56, 119.75, 116.17, 114.33, 116.56], [114.75, 116.43, 117.25, 115.65, 113.74, 115.75], [114.75, 116.13, 117.21, 115.56, 113.72, 115.69], [114.56, 116.13, 117.21, 115.32, 113.65, 115.45], [114.41, 116.06, 117.16, 115.12, 113.34, 115.32]], 'a*': [[1.71, 1.64, 1.85, 1.84, 1.87, 1.82], [1.75, 1.64, 1.82, 1.93, 1.8, 1.73], [1.75, 1.62, 1.81, 1.91, 1.78, 1.71], [1.73, 1.62, 1.8, 1.85, 1.64, 1.53], [1.71, 1.53, 1.73, 1.64, 1.32, 1.21]], 'b*': [[10.08, 11.02, 11.52, 11.16, 11.5, 10.08], [9.87, 10.76, 11.79, 11.2, 10.13, 10.84], [9.87, 10.74, 11.79, 11.18, 10.11, 10.82], [9.85, 10.74, 11.76, 11.14, 10.04, 10.69], [9.83, 10.71, 11.63, 11.04, 10.01, 10.25]], 'Protein': [[8.34, 8.05, 6.96, 8.29, 7.05, 8.21], [8.56, 8.09, 7.01, 8.36, 7.08, 8.24], [8.58, 8.62, 7.18, 8.84, 7.1, 8.32], [8.66, 8.62, 7.25, 9.0, 7.41, 8.65], [8.66, 8.65, 8.51, 9.02, 7.45, 8.7]], 'Fat': [[0.78, 1.03, 1.07, 1.1, 1.6, 1.6], [0.7, 1.03, 1.05, 1.1, 1.4, 1.4], [0.7, 0.9, 0.96, 1.05, 1.1, 1.05], [0.7, 0.9, 0.86, 0.96, 0.86, 0.97], [0.7, 0.89, 0.78, 0.92, 0.85, 0.95]], 'Fibre': [[1.85, 1.4, 1.55, 1.89, 0.97, 1.16], [1.85, 1.4, 1.55, 1.86, 0.97, 1.16], [1.63, 1.1, 1.55, 1.86, 0.97, 1.16], [1.03, 0.6, 1.58, 1.86, 0.98, 1.2], [1.03, 0.8, 1.58, 1.97, 0.98, 1.2]], 'Carbohydrate': [[27.25, 32.12, 31.02, 28.58, 30.91, 22.35], [27.22, 32.25, 30.15, 28.68, 30.95, 22.12], [26.21, 31.41, 30.96, 27.79, 39.98, 21.92], [26.21, 31.76, 30.35, 27.14, 29.39, 21.72], [26.21, 31.23, 30.13, 27.06, 29.24, 21.68]], 'Reducing_Sugars': [[0.41, 0.22, 0.28, 0.4, 0.3, 0.3], [0.52, 0.37, 0.35, 0.4, 0.3, 0.3], [0.52, 0.41, 0.69, 0.73, 0.62, 0.6], [0.73, 0.75, 0.92, 0.81, 0.86, 0.81], [0.82, 0.99, 0.91, 0.83, 0.98, 0.89]], 'Total_Sugars': [[1.92, 2.51, 2.16, 2.79, 2.19, 2.63], [1.86, 2.71, 2.09, 2.45, 2.05, 2.12], [1.86, 2.05, 1.91, 2.45, 1.78, 1.47], [1.56, 1.75, 1.91, 1.82, 1.52, 1.47], [1.32, 0.96, 0.98, 0.92, 1.05, 0.97]], 'pH': [[4.89, 6.64, 5.35, 6.62, 5.17, 5.29], [4.86, 5.62, 4.32, 5.61, 4.15, 4.24], [4.73, 4.57, 4.29, 4.54, 4.15, 4.23], [4.71, 4.54, 4.17, 4.32, 4.11, 4.17], [4.45, 4.0, 4.6, 4.21, 4.07, 4.15]], 'Acidity': [[0.18, 0.14, 0.17, 0.14, 0.16, 0.14], [0.21, 0.18, 0.23, 0.19, 0.2, 0.18], [0.25, 0.29, 0.27, 0.22, 0.23, 0.26], [0.35, 0.32, 0.36, 0.29, 0.31, 0.34], [0.38, 0.35, 0.36, 0.32, 0.35, 0.36]], 'Phenols': [[325.14, 276.38, 235.16, 285.94, 206.65, 247.82], [341.0, 298.13, 241.14, 307.02, 213.87, 264.15], [341.0, 302.58, 263.59, 332.15, 241.0, 297.67], [359.16, 324.03, 274.56, 332.15, 269.17, 311.03], [384.76, 360.14, 280.12, 362.73, 298.75, 321.25]], 'Flavonoids': [[19.23, 11.42, 16.23, 13.15, 10.18, 12.49], [17.05, 12.57, 17.12, 15.03, 11.76, 14.02], [17.05, 14.98, 18.47, 16.59, 13.62, 15.97], [14.05, 13.24, 18.47, 17.12, 14.83, 15.97], [13.16, 13.92, 19.03, 18.16, 16.72, 18.28]], 'Phytate': [[317.83, 412.18, 418.13, 320.26, 412.76, 317.14], [317.17, 371.05, 316.24, 317.45, 363.19, 314.65], [317.17, 311.05, 214.21, 216.74, 210.19, 213.42], [316.42, 210.98, 214.03, 216.24, 210.78, 211.56], [316.42, 210.98, 214.03, 216.24, 210.78, 211.56]], 'Tannin': [[18.23, 19.47, 16.47, 21.44, 18.91, 22.56], [18.62, 18.25, 14.82, 18.83, 16.27, 19.78], [18.62, 17.65, 14.71, 16.92, 14.62, 18.83], [18.62, 17.13, 13.05, 14.15, 13.03, 16.92], [18.73, 16.83, 13.72, 14.15, 13.0, 16.65]], 'Antioxidants': [[52.64, 61.79, 70.7, 72.44, 67.91, 69.87], [52.86, 61.84, 70.85, 72.49, 67.99, 69.9], [52.88, 61.87, 70.97, 72.54, 68.08, 69.96], [55.17, 61.94, 71.07, 72.59, 68.11, 70.05], [55.24, 62.05, 71.14, 73.63, 68.17, 70.13]], 'TSS': [[13.0, 14.0, 14.0, 16.0, 14.0, 14.0], [13.0, 14.0, 14.0, 15.0, 14.0, 14.0], [13.0, 13.0, 13.0, 13.0, 13.0, 13.0], [12.0, 13.0, 13.0, 13.0, 12.0, 13.0], [12.0, 13.0, 13.0, 13.0, 12.0, 13.0]]}

SD_DATA = {'Viscosity': [[0.23, 0.15, 0.75, 1.23, 0.73, 0.42], [0.03, 0.61, 0.97, 0.79, 0.52, 1.25], [1.37, 0.12, 1.12, 0.4, 0.82, 0.22], [0.0, 1.16, 1.19, 0.53, 1.11, 0.08], [0.08, 0.16, 0.3, 0.95, 0.44, 0.62]], 'L*': [[1.15, 0.23, 2.03, 2.68, 3.5, 0.79], [2.73, 2.85, 1.43, 2.12, 0.15, 0.78], [2.42, 2.84, 2.23, 1.65, 3.55, 2.38], [2.8, 3.71, 0.71, 2.98, 1.46, 2.51], [1.79, 1.34, 0.23, 3.44, 2.77, 3.37]], 'a*': [[0.04, 0.01, 0.0, 0.05, 0.0, 0.01], [0.04, 0.04, 0.04, 0.01, 0.04, 0.03], [0.04, 0.03, 0.04, 0.0, 0.03, 0.01], [0.0, 0.01, 0.03, 0.02, 0.04, 0.03], [0.05, 0.03, 0.02, 0.02, 0.02, 0.02]], 'b*': [[0.24, 0.07, 0.34, 0.26, 0.28, 0.24], [0.14, 0.24, 0.26, 0.17, 0.05, 0.1], [0.19, 0.23, 0.1, 0.14, 0.32, 0.3], [0.24, 0.35, 0.28, 0.19, 0.06, 0.09], [0.05, 0.17, 0.11, 0.0, 0.05, 0.27]], 'Protein': [[0.05, 0.18, 0.17, 0.067, 0.09, 0.17], [0.34, 0.19, 0.04, 0.02, 0.05, 0.01], [0.02, 0.2, 0.11, 0.12, 0.16, 0.07], [0.07, 0.26, 0.11, 0.06, 0.03, 0.01], [0.03, 0.14, 0.12, 0.28, 0.09, 0.28]], 'Fat': [[0.01, 0.02, 0.01, 0.0, 0.0, 0.04], [0.0, 0.0, 0.02, 0.01, 0.01, 0.02], [0.0, 0.01, 0.0, 0.01, 0.03, 0.0], [0.0, 0.02, 0.01, 0.01, 0.0, 0.02], [0.01, 0.02, 0.0, 0.01, 0.0, 0.0]], 'Fibre': [[0.01, 0.04, 0.01, 0.03, 0.0, 0.01], [0.02, 0.03, 0.01, 0.0, 0.0, 0.0], [0.02, 0.01, 0.04, 0.05, 0.01, 0.01], [0.0, 0.01, 0.02, 0.04, 0.0, 0.0], [0.01, 0.02, 0.03, 0.05, 0.02, 0.03]], 'Carbohydrate': [[1.51, 1.43, 0.58, 0.75, 0.53, 1.06], [1.63, 0.97, 0.81, 1.75, 1.07, 1.52], [0.8, 1.62, 0.66, 1.53, 1.22, 0.49], [0.61, 2.05, 0.08, 0.73, 1.17, 1.54], [1.03, 1.04, 0.38, 1.08, 0.68, 0.0]], 'Reducing_Sugars': [[0.01, 0.05, 0.06, 0.04, 0.04, 0.05], [0.02, 0.07, 0.09, 0.05, 0.06, 0.01], [0.05, 0.02, 0.07, 0.06, 0.05, 0.05], [0.08, 0.02, 0.07, 0.07, 0.0, 0.0], [0.01, 0.08, 0.04, 0.02, 0.02, 0.06]], 'Total_Sugars': [[0.03, 0.02, 0.01, 0.06, 0.04, 0.09], [0.08, 0.03, 0.09, 0.08, 0.08, 0.08], [0.06, 0.02, 0.03, 0.06, 0.07, 0.03], [0.04, 0.05, 0.08, 0.07, 0.03, 0.01], [0.05, 0.08, 0.04, 0.05, 0.04, 0.01]], 'pH': [[0.09, 0.12, 0.06, 0.07, 0.14, 0.1], [0.13, 0.07, 0.07, 0.14, 0.11, 0.13], [0.15, 0.06, 0.1, 0.13, 0.1, 0.0], [0.0, 0.1, 0.03, 0.02, 0.09, 0.04], [0.1, 0.01, 0.03, 0.09, 0.09, 0.07]], 'Acidity': [[0.02, 0.0, 0.0, 0.0, 0.0, 0.0], [0.01, 0.0, 0.0, 0.0, 0.0, 0.0], [0.03, 0.0, 0.0, 0.0, 0.0, 0.0], [0.01, 0.0, 0.0, 0.0, 0.0, 0.01], [0.02, 0.01, 0.0, 0.0, 0.0, 0.0]], 'Phenols': [[8.18, 5.64, 1.52, 3.11, 0.84, 5.05], [2.55, 0.2, 3.93, 3.76, 5.82, 5.03], [5.1, 8.55, 3.22, 4.51, 3.93, 8.91], [3.17, 8.15, 6.72, 3.16, 2.01, 3.59], [0.68, 11.51, 8.57, 5.67, 2.64, 6.12]], 'Flavonoids': [[0.52, 0.13, 0.06, 0.41, 0.23, 0.1], [0.41, 0.14, 0.01, 0.3, 0.36, 0.45], [0.17, 0.02, 0.31, 0.36, 0.07, 0.51], [0.07, 0.24, 0.46, 0.45, 0.24, 0.19], [0.28, 0.35, 0.52, 0.19, 0.29, 0.34]], 'Phytate': [[8.24, 9.53, 0.28, 10.4, 6.17, 9.49], [0.64, 8.07, 5.37, 3.67, 0.24, 4.71], [2.94, 6.41, 2.52, 2.35, 4.73, 0.86], [1.45, 3.66, 0.71, 3.93, 5.29, 4.73], [3.59, 2.19, 4.54, 2.64, 5.3, 6.18]], 'Tannin': [[0.04, 0.15, 0.44, 0.05, 0.46, 0.36], [0.05, 0.31, 0.21, 0.62, 0.45, 0.47], [0.58, 0.17, 0.25, 0.29, 0.03, 0.42], [0.09, 0.05, 0.29, 0.33, 0.36, 0.25], [0.3, 0.4, 0.08, 0.1, 0.03, 0.27]], 'Antioxidants': [[2.43, 1.05, 1.74, 0.14, 2.07, 1.47], [2.66, 1.68, 1.1, 0.54, 2.08, 0.57], [2.17, 1.45, 0.86, 0.54, 0.74, 0.38], [2.61, 1.43, 1.16, 0.09, 1.11, 1.04], [1.93, 1.13, 1.69, 1.82, 1.06, 1.76]], 'TSS': [[0.08, 0.14, 0.41, 0.01, 0.04, 0.19], [0.06, 0.41, 0.3, 0.18, 0.42, 0.4], [0.15, 0.36, 0.15, 0.07, 0.27, 0.01], [0.36, 0.28, 0.0, 0.38, 0.19, 0.13], [0.26, 0.09, 0.16, 0.1, 0.0, 0.08]]}


def default_arrays():
    """Build the (6, 5, 18) mean/SD arrays from the notebook's verified data."""
    X_mean = np.zeros((len(TREATMENTS), len(DAYS), len(FEATURES)))
    X_sd = np.zeros_like(X_mean)
    for j, feature in enumerate(FEATURES):
        X_mean[:, :, j] = np.asarray(MEAN_DATA[feature], dtype=float).T
        X_sd[:, :, j] = np.asarray(SD_DATA[feature], dtype=float).T
    return X_mean, X_sd


def default_long_dataframe():
    """The notebook's verified dataset as a long-format (Treatment, Day, ...) frame,
    with optional '<Feature>_SD' columns included, matching the upload format."""
    X_mean, X_sd = default_arrays()
    rows = []
    for i, treatment in enumerate(TREATMENTS):
        for t, day in enumerate(DAYS):
            row = {"Treatment": treatment, "Day": day}
            for j, feature in enumerate(FEATURES):
                row[feature] = X_mean[i, t, j]
            for j, feature in enumerate(FEATURES):
                row[f"{feature}_SD"] = X_sd[i, t, j]
            rows.append(row)
    return pd.DataFrame(rows)


# =====================================================================
# MQTF CORE PIPELINE -- PORTED FROM NOTEBOOK (verbatim logic)
# =====================================================================
def calculate_normalized_trajectories(X):
    """Baseline-centered, early-period-standardized trajectories.

    early_mean / sigma are computed only from the early period (days 0, 7, 14),
    pooled across all treatments. Z is the day-0-baselined trajectory divided
    by that pooled early-period SD. This exact definition is what makes it
    valid to reuse (sigma, baseline) unchanged for the forecasting extension
    below: they never depend on days 21/28, so there is no leakage from the
    values being forecast/backtested.
    """
    early = X[:, :3, :]
    early_mean = early.mean(axis=(0, 1))
    sigma = early.std(axis=(0, 1), ddof=1)
    if np.any(sigma == 0):
        raise ValueError("A feature has zero pooled early-period SD.")
    Z = (X - X[:, 0:1, :]) / sigma
    return Z, early_mean, sigma


def _full_redundancy_matrix(X):
    early = X[:, :3, :]
    delta = early - early[:, 0:1, :]
    centered = delta - delta.mean(axis=1, keepdims=True)
    denom = np.sqrt(np.sum(centered * centered, axis=1, keepdims=True))
    corr = np.einsum('tif,tig->ifg', centered, centered) / (
        np.einsum('tif,tig->ifg', denom, denom) + 1e-15
    )
    corr = np.nan_to_num(corr)
    return np.mean(np.abs(corr), axis=0)


def subset_metrics(signal, R, subset):
    subset = tuple(subset)
    mean_signal = float(np.mean(signal[list(subset)]))
    pairs = list(itertools.combinations(subset, 2))
    mean_redundancy = float(np.mean([R[a, b] for a, b in pairs])) if pairs else 0.0
    return mean_signal, mean_redundancy


def select_features(X, domain_names):
    """Pareto-optimal (max signal, min redundancy) feature subset selection."""
    Z, early_mean, sigma = calculate_normalized_trajectories(X)
    signal = np.mean(Z[:, :3, :] ** 2, axis=(0, 1))
    domain = [FEATURES.index(f) for f in domain_names]

    subsets = [s for k in range(1, len(domain) + 1) for s in itertools.combinations(domain, k)]
    R_full = _full_redundancy_matrix(X)
    metrics = {s: subset_metrics(signal, R_full, s) for s in subsets}

    pareto = []
    for s, (sig, red) in metrics.items():
        dominated = False
        for q, (sig2, red2) in metrics.items():
            if q == s:
                continue
            if sig2 >= sig and red2 <= red and (sig2 > sig or red2 < red):
                dominated = True
                break
        if not dominated:
            pareto.append(s)

    sigs = np.array([metrics[s][0] for s in pareto])
    reds = np.array([metrics[s][1] for s in pareto])
    if len(pareto) == 1:
        distances = np.array([0.0])
    else:
        sig_norm = (sigs - sigs.min()) / (sigs.max() - sigs.min()) if sigs.max() > sigs.min() else np.ones(len(sigs))
        red_norm = (reds - reds.min()) / (reds.max() - reds.min()) if reds.max() > reds.min() else np.zeros(len(reds))
        distances = np.sqrt((1 - sig_norm) ** 2 + red_norm ** 2)

    best = pareto[int(np.argmin(distances))]
    return {
        'Z': Z, 'early_mean': early_mean, 'sigma': sigma, 'signal': signal,
        'metrics': metrics, 'pareto': pareto, 'best': best,
        'best_names': [FEATURES[i] for i in best],
    }


def calculate_displacement(Z, selected_indices):
    """Standardized multivariate displacement (STD/FTD) from day-0 baseline."""
    return np.sqrt(np.sum(Z[:, :, list(selected_indices)] ** 2, axis=2))


def run_mqtf(X_mean):
    """Deterministic MQTF pipeline: feature selection + STD/FTD trajectories."""
    A = select_features(X_mean, DOMAIN_A)
    B = select_features(X_mean, DOMAIN_B)
    Z = A['Z']
    std = calculate_displacement(Z, A['best'])
    ftd = calculate_displacement(Z, B['best'])
    return A, B, std, ftd


def monte_carlo_selection(X_mean, X_sd, n_sim=1000, seed=42):
    """Monte Carlo FEATURE-SELECTION STABILITY (ported from notebook).

    IMPORTANT: this answers "how often would the same feature subset be
    re-selected if the reported means were resampled around their standard
    error?" It is a robustness check on the feature-selection step, NOT a
    model of prediction uncertainty. Do not relabel its output as a
    prediction interval anywhere in the UI.
    """
    rng = np.random.default_rng(seed)
    counts_A, counts_B = Counter(), Counter()
    standard_error = X_sd / np.sqrt(N_REPLICATES)

    for _ in range(n_sim):
        X_sim = rng.normal(X_mean, standard_error)
        a = tuple(FEATURES[i] for i in select_features(X_sim, DOMAIN_A)['best'])
        b = tuple(FEATURES[i] for i in select_features(X_sim, DOMAIN_B)['best'])
        counts_A[a] += 1
        counts_B[b] += 1

    return counts_A, counts_B


# =====================================================================
# NEW: FORECASTING EXTENSION
# =====================================================================
# The reproducible notebook contains no future-trajectory forecasting.
# This section is a methodological extension, not pre-validated methodology
# from the paper. It is deliberately the smallest change consistent with
# the existing MQTF machinery:
#
#   1. MQTF's own feature selection (above) already reduced Domain A to a
#      single variable (Acidity) and Domain B to a single variable
#      (Phytate) for this dataset. STD = |Z_Acidity|, FTD = |Z_Phytate|
#      exactly (a displacement over one feature is just its absolute
#      standardized value).
#   2. Because Z is an AFFINE function of the raw feature value
#      (Z = (X - baseline) / sigma, with baseline = day-0 value and sigma
#      fixed from the early period, both independent of day), fitting an
#      ordinary-least-squares line to the raw feature trajectory against
#      Day and then transforming the extrapolated point through the SAME
#      fixed (baseline, sigma) used by the deterministic MQTF pipeline is
#      mathematically identical to extrapolating the Z-trajectory directly.
#      No new normalization is introduced.
#   3. sigma and baseline never use days 21/28, so reusing them for
#      forecasting or for backtesting predictions of days 21/28 introduces
#      no leakage from the values being predicted.
#
# Model: ordinary least squares, degree 1 (a straight line through the
# observed Day -> value points available at forecast time). This is the
# simplest model that can be fit to as few as two time points, which this
# design (5 storage days per treatment) requires.
#
# Uncertainty: Monte Carlo resampling using the SAME standard-error
# definition as the notebook's feature-selection Monte Carlo
# (SE = SD / sqrt(N_REPLICATES)), but here propagated through the
# trend-fit + extrapolation step to produce a genuine forecast interval.
# This is a DIFFERENT Monte Carlo procedure from monte_carlo_selection()
# above and is labelled separately in the UI as a
# "Monte Carlo-derived prediction interval" (not a classical confidence
# interval, and not the feature-selection-stability Monte Carlo).

PREDICTABLE_FEATURES = {"Acidity": DOMAIN_A, "Phytate": DOMAIN_B}


def _ols_line(days, values):
    """Ordinary least squares fit of a straight line, value = a*day + b."""
    days = np.asarray(days, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(days) < 2:
        raise ValueError("At least two time points are required to fit a trend.")
    slope, intercept = np.polyfit(days, values, 1)
    return float(slope), float(intercept)


def extrapolate_feature(X_mean, treatment_idx, feature_name, fit_day_indices, target_day):
    """Fit an OLS line to the observed trajectory of `feature_name` for one
    treatment, using only the days in fit_day_indices, then evaluate the
    line at target_day. Returns the point forecast (float)."""
    j = FEATURES.index(feature_name)
    fit_days = [DAYS[d] for d in fit_day_indices]
    fit_values = [X_mean[treatment_idx, d, j] for d in fit_day_indices]
    slope, intercept = _ols_line(fit_days, fit_values)
    return slope * target_day + intercept


def feature_to_displacement(value, feature_name, sigma, baseline):
    """Transform a raw feature value into the same standardized displacement
    space (|Z|) used by calculate_displacement, using the FIXED sigma/baseline
    already established by the deterministic MQTF pipeline (never refit on
    forecast data)."""
    j = FEATURES.index(feature_name)
    z = (value - baseline[j]) / sigma[j]
    return abs(z)


def extrapolation_risk_label(last_observed_day, target_day):
    """Qualitative flag for how far a forecast extrapolates beyond the
    observed range. Purely descriptive -- does not change the estimate."""
    horizon = target_day - last_observed_day
    if target_day <= last_observed_day:
        return "interpolation"
    if horizon <= 7:
        return "low"       # within one storage-interval step beyond data
    if horizon <= 21:
        return "moderate"  # up to three intervals beyond data
    return "high"


def predict_treatment(X_mean, X_sd, treatment_idx, target_day,
                       fit_day_indices=None, n_sim=1000, seed=42):
    """Forecast Acidity, Phytate, STD and FTD for one treatment at
    target_day, using all observed days up to and including the last
    available day by default (fit_day_indices=None -> use every day in
    X_mean's day axis that is <= target_day's available history, i.e. all
    of them for a genuine forward forecast).

    Returns a dict with point estimates, Monte Carlo interval (5th/95th
    percentile), the number of simulations, and the seed used.
    """
    if fit_day_indices is None:
        fit_day_indices = list(range(len(DAYS)))

    # Fixed normalization constants from the deterministic pipeline
    # (uses only days 0/7/14 -- never depends on the forecast target).
    _, early_mean, sigma = calculate_normalized_trajectories(X_mean)
    baseline = X_mean[treatment_idx, 0, :]

    point = {}
    mc_samples = {name: [] for name in PREDICTABLE_FEATURES}

    for name in PREDICTABLE_FEATURES:
        point[name] = extrapolate_feature(X_mean, treatment_idx, name, fit_day_indices, target_day)

    point["STD"] = feature_to_displacement(point["Acidity"], "Acidity", sigma, baseline)
    point["FTD"] = feature_to_displacement(point["Phytate"], "Phytate", sigma, baseline)

    # Monte Carlo prediction interval: resample observed points around their
    # standard error, refit, re-extrapolate. Uses X_sd; if X_sd is all zero
    # (no SD available -- e.g. an uploaded CSV without SD columns), the
    # interval collapses to the point estimate and this is reported plainly.
    rng = np.random.default_rng(seed)
    se = X_sd / np.sqrt(N_REPLICATES)
    std_draws, ftd_draws = [], []
    for _ in range(n_sim):
        for name in PREDICTABLE_FEATURES:
            j = FEATURES.index(name)
            fit_days = [DAYS[d] for d in fit_day_indices]
            sim_values = [
                rng.normal(X_mean[treatment_idx, d, j], se[treatment_idx, d, j])
                if se[treatment_idx, d, j] > 0 else X_mean[treatment_idx, d, j]
                for d in fit_day_indices
            ]
            slope, intercept = _ols_line(fit_days, sim_values)
            pred = slope * target_day + intercept
            mc_samples[name].append(pred)
        std_draws.append(feature_to_displacement(mc_samples["Acidity"][-1], "Acidity", sigma, baseline))
        ftd_draws.append(feature_to_displacement(mc_samples["Phytate"][-1], "Phytate", sigma, baseline))

    def interval(arr):
        arr = np.asarray(arr)
        if np.allclose(arr, arr[0]):
            return (float(arr[0]), float(arr[0]))
        return (float(np.percentile(arr, 5)), float(np.percentile(arr, 95)))

    intervals = {name: interval(mc_samples[name]) for name in PREDICTABLE_FEATURES}
    intervals["STD"] = interval(std_draws)
    intervals["FTD"] = interval(ftd_draws)

    has_uncertainty = bool(np.any(X_sd[treatment_idx][:, [FEATURES.index(n) for n in PREDICTABLE_FEATURES]] > 0))

    return {
        "treatment": TREATMENTS[treatment_idx],
        "target_day": target_day,
        "point": point,
        "interval_90pct": intervals,
        "n_sim": n_sim,
        "seed": seed,
        "risk": extrapolation_risk_label(max(DAYS[d] for d in fit_day_indices), target_day),
        "has_uncertainty": has_uncertainty,
        "fit_days": [DAYS[d] for d in fit_day_indices],
    }


def predict_all_treatments(X_mean, X_sd, target_day, n_sim=1000, seed=42):
    return [predict_treatment(X_mean, X_sd, i, target_day, n_sim=n_sim, seed=seed)
            for i in range(len(TREATMENTS))]


# ---------------------------------------------------------------------
# Time-ordered backtesting
# ---------------------------------------------------------------------
# Expanding-window backtest, respecting temporal order (no random splits):
#   fold 1: fit on days [0, 7]         -> predict day 14
#   fold 2: fit on days [0, 7, 14]     -> predict day 21
#   fold 3: fit on days [0, 7, 14, 21] -> predict day 28
# Run for every treatment and for both MQTF-selected variables
# (Acidity, Phytate) plus their derived displacements (STD, FTD).
# sigma/baseline reused unchanged (see note above -- no leakage).
BACKTEST_FOLDS = [
    ([0, 1], 14),
    ([0, 1, 2], 21),
    ([0, 1, 2, 3], 28),
]


def backtest(X_mean):
    """Run the time-ordered backtest and return a long-format DataFrame of
    Actual vs Predicted values, one row per (treatment, fold, variable)."""
    _, early_mean, sigma = calculate_normalized_trajectories(X_mean)
    rows = []
    for i, treatment in enumerate(TREATMENTS):
        baseline = X_mean[i, 0, :]
        for fit_idx, target_day in BACKTEST_FOLDS:
            target_day_idx = DAYS.index(target_day)
            for name in PREDICTABLE_FEATURES:
                pred_raw = extrapolate_feature(X_mean, i, name, fit_idx, target_day)
                actual_raw = X_mean[i, target_day_idx, FEATURES.index(name)]
                rows.append({
                    "Treatment": treatment, "Fold_target_day": target_day,
                    "Fit_days": ",".join(str(DAYS[d]) for d in fit_idx),
                    "Variable": name, "Actual": actual_raw, "Predicted": pred_raw,
                })
                pred_disp = feature_to_displacement(pred_raw, name, sigma, baseline)
                actual_disp = feature_to_displacement(actual_raw, name, sigma, baseline)
                disp_name = "STD" if name == "Acidity" else "FTD"
                rows.append({
                    "Treatment": treatment, "Fold_target_day": target_day,
                    "Fit_days": ",".join(str(DAYS[d]) for d in fit_idx),
                    "Variable": disp_name, "Actual": actual_disp, "Predicted": pred_disp,
                })
    return pd.DataFrame(rows)


def compute_metrics(actual, predicted):
    """MAE, RMSE, R² (and MAPE where safe). Returns dict; values are None
    where the sample is too small/degenerate to report honestly."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    n = len(actual)
    if n == 0:
        return {"n": 0, "MAE": None, "RMSE": None, "R2": None, "MAPE": None}

    errors = actual - predicted
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    if n < 3 or ss_tot == 0:
        r2 = None  # undefined / unstable with this few points or zero variance
    else:
        r2 = 1 - ss_res / ss_tot

    if np.any(np.isclose(actual, 0.0, atol=1e-6)):
        mape = None  # unsafe: at least one actual value is at/near zero
    else:
        mape = float(np.mean(np.abs(errors / actual)) * 100)

    return {"n": n, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def metrics_by_variable(backtest_df):
    out = {}
    for variable in backtest_df["Variable"].unique():
        sub = backtest_df[backtest_df["Variable"] == variable]
        out[variable] = compute_metrics(sub["Actual"], sub["Predicted"])
    return out


# =====================================================================
# CSV VALIDATION
# =====================================================================
class CSVValidationError(Exception):
    """Raised with a full, itemized list of problems (not just the first one)."""
    def __init__(self, issues):
        self.issues = issues
        super().__init__("\n".join(issues))


def validate_and_build(df):
    """Validate an uploaded long-format CSV and build the (6, 5, 18) mean
    array (and SD array, if '<Feature>_SD' columns are present).

    Collects every problem it can find rather than stopping at the first,
    so the user gets one useful, complete error report instead of the old
    "Expected exactly one row for Control, day 0; found 0" message.
    """
    issues = []

    if df is None or df.empty:
        raise CSVValidationError(["The uploaded file is empty."])

    # Case/whitespace-tolerant column matching.
    col_lookup = {c.strip().lower(): c for c in df.columns}

    def find_col(name):
        return col_lookup.get(name.strip().lower())

    treatment_col = find_col("Treatment")
    day_col = find_col("Day")
    if treatment_col is None:
        issues.append("Required column 'Treatment' is missing.")
    if day_col is None:
        issues.append("Required column 'Day' is missing.")

    feature_cols = {}
    missing_features = []
    for f in FEATURES:
        c = find_col(f)
        if c is None:
            missing_features.append(f)
        else:
            feature_cols[f] = c
    if missing_features:
        issues.append(
            "Missing required variable column(s): " + ", ".join(missing_features)
        )

    sd_cols = {}
    for f in FEATURES:
        c = find_col(f"{f}_SD")
        if c is not None:
            sd_cols[f] = c
    partial_sd = [f for f in FEATURES if f not in sd_cols]
    have_any_sd = len(sd_cols) > 0
    have_all_sd = len(sd_cols) == len(FEATURES)
    if have_any_sd and not have_all_sd:
        issues.append(
            "Partial SD columns detected (found SD for "
            f"{len(sd_cols)}/{len(FEATURES)} variables). SD columns must be "
            "provided for either ALL 18 variables or none, so Monte Carlo "
            "uncertainty is not silently computed from a mix of real and "
            "assumed-zero SDs. Missing SD columns for: "
            + ", ".join(f"{f}_SD" for f in partial_sd)
        )

    known_cols = {treatment_col, day_col, *feature_cols.values(), *sd_cols.values()}
    known_cols.discard(None)
    extra_cols = [c for c in df.columns if c not in known_cols]

    if issues:
        # Can't safely go further without the basic columns.
        raise CSVValidationError(issues)

    work = df.copy()
    work["_Treatment"] = work[treatment_col].astype(str).str.strip()
    work["_Day"] = pd.to_numeric(work[day_col], errors="coerce")

    unknown_treatments = sorted(set(work["_Treatment"]) - set(TREATMENTS))
    if unknown_treatments:
        issues.append(
            "Unrecognized Treatment value(s): " + ", ".join(unknown_treatments)
            + f". Expected one of: {', '.join(TREATMENTS)}."
        )

    bad_days_mask = work["_Day"].isna()
    if bad_days_mask.any():
        issues.append(f"{int(bad_days_mask.sum())} row(s) have a non-numeric or missing Day value.")
    unknown_days = sorted(set(work.loc[~bad_days_mask, "_Day"].astype(int)) - set(DAYS))
    if unknown_days:
        issues.append(
            "Unrecognized Day value(s): " + ", ".join(str(d) for d in unknown_days)
            + f". Expected one of: {', '.join(str(d) for d in DAYS)}."
        )

    # Duplicate treatment/day combinations.
    dup_counts = work.groupby(["_Treatment", "_Day"]).size()
    duplicates = dup_counts[dup_counts > 1]
    if len(duplicates) > 0:
        dup_list = [f"{t} / day {int(d)} ({n} rows)" for (t, d), n in duplicates.items()]
        issues.append("Duplicate Treatment/Day combination(s) found: " + "; ".join(dup_list))

    # Missing treatment/day combinations (only checked once basic values are clean).
    if not unknown_treatments and not bad_days_mask.any() and not unknown_days:
        present = set(zip(work["_Treatment"], work["_Day"].astype(int)))
        missing_combos = [
            f"{t} / day {d}" for t in TREATMENTS for d in DAYS if (t, d) not in present
        ]
        if missing_combos:
            issues.append(
                f"Missing {len(missing_combos)} required Treatment/Day row(s): "
                + "; ".join(missing_combos)
            )

    # Non-numeric values in feature columns.
    non_numeric_features = []
    numeric_feature_frame = pd.DataFrame(index=work.index)
    for f, c in feature_cols.items():
        numeric_feature_frame[f] = pd.to_numeric(work[c], errors="coerce")
        bad = numeric_feature_frame[f].isna() & work[c].notna()
        if bad.any():
            non_numeric_features.append(f)
    if non_numeric_features:
        issues.append(
            "Non-numeric value(s) found in column(s): " + ", ".join(non_numeric_features)
        )
    na_in_features = numeric_feature_frame.isna().any().any()
    if na_in_features and not non_numeric_features:
        issues.append("Missing (blank) value(s) found in one or more variable columns.")

    if extra_cols:
        issues.append(
            "Unrecognized extra column(s) present and ignored: " + ", ".join(extra_cols)
            + " (this is only a warning if these are not meant to be used)."
        )

    blocking_issues = [
        i for i in issues
        if not i.startswith("Unrecognized extra column")
    ]
    if blocking_issues:
        raise CSVValidationError(issues)

    # All good structurally -- build the arrays.
    X_mean = np.zeros((len(TREATMENTS), len(DAYS), len(FEATURES)))
    X_sd = np.zeros_like(X_mean)
    for i, treatment in enumerate(TREATMENTS):
        for t, day in enumerate(DAYS):
            row = work[(work["_Treatment"] == treatment) & (work["_Day"] == day)]
            r = row.iloc[0]
            for j, f in enumerate(FEATURES):
                X_mean[i, t, j] = float(pd.to_numeric(r[feature_cols[f]]))
            if have_all_sd:
                for j, f in enumerate(FEATURES):
                    X_sd[i, t, j] = float(pd.to_numeric(r[sd_cols[f]]))

    warnings = [i for i in issues if i.startswith("Unrecognized extra column")]
    return X_mean, X_sd, have_all_sd, warnings
