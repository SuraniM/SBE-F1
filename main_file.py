import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, make_scorer, fbeta_score
from scipy.stats import dirichlet
from scipy.stats import beta
from imblearn.datasets import fetch_datasets
from sklearn.preprocessing import StandardScaler
from collections import Counter
import pandas as pd
import os
import itertools


def generate_data(dataset, weights, n_samples, n_features, n_informative):

    if dataset == "synthetic":
        X, y = make_classification(n_samples=n_samples, n_features=n_features,
                                   n_informative=n_informative, n_redundant=0,
                                   weights=weights, random_state=None)
    elif dataset == 'ecoli_small_n':
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data"
        column_names = [
            'sequence_name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'target'
        ]
        df = pd.read_csv(url, delim_whitespace=True, names=column_names)

        df_filtered = df[df['target'].isin(['pp', 'om'])]

        X = df_filtered.drop(columns=['sequence_name', 'target']).values
        y = df_filtered['target'].values

        y = np.where(y == 'om', 1, 0)
        n_samples = len(y)
        print(Counter(y))
        weights = Counter(y)

    else:
        datasets = fetch_datasets()
        X, y = datasets[dataset]['data'], datasets[dataset]['target']
        n_samples = len(y)
        print(Counter(y))
        weights = Counter(y)

        majority_class = weights.most_common(1)[0][0]
        minority_class = [label for label in weights if label != majority_class][0]

        print("Majority class:", majority_class)
        print("Minority class:", minority_class)

        # Create new labels: map majority to 0 and minority to 1
        y = np.array([0 if label == majority_class else 1 for label in y])

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                         test_size=0.3,
                                                         stratify=y,
                                                         random_state=42)

    return X_train, X_test, y_train, y_test


def transform_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def fit_model(X_train, y_train, X_test, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Test Set Point Estimate
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return model, tn, fp, fn, tp, y_pred


def compute_point_estimates(y_test, y_pred):
    f1_point_estimate = f1_score(y_test, y_pred)
    fbeta_point_estimate = fbeta_score(y_test, y_pred, beta=fbeta)

    return f1_point_estimate, fbeta_point_estimate


def compute_mean_ci(sample):
    mean = np.mean(sample)
    ci = np.percentile(sample, [2.5, 97.5])
    return mean, ci


def compute_cross_validation_f1(model, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1_scores = cross_val_score(model, X_train, y_train,
                                   cv=cv, scoring=make_scorer(f1_score))
    return cv_f1_scores


def compute_bootstrap_f1(model, X_test, y_test, num_bootstrap_samples):
    f1_bootstrap_samples = []

    for _ in range(num_bootstrap_samples):
        resample_indices = np.random.choice(len(X_test), len(X_test), replace=True)
        X_resample = X_test[resample_indices]
        y_resample = y_test[resample_indices]

        y_pred_resample = model.predict(X_resample)
        f1_bootstrap = f1_score(y_resample, y_pred_resample)
        f1_bootstrap_samples.append(f1_bootstrap)

    f1_bootstrap_samples = np.array(f1_bootstrap_samples)
    return f1_bootstrap_samples


def compute_dirichlet_multinomial_f1(counts, num_bootstrap_samples, alpha):
    n = np.sum(counts)

    # Dirichlet posterior
    posterior_alpha = alpha + counts

    # Sample from Dirichlet
    dirichlet_samples = dirichlet.rvs(posterior_alpha, size=num_bootstrap_samples)
    tp_, fp_, fn_, tn_ = dirichlet_samples[:, 0], dirichlet_samples[:, 1], dirichlet_samples[:, 2], dirichlet_samples[:,
                                                                                                    3]
    precision_dir = tp_ / (tp_ + fp_)
    recall_dir = tp_ / (tp_ + fn_)

    f1_dirichlet = 2 * tp_ / (2 * tp_ + fp_ + fn_)
    f1_dirichlet = np.nan_to_num(f1_dirichlet, nan=0.0)

    f1_dir_posterior_samples = np.array(f1_dirichlet)
    return precision_dir, recall_dir, f1_dir_posterior_samples


def compute_beta_multinomial_f1(counts, alpha_P, beta_P, num_bootstrap_samples):
    n = np.sum(counts)

    # Sample confusion matrix counts from posterior
    # theta_samples = dirichlet.rvs(posterior_alpha, size=num_bootstrap_samples)
    p_samples = beta.rvs(counts[0] + alpha_P, counts[1] + beta_P, size=num_bootstrap_samples)
    r_samples = beta.rvs(counts[0] + alpha_P, counts[2] + beta_P, size=num_bootstrap_samples)

    f1_beta = 2 * p_samples * r_samples / (p_samples + r_samples)
    f1_posterior_samples = np.nan_to_num(f1_beta, nan=0.0)
    return p_samples, r_samples, f1_posterior_samples


def compute_fbeta(p_samples, r_samples):
    fbeta_beta = ((1 + fbeta ** 2) * p_samples * r_samples) / (fbeta ** 2 * p_samples + r_samples)
    return fbeta_beta


def compute_correlation(p_samples, r_samples):
    corr = np.corrcoef(p_samples, r_samples)[0, 1]
    return corr


def simulate_test_drift(X_test, y_test, drift_levels):
    drifted_sets = []
    for drift_ratio in drift_levels:
        minority_indices = np.where(y_test == 1)[0]
        majority_indices = np.where(y_test == 0)[0]
        n_minority = int(len(y_test) * drift_ratio)
        n_majority = len(y_test) - n_minority

        if n_minority > len(minority_indices) or n_majority > len(majority_indices):
            continue  # Skip if not enough samples to simulate drift

        sampled_minority = np.random.choice(minority_indices, n_minority, replace=False)
        sampled_majority = np.random.choice(majority_indices, n_majority, replace=False)
        combined_indices = np.concatenate([sampled_minority, sampled_majority])
        np.random.shuffle(combined_indices)

        X_drifted = X_test[combined_indices]
        y_drifted = y_test[combined_indices]
        drifted_sets.append((drift_ratio, X_drifted, y_drifted))
    return drifted_sets


def prior_lists():
    # === Beta Prior Grid (Label, α, β) ===
    beta_prior_values = [0.1, 0.5, 1, 2, 5, 10, 50]
    beta_prior_combinations = list(itertools.product(beta_prior_values, beta_prior_values))[:50]  # Take first 50
    beta_prior_list = [
        (f"Beta({a},{b})", a, b) for a, b in beta_prior_combinations
    ]

    # === Dirichlet Prior Grid (Label, Prior) ===
    dirichlet_prior_list = [
        # --- Uniform and Symmetric ---
        ("Uniform(1,1,1,1)", np.array([1, 1, 1, 1])),
        ("Flat(0.1,0.1,0.1,0.1)", np.array([0.1, 0.1, 0.1, 0.1])),
        ("Mild(2,2,2,2)", np.array([2, 2, 2, 2])),
        ("Strong(10,10,10,10)", np.array([10, 10, 10, 10])),
        ("VeryStrong(50,50,50,50)", np.array([50, 50, 50, 50])),
        ("Low(0.5,0.5,0.5,0.5)", np.array([0.5, 0.5, 0.5, 0.5])),
        ("Tiny(0.01,0.01,0.01,0.01)", np.array([0.01, 0.01, 0.01, 0.01])),

        # --- Favor one category strongly ---
        ("Favor TP(5,1,1,1)", np.array([5, 1, 1, 1])),
        ("Favor FP(1,5,1,1)", np.array([1, 5, 1, 1])),
        ("Favor FN(1,1,5,1)", np.array([1, 1, 5, 1])),
        ("Favor TN(1,1,1,5)", np.array([1, 1, 1, 5])),
        ("TP>>others(10,1,1,1)", np.array([10, 1, 1, 1])),
        ("FP>>others(1,10,1,1)", np.array([1, 10, 1, 1])),
        ("FN>>others(1,1,10,1)", np.array([1, 1, 10, 1])),
        ("TN>>others(1,1,1,10)", np.array([1, 1, 1, 10])),
        ("Spike@TP(50,1,1,1)", np.array([50, 1, 1, 1])),
        ("Spike@TN(1,1,1,50)", np.array([1, 1, 1, 50])),
        ("Spike@FP(1,50,1,1)", np.array([1, 50, 1, 1])),
        ("Spike@FN(1,1,50,1)", np.array([1, 1, 50, 1])),

        # --- Dual emphasis ---
        ("TP&TN>>others(5,1,1,5)", np.array([5, 1, 1, 5])),
        ("FP&FN>>others(1,5,5,1)", np.array([1, 5, 5, 1])),
        ("FP&TN>>others(1,5,1,5)", np.array([1, 5, 1, 5])),
        ("TP&FN>>others(5,1,5,1)", np.array([5, 1, 5, 1])),
        ("TP&FP>>others(5,5,1,1)", np.array([5, 5, 1, 1])),
        ("FN&TN>>others(1,1,5,5)", np.array([1, 1, 5, 5])),

        # --- Downweight single category ---
        ("Downweight TP(0.1,2,2,2)", np.array([0.1, 2, 2, 2])),
        ("Downweight FP(2,0.1,2,2)", np.array([2, 0.1, 2, 2])),
        ("Downweight FN(2,2,0.1,2)", np.array([2, 2, 0.1, 2])),
        ("Downweight TN(2,2,2,0.1)", np.array([2, 2, 2, 0.1])),

        # --- Skewed / Mixed Skew ---
        ("Mixed A(2,1,3,1)", np.array([2, 1, 3, 1])),
        ("Mixed B(1,3,1,2)", np.array([1, 3, 1, 2])),
        ("Mixed C(1,2,3,4)", np.array([1, 2, 3, 4])),
        ("Mixed D(4,3,2,1)", np.array([4, 3, 2, 1])),
        ("Mix Skew E(3,1,2,4)", np.array([3, 1, 2, 4])),
        ("Mix Skew F(0.5,2,1,3)", np.array([0.5, 2, 1, 3])),
        ("Mix Skew G(3,0.5,1,2)", np.array([3, 0.5, 1, 2])),

        # --- Different total concentrations ---
        ("LightTotal(0.2,0.3,0.4,0.1)", np.array([0.2, 0.3, 0.4, 0.1])),
        ("ModTotal(2,3,4,1)", np.array([2, 3, 4, 1])),
        ("HighTotal(20,30,40,10)", np.array([20, 30, 40, 10])),

        # --- Realistic human-like asymmetry ---
        ("Realistic A(3,2,1,4)", np.array([3, 2, 1, 4])),
        ("Realistic B(1,2,3,2)", np.array([1, 2, 3, 2])),
        ("Realistic C(2,2,1,3)", np.array([2, 2, 1, 3])),
        ("Realistic D(4,1,3,2)", np.array([4, 1, 3, 2])),
        ("Realistic E(2,1,4,3)", np.array([2, 1, 4, 3])),

        # --- Extreme imbalances ---
        ("UltraTP(100,1,1,1)", np.array([100, 1, 1, 1])),
        ("UltraFN(1,1,100,1)", np.array([1, 1, 100, 1])),
        ("UltraFP(1,100,1,1)", np.array([1, 100, 1, 1])),
        ("UltraTN(1,1,1,100)", np.array([1, 1, 1, 100]))
    ]

    return beta_prior_list, dirichlet_prior_list


def plot_sensitivity_results(df, title, true_f1):
    df_sorted = df.sort_values(by="Mean", ascending=False).reset_index(drop=True)
    plt.figure(figsize=(14, 6))
    plt.errorbar(df_sorted["Prior"], df_sorted["Mean"],
                 yerr=[df_sorted["Mean"] - df_sorted["CI_Lower"],
                       df_sorted["CI_Upper"] - df_sorted["Mean"]],
                 fmt='o', capsize=5, label="95% Credible Interval")
    plt.axhline(true_f1, color='red', linestyle='--', label=f"Point Estimate = {true_f1:.3f}")
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.ylabel("F1 Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def save_results(dataset, trial, num_bootstrap_samples, weights, n_samples, n_features, n_informative,
                                 corr_beta, corr_dirichlet,
                                 f1_point_estimate, fbeta_point_estimate, f1_cv_mean, f1_cv_ci, f1_bootstrap_mean, f1_bootstrap_ci,
                                 f1_dir_posterior_mean, f1_dir_posterior_ci, f1_beta_posterior_mean, f1_beta_posterior_ci):
    df = pd.DataFrame([{
        "dataset": dataset, "trial": trial, "num_bootstrap_samples": num_bootstrap_samples,
        "weights": weights, "n_samples": n_samples, "n_features": n_features, "n_informative": n_informative,
        "corr_beta": corr_beta, "corr_dirichlet": corr_dirichlet,
        "f1_point_estimate": f1_point_estimate, "fbeta_point_estimate": fbeta_point_estimate, "f1_cv_mean": f1_cv_mean,
        "f1_cv_ci": f1_cv_ci, "f1_bootstrap_mean": f1_bootstrap_mean, "f1_bootstrap_ci": f1_bootstrap_ci,
        "f1_dir_posterior_mean": f1_dir_posterior_mean, "f1_dir_posterior_ci": f1_dir_posterior_ci,
        "f1_beta_posterior_mean": f1_beta_posterior_mean, "f1_beta_posterior_ci": f1_beta_posterior_ci
    }])
    df.to_csv('result4-24.csv', mode='a', header=not os.path.exists('result4-24.csv'), index=False)


def save_results_drift(dataset, trial, num_bootstrap_samples, weights,drift_ratio, n_samples, n_features, n_informative,
                                 corr_beta, corr_dirichlet,
                                 f1_point_estimate, fbeta_point_estimate, f1_bootstrap_mean, f1_bootstrap_ci,
                                 f1_dir_posterior_mean, f1_dir_posterior_ci, f1_beta_posterior_mean, f1_beta_posterior_ci):
    df = pd.DataFrame([{
        "dataset": dataset, "trial": trial, "num_bootstrap_samples": num_bootstrap_samples,
        "weights": weights, "drift_ratio": drift_ratio, "n_samples": n_samples, "n_features": n_features, "n_informative": n_informative,
        "corr_beta": corr_beta, "corr_dirichlet": corr_dirichlet,
        "f1_point_estimate": f1_point_estimate, "fbeta_point_estimate": fbeta_point_estimate,
        "f1_bootstrap_mean": f1_bootstrap_mean, "f1_bootstrap_ci": f1_bootstrap_ci,
        "f1_dir_posterior_mean": f1_dir_posterior_mean, "f1_dir_posterior_ci": f1_dir_posterior_ci,
        "f1_beta_posterior_mean": f1_beta_posterior_mean, "f1_beta_posterior_ci": f1_beta_posterior_ci
    }])
    df.to_csv('result4-24.csv', mode='a', header=not os.path.exists('result4-24.csv'), index=False)


if __name__ == '__main__':
    num_bootstrap_samples = 50000

    dataset = "synthetic"  # synthetic, ecoli, yeast_me2, oil, thyroid_sick, satimage, ecoli_small_n
    # n_samples_list = [50, 500]
    # weights_list = [[0.9, 0.1], [0.90, 0.1], [0.5, 0.5]]
    # n_features_list = [20, 200]

    n_samples_list = [50]
    weights_list = [[0.9, 0.1]]
    n_features_list = [20]

    plot = True

    test_drift = False
    drift_levels = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

    prior_sensitivity = False

    for trial in range(0, 50, 1):
        for weights in weights_list: # , 0.95, 0.05
            for n_samples in n_samples_list:
                for n_features in n_features_list:
                    if n_samples == 50 and n_features == 500:
                        pass
                    n_informative = int(n_features / 2)
                    X_train, X_test, y_train, y_test = generate_data(dataset, weights, n_samples, n_features, n_informative)

                    imb_rate = sum(y_train)/(len(y_train) - sum(y_train))
                    imb_ratio = (len(y_train) - sum(y_train))/sum(y_train)
                    print(imb_rate)
                    print(imb_ratio)

                    fbeta = imb_ratio

                    X_train, X_test = transform_data(X_train, X_test)

                    model, tn, fp, fn, tp, y_pred = fit_model(X_train, y_train, X_test, y_test)  # For test data

                    if prior_sensitivity:
                        beta_prior_list, dirichlet_prior_list = prior_lists()
                        counts = np.array([tp, fp, fn, tn])
                        point_estimate = 2 * tp / (2 * tp + fp + fn)

                        # === Compute Results ===
                        beta_results, dirichlet_results = [], []

                        for label, alpha, beta_ in beta_prior_list:
                            # Bayesian Posterior F1 (Beta-Multinomial)
                            p_samples, r_samples, f1_posterior_samples = compute_beta_multinomial_f1(counts, alpha, beta_,
                                                                                                     num_bootstrap_samples)
                            f1_beta_posterior_mean, f1_beta_posterior_ci = compute_mean_ci(f1_posterior_samples)

                            beta_results.append((label, f1_beta_posterior_mean, f1_beta_posterior_ci[0], f1_beta_posterior_ci[1]))

                        for label, prior in dirichlet_prior_list:
                            # Bayesian Posterior F1 (Dirichlet-Multinomial)
                            precision_dir, recall_dir, f1_dir_posterior_samples = compute_dirichlet_multinomial_f1(
                                counts, num_bootstrap_samples, prior)
                            f1_dir_posterior_mean, f1_dir_posterior_ci = compute_mean_ci(f1_dir_posterior_samples)
                            dirichlet_results.append((label, f1_dir_posterior_mean, f1_dir_posterior_ci[0], f1_dir_posterior_ci[1]))

                        # === Convert to DataFrames ===
                        df_beta = pd.DataFrame(beta_results, columns=["Prior", "Mean", "CI_Lower", "CI_Upper"])
                        df_dirichlet = pd.DataFrame(dirichlet_results,
                                                    columns=["Prior", "Mean", "CI_Lower", "CI_Upper"])

                        # === Plot Both Results ===
                        plot_sensitivity_results(df_beta, f'Beta Posterior Sensitivity (F1 Score) - ({dataset}) - n_samples = {n_samples} - n_features = {n_features} - {weights}', point_estimate)
                        plot_sensitivity_results(df_dirichlet, f'Dirichlet Posterior Sensitivity (F1 Score) - ({dataset}) - n_samples = {n_samples} - n_features = {n_features} - {weights}', point_estimate)

                        # === Save to CSV ===
                        df_beta.to_csv("beta_prior_sensitivity.csv", index=False)
                        df_dirichlet.to_csv("dirichlet_prior_sensitivity.csv", index=False)

                    elif test_drift:
                        drifted_test_sets = simulate_test_drift(X_test, y_test, drift_levels)

                        for drift_ratio, X_drifted, y_drifted in drifted_test_sets:
                            y_pred_drifted = model.predict(X_drifted)
                            tn, fp, fn, tp = confusion_matrix(y_drifted, y_pred_drifted).ravel()

                            f1_point_estimate, fbeta_point_estimate = compute_point_estimates(y_test, y_pred_drifted)

                            # Bootstrap
                            f1_bootstrap_samples = compute_bootstrap_f1(model, X_drifted, y_drifted,
                                                                        num_bootstrap_samples)
                            f1_bootstrap_mean, f1_bootstrap_ci = compute_mean_ci(f1_bootstrap_samples)

                            # Dirichlet
                            counts = np.array([tp, fp, fn, tn])
                            precision_dir, recall_dir, f1_dir_posterior_samples = compute_dirichlet_multinomial_f1(
                                counts, num_bootstrap_samples, alpha)
                            f1_dir_posterior_mean, f1_dir_posterior_ci = compute_mean_ci(f1_dir_posterior_samples)

                            # Beta
                            p_samples, r_samples, f1_posterior_samples = compute_beta_multinomial_f1(counts,
                                                                                                     num_bootstrap_samples)
                            f1_beta_posterior_mean, f1_beta_posterior_ci = compute_mean_ci(f1_posterior_samples)

                            # Correlation
                            corr_beta = compute_correlation(p_samples, r_samples)
                            corr_dirichlet = compute_correlation(precision_dir, recall_dir)

                            # Save result with drift info
                            save_results_drift(dataset=f"{dataset}_drift_{drift_ratio}", trial=trial,
                                         num_bootstrap_samples=num_bootstrap_samples, weights=weights,
                                               drift_ratio=drift_ratio,
                                         n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                                         corr_beta=corr_beta, corr_dirichlet=corr_dirichlet,
                                         f1_point_estimate=f1_point_estimate, fbeta_point_estimate=fbeta_point_estimate,
                                         f1_bootstrap_mean=f1_bootstrap_mean, f1_bootstrap_ci=f1_bootstrap_ci,
                                         f1_dir_posterior_mean=f1_dir_posterior_mean,
                                         f1_dir_posterior_ci=f1_dir_posterior_ci,
                                         f1_beta_posterior_mean=f1_beta_posterior_mean,
                                         f1_beta_posterior_ci=f1_beta_posterior_ci)
                    else:
                        # Compute Point estimates
                        f1_point_estimate, fbeta_point_estimate = compute_point_estimates(y_test, y_pred)

                        # Compute Cross-Validation F1 Scores
                        cv_f1_scores = compute_cross_validation_f1(model, X_train, y_train)
                        f1_cv_mean, f1_cv_ci = compute_mean_ci(cv_f1_scores)

                        # Compute Bootstrap F1 Scores
                        f1_bootstrap_samples = compute_bootstrap_f1(model, X_test, y_test, num_bootstrap_samples)
                        f1_bootstrap_mean, f1_bootstrap_ci = compute_mean_ci(f1_bootstrap_samples)

                        # Bayesian Posterior F1 (Dirichlet-Multinomial)
                        counts = np.array([tp, fp, fn, tn])

                        # Dirichlet prior (noninformative)
                        alpha = np.array([1, 1, 1, 1])
                        precision_dir, recall_dir, f1_dir_posterior_samples = compute_dirichlet_multinomial_f1(counts, num_bootstrap_samples, alpha)
                        f1_dir_posterior_mean, f1_dir_posterior_ci = compute_mean_ci(f1_dir_posterior_samples)

                        # Bayesian Posterior F1 (Beta-Multinomial)

                        # Beta prior (noninformative)
                        alpha_P, beta_P = 1, 1
                        p_samples, r_samples, f1_posterior_samples = compute_beta_multinomial_f1(counts, alpha_P, beta_P, num_bootstrap_samples)
                        f1_beta_posterior_mean, f1_beta_posterior_ci = compute_mean_ci(f1_posterior_samples)

                        # Bayesian Posterior F-beta (Beta-Multinomial)

                        fbeta_beta = compute_fbeta(p_samples, r_samples)
                        fbeta_posterior_samples = np.nan_to_num(fbeta_beta, nan=0.0)
                        fbeta_posterior_mean, fbeta_posterior_ci = compute_mean_ci(fbeta_posterior_samples)

                        # Compute Pearson correlation coefficients
                        corr_beta = compute_correlation(p_samples, r_samples)
                        corr_dirichlet = compute_correlation(precision_dir, recall_dir)

                        # Print the results
                        print(f"Correlation (Beta approach): {corr_beta:.3f}")
                        print(f"Correlation (Dirichlet approach): {corr_dirichlet:.3f}")
                        save_results(dataset, trial, num_bootstrap_samples, weights, n_samples, n_features, n_informative,
                                     corr_beta, corr_dirichlet,
                                     f1_point_estimate, fbeta_point_estimate, f1_cv_mean, f1_cv_ci, f1_bootstrap_mean, f1_bootstrap_ci,
                                     f1_dir_posterior_mean, f1_dir_posterior_ci, f1_beta_posterior_mean, f1_beta_posterior_ci)

                        if plot:
                            # --- Plot only 3 methods in subplots ---
                            fig, axes = plt.subplots(2, 2, figsize=(12, 5), constrained_layout=True)

                            # Bootstrap
                            axes[0, 0].hist(f1_bootstrap_samples, bins=50, color='lightgreen', alpha=0.7, density=True,
                                            label='Bootstrap F1 Samples')
                            axes[0, 0].axvline(f1_bootstrap_mean, color='black', linestyle='--',
                                               label=f'Mean = {f1_bootstrap_mean:.3f}')
                            axes[0, 0].axvline(f1_bootstrap_ci[0], color='green', linestyle='--',
                                               label=f'2.5% CI = {f1_bootstrap_ci[0]:.3f}')
                            axes[0, 0].axvline(f1_bootstrap_ci[1], color='green', linestyle='--',
                                               label=f'97.5% CI = {f1_bootstrap_ci[1]:.3f}')
                            axes[0, 0].axvline(f1_point_estimate, color='red', linestyle='--',
                                               label=f'Point Estimate = {f1_point_estimate:.3f}')
                            axes[0, 0].set_title('Bootstrap F1 Distribution')
                            axes[0, 0].set_xlabel('F1 Score')
                            axes[0, 0].set_xlim(0, 1)
                            axes[0, 0].legend()
                            axes[0, 0].grid(True)

                            # Bayesian Posterior
                            axes[0, 1].hist(f1_posterior_samples, bins=50, color='skyblue', alpha=0.7, density=True,
                                            label='Posterior F1 Samples')
                            axes[0, 1].axvline(f1_beta_posterior_mean, color='black', linestyle='--',
                                               label=f'Mean = {f1_beta_posterior_mean:.3f}')
                            axes[0, 1].axvline(f1_beta_posterior_ci[0], color='orange', linestyle='--',
                                               label=f'2.5% CI = {f1_beta_posterior_ci[0]:.3f}')
                            axes[0, 1].axvline(f1_beta_posterior_ci[1], color='orange', linestyle='--',
                                               label=f'97.5% CI = {f1_beta_posterior_ci[1]:.3f}')
                            axes[0, 1].axvline(f1_point_estimate, color='red', linestyle='--',
                                               label=f'Point Estimate = {f1_point_estimate:.3f}')
                            axes[0, 1].set_title('Posterior (Beta) F1 Distribution')
                            axes[0, 1].set_xlabel('F1 Score')
                            axes[0, 1].set_xlim(0, 1)
                            axes[0, 1].legend()
                            axes[0, 1].grid(True)

                            # Dirichlet Prior
                            axes[1, 0].hist(f1_dir_posterior_samples, bins=50, color='violet', alpha=0.7, density=True,
                                            label='Posterior F1 Samples')
                            axes[1, 0].axvline(f1_dir_posterior_mean, color='black', linestyle='--',
                                               label=f'Mean = {f1_dir_posterior_mean:.3f}')
                            axes[1, 0].axvline(f1_dir_posterior_ci[0], color='orange', linestyle='--',
                                               label=f'2.5% CI = {f1_dir_posterior_ci[0]:.3f}')
                            axes[1, 0].axvline(f1_dir_posterior_ci[1], color='orange', linestyle='--',
                                               label=f'97.5% CI = {f1_dir_posterior_ci[1]:.3f}')
                            axes[1, 0].axvline(f1_point_estimate, color='red', linestyle='--',
                                               label=f'Point Estimate = {f1_point_estimate:.3f}')
                            axes[1, 0].set_title('Posterior (Dirichlet) F1 Distribution')
                            axes[1, 0].set_xlabel('F1 Score')
                            axes[1, 0].set_xlim(0, 1)
                            axes[1, 0].legend()
                            axes[1, 0].grid(True)

                            # Remove the unused bottom-right plot
                            fig.delaxes(axes[1, 1])

                            # Final tweaks
                            fig.suptitle(
                                f'Comparison of F1 Estimates - ({dataset}), n_samples = {n_samples}, IR = {weights}',
                                fontsize=14)
                            plt.show()

                            # Plotting joint distributions
                            # ------------------------------
                            fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

                            # Plot for Beta approach
                            axs[0].hexbin(p_samples, r_samples, gridsize=60, cmap='Blues', extent=[0, 1, 0, 1])
                            axs[0].set_title("Beta Posterior (Independent)")
                            axs[0].set_xlabel("Precision")
                            axs[0].set_ylabel("Recall")

                            # Plot for Dirichlet approach
                            axs[1].hexbin(precision_dir, recall_dir, gridsize=60, cmap='Oranges', extent=[0, 1, 0, 1])
                            axs[1].set_title("Dirichlet Posterior (Joint)")
                            axs[1].set_xlabel("Precision")
                            axs[1].set_ylabel("Recall")

                            plt.suptitle(f"Joint Distribution of Precision and Recall  - ({dataset}), n_samples = {n_samples}, IR = {weights}")
                            plt.tight_layout()
                            plt.show()



