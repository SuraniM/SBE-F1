import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score
from scipy.stats import dirichlet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Configuration
batch_size = 50
n_samples = 1000  # Monte Carlo samples for posterior
random_seed = 42

# Load and clean dataset
df = pd.read_csv(
    'C:\\Users\\Surani\\Documents\\PhD Research Work\\PhD Work\\PostDoc\\F1Score\\archive (2)\\water_potability.csv')
df.dropna(inplace=True)

# Features and target
X = df.drop(columns="Potability")
y = df["Potability"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=random_seed, stratify=y
)

# Reset test indices
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(random_state=random_seed),
    "Nearest Neighbors":  KNeighborsClassifier(3),
    "AdaBoost" : AdaBoostClassifier(random_state=random_seed),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=random_seed)
    # "Logistic Regression": LogisticRegression(max_iter=1000),
    # "SVM": SVC(probability=True),
}

# Loop through classifiers
for classif_name, clf in classifiers.items():
    print(f"\nRunning for: {classif_name}")

    # Fit the model
    clf.fit(X_train, y_train)

    # Calculate point estimates
    y_pred = clf.predict(X_test)
    f1_point_estimate = f1_score(y_test, y_pred)
    print(f1_point_estimate)

    # Reset prior for each model
    alpha_prior = np.array([1.0, 1.0, 1.0, 1.0])

    # Initialize storage
    f1_posterior_means = []
    f1_credible_intervals = []
    f1_point_estimates = []
    batch_indices = []

    n_batches = len(X_test) // batch_size

    for t in range(n_batches):
        start = t * batch_size
        end = start + batch_size
        X_batch = X_test.iloc[start:end]
        y_batch = y_test.iloc[start:end]

        # Predictions and confusion matrix
        y_pred = clf.predict(X_batch)
        tn, fp, fn, tp = confusion_matrix(y_batch, y_pred, labels=[0, 1]).ravel()
        x_t = np.array([tp, fp, fn, tn])

        # Update posterior
        alpha_posterior = alpha_prior + x_t

        # Sample from Dirichlet posterior
        samples = dirichlet.rvs(alpha_posterior, size=n_samples)
        theta_TP, theta_FP, theta_FN = samples[:, 0], samples[:, 1], samples[:, 2]
        f1_samples = 2 * theta_TP / (2 * theta_TP + theta_FP + theta_FN)

        # Summarize posterior
        mean_f1 = np.mean(f1_samples)
        ci_lower, ci_upper = np.percentile(f1_samples, [2.5, 97.5])
        f1_bayes = (mean_f1, ci_lower, ci_upper)

        # Point estimate F1
        f1_point = f1_score(y_batch, y_pred, zero_division=0)

        # Store results
        f1_posterior_means.append(mean_f1)
        f1_credible_intervals.append((ci_lower, ci_upper))
        f1_point_estimates.append(f1_point)
        batch_indices.append(t + 1)

        # Set posterior as next prior
        alpha_prior = alpha_posterior

    # Convert to arrays
    mean_f1 = np.array(f1_posterior_means)
    print(mean_f1[19])
    ci_lower = np.array([ci[0] for ci in f1_credible_intervals])
    ci_upper = np.array([ci[1] for ci in f1_credible_intervals])
    point_f1 = np.array(f1_point_estimates)
    print(ci_lower[19], ci_upper[19])

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(batch_indices, mean_f1, marker='o', label=f'{classif_name} – Bayesian Mean F1')
    # plt.plot(batch_indices, point_f1, linestyle='--', marker='x', alpha=0.7,
    #          label=f'{classif_name} – Point Estimate F1')
    plt.fill_between(batch_indices, ci_lower, ci_upper, alpha=0.3, label='95% Credible Interval')
    plt.ylim(0, 0.7)
    plt.xlabel("Batch Number")
    plt.ylabel("F1 Score")
    plt.title(f"Sequential F1 Score Estimation – {classif_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Annotate the last values
    last_x = batch_indices[-1]
    last_y = mean_f1[-1]
    last_ci_lower = ci_lower[-1]
    last_ci_upper = ci_upper[-1]

    plt.text(last_x, last_y + 0.02, f"Mean: {last_y:.3f}", ha='center', fontsize=10, color='black')
    plt.text(last_x, last_ci_upper + 0.02, f"Upper: {last_ci_upper:.3f}", ha='center', fontsize=9, color='gray')
    plt.text(last_x, last_ci_lower - 0.04, f"Lower: {last_ci_lower:.3f}", ha='center', fontsize=9, color='gray')

    # Save and show the plot
    save_file = "sequential_f1_" + classif_name.lower().replace(' ', '_')
    path = "C:/Users/Surani/Documents/PhD Research Work/PhD Work/PostDoc/SBE-F1/R1_HR_Figures/"
    plt.savefig(path + save_file + ".png", format="png", dpi=600)
    # plt.show()

