import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import dirichlet, multinomial

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
n_batches = 700
batch_size = 50
true_theta95 = np.array([0.03, 0.05, 0.02, 0.90])  # [TP, FP, FN, TN] 95:5
true_theta80 = np.array([0.15, 0.05, 0.05, 0.75])  # [TP, FP, FN, TN] 80:20
true_theta98 = np.array([0.01, 0.03, 0.01, 0.95])  # [TP, FP, FN, TN] 98:2
# alpha_prior = np.array([1.0, 1.0, 1.0, 1.0])  # Flat Dirichlet prior

ir = ["[0.80, 0.20]", "[0.99, 0.01]", "[0.95, 0.05]"]
i = 0
for true_theta in [true_theta95, true_theta80, true_theta98]:
    # Storage
    f1_posterior_means = []
    f1_credible_intervals = []
    batch_indices = []
    alpha_prior = np.array([0.5, 0.5, 0.5, 0.5])  # Flat Dirichlet prior

    # Sequential update
    for t in range(n_batches):
        # Simulate a batch of classification results
        x_t = multinomial.rvs(n=batch_size, p=true_theta)

        # Update posterior parameters
        alpha_posterior = alpha_prior + x_t

        # Sample from Dirichlet posterior
        samples = dirichlet.rvs(alpha_posterior, size=1000)
        theta_TP, theta_FP, theta_FN = samples[:, 0], samples[:, 1], samples[:, 2]

        # Compute F1 score for each sample
        f1_samples = 2 * theta_TP / (2 * theta_TP + theta_FP + theta_FN)

        # Summarize posterior
        mean_f1 = np.mean(f1_samples)
        ci_lower = np.percentile(f1_samples, 2.5)
        ci_upper = np.percentile(f1_samples, 97.5)

        f1_posterior_means.append(mean_f1)
        f1_credible_intervals.append((ci_lower, ci_upper))
        batch_indices.append(t + 1)

        # Set current posterior as next prior
        alpha_prior = alpha_posterior

    # Plotting
    mean_f1 = np.array(f1_posterior_means)
    ci_lower = np.array([ci[0] for ci in f1_credible_intervals])
    ci_upper = np.array([ci[1] for ci in f1_credible_intervals])
    true_f1 = 2 * true_theta[0] / (2 * true_theta[0] + true_theta[1] + true_theta[2])

    plt.figure(figsize=(10, 6))
    plt.plot(batch_indices, mean_f1, marker='o', label='Posterior Mean F1')
    plt.fill_between(batch_indices, ci_lower, ci_upper, alpha=0.3, label='95% Credible Interval')
    plt.axhline(true_f1, color='red', linestyle='--', label='True F1 Score')
    plt.xlabel("Batch Number")
    plt.ylabel("F1 Score")
    plt.title(f"Sequential Bayesian Estimation of F1 Score, IR = {ir[i]}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_file = 'SBE - ' + str(i)
    path = "C:/Users/Surani/Documents/PhD Research Work/PhD Work/PostDoc/SBE-F1/R1_HR_Figures/"
    plt.savefig(path + save_file + ".png", format="png", dpi=600)
    plt.savefig(path + save_file + ".pdf", format="pdf", dpi=600)
    i = i+1
# plt.show()
