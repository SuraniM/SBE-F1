# drifting_f1_forgetting_demo.py
# Demonstration of forgetting in Dirichlet–Multinomial sequential F1 estimation.
# Compares cumulative (lambda=1), exponential discounting (lambda<1), and fixed-window updates.

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

def dirichlet_f1_mean(alpha):
    """Posterior mean F1 from Dirichlet parameters alpha=[aTP,aFP,aFN,aTN]."""
    aTP, aFP, aFN, aTN = alpha
    # Use posterior mean of theta, then plug into F1 functional (delta method alternative is fine)
    s = aTP + aFP + aFN + aTN
    tTP, tFP, tFN = aTP/s, aFP/s, aFN/s
    denom = (2*tTP + tFP + tFN)
    return 0.0 if denom <= 0 else 2*tTP/denom

def dirichlet_f1_ci(alpha, draws=20000, alpha_eps=1e-12):
    """Equal-tailed 95% CI from Dirichlet posterior samples (efficient enough for demo)."""
    a = np.array(alpha, dtype=float)
    # keep strictly positive
    a = np.maximum(a, alpha_eps)
    theta = rng.dirichlet(a, size=draws)
    tp, fp, fn = theta[:,0], theta[:,1], theta[:,2]
    denom = 2*tp + fp + fn
    denom[denom == 0] = np.finfo(float).eps
    f1 = 2*tp/denom
    return float(np.quantile(f1, 0.025)), float(np.quantile(f1, 0.975))

def simulate_batch_counts(n_pos, n_neg, recall, precision):
    """
    Simulate a batch confusion matrix given positives/negatives and (recall, precision).
    We sample TP ~ Binom(n_pos, recall). For FP, invert precision: precision = TP / (TP+FP) ⇒ FP ≈ TP*(1-prec)/prec in expectation.
    We sample FP with mean matching that expectation, capped by n_neg.
    """
    TP = rng.binomial(n_pos, recall)
    FN = n_pos - TP
    # expected FP given TP and precision
    exp_FP = TP*(1-precision)/max(precision, 1e-9)
    # sample FP with variance using Poisson (good for counts), capped
    FP = min(int(rng.poisson(max(exp_FP, 0.0))), n_neg)
    TN = n_neg - FP
    return TP, FP, FN, TN

def run_drift_sim(
    T=120,
    batch_pos=30,
    batch_neg=120,
    recall_start=0.90,
    recall_end=0.60,
    precision=0.85,
    alpha0=(1,1,1,1),
    lam1=0.95,
    lam2=0.97,
    lam3=0.99,
    window=None,         # e.g., 25 for fixed-window; if None, ignore
    draws_ci=20000
):
    """
    Simulate a drifting F1 process and track posterior estimates under:
      - cumulative (lambda=1)
      - exponential discounting (lambda=lam < 1)
      - fixed window (last W batches)
    Returns dict with trajectories and CI bands.
    """
    alpha0 = np.array(alpha0, dtype=float)

    # storage
    true_f1 = []
    cum_mean, cum_lo, cum_hi = [], [], []
    disc_mean1, disc_lo1, disc_hi1 = [], [], []
    disc_mean2, disc_lo2, disc_hi2 = [], [], []
    disc_mean3, disc_lo3, disc_hi3 = [], [], []
    win_mean,  win_lo,  win_hi  = [], [], []

    # running state
    alpha_cum = alpha0.copy()
    alpha_disc1 = alpha0.copy()
    alpha_disc2 = alpha0.copy()
    alpha_disc3 = alpha0.copy()
    window_counts = []  # list of count vectors for fixed window

    for t in range(T):
        # linear drift in recall
        recall = recall_start + (recall_end - recall_start)*t/(T-1)
        # true F1 at current operating point
        denom = precision + recall
        true_f1.append(0.0 if denom == 0 else 2*precision*recall/denom)

        # simulate batch counts
        TP, FP, FN, TN = simulate_batch_counts(batch_pos, batch_neg, recall, precision)
        x_t = np.array([TP, FP, FN, TN], dtype=float)

        # --- cumulative update (lambda=1) ---
        alpha_cum = alpha_cum + x_t
        cm = dirichlet_f1_mean(alpha_cum)
        clo, chi = dirichlet_f1_ci(alpha_cum, draws=draws_ci)
        cum_mean.append(cm); cum_lo.append(clo); cum_hi.append(chi)

        # --- exponential discounting ---
        alpha_disc1 = lam1*alpha_disc1 + x_t
        dm1 = dirichlet_f1_mean(alpha_disc1)
        dlo1, dhi1 = dirichlet_f1_ci(alpha_disc1, draws=draws_ci)
        disc_mean1.append(dm1); disc_lo1.append(dlo1); disc_hi1.append(dhi1)

        # --- exponential discounting ---
        alpha_disc2 = lam2 * alpha_disc2 + x_t
        dm2 = dirichlet_f1_mean(alpha_disc2)
        dlo2, dhi2 = dirichlet_f1_ci(alpha_disc2, draws=draws_ci)
        disc_mean2.append(dm2); disc_lo2.append(dlo2); disc_hi2.append(dhi2)

        # --- exponential discounting ---
        alpha_disc3 = lam3 * alpha_disc3 + x_t
        dm3 = dirichlet_f1_mean(alpha_disc3)
        dlo3, dhi3 = dirichlet_f1_ci(alpha_disc3, draws=draws_ci)
        disc_mean3.append(dm3); disc_lo3.append(dlo3); disc_hi3.append(dhi3)

        # --- fixed window (if requested) ---
        if window is not None and window > 0:
            window_counts.append(x_t)
            if len(window_counts) > window:
                window_counts.pop(0)
            alpha_win = alpha0 + np.sum(window_counts, axis=0)
            wm = dirichlet_f1_mean(alpha_win)
            wlo, whi = dirichlet_f1_ci(alpha_win, draws=draws_ci)
            win_mean.append(wm); win_lo.append(wlo); win_hi.append(whi)
        else:
            win_mean.append(np.nan); win_lo.append(np.nan); win_hi.append(np.nan)

    return {
        "true_f1": np.array(true_f1),
        "cum":  {"mean": np.array(cum_mean),  "lo": np.array(cum_lo),  "hi": np.array(cum_hi)},
        "disc1": {"mean": np.array(disc_mean1), "lo": np.array(disc_lo1), "hi": np.array(disc_hi1)},
        "disc2": {"mean": np.array(disc_mean2), "lo": np.array(disc_lo2), "hi": np.array(disc_hi2)},
        "disc3": {"mean": np.array(disc_mean3), "lo": np.array(disc_lo3), "hi": np.array(disc_hi3)},
        "win":  {"mean": np.array(win_mean),  "lo": np.array(win_lo),  "hi": np.array(win_hi)},
        "ess": None if lam1 >= 1 else 1.0/(1.0 - lam1)
    }

if __name__ == "__main__":
    # --- parameters you can tweak ---
    T = 120
    batch_pos, batch_neg = 30, 120        # class imbalance per batch (20% positives)
    recall_start, recall_end = 0.90, 0.60 # drifting recall → drifting F1
    precision = 0.85
    lam1 = 0.95                            # exponential discount; ESS ~ 33
    lam2 = 0.97
    lam3 = 0.99
    window = 25                           # fixed window of last 25 batches
    draws_ci = 20000                      # equal-tailed 95% CI from Dirichlet posterior

    out = run_drift_sim(
        T=T, batch_pos=batch_pos, batch_neg=batch_neg,
        recall_start=recall_start, recall_end=recall_end,
        precision=precision, lam1=lam1, lam2=lam2, lam3=lam3, window=window, draws_ci=draws_ci
    )

    ess = out["ess"]
    if ess is not None:
        print(f"Exponential discounting λ={lam1:.3f} ⇒ ESS ≈ {ess:.1f} batches")

    # --- Plot
    t = np.arange(T)
    plt.figure(figsize=(8,4.2))
    plt.plot(t, out["true_f1"], linestyle="--", label="True $F_1$ (drift)")
    plt.plot(t, out["cum"]["mean"], label="Cumulative ($\\lambda=1$)")
    plt.plot(t, out["disc1"]["mean"], label=f"Discounted ($\\lambda={lam1}$)")
    plt.plot(t, out["disc2"]["mean"], label=f"Discounted ($\\lambda={lam2}$)")
    plt.plot(t, out["disc3"]["mean"], label=f"Discounted ($\\lambda={lam3}$)")
    # if window and window > 0:
    #     plt.plot(t, out["win"]["mean"], label=f"Fixed window (W={window})")

    plt.xlabel("Batch")
    plt.ylabel("$F_1$ estimate")
    plt.title("Tracking drifting $F_1$: cumulative vs discounting")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Drift", dpi=600, bbox_inches="tight")
    plt.show()
