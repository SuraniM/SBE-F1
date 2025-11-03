# burnin_coverage_with_f1_ci.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import dirichlet
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

# ---- Optional ArviZ diagnostics ----
try:
    import arviz as az
    HAVE_ARVIZ = True
except Exception:
    HAVE_ARVIZ = False
    print("⚠️  ArviZ not found. Using heuristic burn-in detection.")

# --------------------------
# Config for simulation
# --------------------------
n_batches_list   = [50, 100, 200]                      # number of batches
batch_size_list  = [10, 20, 50]                        # batch sizes
imbalance_ratios = [[0.80, 0.20], [0.95, 0.05], [0.99, 0.01]]  # class ratios
R                = 100                                 # replications per setting
draws_final      = 20000                               # posterior draws for final CI (coverage)
draws_per_batch  = 4000                                # posterior draws per-batch CI (trajectory)
alpha0           = np.ones(4)                          # Dirichlet(1,1,1,1) prior
rng_master       = np.random.default_rng(7)

# n_batches_list   = [100]                      # number of batches
# batch_size_list  = [50]                        # batch sizes
# imbalance_ratios = [[0.80, 0.20], [0.95, 0.05], [0.99, 0.01]]  # class ratios
# R                = 100                                 # replications per setting
# draws_final      = 20000                               # posterior draws for final CI (coverage)
# draws_per_batch  = 4000                                # posterior draws per-batch CI (trajectory)
# alpha0           = np.ones(4)                          # Dirichlet(1,1,1,1) prior
# alpha0 = [0.5, 0.5, 0.5, 0.5]
# rng_master       = np.random.default_rng(7)

os.makedirs("plots_burnin", exist_ok=True)
os.makedirs("plots_replication_intervals", exist_ok=True)
EXCEL_OUT = "burnin_coverage_summary.xlsx"

# --------------------------
# Helpers
# --------------------------
def dirichlet_f1_samples(alphas, draws=20000, rng=None):
    """Sample theta ~ Dirichlet(alphas) and return F1 samples."""
    theta = dirichlet.rvs(alphas, size=draws, random_state=rng)
    tp, fp, fn = theta[:, 0], theta[:, 1], theta[:, 2]
    denom = 2*tp + fp + fn
    denom = np.where(denom == 0.0, np.finfo(float).eps, denom)
    return 2*tp / denom

def dirichlet_f1_ci(alphas, draws=20000, rng=None):
    """Return (mean, L, U) for F1 from Dirichlet posterior."""
    f1_s = dirichlet_f1_samples(alphas, draws=draws, rng=rng)
    mu = float(np.mean(f1_s))
    L, U = np.quantile(f1_s, [0.025, 0.975])
    return mu, float(L), float(U)

def run_one_replication_final_ci(clf, X_te, y_te, nb, batch_size, alpha0, rng, LAMBDA):
    """Run one sequential replication; return final (mean, L, U)."""
    cum_tp = cum_fp = cum_fn = cum_tn = 0.0
    for _ in range(nb):
        idx = rng.choice(len(y_te), size=batch_size, replace=False)
        y_b  = y_te[idx]
        yhb  = clf.predict(X_te[idx])
        cm = confusion_matrix(y_b, yhb, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        if LAMBDA != 1.0:
            cum_tp *= LAMBDA; cum_fp *= LAMBDA; cum_fn *= LAMBDA; cum_tn *= LAMBDA
        cum_tp += tp; cum_fp += fp; cum_fn += fn; cum_tn += tn
    alphas = alpha0 + np.array([cum_tp, cum_fp, cum_fn, cum_tn], dtype=float)
    return dirichlet_f1_ci(alphas, draws=draws_final, rng=None)

def run_trajectory(clf, X_te, y_te, nb, batch_size, alpha0, draws=4000, rng=None):
    """Per-batch 95% CI trajectory for F1 (returns mean[], L[], U[], alphas_T)."""
    means, lowers, uppers = [], [], []
    alphas_t = alpha0.copy()
    for _ in range(nb):
        idx = rng.choice(len(y_te), size=batch_size, replace=False)
        y_b  = y_te[idx]
        yhb  = clf.predict(X_te[idx])
        cm = confusion_matrix(y_b, yhb, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        alphas_t = alphas_t + np.array([tp, fp, fn, tn], dtype=float)
        mu, L, U = dirichlet_f1_ci(alphas_t, draws=draws, rng=None)
        means.append(mu); lowers.append(L); uppers.append(U)
    return np.array(means), np.array(lowers), np.array(uppers), alphas_t

def detect_burnin_arviz(mean_f1, ci_lower, ci_upper, true_f1=None,
                        window=50, ess_thresh=0.01, acorr_thresh=0.1):
    """Burn-in via Geweke, rolling ESS, and autocorrelation (ArviZ)."""
    if not HAVE_ARVIZ:
        return None, {}
    mean_f1 = np.asarray(mean_f1)
    n = len(mean_f1)

    # Geweke
    zscores = az.geweke(mean_f1, first=0.1, last=0.5, intervals=20)
    zvals = np.array([z for (_, z) in zscores])
    zvals_smooth = np.convolve(np.abs(zvals), np.ones(5)/5, mode='same')
    stable = zvals_smooth < 2.0
    burnin_geweke = None
    if len(stable) >= window:
        conv = np.convolve(stable, np.ones(window, dtype=int), mode='same')
        hits = np.where(conv >= window)[0]
        if hits.size > 0:
            burnin_geweke = int(hits[0])

    # Rolling ESS stabilization
    ess_values = []
    for t in range(window, n):
        try:
            ess = az.ess(np.array(mean_f1[:t]))
        except Exception:
            ess = np.nan
        ess_values.append(ess)
    ess_values = np.array(ess_values, dtype=float)
    burnin_ess = None
    if np.all(np.isfinite(ess_values)) and ess_values.size > 20:
        rel_change = np.abs(np.diff(ess_values) / np.clip(ess_values[:-1], 1e-12, None))
        idx = np.where(rel_change < ess_thresh)[0]
        if idx.size > 0:
            burnin_ess = int(idx[0] + window)

    # Autocorrelation threshold
    try:
        ac = az.autocorr(mean_f1)
        below = np.where(ac < acorr_thresh)[0]
        burnin_autocorr = int(below[0]) if below.size > 0 else None
    except Exception:
        burnin_autocorr = None

    candidates = [b for b in [burnin_geweke, burnin_ess, burnin_autocorr] if b is not None]
    burnin_final = int(np.median(candidates)) if candidates else None
    return burnin_final, {
        "burnin_geweke": burnin_geweke,
        "burnin_ess": burnin_ess,
        "burnin_autocorr": burnin_autocorr
    }

def detect_burnin_heuristic(mean_f1, ci_lower, ci_upper, true_f1=None,
                            W=30, mean_tol_rel=0.25, width_tol_rel=0.25, K_inside=10):
    """Fallback burn-in detector using stability of mean and CI width (and optional inside-CI streak)."""
    mean_f1, lo, hi = map(np.asarray, (mean_f1, ci_lower, ci_upper))
    width = hi - lo
    n = len(mean_f1)
    if n < W:
        return None, {"heuristic": True}
    med_width = np.median(width[max(1, n//10):]) or np.median(width) or 1e-3
    mean_tol = mean_tol_rel * med_width
    width_tol = width_tol_rel * med_width

    for t in range(0, n - W + 1):
        win = slice(t, t+W)
        if true_f1 is not None:
            if not np.all((lo[win] <= true_f1) & (true_f1 <= hi[win])):
                continue
        max_dmean = np.max(np.abs(np.diff(mean_f1[win])))
        wspan = np.max(width[win]) - np.min(width[win])
        if max_dmean <= mean_tol and wspan <= width_tol:
            return t, {"heuristic": True}

    if true_f1 is not None:
        consec = 0
        for i in range(n):
            if lo[i] <= true_f1 <= hi[i]:
                consec += 1
                if consec >= K_inside:
                    return max(0, i - K_inside + 1), {"heuristic": True}
            else:
                consec = 0
    return None, {"heuristic": True}

def save_burnin_plot(x, mean_f1, ci_lower, ci_upper, true_f1, burn_idx, png_path, pdf_path):
    """Per-batch plot WITH 95% credible intervals for F1 and burn-in marker (300 dpi)."""
    plt.figure(figsize=(10, 6))
    # 95% CI band per batch (this is your requested addition)
    plt.fill_between(x, ci_lower, ci_upper, alpha=0.30, label='95% Credible Interval')
    plt.plot(x, mean_f1, marker='o', ms=3, lw=1.2, label='Posterior Mean $F_1$')
    if true_f1 is not None:
        plt.axhline(true_f1, color='red', linestyle='--', lw=1.4, label='True $F_1^*$')
    if burn_idx is not None:
        plt.axvspan(1, burn_idx+1, color='orange', alpha=0.15, label='Burn-in')
        plt.axvline(burn_idx+1, color='orange', linestyle='--', lw=1.6)
        plt.text(burn_idx+1, np.max(ci_upper), f' Burn-in ends @ batch {burn_idx+1}',
                 va='bottom', ha='left', color='orange')
    plt.xlabel("Batch Number"); plt.ylabel("F1 Score")
    plt.title(f"Sequential Bayesian $F_1$ with 95% CIs and Burn-in Detection - Lambda =  {LAMBDA}")
    plt.legend(); plt.grid(True, alpha=0.4); plt.tight_layout()
    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    # plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()

def save_replication_intervals_plot(means, lowers, uppers, true_f1, nb, batch_size, IR, png_path, pdf_path):
    """Final 95% CI per replication (for empirical coverage visualization)."""
    R = len(means)
    x = np.arange(1, R+1)
    hits = (lowers <= true_f1) & (true_f1 <= uppers)

    plt.figure(figsize=(10, 5.5))
    for covered, color in [(True, "#3568d4"), (False, "#c73d2f")]:
        sel = hits == covered
        for xi, lo, hi in zip(x[sel], lowers[sel], uppers[sel]):
            plt.vlines(xi, lo, hi, color=color, alpha=0.85, linewidth=1.4)
        plt.scatter(x[sel], means[sel], s=24, color=color)
    plt.axhline(true_f1, color="gray", linestyle="--", linewidth=1.2)

    plt.title(f"95% CIs for $F_1$ at final step (nb={nb}, batch={batch_size}, IR={IR})")
    plt.xlabel("Replication (SampleID)")
    plt.ylabel("F1")
    plt.xlim(0, R+1)
    legend_elems = [
        Line2D([0], [0], color="#3568d4", lw=2, label="Parameter in CI: Yes"),
        Line2D([0], [0], color="#c73d2f", lw=2, label="Parameter in CI: No"),
    ]
    plt.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, -0.15),
               ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    # plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()

# --------------------------
# Main
# --------------------------
rows = []
LAMBDA = 1.0

for nb in n_batches_list:
    for bs in batch_size_list:
        for IR in imbalance_ratios:
            print(f"== Setting: nb={nb}, batch_size={bs}, IR={IR} ==")

            # Build population + classifier
            X, y = make_classification(
                n_samples=200_000, n_features=20, n_informative=10,
                weights=IR, random_state=42
            )
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.5, stratify=y, random_state=42
            )
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_tr, y_tr)
            true_f1 = f1_score(y_te, clf.predict(X_te))

            # Empirical coverage via final 95% CI across R replications
            means_R, lowers_R, uppers_R = [], [], []
            hits = []
            for r in range(R):
                rng = np.random.default_rng(10_000 + r)
                mu, L, U = run_one_replication_final_ci(clf, X_te, y_te, nb, bs, alpha0, rng, LAMBDA)
                means_R.append(mu); lowers_R.append(L); uppers_R.append(U)
                hits.append(int(L <= true_f1 <= U))
            means_R  = np.array(means_R)
            lowers_R = np.array(lowers_R)
            uppers_R = np.array(uppers_R)
            cov = float(np.mean(hits))
            se  = float(np.sqrt(cov * (1 - cov) / R))

            # Save replication-interval plot (final-step 95% CI per replication)
            tag_ir = f"{IR[0]:.2f}-{IR[1]:.2f}"
            rep_png = f"plots_replication_intervals/finalCIs_nb{nb}_bs{bs}_IR{tag_ir}_{LAMBDA}.png"
            rep_pdf = f"plots_replication_intervals/finalCIs_nb{nb}_bs{bs}_IR{tag_ir}_{LAMBDA}.pdf"
            save_replication_intervals_plot(means_R, lowers_R, uppers_R, true_f1, nb, bs, IR, rep_png, rep_pdf)

            # One full trajectory (per-batch 95% CI of F1) and burn-in
            rng = np.random.default_rng(12345)
            mean_f1, ci_lower, ci_upper, alphas_T = run_trajectory(
                clf, X_te, y_te, nb, bs, alpha0, draws=draws_per_batch, rng=rng
            )

            if HAVE_ARVIZ:
                burn_idx, diag = detect_burnin_arviz(
                    mean_f1, ci_lower, ci_upper, true_f1=true_f1,
                    window=min(50, max(20, nb // 5)), ess_thresh=0.01, acorr_thresh=0.1
                )
            else:
                burn_idx, diag = detect_burnin_heuristic(
                    mean_f1, ci_lower, ci_upper, true_f1=true_f1,
                    W=min(40, max(20, nb // 5)), mean_tol_rel=0.25, width_tol_rel=0.25, K_inside=10
                )

            # Save per-batch plot with 95% CI and burn-in marker (this is the requested CI addition)
            burn_png = f"plots_burnin/burnin_nb{nb}_bs{bs}_IR{tag_ir}_{LAMBDA}.png"
            burn_pdf = f"plots_burnin/burnin_nb{nb}_bs{bs}_IR{tag_ir}_{LAMBDA}.pdf"
            x = np.arange(1, nb + 1)
            save_burnin_plot(x, mean_f1, ci_lower, ci_upper, true_f1, burn_idx, burn_png, burn_pdf)

            # Row summary
            rows.append({
                "nb": nb,
                "batch_size": bs,
                "IR": str(IR),
                "Empirical Coverage": f"{cov:.3f} ± {1.96*se:.3f}",
                "Coverage (raw)": round(cov, 4),
                "SE": round(se, 4),
                "True F1*": round(true_f1, 4),
                "Burn-in Index": (int(burn_idx) + 1) if burn_idx is not None else None,
                "Diag (Geweke)": diag.get("burnin_geweke") if diag else None,
                "Diag (ESS)": diag.get("burnin_ess") if diag else None,
                "Diag (Autocorr)": diag.get("burnin_autocorr") if diag else None,
                "Per-batch F1 CI Figure": burn_png,
                "Final-step CIs Figure": rep_png
            })

            print(f"   → coverage = {cov:.3f} ± {1.96*se:.3f} | burn-in (1-based) = {rows[-1]['Burn-in Index']}")
            print(f"   → saved: {burn_png}")
            print(f"   → saved: {rep_png}")

# Save Excel summary
df = pd.DataFrame(rows, columns=[
    "nb", "batch_size", "IR", "Empirical Coverage", "Coverage (raw)", "SE",
    "True F1*", "Burn-in Index", "Diag (Geweke)", "Diag (ESS)", "Diag (Autocorr)",
    "Per-batch F1 CI Figure", "Final-step CIs Figure"
])
df.to_excel(EXCEL_OUT, index=False)
print(f"✅ Saved Excel: {EXCEL_OUT}")
print(df)
