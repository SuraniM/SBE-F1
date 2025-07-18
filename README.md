Bayesian F1 Score Estimation
================

# Bayesian F1 Score Estimation with Sequential Dirichlet-Multinomial Updates

This repository implements a Bayesian framework for estimating the
posterior distribution of the **F1 score**, particularly in **imbalanced
classification** tasks. The approach uses a **Dirichlet-Multinomial
model** with **sequential updating**, allowing for real-time monitoring
of model performance with uncertainty quantification.

------------------------------------------------------------------------

## Motivation

The F1 score is commonly used to assess classification models,
especially under class imbalance. However, conventional methods provide
only a **single point estimate** and do not account for
**uncertainty**—a limitation in small sample sizes or streaming data.

This project introduces a **Bayesian alternative** that: - Provides
**posterior distributions** for the F1 score - Offers **95% credible
intervals** - Supports **streaming/online learning** via **sequential
updates** - Enhances interpretability in **real-time evaluation**

------------------------------------------------------------------------

## Features

- Bayesian estimation using **Dirichlet-Multinomial posteriors**
- **Sequential update mechanism** (no need for retraining)
- Simulation studies on varying **sample sizes** and **imbalance
  ratios**
- **Visualization** of posterior F1 mean and credible intervals

------------------------------------------------------------------------
