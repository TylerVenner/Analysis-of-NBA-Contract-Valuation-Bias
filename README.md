# 🏀 Mapping the Latent Structure of Economic Bias in the NBA

**A DML-Unfolding Fusion Framework** *STA 160 Capstone Project | University of California, Davis*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](INSERT_YOUR_STREAMLIT_LINK_HERE)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![JAX](https://img.shields.io/badge/JAX-Accelerated-9cf)

---

## 🚀 Executive Summary

Traditional sports analytics often ends with a "wall of coefficients"—static tables that tell you *if* a factor matters, but not how it shapes the market. Furthermore, standard regression fails in the NBA because nearly 50% of salaries are **deterministic** (fixed by Rookie Scales or Max Contract rules) rather than negotiated in a free market.

**Our Solution:** We introduce a **Stratified Double Machine Learning (DML)** framework combined with **Latent Space Mapping**.
1.  **Isolate:** We separate "Free Market" players from "Fixed Contract" players.
2.  **Learn:** We learn the true price of bias (Age, Market Size, Hype) *only* from negotiated contracts.
3.  **Map:** We project these economic forces into a 3D topology, revealing the hidden "neighborhoods" of player valuation.

👉 **[Explore the Interactive Map](INSERT_YOUR_STREAMLIT_LINK_HERE)**

---

## 📊 Methodology

Our pipeline moves beyond simple regression to disentangle "Rule-based Salary" from "Market-based Salary."

### 1. Stratification (The Fix)
We classify every player as either **Statistical** (Free Market) or **Deterministic** (Rookie/Max). We train our models *only* on the Statistical group to prevent the Collective Bargaining Agreement (CBA) from confusing our estimates of market value.

### 2. Double Machine Learning (The Signal)
We use the **Frisch-Waugh-Lovell** theorem implemented via Gradient Boosting. This mathematically "subtracts" a player's on-court performance statistics from their salary, leaving only the "Unexplained Residual"—the pure economic bias.

### 3. Latent Unfolding (The Visualization)
We treat the learned bias impacts as coordinates in a high-dimensional space. Using a probabilistic JAX-optimized multidimensional scaling algorithm, we "unfold" this space into a 3D map.
* **Result:** Players cluster not by how *good* they are, but by *what structural forces* determine their pay.

---

## 📂 Repository Structure

This repo contains the analysis backend and the Streamlit frontend.

```text
├── 📂 data/
│   ├── raw/                # Original data from NBA API, Spotrac, etc.
│   ├── processed/          # Cleaned datasets
│   └── app_data/           # STATIC ARTIFACTS for the website (Map HTML, CSVs)
│
├── 📂 src/                 # Core Analysis Code
│   ├── analysis/           # DML Pipeline (Gradient Boosting + OLS)
│   ├── core/               # JAX Optimization Engine for 3D Mapping
│   └── scripts/            # Orchestration (run_bias_mapping.py)
│
├── 📂 pages/               # Streamlit Page Logic
│   ├── 1_Statistical_vs_Deterministic.py
│   ├── 2_DML_Pipeline.py
│   ├── 3_DML_Results.py
│   ├── 4_Interactive_Unfolding_Map.py
│   └── 5_Assumptions_and_Conclusion.py
│
├── Welcome.py              # Landing Page
└── requirements.txt        # Python Dependencies