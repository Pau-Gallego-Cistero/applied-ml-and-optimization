# 🪐 Exoplanetary Data Analysis & Pipeline Cleaning

Applied data science project analyzing astrophysical data from the **Open Exoplanet Catalogue (OEC)**.

## Key Tasks
* **Data Robustness:** Designed data-cleaning pipelines capable of detecting and handling corrupted/missing values (`oec_corrupted.csv`).
* **Feature Selection & Correlation:** Investigated planetary mass, orbital period, and stellar properties to find governing correlations.
* **Visualization:** Multi-variable distribution plots and feature importance rankings.

| Feature / Variable | Associated Detection Method |
| :--- | :--- |
| **Short Orbital Period** | Transit Method |
| **High Planetary Mass** | Radial Velocity / Astrometry |
| **Proximity to Earth (Small Distance)** | Direct Imaging / Gravitational Microlensing |
| **Cool Host Star (Low Temperature)** | Direct Imaging (Enhanced Contrast) |
| **Massive Host Star** | Radial Velocity (Harder to induce reflex motion) |
