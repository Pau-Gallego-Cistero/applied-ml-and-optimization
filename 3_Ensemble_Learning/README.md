# 🌲 Ensemble Learning & Applied Physics

Evaluating Bagging, Boosting, and Voting architectures across medical diagnostics and statistical mechanics.

---

### 1. Clinical Diagnosis: Breast Cancer Classification
* **Models:** Random Forest, AdaBoost, Gradient Boosting, and Voting Classifiers (Hard vs. Soft).
* **Key Finding:** **Soft Voting** (Logistic Regression + Random Forest) proved the most stable method (~98% accuracy), combining linear stability with non-linear flexibility to avoid overfitting small sample splits.

### 2. Statistical Mechanics: 2D Ising Model Phase Transition
* **Objective:** Detecting the critical phase transition ($T_c$) between ordered ferromagnetic and disordered paramagnetic states directly from 2D spin configurations.
* **Method:** Supervised classification using tree-based ensembles trained on Monte Carlo lattice states.

---

## 🤖 AI Integration
AI tools were used as a technical assistant to:
* Scaffold and debug `scikit-learn` preprocessing pipelines (preventing data leakage with `StandardScaler`).
* Enhance comparative Matplotlib charts (automating percentage labels on model accuracy bars).
* Refine academic phrasing and structure across the technical PDF reports.

---

## 🚀 How to Run

```bash
python breast_cancer_ensemble_classification.py
python ising_model_ensemble_classification.py
