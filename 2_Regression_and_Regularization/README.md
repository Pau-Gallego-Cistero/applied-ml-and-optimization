# 📉 Linear & Regularized Regression on Classical Mechanics

This module explores linear modeling, polynomial complexity, and penalty regularization applied to classical kinematic and mechanical physical laws.

---

## 🔬 Experiments & Projects

### 1. Hooke’s Law: Model Complexity & Overfitting Analysis
* **Physical System:** Ideal 1D spring force ($F = -kx$, $k = 5\text{ N/m}$) subjected to Gaussian noise $\eta \sim \mathcal{N}(0, \sigma^2)$.
* **Core Machine Learning Concepts:**
  * **Polynomial Regression:** Fitting degrees $d \in [1, 15]$ to evaluate model flexibility versus noise memorization.
  * **Train/Test Partitions & Shuffling:** Demonstrates that non-shuffled splits lead to catastrophic out-of-distribution extrapolation errors ($R^2$ dropping from $0.99$ to $0.08$).
  * **Bias-Variance Tradeoff:** Comparing low-variance/high-bias models ($d=1$) against high-variance/low-bias models ($d=15$).

---

### 2. Free-Fall Kinematics: OLS vs. Ridge vs. LASSO
* **Physical System:** 1D vertical displacement under uniform gravitational acceleration ($y = \frac{1}{2}gt^2$).
* **Implementation:** Built both **from scratch** using closed-form linear algebra and benchmarked against **`scikit-learn`**.

#### 📐 Mathematical Formulations Implemented:
* **Ordinary Least Squares (OLS):** Closed-form minimization:
  $$\hat{\mathbf{w}}_{\text{OLS}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$
* **Ridge Regression ($L_2$ Regularization):** Shrinkage penalty to control parameter magnitude:
  $$\hat{\mathbf{w}}_{\text{Ridge}} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$$
* **LASSO Regression ($L_1$ Regularization):** Feature selection and sparsity via convex optimization:
  $$\hat{\mathbf{w}}_{\text{LASSO}} = \arg\min_{\mathbf{w}} \left( \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2 + \lambda\|\mathbf{w}\|_1 \right)$$

#### 📊 Key Findings:
* Under severe stochastic noise ($\sigma \ge 100$), Ridge regression provided superior stability by dampening coefficient growth, whereas extreme $\lambda$ values in LASSO caused excessive shrinkage towards flat predictions.

---

## 🚀 How to Run

```bash
# Run polynomial regression on Hooke's Law
python hookes_law_polynomial_regression.py

# Run comparison between OLS, Ridge, and LASSO on Free Fall
python free_fall_ols_ridge_lasso.py
