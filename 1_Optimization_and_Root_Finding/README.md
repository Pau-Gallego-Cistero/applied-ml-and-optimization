# 📈 Optimization & Numerical Root-Finding

Numerical algorithms implemented from scratch and applied to fundamental problems in quantum mechanics and high-energy physics:

---

### 1. Root-Finding: Quantum Confinement in Semiconductors
* **Physical System:** 1D finite potential well ($a = 3\text{ \AA}$, $V_0 = 3\text{ eV}$) modeling electron confinement and quantum tunneling in semiconductor structures.
* **Method:** Newton-Raphson root-finding applied to coupled transcendental equations for symmetric ($\xi \tan\xi = \eta$) and antisymmetric ($-\xi \cot\xi = \eta$) wavefunctions.
* **Outputs:** Determines discrete bound-state energy levels and calibrates the semiconductor's effective electron mass ($m^* = x \cdot m_e$).

---

### 2. First-Order Optimizers: Higgs Potential & Symmetry Breaking
* **Physical System:** Spontaneous Symmetry Breaking (SSB) in the 1D Standard Model ($V(\phi)$) and the 2D Two Higgs Doublet Model (THDM, $V(\Phi_1, \Phi_2)$).
* **Algorithms:** Comparative convergence and stability benchmark of:
  * **Gradient Descent (GD):** Evaluates learning-rate sensitivity ($\eta$) and asymptotic decay.
  * **GD with Momentum (GDM):** Analyzes inertial speedup and multi-dimensional oscillatory ringing.
  * **Nesterov Accelerated Gradient (NAG):** Implements look-ahead predictive damping for coupled potential surfaces.
* **Validation & Outputs:** Critical points confirmed as true minima via the **Hessian determinant test** ($\det(H) > 0$). Accurately recovers Vacuum Expectation Values (VEVs) to compute physical boson masses ($m_h \approx 125\text{ GeV}$, $m_{H^0}$, $m_{h^0}$, $m_{H^\pm}$).
