# Detailed Methodology - Ensemble Learning for Seoul Bike Sharing Demand Prediction

## 4. Methodology

### 4.1 Dataset Description and Preprocessing

#### 4.1.1 Dataset Characteristics
This study utilizes the Seoul Bike Sharing System dataset, containing hourly rental records from December 1, 2017, to November 30, 2018. The original dataset comprises 8,760 records with 14 feature variables.

**Original Feature Variables:**
- Temporal features: Date, Hour, Seasons, Holiday, Functioning Day
- Meteorological features: Temperature (°C), Humidity (%), Wind speed (m/s), Visibility (10m), Dew point temperature (°C), Solar Radiation (MJ/m²), Rainfall (mm), Snowfall (cm)
- Target variable: Rented Bike Count

#### 4.1.2 Data Cleaning and Outlier Treatment

**Non-operating Day Exclusion:**
Identification and exclusion of system non-operating days (abnormal dates with zero rentals):
```
Non-operating condition: Rented_Bike_Count = 0 AND Functioning_Day = "Yes"
Excluded records: 295 entries
Valid records: 8,465 entries
```

**Outlier Detection and Treatment:**
Winsorization method based on Interquartile Range (IQR) approach:

```math
IQR = Q_3 - Q_1
Lower\ bound = Q_1 - 1.5 \times IQR
Upper\ bound = Q_3 + 1.5 \times IQR
```

Where Q₁ and Q₃ represent the first and third quartiles, respectively. Values exceeding boundaries are replaced with boundary values:

```math
x_{winsorized} = \begin{cases}
Lower\ bound, & \text{if } x < Lower\ bound \\
Upper\ bound, & \text{if } x > Upper\ bound \\
x, & \text{otherwise}
\end{cases}
```

A total of 152 outliers were treated, primarily concentrated in rental counts under extreme weather conditions.

#### 4.1.3 Feature Engineering

Based on domain knowledge and time series characteristics, 90 enhanced features were constructed:

**1. Temporal Features (24 features):**
- Cyclical encoding:
```math
Hour_{sin} = \sin\left(\frac{2\pi \times Hour}{24}\right)
Hour_{cos} = \cos\left(\frac{2\pi \times Hour}{24}\right)
```
- Similar cyclical encoding for month, weekday, and season
- Binary features: weekday, weekend, holiday, peak hours

**2. Enhanced Meteorological Features (28 features):**
- Temperature-related:
```math
Temperature_{normalized} = \frac{Temperature - T_{min}}{T_{max} - T_{min}}
Temperature_{squared} = Temperature^2
```
- Humidity-temperature interaction:
```math
Humidity\_Temperature = Humidity \times Temperature
```
- Wind chill index:
```math
WindChill = 13.12 + 0.6215 \times T - 11.37 \times V^{0.16} + 0.3965 \times T \times V^{0.16}
```
where T is temperature (°C) and V is wind speed (km/h)

**3. Lag Features (24 features):**
Considering time series autocorrelation, lag features for 1-24 hours were constructed:
```math
X_{lag-k} = X_{t-k}, \quad k \in \{1, 2, 3, 6, 12, 24\}
```

**4. Rolling Window Statistical Features (14 features):**
```math
RollingMean_{w} = \frac{1}{w}\sum_{i=0}^{w-1} X_{t-i}
RollingStd_{w} = \sqrt{\frac{1}{w}\sum_{i=0}^{w-1} (X_{t-i} - RollingMean_{w})^2}
```
where w ∈ {3, 6, 12, 24} hours

### 4.2 Base Learning Algorithms

This study selected four different types of machine learning algorithms as base learners to ensure algorithmic diversity:

#### 4.2.1 Linear Regression Methods

**1. Ordinary Least Squares (OLS):**
```math
\hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2
```

Analytical solution:
```math
\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
```

**2. Ridge Regression:**
```math
\hat{\boldsymbol{\beta}}_{ridge} = \arg\min_{\boldsymbol{\beta}} \left\{ \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda \|\boldsymbol{\beta}\|_2^2 \right\}
```

Analytical solution:
```math
\hat{\boldsymbol{\beta}}_{ridge} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}
```

Parameter setting: λ = 0.01 (optimized for 90-dimensional feature space)

**3. Lasso Regression:**
```math
\hat{\boldsymbol{\beta}}_{lasso} = \arg\min_{\boldsymbol{\beta}} \left\{ \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \alpha \|\boldsymbol{\beta}\|_1 \right\}
```

Parameter setting: α = 0.001 (smaller regularization strength to retain more features)

#### 4.2.2 K-Nearest Neighbors Regression

**Algorithm Principle:**
```math
\hat{f}(x) = \frac{1}{k} \sum_{x_i \in N_k(x)} y_i
```

**Distance-weighted Version:**
```math
\hat{f}(x) = \frac{\sum_{x_i \in N_k(x)} w_i y_i}{\sum_{x_i \in N_k(x)} w_i}
```

where weights are defined as:
```math
w_i = \frac{1}{d(x, x_i) + \epsilon}
```

**Parameter Settings:**
- Number of neighbors: k = 20 (based on √n heuristic, n=5,925 training samples)
- Distance metric: Euclidean distance
- Weighting scheme: Distance weighting
- Algorithm: auto (automatic selection of optimal search algorithm)

#### 4.2.3 Neural Network Regression

**Multi-layer Perceptron Architecture:**

**Basic Network:**
```math
\begin{align}
\mathbf{h}_1 &= \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) \quad \text{(128 neurons)} \\
\mathbf{h}_2 &= \sigma(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2) \quad \text{(64 neurons)} \\
\hat{y} &= \mathbf{W}_3 \mathbf{h}_2 + b_3
\end{align}
```

**Deep Network:**
```math
\begin{align}
\mathbf{h}_1 &= \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) \quad \text{(180 neurons)} \\
\mathbf{h}_2 &= \sigma(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2) \quad \text{(90 neurons)} \\
\mathbf{h}_3 &= \sigma(\mathbf{W}_3 \mathbf{h}_2 + \mathbf{b}_3) \quad \text{(45 neurons)} \\
\mathbf{h}_4 &= \sigma(\mathbf{W}_4 \mathbf{h}_3 + \mathbf{b}_4) \quad \text{(22 neurons)} \\
\hat{y} &= \mathbf{W}_5 \mathbf{h}_4 + b_5
\end{align}
```

where σ is the ReLU activation function:
```math
\sigma(z) = \max(0, z)
```

**Optimization Algorithm:**
Adam optimizer parameter update rules:
```math
\begin{align}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1-\beta_2^t} \\
\theta_t &= \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
\end{align}
```

**Parameter Settings:**
- Learning rate: α = 0.001 (basic) / 0.0005 (deep)
- L2 regularization: λ = 0.001 (basic) / 0.0001 (deep)
- Early stopping: Stop when validation loss shows no improvement for 50-60 epochs

#### 4.2.4 Support Vector Regression

**SVR Optimization Problem:**
```math
\min_{\mathbf{w}, b, \boldsymbol{\xi}, \boldsymbol{\xi}^*} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^n (\xi_i + \xi_i^*)
```

Subject to constraints:
```math
\begin{align}
y_i - \mathbf{w}^T \phi(\mathbf{x}_i) - b &\leq \epsilon + \xi_i \\
\mathbf{w}^T \phi(\mathbf{x}_i) + b - y_i &\leq \epsilon + \xi_i^* \\
\xi_i, \xi_i^* &\geq 0
\end{align}
```

**Prediction Function:**
```math
f(\mathbf{x}) = \sum_{i=1}^n (\alpha_i - \alpha_i^*) K(\mathbf{x}_i, \mathbf{x}) + b
```

**Kernel Function Types:**

1. **Linear Kernel:**
```math
K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j
```

2. **Radial Basis Function (RBF) Kernel:**
```math
K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)
```
where γ = 1/(n_features × X.var()) = scale

3. **Polynomial Kernel:**
```math
K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \mathbf{x}_i^T \mathbf{x}_j + r)^d
```

**Parameter Settings:**
- Penalty parameter: C = 1.0 (linear) / 10.0 (RBF) / 1.0 (polynomial)
- Tolerance for error: ε = 0.01
- Polynomial degree: d = 3

### 4.3 Ensemble Learning Methods

#### 4.3.1 Bagging Method

**Algorithm Principle:**
Bagging generates multiple training subsets through Bootstrap sampling, trains multiple base learners, and averages their predictions:

```math
\hat{f}_{bag}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} \hat{f}_b(\mathbf{x})
```

where $\hat{f}_b(\mathbf{x})$ is the model trained on the b-th Bootstrap sample.

**Bootstrap Sampling:**
Sample n instances with replacement from the original training set D = {(x₁,y₁), ..., (xₙ,yₙ)}:
```math
D_b = \{(x_{i_1}, y_{i_1}), ..., (x_{i_n}, y_{i_n})\}
```
where $i_j \sim \text{Uniform}(1, n)$

**Feature Sampling:**
For the 90-dimensional feature space, feature subspace sampling is employed:
```math
\text{Feature subset size} = \lfloor 0.7 \times 90 \rfloor = 63
```

**Parameter Configuration:**
- Bootstrap sample ratio: 80%
- Feature sampling ratio: 70% (linear models) / 60% (KNN) / 50% (SVR)
- Number of base learners: 100 (linear) / 50 (KNN) / 30 (SVR) / 20 (neural networks)

#### 4.3.2 Voting Method

**Weighted Voting:**
```math
\hat{f}_{vote}(\mathbf{x}) = \sum_{i=1}^{M} w_i \hat{f}_i(\mathbf{x})
```

Subject to constraints: $\sum_{i=1}^{M} w_i = 1$, $w_i \geq 0$

**Weight Optimization:**
Optimal weights determined based on validation set performance:
```math
\mathbf{w}^* = \arg\min_{\mathbf{w}} \sum_{j=1}^{n_{val}} \left(y_j - \sum_{i=1}^{M} w_i \hat{f}_i(\mathbf{x}_j)\right)^2
```

**Weight Configurations:**
1. Linear model combination: [1.0, 1.2] (ordinary linear, ridge regression)
2. Four-algorithm combination: [1.0, 0.8, 1.3, 1.1] (linear, KNN, neural network, SVR)
3. Linear + neural network: [0.8, 1.2] (ridge regression, neural network)
4. KNN + SVR: [1.0, 1.0] (equal weights)

#### 4.3.3 Stacking Method

**Two-layer Architecture:**

**First Layer (Base Learners):**
Train M different base learners:
```math
\hat{f}_1, \hat{f}_2, ..., \hat{f}_M
```

**Cross-validation Predictions:**
Use k-fold cross-validation to obtain base learner predictions, avoiding overfitting:
```math
\hat{\mathbf{Z}} = [\hat{z}_1, \hat{z}_2, ..., \hat{z}_M]
```
where $\hat{z}_i$ is the cross-validation prediction of the i-th base learner.

**Second Layer (Meta-learner):**
```math
\hat{f}_{meta}(\mathbf{z}) = g(\mathbf{z})
```

**Final Prediction:**
```math
\hat{f}_{stack}(\mathbf{x}) = \hat{f}_{meta}([\hat{f}_1(\mathbf{x}), ..., \hat{f}_M(\mathbf{x})])
```

**Architecture Configuration:**
- First layer: 9 base learners (3 linear + 2 KNN + 2 SVR + 2 neural networks)
- Second layer: 6 meta-learners (linear regression, ridge regression, Lasso, KNN, SVR, neural network)

### 4.4 Experimental Design

#### 4.4.1 Data Splitting Strategy

**Time Series Split:**
Considering the time series nature of the data, temporal order splitting is adopted:

```
Training set: December 1, 2017 - August 31, 2018 (70%, 5,925 samples)
Validation set: September 1, 2018 - October 15, 2018 (15%, 1,270 samples)
Test set: October 16, 2018 - November 30, 2018 (15%, 1,270 samples)
```

**Feature Standardization:**
Z-score standardization is used:
```math
x_{scaled} = \frac{x - \mu_{train}}{\sigma_{train}}
```
where μ_train and σ_train are the mean and standard deviation of the training set, respectively.

#### 4.4.2 Evaluation Metrics

**1. Root Mean Square Error (RMSE):**
```math
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
```

**2. Coefficient of Determination (R²):**
```math
R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
```

**3. Mean Absolute Error (MAE):**
```math
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
```

**4. Mean Absolute Percentage Error (MAPE):**
```math
MAPE = \frac{100\%}{n} \sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right|
```

**5. Symmetric Mean Absolute Percentage Error (SMAPE):**
```math
SMAPE = \frac{100\%}{n} \sum_{i=1}^{n} \frac{2|y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i|}
```

#### 4.4.3 Statistical Significance Testing

**Friedman Test:**
Test whether there are significant differences in performance among multiple algorithms:
```math
\chi_F^2 = \frac{12n}{k(k+1)} \left[ \sum_{j=1}^{k} R_j^2 - \frac{k(k+1)^2}{4} \right]
```
where n is the number of datasets, k is the number of algorithms, and R_j is the average rank of the j-th algorithm.

**Wilcoxon Signed-rank Test:**
Pairwise comparison of two algorithms:
```math
W = \sum_{i=1}^{n} \text{sgn}(d_i) \cdot R_i
```
where d_i = x_i - y_i is the difference, and R_i is the rank of |d_i|.

#### 4.4.4 Computational Complexity Analysis

**Time Complexity:**
- Linear regression: O(p³ + np²) where p=90 is the number of features
- KNN: O(nd) where d is the feature dimension
- Neural networks: O(W × E) where W is the number of weights and E is the number of epochs
- SVR: O(n² × p) to O(n³ × p)

**Space Complexity:**
- Base models: O(p) to O(np)
- Bagging: O(B × model complexity) where B is the number of base learners
- Stacking: O(M × model complexity) where M is the number of base learners

### 4.5 Hyperparameter Optimization Strategy

**Grid Search Ranges:**
- Ridge regression α: [0.001, 0.01, 0.1, 1.0]
- KNN neighbors: [10, 15, 20, 25, 30]
- SVR penalty parameter C: [0.1, 1.0, 10.0, 100.0]
- Neural network learning rate: [0.0001, 0.0005, 0.001, 0.005]

**Cross-validation:**
5-fold time series cross-validation is used for hyperparameter selection, ensuring temporal order is not violated.

### 4.6 Experimental Environment and Implementation

**Hardware Environment:**
- CPU: Intel Core i7-9750H
- Memory: 16GB DDR4
- Storage: 512GB SSD

**Software Environment:**
- Python 3.8+
- scikit-learn 1.0+
- NumPy 1.21+
- Pandas 1.3+
- Matplotlib 3.5+

**Parallelization Strategy:**
- Base model training: Single-threaded (avoiding nested parallelism)
- Bagging: n_jobs=-1 (using all CPU cores)
- Grid search: Parallelized hyperparameter combination evaluation

This methodology ensures the scientific rigor, reproducibility, and reliability of the experiment, providing a solid theoretical and technical foundation for subsequent result analysis and discussion. 