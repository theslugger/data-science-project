# 详细方法论 - 基于集成学习的首尔自行车需求预测研究

## 4. 方法论 (Methodology)

### 4.1 数据集描述与预处理 (Dataset Description and Preprocessing)

#### 4.1.1 数据集特征
本研究使用首尔自行车共享系统数据集，包含2017年12月1日至2018年11月30日的每小时租赁记录。原始数据集包含8,760条记录和14个特征变量。

**原始特征变量：**
- 时间特征：日期(Date)、小时(Hour)、季节(Seasons)、假期(Holiday)、工作日(Functioning Day)
- 气象特征：温度(Temperature, °C)、湿度(Humidity, %)、风速(Wind speed, m/s)、能见度(Visibility, 10m)、露点温度(Dew point temperature, °C)、太阳辐射(Solar Radiation, MJ/m²)、降雨量(Rainfall, mm)、降雪量(Snowfall, cm)
- 目标变量：租赁自行车数量(Rented Bike Count)

#### 4.1.2 数据清洗与异常值处理

**非运营日排除：**
识别并排除系统非运营日（租赁数量为0的异常日期）：
```
非运营日条件: Rented_Bike_Count = 0 AND Functioning_Day = "Yes"
排除记录数: 295条
有效记录数: 8,465条
```

**异常值检测与处理：**
采用Winsorization方法处理异常值，基于四分位距(IQR)方法：

IQR = Q₃ - Q₁
下界 = Q₁ - 1.5 × IQR  
上界 = Q₃ + 1.5 × IQR

其中Q₁和Q₃分别为第一和第三四分位数。对于超出边界的值，采用边界值替换：

当 x < 下界时，x_winsorized = 下界
当 x > 上界时，x_winsorized = 上界  
其他情况，x_winsorized = x

共处理152个异常值，主要集中在极端天气条件下的租赁数量。

#### 4.1.3 特征工程 (Feature Engineering)

基于领域知识和时间序列特性，构建90个增强特征：

**1. 时间特征 (Temporal Features, 24个)：**
- 周期性编码：
  Hour_sin = sin(2π × Hour / 24)
  Hour_cos = cos(2π × Hour / 24)
- 类似地对月份、星期、季节进行周期性编码
- 二进制特征：是否为工作日、周末、假期、高峰时段

**2. 气象特征增强 (Meteorological Features, 28个)：**
- 温度相关：
  Temperature_normalized = (Temperature - T_min) / (T_max - T_min)
  Temperature_squared = Temperature²
- 湿度-温度交互项：
  Humidity_Temperature = Humidity × Temperature
- 风寒指数：
  WindChill = 13.12 + 0.6215 × T - 11.37 × V^0.16 + 0.3965 × T × V^0.16
其中T为温度(°C)，V为风速(km/h)

**3. 滞后特征 (Lag Features, 24个)：**
考虑时间序列的自相关性，构建1-24小时的滞后特征：
```math
X_{lag-k} = X_{t-k}, \quad k \in \{1, 2, 3, 6, 12, 24\}
```

**4. 滑动窗口统计特征 (Rolling Statistics, 14个)：**
```math
RollingMean_{w} = \frac{1}{w}\sum_{i=0}^{w-1} X_{t-i}
RollingStd_{w} = \sqrt{\frac{1}{w}\sum_{i=0}^{w-1} (X_{t-i} - RollingMean_{w})^2}
```
其中w ∈ {3, 6, 12, 24}小时

### 4.2 基础学习算法 (Base Learning Algorithms)

本研究选择四种不同类型的机器学习算法作为基础学习器，以确保算法多样性：

#### 4.2.1 线性回归方法

**1. 普通最小二乘回归 (Ordinary Least Squares, OLS)：**
```math
\hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2
```

解析解：
```math
\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
```

**2. 岭回归 (Ridge Regression)：**
```math
\hat{\boldsymbol{\beta}}_{ridge} = \arg\min_{\boldsymbol{\beta}} \left\{ \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda \|\boldsymbol{\beta}\|_2^2 \right\}
```

解析解：
```math
\hat{\boldsymbol{\beta}}_{ridge} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}
```

参数设置：λ = 0.01（针对90维特征空间优化）

**3. Lasso回归 (Lasso Regression)：**
```math
\hat{\boldsymbol{\beta}}_{lasso} = \arg\min_{\boldsymbol{\beta}} \left\{ \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \alpha \|\boldsymbol{\beta}\|_1 \right\}
```

参数设置：α = 0.001（保留更多特征的较小正则化强度）

#### 4.2.2 K近邻回归 (K-Nearest Neighbors Regression)

**算法原理：**
```math
\hat{f}(x) = \frac{1}{k} \sum_{x_i \in N_k(x)} y_i
```

**距离加权版本：**
```math
\hat{f}(x) = \frac{\sum_{x_i \in N_k(x)} w_i y_i}{\sum_{x_i \in N_k(x)} w_i}
```

其中权重定义为：
```math
w_i = \frac{1}{d(x, x_i) + \epsilon}
```

**参数设置：**
- 邻居数量：k = 20（基于√n经验法则，n=5,925训练样本）
- 距离度量：欧几里得距离
- 权重方式：距离加权
- 算法：auto（自动选择最优搜索算法）

#### 4.2.3 神经网络回归 (Neural Network Regression)

**多层感知机架构：**

**基础网络：**
```math
\begin{align}
\mathbf{h}_1 &= \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) \quad \text{(128 neurons)} \\
\mathbf{h}_2 &= \sigma(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2) \quad \text{(64 neurons)} \\
\hat{y} &= \mathbf{W}_3 \mathbf{h}_2 + b_3
\end{align}
```

**深度网络：**
```math
\begin{align}
\mathbf{h}_1 &= \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) \quad \text{(180 neurons)} \\
\mathbf{h}_2 &= \sigma(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2) \quad \text{(90 neurons)} \\
\mathbf{h}_3 &= \sigma(\mathbf{W}_3 \mathbf{h}_2 + \mathbf{b}_3) \quad \text{(45 neurons)} \\
\mathbf{h}_4 &= \sigma(\mathbf{W}_4 \mathbf{h}_3 + \mathbf{b}_4) \quad \text{(22 neurons)} \\
\hat{y} &= \mathbf{W}_5 \mathbf{h}_4 + b_5
\end{align}
```

其中σ为ReLU激活函数：
```math
\sigma(z) = \max(0, z)
```

**优化算法：**
Adam优化器的参数更新规则：
```math
\begin{align}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1-\beta_2^t} \\
\theta_t &= \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
\end{align}
```

**参数设置：**
- 学习率：α = 0.001（基础）/ 0.0005（深度）
- L2正则化：λ = 0.001（基础）/ 0.0001（深度）
- 早停策略：验证损失50-60轮无改善时停止

#### 4.2.4 支持向量回归 (Support Vector Regression)

**SVR优化问题：**
```math
\min_{\mathbf{w}, b, \boldsymbol{\xi}, \boldsymbol{\xi}^*} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^n (\xi_i + \xi_i^*)
```

约束条件：
```math
\begin{align}
y_i - \mathbf{w}^T \phi(\mathbf{x}_i) - b &\leq \epsilon + \xi_i \\
\mathbf{w}^T \phi(\mathbf{x}_i) + b - y_i &\leq \epsilon + \xi_i^* \\
\xi_i, \xi_i^* &\geq 0
\end{align}
```

**预测函数：**
```math
f(\mathbf{x}) = \sum_{i=1}^n (\alpha_i - \alpha_i^*) K(\mathbf{x}_i, \mathbf{x}) + b
```

**核函数类型：**

1. **线性核：**
```math
K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j
```

2. **径向基函数核 (RBF)：**
```math
K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)
```
其中γ = 1/(n_features × X.var()) = scale

3. **多项式核：**
```math
K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \mathbf{x}_i^T \mathbf{x}_j + r)^d
```

**参数设置：**
- 惩罚参数：C = 1.0（线性）/ 10.0（RBF）/ 1.0（多项式）
- 容忍误差：ε = 0.01
- 多项式次数：d = 3

### 4.3 集成学习方法 (Ensemble Learning Methods)

#### 4.3.1 Bagging方法

**算法原理：**
Bagging通过Bootstrap采样生成多个训练子集，训练多个基学习器并平均其预测：

```math
\hat{f}_{bag}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} \hat{f}_b(\mathbf{x})
```

其中$\hat{f}_b(\mathbf{x})$是第b个Bootstrap样本训练的模型。

**Bootstrap采样：**
从原始训练集D = {(x₁,y₁), ..., (xₙ,yₙ)}中有放回地采样n个样本：
```math
D_b = \{(x_{i_1}, y_{i_1}), ..., (x_{i_n}, y_{i_n})\}
```
其中$i_j \sim \text{Uniform}(1, n)$

**特征采样：**
对于90维特征空间，采用特征子空间采样：
```math
\text{特征子集大小} = \lfloor 0.7 \times 90 \rfloor = 63
```

**参数配置：**
- Bootstrap样本比例：80%
- 特征采样比例：70%（线性模型）/ 60%（KNN）/ 50%（SVR）
- 基学习器数量：100（线性）/ 50（KNN）/ 30（SVR）/ 20（神经网络）

#### 4.3.2 Voting方法

**加权投票：**
```math
\hat{f}_{vote}(\mathbf{x}) = \sum_{i=1}^{M} w_i \hat{f}_i(\mathbf{x})
```

约束条件：$\sum_{i=1}^{M} w_i = 1$, $w_i \geq 0$

**权重优化：**
基于验证集性能确定最优权重：
```math
\mathbf{w}^* = \arg\min_{\mathbf{w}} \sum_{j=1}^{n_{val}} \left(y_j - \sum_{i=1}^{M} w_i \hat{f}_i(\mathbf{x}_j)\right)^2
```

**权重配置：**
1. 线性模型组合：[1.0, 1.2] (普通线性, 岭回归)
2. 四算法组合：[1.0, 0.8, 1.3, 1.1] (线性, KNN, 神经网络, SVR)
3. 线性+神经网络：[0.8, 1.2] (岭回归, 神经网络)
4. KNN+SVR：[1.0, 1.0] (等权重)

#### 4.3.3 Stacking方法

**两层架构：**

**第一层（基学习器）：**
训练M个不同的基学习器：
```math
\hat{f}_1, \hat{f}_2, ..., \hat{f}_M
```

**交叉验证预测：**
使用k折交叉验证获得基学习器的预测，避免过拟合：
```math
\hat{\mathbf{Z}} = [\hat{z}_1, \hat{z}_2, ..., \hat{z}_M]
```
其中$\hat{z}_i$是第i个基学习器的交叉验证预测。

**第二层（元学习器）：**
```math
\hat{f}_{meta}(\mathbf{z}) = g(\mathbf{z})
```

**最终预测：**
```math
\hat{f}_{stack}(\mathbf{x}) = \hat{f}_{meta}([\hat{f}_1(\mathbf{x}), ..., \hat{f}_M(\mathbf{x})])
```

**架构配置：**
- 第一层：9个基学习器（3个线性 + 2个KNN + 2个SVR + 2个神经网络）
- 第二层：6种元学习器（线性回归、岭回归、Lasso、KNN、SVR、神经网络）

### 4.4 实验设计 (Experimental Design)

#### 4.4.1 数据分割策略

**时间序列分割：**
考虑到数据的时间序列性质，采用时间顺序分割：

```
训练集：2017年12月1日 - 2018年8月31日 (70%, 5,925样本)
验证集：2018年9月1日 - 2018年10月15日 (15%, 1,270样本)
测试集：2018年10月16日 - 2018年11月30日 (15%, 1,270样本)
```

**特征标准化：**
使用Z-score标准化：
```math
x_{scaled} = \frac{x - \mu_{train}}{\sigma_{train}}
```
其中μ_train和σ_train分别为训练集的均值和标准差。

#### 4.4.2 评估指标 (Evaluation Metrics)

**1. 均方根误差 (RMSE)：**
```math
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
```

**2. 决定系数 (R²)：**
```math
R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
```

**3. 平均绝对误差 (MAE)：**
```math
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
```

**4. 平均绝对百分比误差 (MAPE)：**
```math
MAPE = \frac{100\%}{n} \sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right|
```

**5. 对称平均绝对百分比误差 (SMAPE)：**
```math
SMAPE = \frac{100\%}{n} \sum_{i=1}^{n} \frac{2|y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i|}
```

#### 4.4.3 统计显著性检验

**Friedman检验：**
检验多个算法性能是否存在显著差异：
```math
\chi_F^2 = \frac{12n}{k(k+1)} \left[ \sum_{j=1}^{k} R_j^2 - \frac{k(k+1)^2}{4} \right]
```
其中n为数据集数量，k为算法数量，R_j为第j个算法的平均排名。

**Wilcoxon符号秩检验：**
对两个算法进行成对比较：
```math
W = \sum_{i=1}^{n} \text{sgn}(d_i) \cdot R_i
```
其中d_i = x_i - y_i为差值，R_i为|d_i|的排名。

#### 4.4.4 计算复杂度分析

**时间复杂度：**
- 线性回归：O(p³ + np²) 其中p=90为特征数
- KNN：O(nd) 其中d为特征维度
- 神经网络：O(W × E) 其中W为权重数量，E为训练轮数
- SVR：O(n² × p) 到 O(n³ × p)

**空间复杂度：**
- 基础模型：O(p) 到 O(np)
- Bagging：O(B × 模型复杂度) 其中B为基学习器数量
- Stacking：O(M × 模型复杂度) 其中M为基学习器数量

### 4.5 超参数优化策略

**网格搜索范围：**
- 岭回归α：[0.001, 0.01, 0.1, 1.0]
- KNN邻居数：[10, 15, 20, 25, 30]
- SVR惩罚参数C：[0.1, 1.0, 10.0, 100.0]
- 神经网络学习率：[0.0001, 0.0005, 0.001, 0.005]

**交叉验证：**
使用5折时间序列交叉验证进行超参数选择，确保时间顺序不被破坏。

### 4.6 实验环境与实现

**硬件环境：**
- CPU：Intel Core i7-9750H
- 内存：16GB DDR4
- 存储：512GB SSD

**软件环境：**
- Python 3.8+
- scikit-learn 1.0+
- NumPy 1.21+
- Pandas 1.3+
- Matplotlib 3.5+

**并行化策略：**
- 基础模型训练：单线程（避免嵌套并行）
- Bagging：n_jobs=-1（使用所有CPU核心）
- 网格搜索：并行化超参数组合评估

本方法论确保了实验的科学性、可重现性和结果的可靠性，为后续的结果分析和讨论提供了坚实的理论和技术基础。 