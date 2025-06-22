# Application of Ensemble Learning Methods in Seoul Bike Sharing Demand Prediction

## Abstract

This chapter presents an in-depth study of ensemble learning methods applied to Seoul bike sharing demand prediction. Based on a dataset containing 90 enhanced features and 8,465 valid records, we systematically compared 24 different ensemble learning models, covering three major categories: Bagging, Voting, and Stacking methods. Experimental results demonstrate that the Voting Linear+NN ensemble model achieved the best performance with a test RMSE of 201.98 and R² of 0.8635, significantly outperforming individual base models. This research provides an effective ensemble learning solution for urban transportation demand forecasting.

## 1. Introduction

Ensemble learning, as an important branch of machine learning, improves overall performance by combining predictions from multiple base learners. In regression problems, ensemble methods can effectively reduce prediction variance and enhance model generalization ability and robustness. This chapter focuses on studying the performance of ensemble learning methods in the practical application scenario of Seoul bike sharing demand prediction.

Seoul bike sharing system demand prediction has the following characteristics: (1) data exhibits obvious temporal and periodic patterns; (2) influencing factors are complex, including weather, time, holidays, and other multiple dimensions; (3) demand changes show nonlinear characteristics. These characteristics make it difficult for single models to fully capture the complex patterns in the data, providing an ideal scenario for ensemble learning applications.

## 2. Theoretical Foundation of Ensemble Learning

### 2.1 Mathematical Framework of Ensemble Learning

The core idea of ensemble learning is to construct stronger prediction models by combining multiple base learners. For regression problems, ensemble prediction can be expressed as:

f̂_ensemble(x) = Σ(i=1 to M) w_i × f̂_i(x)

where f̂_i(x) is the prediction of the i-th base learner, w_i is the corresponding weight, M is the number of base learners, and Σ(i=1 to M) w_i = 1.

### 2.2 Bagging Method

Bootstrap Aggregating (Bagging) trains multiple base models through bootstrap sampling of training data, then averages their predictions:

f̂_bag(x) = (1/B) × Σ(b=1 to B) f̂_b(x)

where B is the number of bootstrap samples, and f̂_b(x) is the model prediction trained on the b-th bootstrap sample.

The variance reduction effect of Bagging can be understood through the following formula. Assuming the prediction errors of base learners are independent and identically distributed with variance σ², the prediction variance of Bagging is:

Var[f̂_bag(x)] = σ²/B

This indicates that as the number of base learners increases, the variance of the ensemble model decreases accordingly.

### 2.3 Voting Method

Voting methods combine different types of base learners through weighted averaging:

f̂_vote(x) = Σ(i=1 to M) w_i × f̂_i(x), where Σ(i=1 to M) w_i = 1

The weights w_i can be determined through the following approaches:

| Approach | Formula/Description |
|----------|-------------------|
| Equal weights | w_i = 1/M |
| Validation-based | w_i = (1/RMSE_i) / Σ(j=1 to M) (1/RMSE_j) |
| Cross-validation | Optimizing weight combinations through grid search |

*Table 1: Weight determination approaches for voting ensemble methods. The table shows three common strategies for assigning weights to base learners in voting ensembles.*

### 2.4 Stacking Method

Stacking adopts a two-layer learning structure, where the first layer contains multiple base learners, and the second layer uses a meta-learner to combine the outputs of base learners:

f̂_stack(x) = g(f̂_1(x), f̂_2(x), ..., f̂_M(x))

where g(·) is the meta-learner function. The training data for the meta-learner consists of predictions from base learners on the validation set:

D_meta = {(f̂_1(x_i), f̂_2(x_i), ..., f̂_M(x_i), y_i)}_{i=1}^{n}

## 3. Base Learning Algorithms

### 3.1 Linear Regression and Regularized Variants

Linear regression forms the foundation of many machine learning algorithms by modeling the relationship between input features and target variables through a linear combination. The basic linear regression model can be expressed as:

ŷ = β₀ + Σ(j=1 to p) βⱼ × xⱼ

To address overfitting issues in high-dimensional feature spaces, regularized variants of linear regression are commonly employed. Ridge regression incorporates L2 regularization by adding a penalty term proportional to the sum of squared coefficients:

β̂_ridge = argmin_β {Σ(i=1 to n) (yᵢ - xᵢᵀβ)² + λ × Σ(j=1 to p) βⱼ²}

Alternatively, Lasso regression uses L1 regularization, which not only prevents overfitting but also performs automatic feature selection by driving some coefficients to zero:

β̂_lasso = argmin_β {Σ(i=1 to n) (yᵢ - xᵢᵀβ)² + λ × Σ(j=1 to p) |βⱼ|}

In both regularized approaches, λ is the regularization parameter that controls the trade-off between model complexity and fitting accuracy.

### 3.2 K-Nearest Neighbors Regression

K-nearest neighbors regression makes predictions through local averaging:

f̂(x) = (1/k) × Σ(xᵢ ∈ Nₖ(x)) yᵢ

For the distance-weighted version:

f̂(x) = [Σ(xᵢ ∈ Nₖ(x)) wᵢ × yᵢ] / [Σ(xᵢ ∈ Nₖ(x)) wᵢ]

where wᵢ = 1/(d(x, xᵢ) + ε), and d(x, xᵢ) is the distance function.

### 3.3 Neural Networks

Forward propagation of multilayer perceptron can be expressed as:

h^(l+1) = σ(W^(l) × h^(l) + b^(l))

For a network with L layers, the final output is:

ŷ = W^(L) × h^(L) + b^(L)

where σ(·) is the activation function (such as ReLU), and W^(l) and b^(l) are the weight matrix and bias vector of the l-th layer, respectively.

### 3.4 Support Vector Regression

The optimization objective of SVR is:

min_(w,b,ξ,ξ*) {(1/2)||w||² + C × Σ(i=1 to n) (ξᵢ + ξᵢ*)}

Subject to constraints:
yᵢ - wᵀφ(xᵢ) - b ≤ ε + ξᵢ
wᵀφ(xᵢ) + b - yᵢ ≤ ε + ξᵢ*
ξᵢ, ξᵢ* ≥ 0

The final prediction function is:

f(x) = Σ(i=1 to n) (αᵢ - αᵢ*) × K(xᵢ, x) + b

where K(xᵢ, x) is the kernel function. Common kernel functions include:

| Kernel Type | Mathematical Expression |
|-------------|------------------------|
| Linear | K(xᵢ, xⱼ) = xᵢᵀxⱼ |
| RBF | K(xᵢ, xⱼ) = exp(-γ × ||xᵢ - xⱼ||²) |
| Polynomial | K(xᵢ, xⱼ) = (γ × xᵢᵀxⱼ + r)^d |

*Table 2: Common kernel functions used in Support Vector Regression. Each kernel function transforms the input space to capture different types of relationships between data points.*

## 4. Experimental Design

### 4.1 Dataset Description

This study uses the Seoul bike sharing dataset, which after comprehensive feature engineering contains multiple categories of predictive variables. The original meteorological features include temperature, humidity, wind speed, visibility, dew point temperature, solar radiation, rainfall, and snowfall measurements that directly influence cycling behavior. 

Temporal features capture the cyclical patterns inherent in bike sharing demand, incorporating hour of day, day of week, month, season, and weekend indicators to model the regular patterns in urban mobility. 

Enhanced features were engineered to capture complex relationships not explicitly present in the raw data, including comfort index calculations, extreme weather event indicators, and interaction terms between meteorological and temporal variables. 

Lag features provide historical context by incorporating demand values from 1 hour, 24 hours, and 168 hours (1 week) prior, enabling the model to learn from recent usage patterns and weekly cycles.

The final dataset contains 8,465 valid records with 90 features, split chronologically into training set (70%, 5,925 records), validation set (15%, 1,270 records), and test set (15%, 1,270 records).

### 4.2 Model Configuration

Based on the characteristics of the 90-dimensional feature space, we performed targeted optimization for each type of model:

**Base Model Configuration:**

| Model Type | Configuration Details |
|------------|---------------------|
| Linear Regression | Ridge regularization α=0.01, Lasso regularization α=0.001 |
| KNN Regression | k=20, distance weighting, optimized for 5,925 training samples |
| Neural Networks | Basic network (128,64), deep network (180,90,45,22), designed considering feature dimensions |
| SVR | Linear kernel C=1.0, RBF kernel C=10.0, polynomial kernel degree=3 |

*Table 3: Base model configuration parameters. All parameters were optimized for the 90-dimensional feature space and 5,925 training samples.*

**Ensemble Method Configuration:**

| Ensemble Method | Configuration Details |
|----------------|---------------------|
| Bagging | 100 estimators, 70% feature sampling, 80% sample sampling |
| Voting | Weighting strategy based on validation set performance |
| Stacking | 9 base learners, 6 types of meta-learners |

*Table 4: Ensemble method configuration parameters. Each ensemble method was configured to maximize diversity while maintaining computational efficiency.*

### 4.3 Evaluation Metrics

Multiple regression metrics were employed to provide a comprehensive evaluation of model performance across different aspects of prediction accuracy. Root Mean Square Error (RMSE) serves as the primary metric, penalizing larger errors more heavily and providing a measure in the same units as the target variable:

RMSE = √[(1/n) × Σ(i=1 to n) (yᵢ - ŷᵢ)²]

The Coefficient of Determination (R²) quantifies the proportion of variance in the target variable explained by the model, offering an intuitive measure of model effectiveness:

R² = 1 - [Σ(i=1 to n) (yᵢ - ŷᵢ)²] / [Σ(i=1 to n) (yᵢ - ȳ)²]

Mean Absolute Error (MAE) provides a robust measure of average prediction error that is less sensitive to outliers compared to RMSE:

MAE = (1/n) × Σ(i=1 to n) |yᵢ - ŷᵢ|

Mean Absolute Percentage Error (MAPE) offers a scale-independent measure that expresses prediction accuracy as a percentage, facilitating comparison across different datasets and problem domains:

MAPE = (100%/n) × Σ(i=1 to n) |(yᵢ - ŷᵢ)/yᵢ|

## 5. Experimental Results and Analysis

### 5.1 Overall Performance Analysis

**[Figure 1 Position: Insert performance_comparison.png here]**
*Figure 1: Performance comparison analysis of ensemble learning methods. (a) Average test RMSE comparison by method type; (b) Average test R² comparison by method type; (c) RMSE ranking of top 10 models; (d) Trade-off analysis between training time and performance.*

Experimental results show that among the 24 tested models, the Voting Linear+NN model achieved the best performance:

| Performance Metric | Value |
|-------------------|-------|
| Test RMSE | 201.98 |
| Test R² | 0.8635 |
| Test MAE | 151.32 |
| Test MAPE | 53.21% |
| Training Time | 20.63 seconds |

*Table 5: Performance metrics of the best performing model (Voting Linear+NN). The model achieved the lowest test RMSE among all 24 tested ensemble configurations.*

**[Table 6 Position: Insert detailed_performance_table.png here]**
*Table 6: Detailed performance comparison of top 15 models. The table is sorted by test RMSE in ascending order, with the best model highlighted.*

### 5.2 Comparison of Ensemble Method Types

**[Figure 2 Position: Insert method_comparison_boxplot.png here]**
*Figure 2: Performance distribution boxplots of different ensemble method types. Shows statistical distribution characteristics of each method on four metrics: RMSE, R², MAE, and MAPE.*

Analysis results by ensemble method type are as follows:

| Method Type | Model Count | Average Test RMSE | Best Model | Best RMSE | Key Characteristics |
|-------------|-------------|-------------------|------------|-----------|-------------------|
| Voting | 4 | 291.62 ± 89.45 | Voting Linear+NN | 201.98 | Effectively combines advantages of different algorithms through reasonable weight allocation |
| Base Models | 9 | 331.12 ± 162.34 | Lasso Regression | 214.71 | Lasso regression performs excellently in high-dimensional feature space through feature selection |
| Stacking | 6 | 329.34 ± 41.28 | Stacking KNN | 258.12 | Meta-learner can learn complementary relationships between base models |
| Bagging | 5 | 420.96 ± 94.87 | Bagging KNN | 331.05 | Feature sampling effects are not as expected in high-dimensional feature space |

*Table 7: Comprehensive comparison of ensemble method types. Results show voting methods achieved the best overall performance, while bagging methods struggled in the high-dimensional feature space.*

### 5.3 Effectiveness Analysis of Ensemble Learning

**[Figure 3 Position: Insert ensemble_effectiveness_analysis.png here]**
*Figure 3: Effectiveness analysis of ensemble learning. (a) RMSE distribution comparison between base models vs ensemble models; (b) Average performance of different ensemble methods; (c) Performance improvement relative to best base model; (d) Trade-off between training time and performance improvement.*

The effectiveness of ensemble learning is demonstrated in the following aspects:

**Variance Reduction Effect:** The best ensemble model (Voting Linear+NN) achieved a 5.9% RMSE reduction compared to the best base model (Lasso), from 214.71 to 201.98.

**Complementarity Utilization:** The combination of linear models and neural networks can simultaneously capture linear and nonlinear patterns.

**Robustness Enhancement:** Ensemble methods show relatively small performance fluctuations with significantly lower standard deviations than base models.

### 5.4 Impact of Feature Space Dimensionality

In the 90-dimensional feature space, different algorithms demonstrated markedly different capabilities in handling high-dimensional data. Linear methods, particularly Ridge and Lasso regression, proved highly effective at managing high-dimensional features through their regularization mechanisms. Lasso regression's inherent feature selection capability enabled it to automatically identify and emphasize the most important predictive factors among the numerous available features, contributing to its superior performance among base models.

Non-parametric methods, exemplified by K-Nearest Neighbors, suffered significantly from the "curse of dimensionality" phenomenon. In high-dimensional spaces, the concept of proximity becomes less meaningful as all points tend to become equidistant, resulting in relatively poor performance with an RMSE of 349.19.

Neural networks demonstrated considerable potential through their deep network architecture (180-90-45-22 configuration). This carefully designed structure effectively captured complex interactions between features, with the layer sizes progressively reducing to match the feature dimensionality, significantly outperforming basic network configurations.

Support Vector Regression exhibited varied performance depending on kernel selection. The linear kernel maintained stable and competitive performance (RMSE: 230.41), while RBF and polynomial kernels showed signs of overfitting in the high-dimensional space, highlighting the importance of appropriate kernel selection for high-dimensional regression problems.

### 5.5 Performance Metric Correlation Analysis

**[Figure 4 Position: Insert correlation_heatmap.png here]**
*Figure 4: Performance metric correlation heatmap. Shows correlation relationships between RMSE, MAE, R², MAPE, SMAPE, and training time.*

Correlation analysis reveals:

| Metric Pair | Correlation Coefficient | Interpretation |
|-------------|------------------------|----------------|
| RMSE vs MAE | r = 0.95 | Highly positive correlation, indicating basic consistency in model ranking |
| R² vs RMSE | r = -0.89 | Strong negative correlation, consistent with theoretical expectations |
| Training Time vs Performance | r < 0.3 | Weak correlation, indicating complex models do not necessarily bring better performance |

*Table 8: Correlation analysis between performance metrics. The high correlation between RMSE and MAE validates the consistency of model rankings across different error measures.*

### 5.6 Model Ranking Analysis

**[Figure 5 Position: Insert model_ranking_analysis.png here]**
*Figure 5: Model performance ranking analysis. (a) RMSE ranking and detailed metrics of top 10 models; (b) Multi-dimensional performance scatter plot of top 5 models, with bubble size representing MAE performance.*

Ranking analysis reveals several important patterns in model performance across the ensemble learning landscape. Voting methods demonstrated a clear advantage in the overall rankings, with 2 out of the top 5 performing models utilizing voting strategies. This success underscores the effectiveness of weighted combination approaches in leveraging the complementary strengths of different algorithms.

Regularized linear models, particularly Lasso regression, achieved outstanding performance by ranking first among all base models. This exceptional performance highlights the critical importance of feature selection mechanisms in high-dimensional regression problems, where automatic identification of relevant features can significantly impact prediction accuracy.

Neural networks showed great potential for capturing complex nonlinear relationships, with deep neural network configurations ranking second among base models. This performance demonstrates the substantial value of nonlinear modeling approaches in capturing the intricate patterns present in urban bike sharing demand data.

The selection of appropriate kernel functions proved critical for Support Vector Regression performance. While the linear kernel maintained competitive performance levels, RBF and polynomial kernels suffered from severe overfitting on this particular dataset, emphasizing the importance of careful kernel selection in high-dimensional regression scenarios.

## 6. Discussion

### 6.1 Advantages and Limitations of Ensemble Learning

| Aspect | Advantages | Limitations |
|--------|------------|-------------|
| Performance | Best ensemble model shows significant improvement over single models | Need to balance complexity and effectiveness |
| Robustness | Ensemble methods are more robust to outliers and noise | May not always guarantee better performance |
| Complementarity | Advantages of different algorithms are effectively combined | Requires careful selection of base learners |
| Computational Cost | Can parallelize training of base learners | Training and prediction time significantly increase |
| Interpretability | Can provide insights through model diversity | Black-box nature makes result interpretation difficult |
| Parameter Tuning | Can leverage strengths of multiple algorithms | Need to simultaneously optimize parameters of multiple sub-models |

*Table 9: Advantages and limitations of ensemble learning methods. The table provides a balanced view of the trade-offs involved in ensemble learning applications.*

### 6.2 Challenges in High-Dimensional Feature Space

In the 90-dimensional feature space, traditional ensemble methods encountered several significant challenges that affected their overall effectiveness. The curse of dimensionality emerged as a fundamental obstacle, where distance measures become increasingly ineffective as dimensionality increases. This phenomenon particularly impacted distance-based methods like K-Nearest Neighbors, where the concept of "nearest" loses meaning when all points become approximately equidistant in high-dimensional space.

Overfitting risk increased substantially in the high-dimensional environment, where complex models with numerous parameters could easily memorize training data rather than learning generalizable patterns. This challenge was particularly evident in non-linear kernel methods and complex ensemble configurations that struggled to maintain generalization performance.

The importance of effective feature selection mechanisms became paramount in this high-dimensional setting. Models that incorporated automatic feature selection, such as Lasso regression, significantly outperformed those that attempted to utilize all available features without discrimination, highlighting the critical need for dimensionality reduction and feature relevance assessment in ensemble learning applications.

### 6.3 Practical Application Recommendations

Based on the comprehensive experimental results obtained from this study, several practical application recommendations emerge for practitioners working on similar time series regression problems. For model selection in urban transportation demand forecasting scenarios, the Voting Linear+NN combination represents the optimal choice, effectively balancing prediction accuracy with computational efficiency while leveraging the complementary strengths of linear and nonlinear modeling approaches.

Feature engineering strategies should prioritize feature selection and regularization techniques over simple feature accumulation. The exceptional performance of Lasso regression in this high-dimensional setting provides compelling evidence for the superiority of selective feature utilization over comprehensive feature inclusion, particularly in scenarios where interpretability and generalization are important considerations.

Computational resource allocation requires careful consideration of the trade-off between model complexity and practical implementation constraints. Under limited computational resources, simple weighted voting strategies may prove more practical and cost-effective than complex Stacking approaches, while still delivering substantial performance improvements over individual base models. This finding is particularly relevant for real-time prediction systems where computational efficiency is paramount.

## 7. Conclusion

This study systematically compared the performance of 24 ensemble learning models in Seoul bike sharing demand prediction through comprehensive experiments. Main findings include:

**Optimal Voting Method:** Voting Linear+NN model achieved the best performance with test RMSE of 201.98 and R² of 0.8635.

**Importance of Regularization:** In the 90-dimensional feature space, Lasso regression achieved the best performance among base models through feature selection.

**Effectiveness of Ensemble Learning:** Appropriate ensemble strategies can significantly improve prediction performance, but need to balance complexity and effectiveness.

**Algorithm Complementarity:** The combination of linear models and neural networks can simultaneously capture linear and nonlinear characteristics of data.

**High-Dimensional Challenges:** In high-dimensional feature space, feature selection and regularization are more important than simple model ensembling.

These findings provide valuable guidance for urban transportation demand prediction and related time series regression problems. Future research can explore more advanced ensemble strategies, such as dynamic weight adjustment and online learning methods, to further improve prediction performance.

## References

[1] Breiman, L. (1996). Bagging predictors. Machine learning, 24(2), 123-140.

[2] Wolpert, D. H. (1992). Stacked generalization. Neural networks, 5(2), 241-259.

[3] Kuncheva, L. I. (2004). Combining pattern classifiers: methods and algorithms. John Wiley & Sons.

[4] Zhou, Z. H. (2012). Ensemble methods: foundations and algorithms. CRC press.

[5] Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning: data mining, inference, and prediction. Springer Science & Business Media. 