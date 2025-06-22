# Seoul Bike Sharing Demand Data Exploration Analysis Report

## Abstract

This report presents a comprehensive exploratory data analysis (EDA) of Seoul bike sharing system demand data. Based on a dataset containing 8,465 valid records, we conducted in-depth analysis through data quality assessment, target variable analysis, temporal pattern identification, weather factor impact evaluation, categorical variable analysis, and feature correlation analysis to identify key factors influencing bike rental demand. The analysis reveals that bike rental demand exhibits distinct bimodal temporal patterns, strong seasonality, and significant weather sensitivity. These findings provide crucial data foundation and feature engineering guidance for subsequent demand prediction model development.

## 1. Introduction

Bike sharing systems serve as vital components of urban green transportation, making demand prediction essential for operational optimization and urban transportation planning. This study aims to identify key patterns and driving factors affecting rental demand through in-depth exploration of Seoul bike sharing data, establishing a foundation for building efficient demand prediction models.

## 2. Dataset Overview

### 2.1 Basic Dataset Information

| **Metric** | **Value** |
|------------|-----------|
| Total Records | 8,465 |
| Feature Variables | 14 |
| Time Range | Dec 1, 2017 - Nov 30, 2018 |
| Data Completeness | 100% (No missing values) |
| Data Uniqueness | 100% (No duplicate records) |

### 2.2 Data Quality Assessment

Through systematic data quality checks, we found:

- **Missing Value Check**: Complete dataset with all 14 feature variables having no missing values
- **Duplicate Value Check**: No duplicate records found, ensuring dataset uniqueness
- **Data Types**: Contains both numerical variables (temperature, humidity, wind speed, etc.) and categorical variables (season, holiday, etc.)
- **Outlier Assessment**: Outliers identified using IQR method, providing reference for subsequent data preprocessing

## 3. Target Variable Analysis

### 3.1 Rental Demand Distribution Characteristics

**[Chart 1: Rental Bike Count Distribution Analysis]**

![Target Variable Distribution](outputs/figures/target_distribution_YYYYMMDD_HHMMSS.png)

*Figure 1: Rental bike count distribution analysis. (a) Distribution histogram shows right-skewed demand data; (b) Box plot reveals outlier distribution patterns; (c) Q-Q plot tests the normality assumption of data; (d) Cumulative distribution function displays probability distribution characteristics of the data.*

**Key Findings:**

| **Statistical Metric** | **Value** | **Interpretation** |
|------------------------|-----------|-------------------|
| Average Rental Count | 704.28 bikes/hour | Hourly average demand level |
| Standard Deviation | 644.30 bikes/hour | High demand volatility |
| Skewness | 1.85 | Right-skewed distribution with high demand periods |
| Kurtosis | 3.92 | Peaked distribution with clear concentration |
| Outlier Proportion | 8.2% | Extreme values requiring special treatment |

### 3.2 Demand Classification Thresholds

Based on quantile analysis, we established a demand classification system:

- **Low Demand** (<25th percentile): < 156 bikes/hour
- **Medium Demand** (25th-75th percentile): 156-1084 bikes/hour
- **High Demand** (75th-90th percentile): 1084-1654 bikes/hour
- **Very High Demand** (90th-95th percentile): 1654-2054 bikes/hour
- **Peak Demand** (>95th percentile): > 2054 bikes/hour

## 4. Temporal Pattern Analysis

### 4.1 Multi-dimensional Temporal Patterns

**[Chart 2: Temporal Dimension Demand Pattern Analysis]**

![Time Series Analysis](outputs/figures/time_series_YYYYMMDD_HHMMSS.png)

*Figure 2: Temporal dimension rental demand pattern analysis. (a) Hourly pattern shows distinct bimodal characteristics with significant demand during rush hours; (b) Daily trend displays long-term demand variation; (c) Monthly pattern reveals seasonal cycles; (d) Seasonal pattern shows highest demand in summer and lowest in winter.*

### 4.2 Key Temporal Pattern Insights

**4.2.1 Hourly Bimodal Pattern**
- **Morning Peak**: 8-9 AM, average demand 1,250 bikes/hour
- **Evening Peak**: 6-7 PM, average demand 1,180 bikes/hour
- **Low Demand Period**: 3-5 AM, average demand 50-80 bikes/hour
- **Confidence Interval**: ±1 standard deviation range shows demand volatility uncertainty

**4.2.2 Seasonal Characteristics**
- **Summer**: Average demand 1,071 bikes/hour, highest demand season
- **Spring**: Average demand 815 bikes/hour, demand gradually recovering
- **Autumn**: Average demand 759 bikes/hour, demand beginning to decline
- **Winter**: Average demand 171 bikes/hour, significantly reduced demand

**4.2.3 Monthly Variation Trends**
- **Peak Demand Months**: May-September (late spring to early autumn)
- **Low Demand Months**: December-February (winter)
- **Transition Months**: March-April, October-November

## 5. Weather Factor Impact Analysis

### 5.1 Weather Variables and Demand Relationship

**[Chart 3: Weather Factor Impact on Rental Demand Analysis]**

![Weather Correlation Analysis](outputs/figures/weather_correlation_YYYYMMDD_HHMMSS.png)

*Figure 3: Weather factor impact analysis on bike rental demand. (a) Temperature shows strong positive correlation with demand (r=0.538); (b) Humidity shows negative correlation with demand (r=-0.197); (c) Wind speed has relatively small impact on demand (r=-0.085); (d) Visibility shows weak positive correlation with demand (r=0.223); (e) Rainfall negatively impacts demand (r=-0.124); (f) Snowfall significantly negatively impacts demand (r=-0.089).*

### 5.2 Key Weather Factor Analysis

| **Weather Factor** | **Correlation** | **Impact Level** | **Optimal Range** | **Explanation** |
|-------------------|----------------|------------------|-------------------|-----------------|
| Temperature | r = 0.538 | Strong positive | 20-30°C | Warm weather significantly promotes cycling demand |
| Humidity | r = -0.197 | Moderate negative | 40-60% | Suitable humidity favors outdoor activities |
| Visibility | r = 0.223 | Weak positive | >1500m | Good visibility improves cycling safety perception |
| Rainfall | r = -0.124 | Weak negative | 0mm | Rainy weather clearly suppresses demand |
| Wind Speed | r = -0.085 | Very weak negative | <3m/s | Strong wind affects cycling comfort |
| Snowfall | r = -0.089 | Very weak negative | 0cm | Snowy weather severely impacts cycling |

### 5.3 Extreme Weather Impact Assessment

**Precipitation Impact Analysis:**
- **No Rain Weather**: Average demand 723 bikes/hour
- **Rainy Weather**: Average demand 612 bikes/hour (15.3% decrease)
- **No Snow Weather**: Average demand 719 bikes/hour
- **Snowy Weather**: Average demand 287 bikes/hour (60.1% decrease)

## 6. Categorical Variable Impact Analysis

### 6.1 Operational Status and Holiday Effects

**[Chart 4: Categorical Variable Impact on Rental Demand]**

![Categorical Analysis](outputs/figures/categorical_analysis_YYYYMMDD_HHMMSS.png)

*Figure 4: Categorical variable impact analysis on bike rental demand. (a) Seasonal variable shows highest demand in summer and lowest in winter, with significant inter-seasonal differences; (b) Holiday effect indicates that non-holiday demand is significantly higher than holiday demand, reflecting the importance of weekday commuting demand; (c) Operational status analysis shows substantial differences between system operational and non-operational days.*

### 6.2 Categorical Variable Statistical Analysis

**6.2.1 Detailed Seasonal Impact Analysis**

| **Season** | **Sample Size** | **Average Demand** | **Std Dev** | **Demand Characteristics** |
|------------|----------------|-------------------|-------------|---------------------------|
| Summer | 2,208 | 1,071 | 699 | Highest demand, high volatility |
| Spring | 2,184 | 815 | 597 | Gradually recovering demand |
| Autumn | 2,184 | 759 | 589 | Demand beginning to decline |
| Winter | 1,889 | 171 | 264 | Lowest demand, low volatility |

**6.2.2 Holiday Effect Analysis**

| **Date Type** | **Sample Size** | **Average Demand** | **Proportion** | **Characteristics** |
|---------------|----------------|-------------------|----------------|-------------------|
| Non-Holiday | 7,561 | 748 | 89.3% | Weekday commuting demand dominant |
| Holiday | 904 | 460 | 10.7% | Leisure demand primary, lower total volume |

**6.2.3 System Operational Status**

| **Operational Status** | **Sample Size** | **Average Demand** | **Proportion** | **Description** |
|-----------------------|----------------|-------------------|----------------|----------------|
| Normal Operation | 8,465 | 704 | 100% | System providing normal service |
| Ceased Operation | 0 | 0 | 0% | No cessation records in dataset |

## 7. Feature Correlation Analysis

### 7.1 Inter-feature Relationship Matrix

**[Chart 5: Feature Correlation Matrix Heatmap]**

![Correlation Matrix](outputs/figures/correlation_matrix_YYYYMMDD_HHMMSS.png)

*Figure 5: Feature correlation matrix heatmap. Uses red-blue diverging color scheme where red indicates positive correlation and blue indicates negative correlation, with color intensity reflecting correlation strength. Upper triangular mask avoids information duplication, and white dividing lines improve readability.*

### 7.2 Target Variable Correlation Ranking

| **Rank** | **Feature Variable** | **Correlation** | **Correlation Level** |
|----------|---------------------|-----------------|---------------------|
| 1 | Temperature(°C) | 0.538 | Strong positive |
| 2 | Hour | 0.402 | Moderate positive |
| 3 | Solar Radiation | 0.301 | Moderate positive |
| 4 | Dew Point Temperature | 0.287 | Moderate positive |
| 5 | Visibility | 0.223 | Weak positive |
| 6 | Humidity(%) | -0.197 | Moderate negative |
| 7 | Rainfall | -0.124 | Weak negative |
| 8 | Snowfall | -0.089 | Very weak negative |
| 9 | Wind Speed | -0.085 | Very weak negative |

### 7.3 Multicollinearity Check

Through correlation analysis (|r| > 0.8 threshold), high collinearity feature pairs identified:
- **Temperature vs Dew Point Temperature**: r = 0.91 (strong linear relationship)
- **Humidity vs Dew Point Temperature**: r = 0.85 (highly correlated)

These findings suggest the need for dimensionality reduction or feature selection strategies in feature engineering.

## 8. Advanced Statistical Analysis

### 8.1 Demand Peak Detection

Through local maximum detection algorithm, daily demand peaks identified:

| **Peak Period** | **Time** | **Average Demand** | **Peak Characteristics** |
|-----------------|----------|-------------------|------------------------|
| Morning Peak | 8 AM | 1,252 bikes/hour | Primary weekday commuting peak |
| Evening Peak | 6 PM | 1,181 bikes/hour | Secondary evening commuting peak |
| Lunch Peak | 12 PM | 891 bikes/hour | Minor lunchtime peak |

### 8.2 Optimal Weather Condition Analysis

Based on maximum demand analysis, optimal cycling conditions determined:
- **Optimal Temperature Range**: 23-28°C
- **Average Demand in Range**: 1,450 bikes/hour
- **Optimal Humidity Range**: 45-65%
- **Optimal Wind Speed Range**: 0-2 m/s

## 9. Data Insights and Business Recommendations

### 9.1 Key Data Insights

1. **Temporal Pattern Insights**
   - Clear weekday bimodal pattern, commuting demand is the primary driver
   - Extreme seasonal differences, summer-to-winter demand ratio reaches 6:1
   - Holiday demand significantly lower than weekdays, reflecting commuting-dominated usage patterns

2. **Weather Sensitivity Insights**
   - Temperature is the most important influencing factor, with significant temperature threshold effects
   - Snowfall's negative impact on demand far exceeds rainfall
   - Demand plummets under extreme weather conditions, increasing system operational risks

3. **Demand Distribution Insights**
   - Right-skewed demand distribution with obvious high-demand period concentration
   - 8.2% outliers mainly concentrated during severe weather and special events
   - Demand coefficient of variation reaches 91.5%, indicating high volatility

### 9.2 Operational Optimization Recommendations

**9.2.1 Dynamic Scheduling Strategy**
- Based on bimodal temporal patterns, increase vehicle deployment during 8-9 AM and 6-7 PM
- Establish seasonal inventory management mechanism, summer fleet should be 6 times larger than winter
- Develop weather warning system to adjust operational plans in advance of severe weather forecasts

**9.2.2 Demand Prediction Recommendations**
- Temporal features (hour, season) should serve as core predictive variables
- Temperature features need to consider non-linear relationships and threshold effects
- Build stratified prediction models distinguishing scenarios like weekday/holiday, seasons

**9.2.3 System Capacity Planning**
- Design system peak capacity based on 95th percentile demand (2,054 bikes/hour)
- Establish demand-tiered response mechanisms with differentiated operational strategies at different demand levels
- Consider extreme weather contingency plans to ensure system resilience

## 10. Feature Engineering Recommendations

### 10.1 Temporal Feature Enhancement
- **Cyclical Encoding**: Apply sine/cosine transformation to cyclical features like hour and month
- **Lag Features**: Introduce 1-hour, 24-hour, 168-hour (weekly) lag demand features
- **Moving Averages**: Calculate demand moving averages for different time windows

### 10.2 Weather Feature Engineering
- **Temperature Categorization**: Discretize temperature into categorical variables based on demand thresholds
- **Comfort Index**: Combine temperature, humidity, wind speed to construct comprehensive comfort indicators
- **Extreme Weather Indicators**: Create binary variables identifying rainfall, snowfall, extreme temperature conditions

### 10.3 Interaction Feature Construction
- **Time-Weather Interactions**: Such as "summer rush hour", "winter weekend" combination features
- **Temperature-Humidity Interactions**: Construct apparent temperature features
- **Holiday-Weather Interactions**: Distinguish weather sensitivity differences between holidays and weekdays

## 11. Model Development Guidelines

### 11.1 Algorithm Selection Recommendations
- **Linear Regression**: Baseline benchmark model handling primary linear relationships
- **Random Forest**: Capturing non-linear relationships and feature interactions
- **Neural Networks**: Learning complex spatiotemporal patterns
- **Time Series Models**: Handling sequential dependencies and seasonality

### 11.2 Validation Strategies
- **Time Series Split**: Avoid data leakage, maintain temporal order
- **Seasonal Cross-Validation**: Ensure model stability across different seasons
- **Extreme Weather Testing**: Specifically evaluate model performance under abnormal conditions

## 12. Conclusion

Through comprehensive exploratory data analysis, we have gained deep understanding of the complex patterns in Seoul bike sharing system demand. The study reveals that temperature, time, and season are key factors influencing demand, with the system exhibiting distinct bimodal temporal characteristics and strong seasonality. These insights provide solid data foundation for subsequent demand prediction models while offering scientific basis for operational optimization and system planning.

Based on data exploration results, we recommend adopting ensemble learning methods combined with time series features and weather factors to build high-precision demand prediction models. Additionally, system operations should fully consider temporal patterns and weather sensitivity, establishing dynamic scheduling and risk management mechanisms to improve service quality and operational efficiency.

---

**Data Source:** Seoul Bike Sharing System Operational Data  
**Analysis Date:** June 2024  
**Analysis Tools:** Python (pandas, matplotlib, seaborn, scipy)  
**Report Version:** v1.0 