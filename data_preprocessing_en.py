#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data preprocessing module
CDS503 Group Project - Seoul Bike Demand Prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

class BikeDataPreprocessor:
    """Bike Data Preprocessor Class"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def load_data(self, file_path='SeoulBikeData.csv'):
        """Load data"""
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='latin-1')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='cp1252')
        
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def create_time_features(self, df):
        """Create time features"""
        df = df.copy()
        
        # Convert date format
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        
        # Basic time features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['Weekday'] = df['Date'].dt.weekday  # 0=Monday
        df['Quarter'] = df['Date'].dt.quarter
        
        # Weekend indicator
        df['Is_Weekend'] = (df['Weekday'] >= 5).astype(int)
        
        # Cyclical encoding for hour
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        
        # Day of the year
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['DayOfYear_Sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
        df['DayOfYear_Cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
        
        print("Time features created")
        return df
    
    def create_interaction_features(self, df):
        """Create interaction features"""
        df = df.copy()
        
        # Temperature interaction features
        df['Temp_Humidity'] = df['Temperature(°C)'] * df['Humidity(%)']
        df['Temp_Hour'] = df['Temperature(°C)'] * df['Hour']
        df['Temp_Weekend'] = df['Temperature(°C)'] * df['Is_Weekend']
        
        # Weather combination features
        df['Weather_Index'] = (df['Temperature(°C)'] + 20) / 60 * \
                             (100 - df['Humidity(%)']) / 100 * \
                             (df['Visibility (10m)'] / 2000)
        
        # Commuting time indicator
        df['Rush_Hour'] = ((df['Hour'].between(7, 9)) | 
                          (df['Hour'].between(17, 19))).astype(int)
        
        # Precipitation related
        df['Has_Rain'] = (df['Rainfall(mm)'] > 0).astype(int)
        df['Has_Snow'] = (df['Snowfall (cm)'] > 0).astype(int)
        df['Precipitation'] = df['Rainfall(mm)'] + df['Snowfall (cm)']
        
        print("Interaction features created")
        return df
    
    def create_lag_features(self, df, target_col='Rented Bike Count'):
        """Create lag features(beware data leakage)"""
        df = df.copy()
        df = df.sort_values(['Date', 'Hour']).reset_index(drop=True)
        
        # Rental count 1 hour ago
        df['Rent_Lag_1h'] = df[target_col].shift(1)
        
        # Rental count 24 hours ago (same time yesterday)(same time yesterday)
        df['Rent_Lag_24h'] = df[target_col].shift(24)
        
        # Rental count same time 7 days ago
        df['Rent_Lag_168h'] = df[target_col].shift(168)
        
        # Fill missing values
        df['Rent_Lag_1h'].fillna(df[target_col].mean(), inplace=True)
        df['Rent_Lag_24h'].fillna(df[target_col].mean(), inplace=True)
        df['Rent_Lag_168h'].fillna(df[target_col].mean(), inplace=True)
        
        print("Lag features created")
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical features"""
        df = df.copy()
        
        # One-hot encode categorical features
        categorical_cols = ['Seasons', 'Holiday', 'Functioning Day']
        
        for col in categorical_cols:
            # Create one-hot encoding
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
        
        # Drop original categorical columns
        df.drop(categorical_cols, axis=1, inplace=True)
        
        print("Categorical feature encoding completed")
        return df
    
    def remove_outliers(self, df, target_col='Rented Bike Count', method='iqr', factor=1.5):
        """Remove outliers"""
        df = df.copy()
        
        if method == 'iqr':
            Q1 = df[target_col].quantile(0.25)
            Q3 = df[target_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            outliers_mask = (df[target_col] < lower_bound) | (df[target_col] > upper_bound)
            outliers_count = outliers_mask.sum()
            
            print(f"Detected {outliers_count} outliers ({outliers_count/len(df)*100:.2f}%)")
            
            # Do not delete outliers; apply winsorization instead
            df.loc[df[target_col] < lower_bound, target_col] = lower_bound
            df.loc[df[target_col] > upper_bound, target_col] = upper_bound
            
            print("Outlier handling completed(using winsorization)")
        
        return df
    
    def handle_non_operating_days(self, df):
        """Processing non-operating day data"""
        print("🚫 Processing non-operating day data...")
        
        # Mark non-operating day
        non_operating_mask = df['Functioning Day'] == 'No'
        non_operating_count = non_operating_mask.sum()
        
        print(f"found {non_operating_count} samples non-operating day records ({non_operating_count/len(df)*100:.2f}%)")
        
        # Exclude non-operating day data
        df_filtered = df[~non_operating_mask].copy()
        
        print(f"Remaining after exclusion {len(df_filtered)} records")
        return df_filtered
    
    def create_enhanced_weather_features(self, df):
        """Create enriched weather features based on data insights"""
        print("🌤️ Creating enriched weather features...")
        df = df.copy()
        
        # 1. temperature bin feature(thresholds based on data analysis)
        df['Temp_Severe_Cold'] = (df['Temperature(°C)'] < 0).astype(int)
        df['Temp_Cold'] = ((df['Temperature(°C)'] >= 0) & (df['Temperature(°C)'] < 10)).astype(int)
        df['Temp_Cool'] = ((df['Temperature(°C)'] >= 10) & (df['Temperature(°C)'] < 20)).astype(int)
        df['Temp_Warm'] = ((df['Temperature(°C)'] >= 20) & (df['Temperature(°C)'] < 30)).astype(int)
        df['Temp_Hot'] = (df['Temperature(°C)'] >= 30).astype(int)
        
        # 2. humidity threshold feature(based on data analysis)
        df['Humidity_Low'] = (df['Humidity(%)'] < 30).astype(int)
        df['Humidity_Medium'] = ((df['Humidity(%)'] >= 30) & (df['Humidity(%)'] < 50)).astype(int)
        df['Humidity_High'] = ((df['Humidity(%)'] >= 50) & (df['Humidity(%)'] < 70)).astype(int)
        df['Humidity_Very_High'] = (df['Humidity(%)'] >= 70).astype(int)
        
        # 3. precipitation threshold feature
        df['Has_Rain'] = (df['Rainfall(mm)'] > 0).astype(int)
        df['Has_Snow'] = (df['Snowfall (cm)'] > 0).astype(int)
        df['Has_Precipitation'] = ((df['Rainfall(mm)'] > 0) | (df['Snowfall (cm)'] > 0)).astype(int)
        
        # 4. wind speed binning
        df['Wind_Calm'] = (df['Wind speed (m/s)'] < 2).astype(int)
        df['Wind_Light'] = ((df['Wind speed (m/s)'] >= 2) & (df['Wind speed (m/s)'] < 4)).astype(int)
        df['Wind_Moderate'] = ((df['Wind speed (m/s)'] >= 4) & (df['Wind speed (m/s)'] < 6)).astype(int)
        df['Wind_Strong'] = (df['Wind speed (m/s)'] >= 6).astype(int)
        
        print("Enriched weather features created")
        return df
    
    def create_enhanced_time_features(self, df):
        """Create enriched time features based on bimodal pattern"""
        print("⏰ Creating enriched time features(bimodal pattern)...")
        df = df.copy()
        
        # 1. time segment partitioning based on data analysis
        df['Hour_Deep_Night'] = (df['Hour'].isin([0, 1, 2, 3, 4, 5])).astype(int)  #deep night low
        df['Hour_Early_Morning'] = (df['Hour'].isin([6, 7])).astype(int)  #early morning rise
        df['Hour_Morning_Peak'] = (df['Hour'].isin([8, 9])).astype(int)  #morning peak
        df['Hour_Morning_Decline'] = (df['Hour'].isin([10, 11, 12])).astype(int)  #mid-morning decline
        df['Hour_Afternoon'] = (df['Hour'].isin([13, 14, 15, 16])).astype(int)  #afternoon steady
        df['Hour_Evening_Peak'] = (df['Hour'].isin([17, 18, 19])).astype(int)  #evening peak
        df['Hour_Evening_Decline'] = (df['Hour'].isin([20, 21, 22, 23])).astype(int)  #evening decline
        
        # 2. bimodal features
        df['Is_Peak_Hour'] = ((df['Hour'].isin([8, 17, 18, 19]))).astype(int)
        df['Is_Low_Hour'] = ((df['Hour'].isin([3, 4, 5]))).astype(int)
        
        # 3. seasonal cyclic features(seasonal differences based on data analysis)
        df['Season_Summer'] = (df['Seasons'] == 'Summer').astype(int)
        df['Season_Winter'] = (df['Seasons'] == 'Winter').astype(int)
        df['Season_Spring'] = (df['Seasons'] == 'Spring').astype(int)
        df['Season_Autumn'] = (df['Seasons'] == 'Autumn').astype(int)
        
        # 4. cyclical encoding of month(capture seasonal change)
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        # 5. weekday vs weekend subdivision
        df['Is_Weekday'] = (df['Weekday'] < 5).astype(int)
        df['Is_Weekend'] = (df['Weekday'] >= 5).astype(int)
        df['Is_Friday'] = (df['Weekday'] == 4).astype(int)  # Friday special
        df['Is_Monday'] = (df['Weekday'] == 0).astype(int)  # Monday special
        
        print("Enriched time features created")
        return df
    
    def create_comfort_and_extreme_features(self, df):
        """Creating comfort and extreme weather features"""
        print("😌 Creating comfort and extreme weather features...")
        df = df.copy()
        
        # 1. comfort index(based ontemperatureandhumidity)
        # Based on data analysis:optimal temperature20-30°C，humidity30-70%
        df['Comfort_Index'] = np.where(
            (df['Temperature(°C)'].between(20, 30)) & (df['Humidity(%)'].between(30, 70)),
            1.0,  # most comfortable
            np.where(
                (df['Temperature(°C)'].between(10, 35)) & (df['Humidity(%)'].between(20, 80)),
                0.7,  # moderately comfortable
                np.where(
                    (df['Temperature(°C)'].between(0, 40)) & (df['Humidity(%)'].between(10, 90)),
                    0.4,  # average
                    0.1   # uncomfortable
                )
            )
        )
        
        # 2. perceived temperature(Heat Indexsimplified version)
        df['Heat_Index'] = df['Temperature(°C)'] + 0.5 * (df['Humidity(%)'] - 50) / 100 * df['Temperature(°C)']
        
        # 3. extreme weather indicator
        df['Extreme_Cold'] = (df['Temperature(°C)'] < -10).astype(int)
        df['Extreme_Hot'] = (df['Temperature(°C)'] > 35).astype(int)
        df['Extreme_Humid'] = (df['Humidity(%)'] > 90).astype(int)
        df['Extreme_Dry'] = (df['Humidity(%)'] < 20).astype(int)
        df['Heavy_Rain'] = (df['Rainfall(mm)'] > 10).astype(int)
        df['Heavy_Snow'] = (df['Snowfall (cm)'] > 5).astype(int)
        
        # 4. Bad weather combination
        df['Bad_Weather'] = (
            (df['Extreme_Cold'] == 1) |
            (df['Extreme_Hot'] == 1) |
            (df['Heavy_Rain'] == 1) |
            (df['Heavy_Snow'] == 1) |
            (df['Extreme_Humid'] == 1)
        ).astype(int)
        
        # 5. perfect weather(high demand expected)
        df['Perfect_Weather'] = (
            (df['Temperature(°C)'].between(20, 28)) &
            (df['Humidity(%)'].between(40, 60)) &
            (df['Rainfall(mm)'] == 0) &
            (df['Snowfall (cm)'] == 0) &
            (df['Wind speed (m/s)'] < 3)
        ).astype(int)
        
        print("Comfort and extreme weather features created")
        return df
    
    def create_enhanced_interaction_features(self, df):
        """Create enriched interaction features"""
        print("🔗 Create enriched interaction features...")
        df = df.copy()
        
        # 1. temperature × season interaction feature(key optimization)
        df['Temp_Summer'] = df['Temperature(°C)'] * df['Season_Summer']
        df['Temp_Winter'] = df['Temperature(°C)'] * df['Season_Winter']
        df['Temp_Spring'] = df['Temperature(°C)'] * df['Season_Spring']
        df['Temp_Autumn'] = df['Temperature(°C)'] * df['Season_Autumn']
        
        # 2. time × weather interaction
        df['Peak_Hour_Good_Weather'] = df['Is_Peak_Hour'] * (1 - df['Bad_Weather'])
        df['Peak_Hour_Bad_Weather'] = df['Is_Peak_Hour'] * df['Bad_Weather']
        df['Weekend_Good_Weather'] = df['Is_Weekend'] * (1 - df['Bad_Weather'])
        
        # 3. comfort × time interaction
        df['Comfort_Peak'] = df['Comfort_Index'] * df['Is_Peak_Hour']
        df['Comfort_Weekend'] = df['Comfort_Index'] * df['Is_Weekend']
        
        # 4. season × time interaction(capture seasonal usage pattern differences)
        df['Summer_Peak'] = df['Season_Summer'] * df['Is_Peak_Hour']
        df['Winter_Peak'] = df['Season_Winter'] * df['Is_Peak_Hour']
        df['Summer_Weekend'] = df['Season_Summer'] * df['Is_Weekend']
        df['Winter_Weekend'] = df['Season_Winter'] * df['Is_Weekend']
        
        print("Enriched interaction features created")
        return df
    
    def prepare_features(self, df, use_lag_features=False, exclude_non_operating=True):
        """Enhanced feature engineering based on data insights"""
        print("🚀 Starting enhanced feature engineering based on data insights...")
        df = df.copy()
        
        # 1. Processing non-operating day data
        if exclude_non_operating:
            df = self.handle_non_operating_days(df)
        
        # 2. Basic time features
        df = self.create_time_features(df)
        
        # 3. enriched weather features
        df = self.create_enhanced_weather_features(df)
        
        # 4. enriched time features(bimodal pattern)
        df = self.create_enhanced_time_features(df)
        
        # 5. comfort and extreme weather features
        df = self.create_comfort_and_extreme_features(df)
        
        # 6. existing interaction features
        df = self.create_interaction_features(df)
        
        # 7. enriched interaction features
        df = self.create_enhanced_interaction_features(df)
        
        # 8. lag features
        if use_lag_features:
            df = self.create_lag_features(df)
        
        # 9. Encode categorical features
        df = self.encode_categorical_features(df)
        
        # 10. handle outliers
        df = self.remove_outliers(df)
        
        # select feature columns(excluding raw date and target)
        exclude_cols = ['Date', 'Rented Bike Count']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        self.feature_names = feature_cols
        
        print(f"✅ Final feature count: {len(feature_cols)}")
        print(f"📋 Feature list: {feature_cols}")
        
        return df
    
    def split_data_temporal(self, df, target_col='Rented Bike Count', 
                           train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Time-series-friendly data split"""
        
        # sort by time
        df = df.sort_values(['Date', 'Hour']).reset_index(drop=True)
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        # Split data
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        print(f"Data split completed:")
        print(f"  Training set: {len(train_df)} samples ({len(train_df)/n*100:.1f}%)")
        print(f"  Validation set: {len(val_df)} samples ({len(val_df)/n*100:.1f}%)")
        print(f"  Test set: {len(test_df)} samples ({len(test_df)/n*100:.1f}%)")
        
        # extract features and target
        feature_cols = self.feature_names
        
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        
        X_val = val_df[feature_cols]
        y_val = val_df[target_col]
        
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train, X_val, X_test):
        """Feature standardization"""
        # standardize only numerical features
        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        X_test_scaled = X_test.copy()
        
        # fit scaler
        self.scaler.fit(X_train[numeric_features])
        
        # apply scaling
        X_train_scaled[numeric_features] = self.scaler.transform(X_train[numeric_features])
        X_val_scaled[numeric_features] = self.scaler.transform(X_val[numeric_features])
        X_test_scaled[numeric_features] = self.scaler.transform(X_test[numeric_features])
        
        print("Feature standardization completed")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def get_time_series_cv(self, n_splits=5):
        """Get time series cross-validator"""
        return TimeSeriesSplit(n_splits=n_splits)
    
    def prepare_stratified_data(self, df):
        """Prepare data for stratified modeling"""
        print("🎯 Preparing stratified modeling data...")
        
        # Create stratification indicator
        stratified_data = {}
        
        # 1. Stratify by operating status(if non-operating day)
        if 'Functioning Day' in df.columns:
            operating_data = df[df['Functioning Day'] == 'Yes'].copy()
        else:
            operating_data = df.copy()
        
        stratified_data['operating'] = operating_data
        
        # 2. Stratify by season(using season features after one-hot encoding)
        season_mapping = {
            'spring': 'Seasons_Spring',
            'summer': 'Seasons_Summer', 
            'autumn': 'Seasons_Winter',  # Note: because drop_first=True, Autumn is represented by all other season columns being 0
            'winter': 'Seasons_Winter'
        }
        
        for season_name, season_col in season_mapping.items():
            if season_name == 'autumn':
                # Autumn is the reference category; all other season columns are 0
                season_data = operating_data[
                    (operating_data.get('Seasons_Spring', 0) == 0) & 
                    (operating_data.get('Seasons_Summer', 0) == 0) & 
                    (operating_data.get('Seasons_Winter', 0) == 0)
                ].copy()
            elif season_col in operating_data.columns:
                season_data = operating_data[operating_data[season_col] == 1].copy()
            else:
                print(f"  ⚠️  {season_name.title()} columns {season_col} does not exist，skipping")
                continue
                
            if len(season_data) > 0:
                stratified_data[f'season_{season_name}'] = season_data
                print(f"  {season_name.title()}: {len(season_data)} records")
            else:
                print(f"  ⚠️  {season_name.title()}: 0 records")
        
        # 3. Stratify by weather condition
        good_weather = operating_data[
            (operating_data['Temperature(°C)'].between(10, 30)) &
            (operating_data['Humidity(%)'].between(30, 70)) &
            (operating_data['Rainfall(mm)'] == 0) &
            (operating_data['Snowfall (cm)'] == 0)
        ].copy()
        
        bad_weather = operating_data[
            (operating_data['Temperature(°C)'] < 0) |
            (operating_data['Temperature(°C)'] > 35) |
            (operating_data['Rainfall(mm)'] > 5) |
            (operating_data['Snowfall (cm)'] > 2)
        ].copy()
        
        stratified_data['good_weather'] = good_weather
        stratified_data['bad_weather'] = bad_weather
        
        print(f"  Good weather: {len(good_weather)} records")
        print(f"  Bad weather: {len(bad_weather)} records")
        
        # 4. Stratify by time segment
        peak_hours = operating_data[operating_data['Hour'].isin([8, 17, 18, 19])].copy()
        off_peak_hours = operating_data[~operating_data['Hour'].isin([8, 17, 18, 19])].copy()
        
        stratified_data['peak_hours'] = peak_hours
        stratified_data['off_peak_hours'] = off_peak_hours
        
        print(f"  Peak hours: {len(peak_hours)} records")
        print(f"  Off-peak hours: {len(off_peak_hours)} records")
        
        return stratified_data

def main():
    """Main function - Demonstrate data preprocessing workflow"""
    
    # Initialize preprocessor
    preprocessor = BikeDataPreprocessor()
    
    # Load data
    df = preprocessor.load_data()
    
    # Prepare features
    df_processed = preprocessor.prepare_features(df, use_lag_features=True)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data_temporal(df_processed)
    
    # Standardize features
    X_train_scaled, X_val_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_val, X_test)
    
    print("\nData preprocessing completed!")
    print(f"Training feature shape: {X_train_scaled.shape}")
    print(f"Training target shape: {y_train.shape}")
    
    # Save processed data
    print("\nSaving preprocessed data...")
    np.save('X_train.npy', X_train_scaled.values)
    np.save('X_val.npy', X_val_scaled.values)
    np.save('X_test.npy', X_test_scaled.values)
    np.save('y_train.npy', y_train.values)
    np.save('y_val.npy', y_val.values)
    np.save('y_test.npy', y_test.values)
    
    # Save feature names
    with open('feature_names.txt', 'w') as f:
        for name in preprocessor.feature_names:
            f.write(f"{name}\n")
    
    print("Preprocessed data saved to file")

if __name__ == "__main__":
    main() 