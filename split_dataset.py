import pandas as pd
from sklearn.model_selection import train_test_split

# 载入数据集
file_path = '/Users/tianguoguo/Desktop/SeoulBikeData.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# 将数据集分为特征（X）和目标变量（y）
X = df.drop(columns=['Rented Bike Count'])  # 特征
y = df['Rented Bike Count']  # 目标

# 按照70%训练集，15%验证集，15%测试集进行数据拆分
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# 再将剩余的30%数据拆分成15%的验证集和15%的测试集
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 合并特征和目标变量，得到完整的数据集
train_data = pd.concat([X_train, y_train], axis=1)
val_data = pd.concat([X_val, y_val], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# 保存拆分后的数据集为 CSV 文件
train_data.to_csv('/Users/tianguoguo/Desktop/train_data.csv', index=False)
val_data.to_csv('/Users/tianguoguo/Desktop/val_data.csv', index=False)
test_data.to_csv('/Users/tianguoguo/Desktop/test_data.csv', index=False)

# 打印拆分后的数据集形状
print(f"训练集：X_train {X_train.shape}, y_train {y_train.shape}")
print(f"验证集：X_val {X_val.shape}, y_val {y_val.shape}")
print(f"测试集：X_test {X_test.shape}, y_test {y_test.shape}")
