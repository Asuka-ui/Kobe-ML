# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import warnings
import time
from packaging import version
import sklearn

# 忽略相关的警告提醒消息
warnings.filterwarnings('ignore')

# 1. 读取原始数据
raw_original = pd.read_csv('kobe_data.csv')

# 为了确保原始数据不被修改，创建一个副本进行处理
raw = raw_original.copy()

# 2. 数据探索性分析（EDA）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 2.1 单变量分析
plt.figure(figsize=(10,6))
raw['combined_shot_type'].value_counts().plot(kind='bar')
plt.xlabel('出手类型')
plt.ylabel('出手次数')
plt.title('科比职业生涯不同出手类型的次数统计')
plt.show()

plt.figure(figsize=(8,6))
raw['shot_type'].value_counts().plot(kind='bar')
plt.xlabel('投篮类型')
plt.ylabel('出手次数')
plt.title('科比职业生涯两分球和三分球的出手数')
plt.xticks(rotation=0)
plt.show()

plt.figure(figsize=(8,6))
raw['shot_distance'].hist(bins=100)
plt.xlabel('出手距离 (英尺)')
plt.ylabel('出手次数')
plt.title('科比出手距离的分布')
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(data=raw, y='shot_distance')
plt.xlabel('出手距离 (英尺)')
plt.ylabel('出手次数')
plt.title('科比出手距离的箱型图')
plt.show()

plt.figure(figsize=(20,10))

def scatter_plot_by_category(feat):
    alpha = 0.1
    gs = raw.groupby(feat)
    cs = plt.cm.rainbow(np.linspace(0,1,len(gs)))
    for g, c in zip(gs, cs):
        plt.scatter(g[1].loc_x, g[1].loc_y, color=c, alpha=alpha)

plt.subplot(1,2,1)
scatter_plot_by_category(raw['shot_zone_area'])
plt.title('出手区域 (shot_zone_area)')

plt.subplot(1,2,2)
scatter_plot_by_category(raw['shot_zone_range'])
plt.title('出手区域范围 (shot_zone_range)')

plt.suptitle('科比职业生涯不同出手区域的散点图')
plt.show()

# 2.2 双变量分析
kobe = raw[pd.notnull(raw['shot_made_flag'])].copy()
print('训练集的大小:', kobe.shape)

plt.figure(figsize=(6,4))
kobe['shot_made_flag'].value_counts(normalize=True).plot(kind='bar')
plt.xlabel('命中情况')
plt.ylabel('命中比例')
plt.title('科比的出手命中率')
plt.show()

plt.figure(figsize=(12,6))
sns.barplot(data=kobe, x='combined_shot_type', y='shot_made_flag')
plt.xlabel('出手类型')
plt.ylabel('命中率')
plt.title('不同出手类型的命中率')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(6,4))
sns.barplot(data=kobe, x='shot_type', y='shot_made_flag')
plt.xlabel('投篮类型')
plt.ylabel('命中率')
plt.title('两分球与三分球的命中率')
plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(data=kobe, x='shot_distance', y='shot_made_flag', alpha=0.3)
plt.xlabel('出手距离 (英尺)')
plt.ylabel('是否命中')
plt.title('出手距离与命中率的关系')
plt.show()

plt.figure(figsize=(10,6))
sns.violinplot(data=kobe, x='shot_made_flag', y='shot_distance')
plt.xlabel('是否命中')
plt.ylabel('出手距离 (英尺)')
plt.title('出手距离与命中情况的分布')
plt.show()

# 3. 数据预处理
preprocessed = raw.copy()

# 创建新的特征 time_remaining，用于替代 minutes_remaining 和 seconds_remaining
preprocessed['time_remaining'] = preprocessed['minutes_remaining'] * 60 + preprocessed['seconds_remaining']

# 删除 minutes_remaining 和 seconds_remaining 特征
preprocessed = preprocessed.drop(['minutes_remaining', 'seconds_remaining'], axis=1)

# 划分训练集和测试集
train_data = preprocessed[pd.notnull(preprocessed['shot_made_flag'])].copy()
test_data = preprocessed[pd.isnull(preprocessed['shot_made_flag'])].copy()

print('训练集的大小:', train_data.shape)
print('测试集的大小:', test_data.shape)

# 处理异常值去掉（仅对训练集）
train_data = train_data[train_data['shot_distance'] <= 50]
print('去除异常值后的训练集大小:', train_data.shape)

# 独热编码（One-Hot Encoding）处理分类变量
categorical_features = train_data.select_dtypes(include=['object', 'category']).columns.tolist()

# 检查 scikit-learn 版本以决定使用 'sparse' 还是 'sparse_output'
sklearn_version = version.parse(sklearn.__version__)
if sklearn_version >= version.parse("1.2"):
    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
else:
    encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')

encoded_train = pd.DataFrame(encoder.fit_transform(train_data[categorical_features]),
                             columns=encoder.get_feature_names_out(categorical_features),
                             index=train_data.index)

encoded_test = pd.DataFrame(encoder.transform(test_data[categorical_features]),
                            columns=encoder.get_feature_names_out(categorical_features),
                            index=test_data.index)

train_data = train_data.drop(categorical_features, axis=1).join(encoded_train)
test_data = test_data.drop(categorical_features, axis=1).join(encoded_test)

# 数值型特征
numerical_features = ['loc_x', 'loc_y', 'period', 'playoffs', 'shot_distance', 'time_remaining']

# 数据标准化
scaler = StandardScaler()
train_data[numerical_features] = scaler.fit_transform(train_data[numerical_features])
test_data[numerical_features] = scaler.transform(test_data[numerical_features])

# 准备数据，将训练数据分为特征和标签，删掉测试数据中的 shot_made_flag 列
train_labels = train_data['shot_made_flag']
train_features = train_data.drop('shot_made_flag', axis=1)

test_features = test_data.drop('shot_made_flag', axis=1)

# 保存处理后的数据集
train_features.to_csv('kobe_train_preprocessed_features.csv', index=False)
train_labels.to_csv('kobe_train_preprocessed_labels.csv', index=False)
test_features.to_csv('kobe_test_preprocessed_features.csv', index=False)

print("数据预处理完成，处理后的数据已保存为 'kobe_train_preprocessed_features.csv', 'kobe_train_preprocessed_labels.csv' 和 'kobe_test_preprocessed_features.csv'。")

# 4. 训练模型
print("加载预处理后的数据...")
train_features = pd.read_csv('kobe_train_preprocessed_features.csv')
train_labels = pd.read_csv('kobe_train_preprocessed_labels.csv').squeeze()
test_features = pd.read_csv('kobe_test_preprocessed_features.csv')

print('训练特征的形状:', train_features.shape)
print('训练标签的形状:', train_labels.shape)
print('测试特征的形状:', test_features.shape)

print("\n划分训练集和验证集...")
X_train, X_val, y_train, y_val = train_test_split(
    train_features, train_labels, test_size=0.2, random_state=42, stratify=train_labels
)

print('训练子集特征的形状:', X_train.shape)
print('验证子集特征的形状:', X_val.shape)
print('训练子集标签的形状:', y_train.shape)
print('验证子集标签的形状:', y_val.shape)

print("\n应用SMOTE进行数据平衡...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print('平衡后的训练集特征的形状:', X_train_res.shape)
print('平衡后的训练集标签的形状:', y_train_res.shape)
print('平衡后的训练标签分布:\n', pd.Series(y_train_res).value_counts())

# 4.1 初始化并训练随机森林模型
print("\n训练随机森林模型...")
best_params_rfc = {'n_estimators': 522, 'max_depth': 21, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'sqrt'}

best_model_rfc = RandomForestClassifier(
    n_estimators=best_params_rfc['n_estimators'],
    max_depth=best_params_rfc['max_depth'],
    min_samples_split=best_params_rfc['min_samples_split'],
    min_samples_leaf=best_params_rfc['min_samples_leaf'],
    max_features=best_params_rfc['max_features'],
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

# 记录训练开始时间
start_time_rfc = time.time()

# 训练模型
best_model_rfc.fit(X_train_res, y_train_res)

# 记录训练结束时间
end_time_rfc = time.time()

print(f"随机森林模型的训练时间: {end_time_rfc - start_time_rfc:.2f} 秒")

# 5. 评估随机森林模型性能
print("\n在验证集上评估随机森林模型性能...")
y_pred_rfc = best_model_rfc.predict(X_val)
y_pred_proba_rfc = best_model_rfc.predict_proba(X_val)[:, 1]

accuracy_rfc = accuracy_score(y_val, y_pred_rfc)
precision_rfc = precision_score(y_val, y_pred_rfc)
recall_rfc = recall_score(y_val, y_pred_rfc)
f1_rfc = f1_score(y_val, y_pred_rfc)
roc_auc_rfc = roc_auc_score(y_val, y_pred_proba_rfc)

print("\n随机森林模型在验证集上的表现:")
print(f"准确率 (Accuracy): {accuracy_rfc:.4f}")
print(f"精确率 (Precision): {precision_rfc:.4f}")
print(f"召回率 (Recall): {recall_rfc:.4f}")
print(f"F1分数 (F1 Score): {f1_rfc:.4f}")
print(f"ROC-AUC: {roc_auc_rfc:.4f}")

print("\n随机森林分类报告:")
print(classification_report(y_val, y_pred_rfc, target_names=['未命中', '命中']))

conf_matrix_rfc = confusion_matrix(y_val, y_pred_rfc)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix_rfc, annot=True, fmt='d', cmap='Blues', xticklabels=['未命中', '命中'], yticklabels=['未命中', '命中'])
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('随机森林混淆矩阵')
plt.show()

fpr_rfc, tpr_rfc, thresholds_rfc = roc_curve(y_val, y_pred_proba_rfc)
roc_auc_value_rfc = auc(fpr_rfc, tpr_rfc)

plt.figure(figsize=(8,6))
plt.plot(fpr_rfc, tpr_rfc, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc_value_rfc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率 (False Positive Rate)')
plt.ylabel('真正率 (True Positive Rate)')
plt.title('随机森林接收者操作特征曲线 (ROC)')
plt.legend(loc="lower right")
plt.show()

# 4.2 初始化并训练XGBoost模型
print("\n训练XGBoost模型...")
best_params_xgb = {
    'n_estimators': 573,
    'max_depth': 9,
    'learning_rate': 0.06654490124263246,
    'subsample': 0.8063558350286024,
    'colsample_bytree': 0.8310982321715663
}

best_model_xgb = XGBClassifier(
    n_estimators=best_params_xgb['n_estimators'],
    max_depth=best_params_xgb['max_depth'],
    learning_rate=best_params_xgb['learning_rate'],
    subsample=best_params_xgb['subsample'],
    colsample_bytree=best_params_xgb['colsample_bytree'],
    random_state=42,
    # use_label_encoder=False,  # 已移除此参数以消除警告
    eval_metric='auc',
    n_jobs=-1
)

# 记录训练开始时间
start_time_xgb = time.time()

# 训练模型
best_model_xgb.fit(X_train_res, y_train_res)

# 记录训练结束时间
end_time_xgb = time.time()

print(f"XGBoost模型的训练时间: {end_time_xgb - start_time_xgb:.2f} 秒")

# 在验证集上进行预测
y_pred_xgb = best_model_xgb.predict(X_val)
y_pred_proba_xgb = best_model_xgb.predict_proba(X_val)[:, 1]

accuracy_xgb = accuracy_score(y_val, y_pred_xgb)
precision_xgb = precision_score(y_val, y_pred_xgb)
recall_xgb = recall_score(y_val, y_pred_xgb)
f1_xgb = f1_score(y_val, y_pred_xgb)
roc_auc_xgb = roc_auc_score(y_val, y_pred_proba_xgb)

print("\nXGBoost模型在验证集上的表现:")
print(f"准确率 (Accuracy): {accuracy_xgb:.4f}")
print(f"精确率 (Precision): {precision_xgb:.4f}")
print(f"召回率 (Recall): {recall_xgb:.4f}")
print(f"F1分数 (F1 Score): {f1_xgb:.4f}")
print(f"ROC-AUC: {roc_auc_xgb:.4f}")

print("\nXGBoost分类报告:")
print(classification_report(y_val, y_pred_xgb, target_names=['未命中', '命中']))

conf_matrix_xgb = confusion_matrix(y_val, y_pred_xgb)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix_xgb, annot=True, fmt='d', cmap='Greens', xticklabels=['未命中', '命中'], yticklabels=['未命中', '命中'])
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('XGBoost混淆矩阵')
plt.show()

fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_val, y_pred_proba_xgb)
roc_auc_value_xgb = auc(fpr_xgb, tpr_xgb)

plt.figure(figsize=(8,6))
plt.plot(fpr_xgb, tpr_xgb, color='green', lw=2, label=f'ROC曲线 (AUC = {roc_auc_value_xgb:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率 (False Positive Rate)')
plt.ylabel('真正率 (True Positive Rate)')
plt.title('XGBoost接收者操作特征曲线 (ROC)')
plt.legend(loc="lower right")
plt.show()

# 5. 实施堆叠（Stacking）集成方法
print("\n实施堆叠（Stacking）集成方法...")

# 定义基模型
base_models = [
    ('rf', best_model_rfc),
    ('xgb', best_model_xgb)
]

# 定义元模型
meta_model = LogisticRegression(random_state=42, n_jobs=-1)

# 定义堆叠模型
stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    passthrough=False,
    n_jobs=-1
)

# 记录训练开始时间
start_time_stack = time.time()

# 训练堆叠模型
stacking.fit(X_train_res, y_train_res)

# 记录训练结束时间
end_time_stack = time.time()

print(f"堆叠模型的训练时间: {end_time_stack - start_time_stack:.2f} 秒")

# 在验证集上进行预测
y_pred_stack = stacking.predict(X_val)
y_pred_proba_stack = stacking.predict_proba(X_val)[:, 1]

# 计算评估指标
accuracy_stack = accuracy_score(y_val, y_pred_stack)
precision_stack = precision_score(y_val, y_pred_stack)
recall_stack = recall_score(y_val, y_pred_stack)
f1_stack = f1_score(y_val, y_pred_stack)
roc_auc_stack = roc_auc_score(y_val, y_pred_proba_stack)

print("\n堆叠模型在验证集上的表现:")
print(f"准确率 (Accuracy): {accuracy_stack:.4f}")
print(f"精确率 (Precision): {precision_stack:.4f}")
print(f"召回率 (Recall): {recall_stack:.4f}")
print(f"F1分数 (F1 Score): {f1_stack:.4f}")
print(f"ROC-AUC: {roc_auc_stack:.4f}")

print("\n堆叠分类报告:")
print(classification_report(y_val, y_pred_stack, target_names=['未命中', '命中']))

conf_matrix_stack = confusion_matrix(y_val, y_pred_stack)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix_stack, annot=True, fmt='d', cmap='RdPu', xticklabels=['未命中', '命中'], yticklabels=['未命中', '命中'])
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('堆叠混淆矩阵')
plt.show()

fpr_stack, tpr_stack, thresholds_stack = roc_curve(y_val, y_pred_proba_stack)
roc_auc_value_stack = auc(fpr_stack, tpr_stack)

plt.figure(figsize=(8,6))
plt.plot(fpr_stack, tpr_stack, color='red', lw=2, label=f'ROC曲线 (AUC = {roc_auc_value_stack:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率 (False Positive Rate)')
plt.ylabel('真正率 (True Positive Rate)')
plt.title('堆叠接收者操作特征曲线 (ROC)')
plt.legend(loc="lower right")
plt.show()

# 6. 比较所有模型的性能
print("\n比较所有模型的性能:")

model_performance = pd.DataFrame({
    '模型': ['随机森林', 'XGBoost', '堆叠'],
    '准确率 (Accuracy)': [accuracy_rfc, accuracy_xgb, accuracy_stack],
    '精确率 (Precision)': [precision_rfc, precision_xgb, precision_stack],
    '召回率 (Recall)': [recall_rfc, recall_xgb, recall_stack],
    'F1 分数 (F1 Score)': [f1_rfc, f1_xgb, f1_stack],
    'ROC-AUC': [roc_auc_rfc, roc_auc_xgb, roc_auc_stack]
})

print(model_performance)

plt.figure(figsize=(12,8))
metrics = ['准确率 (Accuracy)', '精确率 (Precision)', '召回率 (Recall)', 'F1 分数 (F1 Score)', 'ROC-AUC']
for i, metric in enumerate(metrics):
    plt.subplot(2, 3, i+1)
    sns.barplot(x='模型', y=metric, data=model_performance)
    plt.title(metric)
    plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# 7. 预测测试集并保存结果
print("\n对测试集进行预测并保存结果...")

# 使用堆叠模型预测
test_predictions_stack = stacking.predict(test_features)
test_predictions_proba_stack = stacking.predict_proba(test_features)[:, 1]

# 将预测结果添加到测试集
test_data_with_predictions = test_features.copy()
test_data_with_predictions['shot_made_flag_RF'] = best_model_rfc.predict(test_features)
test_data_with_predictions['shot_made_flag_XGB'] = best_model_xgb.predict(test_features)
test_data_with_predictions['shot_made_flag_Stacking'] = test_predictions_stack

# 保存预测结果为CSV文件
test_data_with_predictions.to_csv('kobe_test_predictions.csv', index=False)

print("测试集的预测结果已保存为 'kobe_test_predictions.csv'。")
