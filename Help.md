# IceBird ML Toolkit

一个基于conda环境的自动化机器学习工具箱，用于训练模型并批量预测IceBird数据。无需编写代码，通过配置文件即可运行。

## 快速开始

### 1. 环境准备

```bash
# 创建并激活conda环境
conda create -n icebird python=3.8
conda activate icebird

# 安装依赖（假设requirement.txt已存在）
pip install -r requirement.txt
```

### 2.准备配置文件
将整个代码库下载到本地，或者通过自定义的方式，新建文件夹，下载*core_process.py*和*user_config.txt*到本地文件夹中。根据需求修改*user_config.txt*文件。

```ini
# 训练数据路径（支持.xlsx或.csv）
input_data = 'training/ML_IceBird_Train.xlsx'

# 模型保存路径
model_save = 'model/'

# 训练特征列索引（从1开始）
Train_feature_input_idx = [7,11,15,14]

# 输出标签列索引
Feature_output_idx = 2

# 选择模型编号和命名
model_selection = [3,7,10]
model_name = ['knn', 'catboost', 'tabnet']

# 数据清洗（可选）
if_need_clean_data = False
Clean_data_devide = [5.6, 12.3, 18.96, 25.61, 38.95]
Num_of_data_per_bin = 36
Randomseed = 42

# 预测数据路径
input_predict_data = 'predicting/Feature/'
output_predict_data = 'predicting/'
Predict_feature_input_idx = [4,8,12,11]
```
