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
将整个代码库下载到本地，或者通过自定义的方式，新建文件夹，下载`core_process.py`和`user_config.txt`到本地文件夹中。根据需求修改`user_config.txt`文件。

```ini
# 训练数据路径（支持.xlsx或.csv）
input_data = 'training/ML_IceBird_Train.xlsx'

# 模型保存路径
model_save = 'model/'

# 特征在训练数据中的列索引（从1开始），先后顺序分别为7V,19V,37V,37H
Train_feature_input_idx = [7,11,15,14]

# 输出标签列索引
Feature_output_idx = 2

# 选择模型编号和命名
model_selection = [3,7,10] #模型编号，参考下文表格
model_name = ['knn', 'catboost', 'tabnet'] #模型命名，可自定义；该名称将作为后续输出文件所在子文件夹的名称，便于您识别

# 数据清洗（可选）
if_need_clean_data = False #是否需要采用直方图均衡方法以清洗数据；如果为False，那么后三个数据将不起作用
Clean_data_devide = [5.6, 12.3, 18.96, 25.61, 38.95] #选择分割直方图的边界；默认为[5.6, 12.3, 18.96, 25.61, 38.95]
Num_of_data_per_bin = 36 #每个直方图区间内经过均衡后剩余的最大数据数目；默认为36
Randomseed = 42 #固定随机种子，便于复现结果；如果删去此输入，则随机生成随机种子

# 预测数据路径
input_predict_data = 'predicting/Feature/' #预测数据特征文件保存路径
output_predict_data = 'predicting/' #预测雪深结果输出路径
Predict_feature_input_idx = [4,8,12,11] #特征在预测数据中的列索引（从1开始），先后顺序分别为7V,19V,37V,37H
```

### 训练模型
在拥有依赖的conda环境下，输入：

```bash
# 训练并保存模型
python core_process.py --train --config user_config.txt
```

运行后：
- 模型文件保存在 `model/` 目录，采用`model_name`作为各自的命名
- 自动生成 `model/minmax.csv`（数据归一化参数）作为中间数据，无需操作

### 批量预测

```bash
# 使用已训练模型进行预测
python core_process.py --predict --config user_config.txt
```

或同时训练和预测：

```bash
python core_process.py --train --predict --config user_config.txt
```

预测结果将按`model_name`模型名称分别保存在 `output_predict_data/` 各子目录中。
