# IceBird ML Toolkit

一个基于conda环境的自动化机器学习工具箱，用于训练模型并批量预测IceBird数据。无需编写代码，通过配置文件即可运行。

## 快速开始

### 1. 环境准备
创建并激活conda环境
```bash
conda create -n icebird python=3.8
conda activate icebird

```
安装依赖（假设requirement.txt已存在）

```bash
pip install -r requirement.txt
```

### 2.准备配置文件
将整个代码库下载到本地，或者通过自定义的方式，新建文件夹，下载`core_process.py`和`user_config.txt`到本地文件夹中。根据需求修改`user_config.txt`文件。

```ini
# 训练数据选项
input_data = 'training/ML_IceBird_Train.xlsx'
model_save = 'model/'
Train_feature_input_idx = [7,11,15,14]
Feature_output_idx = 2

# 模型选择选项
model_selection = [3,7,10] 
model_name = ['knn', 'catboost', 'tabnet']

# 数据清洗
if_need_clean_data = False 
Clean_data_devide = [5.6, 12.3, 18.96, 25.61, 38.95] 
Num_of_data_per_bin = 36 
Randomseed = 42 

# 预测数据选项
input_predict_data = 'predicting/Feature/'
output_predict_data = 'predicting/'
Predict_feature_input_idx = [4,8,12,11]
is_need_lonlat = True
```

### 3.准备数据
训练数据请整合为一个xlsx文件或csv文件，如本研究案例`training/ML_IceBird_Train.xlsx`所示。其中要求至少包括1列训练标签和4列训练特征（7V,19V,37V,37H）。

预测数据请放在`input_predict_data`所给出的文件夹中，要求至少含有2列地理坐标系（必须放置在前两列）和4列预测特征（7V,19V,37V,37H），支持多文件，但所有文件特征列的存放顺序必须一致。输出数据和输入数据的文件名称一一对应。预测文件只支持csv格式。

### 4.训练模型
在拥有依赖的conda环境下，训练并保存模型：

```bash
python core_process.py --train --config user_config.txt
```

运行后：
- 模型文件保存在 `model/` 目录，采用`model_name`作为各自的命名
- 自动生成 `model/minmax.csv`（数据归一化参数）作为中间数据，无需操作

### 5.批量预测
使用已训练模型进行预测
```bash
python core_process.py --predict --config user_config.txt
```

或同时训练和预测：
```bash
python core_process.py --train --predict --config user_config.txt
```

预测结果将按`model_name`模型名称分别保存在 `output_predict_data/` 各子目录中。

### 配置参数说明
| 参数 | 说明 | 示例 | 默认 |
|------|------|------|------|
| `input_data` | 训练数据文件路径（支持.xlsx或.csv） | `'training/data.xlsx'` | 如果`--train`模式开启，那么该数据强制输入，并要求该文件夹存在 |
| `model_save` | 模型保存路径 | `'model/'` | 该数据强制输入；如果`--predict`模式开启，那么要求该文件夹存在 |
| `Train_feature_input_idx` | 特征在训练数据中的列索引（从1开始），先后顺序分别为7V,19V,37V,37H | `[7,11,15,14]` | `[7,11,15,14]` | 
| `Feature_output_idx` | 真实数据标签在训练数据中的列索引（从1开始） | `2` | `2` |
|------|------|------|------|
| `model_selection` | 模型编号列表，参考下文表格 | `[3,7,10]` | `[0]` |
| `model_name` | 自定义模型名称；该名称将作为后续输出文件所在子文件夹的名称，便于您识别 | `['knn','catboost','tabnet']` | `['model_1', ..., 'model_i']` |
|------|------|------|------|
| `if_need_clean_data` | 是否按标签值分箱采样；如果为False，那么后三个数据将不起作用 | `False` | `False` |
| `Clean_data_devide` | 选择分割直方图的边界 | `[5.6, 12.3, 18.96, 25.61, 38.95]` | `[5.6, 12.3, 18.96, 25.61, 38.95]` |
| `Num_of_data_per_bin` | 每个区间内经过均衡后剩余的最大数据数目 | `36` | `36` |
| `Randomseed` | 固定随机种子 | `42` | 如果删去该数据段或输入None，那么随机挑选随机种子 |
|------|------|------|------|
| `input_predict_data` | 预测数据特征文件保存路径 | `'predicting/Feature/'` | 如果`--predict`模式开启，那么该数据强制输入，并要求该文件夹存在 |
| `output_predict_data` | 预测雪深结果输出路径 | `'predicting/'` | 如果`--predict`模式开启，那么该数据强制输入，并要求该文件夹存在 |
| `Predict_feature_input_idx` | 特征在预测数据中的列索引（从1开始），先后顺序分别为7V,19V,37V,37H | `[4,8,12,11]` | `[4,8,12,11]` |
| `is_need_lonlat` | 预测结果是否保留经纬度 | `True` | `True` |

### 支持模型列表
| 编号 | 模型名称 |
|------|----------|
| 0 | 线性回归 |
| 1 | 支持向量机 |
| 2 | XGBoost |
| 3 | K最近邻 |
| 4 | 梯度提升 |
| 5 | 随机森林 |
| 6 | LightGBM |
| 7 | CatBoost |
| 8 | ExtraTrees |
| 9 | AdaBoost |
| 10 | TabNet |
| 11 | LCE |
| 12 | 神经网络 |
| 13 | 残差网络 |

### 典型工作流程
1. **准备数据**：确保训练数据和预测数据格式一致（列对应）
2. **配置模型**：在user_config.txt中选择1-3个模型
3. **训练**：运行--train命令或同时进行预测与训练
4. **预测**：运行--predict命令

### 注意事项
- 特征列索引从1开始计数
- 预测数据文件夹内的文件需为.csv格式
- 模型名称将作为模型文件和预测结果子文件夹名称
- 临时文件minmax.csv不可删除，否则无法预测
