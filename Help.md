# ML Toolkit: Result from AMSR2 and IceBird

An automated machine learning toolkit based on conda environment for training models and batch predicting IceBird data. No coding required, just run with configuration files.

## Quick Start

### 1. Environment Preparation

Create and activate conda environment:
```bash
conda create -n icebird python=3.8
conda activate icebird

```
Install dependencies:

```bash
pip install -r requirement.txt
```

### 2.Prepare Configuration File
Download the entire codebase locally, or create a new folder in a custom way, download `core_process.py` and `user_config.txt` to your local folder. Modify the `user_config.txt` file according to your needs.

```ini
# Training Data Options
input_data = 'training/ML_IceBird_Train.xlsx'
model_save = 'model/'
Train_feature_input_idx = [7,11,15,14]
Feature_output_idx = 2

# Model Selection Options
model_selection = [3,7,10] 
model_name = ['knn', 'catboost', 'tabnet']

# Data Cleaning
if_need_clean_data = False 
Clean_data_devide = [5.6, 12.3, 18.96, 25.61, 38.95] 
Num_of_data_per_bin = 36 
Randomseed = 42 

# Prediction Data Options
input_predict_data = 'predicting/Feature/'
output_predict_data = 'predicting/'
Predict_feature_input_idx = [4,8,12,11]
is_need_lonlat = True
```

### 3.Prepare Data
Please consolidate training data into a single `xlsx` or `csv` file, as shown in the case study `training/ML_IceBird_Train.xlsx`. It must include at least 1 training label column and 4 training feature columns (7V, 19V, 37V, 37H).

Place prediction data in the folder specified by `input_predict_data`. It must contain at least 2 geographic coordinate columns (must be placed in the first two columns) and 4 prediction feature columns (7V, 19V, 37V, 37H). Multiple files are supported, but the order of feature columns must be consistent across all files. Output and input file names correspond one-to-one. Prediction files only support `csv` format. As an example, refer to the files in `predicting/Feature`. To use the complete Spring 2024 dataset, download the CSV files in `Feature/` hosted on [Hugging Face]:

(https://huggingface.co/datasets/RTC0Ritchie/snow_depth_ML)

### 4.Train Model
In the conda environment with dependencies installed, train and save the model:

```bash
python core_process.py --train --config user_config.txt
```

After running:
- Model files are saved in the `model/` directory, using `model_name` as their respective names
- Automatically generates `model/minmax.csv` (data normalization parameters) as intermediate data, no action required

### 5.Batch Prediction
Use trained models for prediction
```bash
python core_process.py --predict --config user_config.txt
```

Or train and predict simultaneously:
```bash
python core_process.py --train --predict --config user_config.txt
```

Prediction results will be saved in respective subdirectories under `output_predict_data/` according to the `model_name`.

### Configuration Parameter Description
| Parameter | Description | Example | Default |
|------|------|------|------|
| `input_data` | Training data file path (supports .xlsx or .csv) | `'training/data.xlsx'` | If `--train` mode is enabled, this data is mandatory and the file must exist |
| `model_save` | Model save path | `'model/'` | This data is mandatory; if `--predict` mode is enabled, the folder must exist |
| `Train_feature_input_idx` | Column index of features in training data (1-indexed), order: 7V, 19V, 37V, 37H | `[7,11,15,14]` | `[7,11,15,14]` | 
| `Feature_output_idx` | Column index of ground truth labels in training data (1-indexed) | `2` | `2` |
|------|------|------|------|
| `model_selection` | Model index list, refer to table below | `[3,7,10]` | `[0]` |
| `model_name` | Custom model name; this name will be used as subdirectory name for output files for easy identification | `['knn','catboost','tabnet']` | `['model_1', ..., 'model_i']` |
|------|------|------|------|
| `if_need_clean_data` | Whether to sample by binning label values; if False, the next three parameters have no effect | `False` | `False` |
| `Clean_data_devide` | Histogram split boundaries | `[5.6, 12.3, 18.96, 25.61, 38.95]` | `[5.6, 12.3, 18.96, 25.61, 38.95]` |
| `Num_of_data_per_bin` | Maximum number of data points remaining per bin after balancing | `36` | `36` |
| `Randomseed` | Fixed random seed | `42` | If this parameter is removed or set to None, a random seed will be selected randomly |
|------|------|------|------|
| `input_predict_data` | Prediction data feature files folder path | `'predicting/Feature/'` | If `--predict` mode is enabled, this data is mandatory and the folder must exist |
| `output_predict_data` | Predicted snow depth output path | `'predicting/'` | If `--predict` mode is enabled, this data is mandatory and the folder must exist |
| `Predict_feature_input_idx` | Column index of features in prediction data (1-indexed), order: 7V, 19V, 37V, 37H | `[4,8,12,11]` | `[4,8,12,11]` |
| `is_need_lonlat` | Whether to retain longitude/latitude in prediction results | `True` | `True` |

### Supported Models List
| Index | Model Name |
|------|----------|
| 0 | Linear Regression |
| 1 | Support Vector Machine |
| 2 | XGBoost |
| 3 | K-Nearest Neighbors |
| 4 | Gradient Boosting |
| 5 | Random Forest |
| 6 | LightGBM |
| 7 | CatBoost |
| 8 | ExtraTrees |
| 9 | AdaBoost |
| 10 | TabNet |
| 11 | LCE |
| 12 | Neural Network |
| 13 | ResNet |

### Typical Workflow

1. **Prepare data**: Ensure training and prediction data have consistent format (column correspondence)
2. **Configure models**: Select models in `user_config.txt` and do other setup
3. **Train**: Run `--train` command or train and predict simultaneously
4. **Predict**: Run `--predict` command

### Notes
- Feature column indices are 1-indexed
- Files in prediction data folder must be in `.csv` format
- Model names will be used for model files and prediction result subfolder names
- Temporary file `minmax.csv` must not be deleted, otherwise prediction will fail
