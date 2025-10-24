import numpy as np
import pandas as pd
#model
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

import sklearn.linear_model as LM
from sklearn import svm
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.augmentations import RegressionSMOTE
from lce import LCERegressor

import torch
from torch import nn
from torch.utils import data

import os
import re
import argparse
import ast

import pickle

# For model config
SVR_C = 30; SVR_E = 0.06; SVR_G = 0.4 # 1-SVR
KNN_n_neighbors = 5; KNN_weights = 'uniform' # 3-KNN
GB_learning_rate=0.01; GB_n_estimators = 100 # 4-GB
RF_n_estimators = 100 # 5-RF
LGBM_metric = 'rmse' # 6-LGBM
CBR_iterations = 1000; CBR_learning_rate = 0.1; CBR_depth = 7; CBR_loss_function = 'RMSE'; CBR_verbose=100 # 7-CBR
ET_n_estimators = 100 # 8-ET
ABR_max_depth = 4; ABR_n_estimators = 500 # 9-ABR
Tab_max_epochs = 100; Tab_patience = 20; Tab_batch_size = 12; Tab_virtual_batch_size = 12; Tab_num_workers = 0; Tab_drop_last = False; Tab_aug_p = 0.2 # 10-TabNet
NN_batch_size = 12; NN_lr = 0.0005; NN_num_epochs = 1000 # 12-NN
RNN_batch_size = 12; RNN_lr = 0.0005; RNN_num_epochs = 1000 # 13-RNN

##--------------------- Auxiliary functions -----------------------##

def splitlabel(data,input_idx,output_idx):
    X = data[:,input_idx]
    y = data[:,output_idx]
    return X,y

def average_choose_data(data_x,data_y,dividearr,divide_n,aver_num,seed=None):
    datanew_x,datanew_y = np.zeros((0,data_x.shape[1])),np.zeros(0)
    for ii in range(divide_n):
        low,big = dividearr[ii],dividearr[ii+1]
        cond = np.where(np.logical_and(data_y>=low,data_y<=big))[0]
        tmpx,tmpy = data_x[cond,:],data_y[cond]
        if seed != None:
            np.random.seed(seed)
        verifyidx = np.random.choice(range(tmpy.shape[0]), size=aver_num, replace=False)
        datanew_x = np.vstack((datanew_x,tmpx[verifyidx,:]))
        datanew_y = np.hstack((datanew_y,tmpy[verifyidx]))

    return datanew_x,datanew_y

def raw2process(datax): # make sure the features should have the column: *,7v,*,*,*,19v,*,*,37h,37v
    datanew = np.zeros((datax.shape[0],6))
    # datanew[:,0] = datax[:,1] # 7v
    # datanew[:,1] = datax[:,5] # 19V
    # datanew[:,2] = datax[:,-1] # 37v
    # datanew[:,3] = (datax[:,-1]-datax[:,5])/(datax[:,-1]+datax[:,5]) #gr 37v/19v
    # datanew[:,4] = (datax[:,5]-datax[:,1])/(datax[:,5]+datax[:,1]) #gr 19v/7v
    # datanew[:,5] = (datax[:,-1]-datax[:,-2])/(datax[:,-1]+datax[:,-2]) #pr37

    datanew[:, 0] = datax[:, 0]  # 7v
    datanew[:, 1] = datax[:, 1]  # 19V
    datanew[:, 2] = datax[:, 2]  # 37v
    datanew[:, 3] = (datax[:, 2] - datax[:, 1]) / (datax[:, 2] + datax[:, 1])  # gr 37v/19v
    datanew[:, 4] = (datax[:, 1] - datax[:, 0]) / (datax[:, 1] + datax[:, 0])  # gr 19v/7v
    datanew[:, 5] = (datax[:, 2] - datax[:, 3]) / (datax[:, 2] + datax[:, 3])  # pr37
    return datanew

def normal(data_xraw,data_yraw):
    data_x = np.zeros_like(data_xraw)
    minmax = np.zeros((2,data_x.shape[1]+1))
    for jj in range(data_x.shape[1]):
    # for jj in range(3):
        minmax[0,jj],minmax[1,jj] = np.min(data_xraw[:,jj]),np.max(data_xraw[:,jj])
        data_x[:,jj] = 2*(data_xraw[:,jj]-minmax[0,jj])/(minmax[1,jj]-minmax[0,jj])-1

    minmax[0,-1],minmax[1,-1] = np.min(data_yraw),np.max(data_yraw)
    data_y = 2*(data_yraw-minmax[0,-1])/(minmax[1,-1]-minmax[0,-1])-1

    return data_x, data_y, minmax

def data_MSE(model,X_train,X_test,y_train,y_test):
    y_train_pred = model.predict(X_train)
    MSEtrain,R2train = mean_squared_error(y_train_pred,y_train),r2_score(y_train_pred,y_train)
    MAEtrain = mean_absolute_error(y_train_pred,y_train)
    biastrain = np.mean(y_train_pred-y_train)
    print('train-MSE ',MSEtrain,' r2 ',R2train,'MAE',MAEtrain,'bias',biastrain)
    y_test_pred = model.predict(X_test)
    MSEtest,R2test = mean_squared_error(y_test_pred,y_test),r2_score(y_test_pred,y_test)
    MAEtest = mean_absolute_error(y_test_pred,y_test)
    biastest = np.mean(y_test_pred-y_test)
    print('test-MSE ',MSEtest,' r2 ',R2test,'MAE',MAEtest,'bias',biastest)
    return [np.sqrt(MSEtest),R2test,MAEtest,biastest]

def keep_digits(s):
    return re.sub(r'\D', '', s)

def get_all_files(folder):
    all_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            all_files.append(keep_digits(file))
    return all_files

def normal_out_func(pre,normal_out):
    return (pre+1)/2*(normal_out[1]-normal_out[0])+normal_out[0]

def get_predict(model,datan,normalize_out):
    pre_1 = model.predict(datan)
    pre = normal_out_func(pre_1,normalize_out)
    pre[np.where(pre<0)[0]]=np.nan
    return pre.reshape((-1,1))

def process_predict(data_s,normalize_in,normalize_out,models):
    data_s2 = 2*(data_s-normalize_in[0,:])/(normalize_in[1,:]-normalize_in[0,:])-1
    all_results = np.zeros((data_s.shape[0],0))
    for model in models:
        all_results = np.hstack((all_results,get_predict(model,data_s2,normalize_out)))
    return all_results

##--------------------- Process functions -----------------------##

# Placeholder
# # For user
# input_data = None
#
# if_need_clean_data = False
# Clean_data_devide = [5.6, 12.3 , 18.96, 25.61, 38.95]
# Num_of_data_per_bin = 36
# Randomseed = 42
#
# # For system
# Feature_input_idx = [7,11,15,14]
# Feature_output_idx = 2


def data_process(input_data,Feature_input_idx,Feature_output_idx,
                 if_need_clean_data,Clean_data_devide,Num_of_data_per_bin,Randomseed):
    data_x_p1,data_y_p1 = splitlabel(input_data,np.array(Feature_input_idx)-1,np.array(Feature_output_idx)-1)
    if if_need_clean_data:
        Clean_data_devide_num = len(Clean_data_devide) - 1
        data_x_p1,data_y_p1 = average_choose_data(data_x_p1,data_y_p1,
                                                  Clean_data_devide,
                                                  Clean_data_devide_num,Num_of_data_per_bin,Randomseed) # clean data
    data_x_p1 = raw2process(data_x_p1) # combine the features
    data_x,data_y,minmax = normal(data_x_p1,data_y_p1) # normalization
    return data_x, data_y, minmax
def train_model(X_train,y_train,model_selection:list,models_config:list):
    if model_selection is [] or model_selection is None:
        model_selection = [0]
    models_function = []

    def model_func0(models_function):
        # 0--Multi-Linear Regression
        model0 = LM.LinearRegression()
        model0.fit(X_train, y_train)
        models_function.append(model0)
        return models_function
    def model_func1(models_function,model1_config):
        # 1--Support Vector Machine
        SVR_C, SVR_E, SVR_G = model1_config
        model1 = svm.SVR(C=SVR_C, epsilon=SVR_E, gamma=SVR_G)
        model1.fit(X_train, y_train)
        models_function.append(model1)
        return models_function
    def model_func2(models_function):
        # 2--XGBoost
        model2 = XGBRegressor()
        model2.fit(X_train, y_train)
        models_function.append(model2)
        return models_function
    def model_func3(models_function,model3_config):
        # 3--KNN
        KNN_n_neighbors, KNN_weights = model3_config
        model3 = KNeighborsRegressor(n_neighbors=KNN_n_neighbors, weights=KNN_weights)
        model3.fit(X_train, y_train)
        models_function.append(model3)
        return models_function
    def model_func4(models_function,model4_config):
        # 4--Gradient Boosting
        GB_learning_rate, GB_n_estimators = model4_config
        model4 = GradientBoostingRegressor(learning_rate=GB_learning_rate, n_estimators=GB_n_estimators)
        model4.fit(X_train, y_train)
        models_function.append(model4)
        return models_function
    def model_func5(models_function,model5_config):
        # 5--Random Forest
        RF_n_estimators = model5_config
        model5 = RandomForestRegressor(n_estimators=RF_n_estimators)
        model5.fit(X_train, y_train)
        models_function.append(model5)
        return models_function
    def model_func6(models_function,model6_config):
        # 6--LightGBM
        LGBM_metric = model6_config
        model6 = lgb.LGBMRegressor(metric=LGBM_metric)
        model6.fit(X_train, y_train)
        models_function.append(model6)
    def model_func7(models_function,model7_config):
        # 7--Catboost
        CBR_iterations, CBR_learning_rate, CBR_depth, CBR_loss_function, CBR_verbose = model7_config
        model7 = CatBoostRegressor(
            iterations=CBR_iterations,  # iteration
            learning_rate=CBR_learning_rate,  # lr
            depth=CBR_depth,  # depth
            loss_function=CBR_loss_function,  # loss function
            verbose=CBR_verbose
        )
        model7.fit(X_train, y_train)
        models_function.append(model7)
        return models_function
    def model_func8(models_function,model8_config):
        # 8--Extra Trees
        ET_n_estimators = model8_config
        model8 = ExtraTreesRegressor(n_estimators=ET_n_estimators)
        model8.fit(X_train, y_train)
        models_function.append(model8)
        return models_function
    def model_func9(models_function,model9_config):
        # 9--AdaBoost
        ABR_max_depth, ABR_n_estimators = model9_config
        model9 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=ABR_max_depth), n_estimators=ABR_n_estimators)
        model9.fit(X_train, y_train)
        models_function.append(model9)
        return models_function
    def model_func10(models_function,model10_config):
        # 10--TabNet
        Tab_max_epochs, Tab_patience, Tab_batch_size, Tab_virtual_batch_size, Tab_num_workers, Tab_drop_last, Tab_aug_p = model10_config
        model10 = TabNetRegressor()
        model10.fit(X_train, y_train.reshape(-1, 1),
                    max_epochs=Tab_max_epochs,
                    patience=Tab_patience,
                    batch_size=Tab_batch_size, virtual_batch_size=Tab_virtual_batch_size,
                    num_workers=Tab_num_workers,
                    drop_last=Tab_drop_last,
                    augmentations=RegressionSMOTE(p=Tab_aug_p),  # aug
                    )
        models_function.append(model10)
        return models_function
    def model_func11(models_function):
        # 11--LCE
        model11 = LCERegressor()
        model11.fit(X_train, y_train)
        models_function.append(model11)
        return models_function
    def model_func12_13(models_function,model12_13_config,is_NN=True):
        def np2torch(arr):
            arr1 = torch.from_numpy(arr)
            arr1 = arr1.float()
            return arr1
        def torch2np(arr):
            return arr.detach().numpy()
        def load_array(data_arrays, batch_size, is_train=True):
            dataset = data.TensorDataset(*data_arrays)
            return data.DataLoader(dataset, batch_size, shuffle=is_train)
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, std=0.01)
        class Network():
            def __init__(self, net):
                self.net = net
                self.net = self.net.eval()

            def predict(self, X):
                Xtorch = np2torch(X)
                Ytorch = self.net(Xtorch)
                return torch2np(Ytorch).reshape((-1))
        features, labels = np2torch(X_train), np2torch(np.reshape(y_train, (-1, 1)))
        loss = nn.MSELoss()

        if is_NN:
            # 12--NN
            NN_batch_size, NN_lr, NN_num_epochs = model12_13_config

            batch_size = NN_batch_size
            data_iter = load_array((features, labels), batch_size)

            net1 = nn.Sequential(
                nn.Linear(6, 64), nn.ReLU(), nn.Dropout(0.25),
                nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.25),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 1)
            )
            net1.apply(init_weights)
            trainer1 = torch.optim.Adam(net1.parameters(), lr=NN_lr)
            num_epochs = NN_num_epochs
            for epoch in range(num_epochs):
                net1 = net1.train()
                for X, y in data_iter:
                    trainer1.zero_grad()
                    l = loss(net1(X), y)
                    l.backward()
                    trainer1.step()
                train_l = loss(net1(features), labels)
                if (epoch + 1) % 100 == 0:
                    print(f'epoch {epoch + 1}, loss {train_l:f}')
            model12 = Network(net1)
            models_function.append(model12)

        else:
            # 13--RNN
            RNN_batch_size, RNN_lr, RNN_num_epochs = model12_13_config
            class ResNet(nn.Module):
                def __init__(self, in_features_dim, out_features_dim):
                    super(ResNet, self).__init__()
                    self.net2_1 = nn.Sequential(nn.Linear(in_features_dim, 64), nn.ReLU(), nn.Dropout(0.25))
                    self.net2_2 = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Dropout(0.25),
                                                nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.25),
                                                nn.Linear(256, 64), nn.ReLU())
                    self.net2_3 = nn.Sequential(nn.Linear(64, out_features_dim))

                def forward(self, x):
                    x1 = self.net2_1(x)
                    residual = x1
                    x2 = self.net2_2(x1)
                    x2 += residual
                    return self.net2_3(x2)

            batch_size = RNN_batch_size
            data_iter = load_array((features, labels), batch_size)
            net2 = ResNet(6, 1)
            net2.apply(init_weights)
            num_epochs = RNN_num_epochs
            trainer2 = torch.optim.Adam([{
                "params": net2[0].weight,
                'weight_decay': 1}, {
                "params": net2[0].bias
            }
            ], lr=RNN_lr)
            for epoch in range(num_epochs):
                net2 = net2.train()
                for X, y in data_iter:
                    trainer2.zero_grad()
                    l = loss(net2(X), y)
                    l.backward()
                    trainer2.step()
                train_l = loss(net2(features), labels)
                if (epoch + 1) % 100 == 0:
                    print(f'epoch {epoch + 1}, loss {train_l:f}')
            model13 = Network(net2)
            models_function.append(model13)
        return models_function

    for iter,model_idx in enumerate(model_selection):
        if model_idx == 0:
            models_function = model_func0(models_function)
        elif model_idx == 1:
            models_function = model_func1(models_function,models_config[iter])
        elif model_idx == 2:
            models_function = model_func2(models_function)
        elif model_idx == 3:
            models_function = model_func3(models_function,models_config[iter])
        elif model_idx == 4:
            models_function = model_func4(models_function,models_config[iter])
        elif model_idx == 5:
            models_function = model_func5(models_function,models_config[iter])
        elif model_idx == 6:
            models_function = model_func6(models_function,models_config[iter])
        elif model_idx == 7:
            models_function = model_func7(models_function,models_config[iter])
        elif model_idx == 8:
            models_function = model_func8(models_function,models_config[iter])
        elif model_idx == 9:
            models_function = model_func9(models_function,models_config[iter])
        elif model_idx == 10:
            models_function = model_func10(models_function,models_config[iter])
        elif model_idx == 11:
            models_function = model_func11(models_function)
        elif model_idx == 12:
            models_function = model_func12_13(models_function,models_config[iter],is_NN=True)
        elif model_idx == 13:
            models_function = model_func12_13(models_function,models_config[iter],is_NN=False)

    return models_function
def save_model(models_function,models_savepath,minmax_data,minmax_savepath):
    for idx in range(len(models_function)):
        with open(models_savepath[idx], 'wb') as file:
            pickle.dump(models_function[idx], file)
    df = pd.DataFrame(minmax_data)
    df.to_csv(minmax_savepath, index=False)
def load_model(models_savepath,minmax_path):
    models_function = []
    for idx in range(len(models_savepath)):
        with open(models_savepath[idx],'rb') as file:
            models_function.append(pickle.load(file))
    return models_function, np.array(pd.read_csv(minmax_path))
def predict(models_function,minmax,predict_input_idx,input_predict_data_folder_path,output_predict_data_folder_path,model_name,
            is_need_lonlat=True):
    filein = get_all_files(input_predict_data_folder_path)
    normalize_in = minmax[:, :-1]
    normalize_out = minmax[:, -1]
    for file in filein:
        datain = pd.read_csv(os.path.join(input_predict_data_folder_path, file + '.csv'))
        datain = np.array(datain)
        datain_x = datain[:, np.array(predict_input_idx)-1]
        datain_x = raw2process(datain_x)
        dataout_y = np.zeros((datain_x.shape[0], len(models_function)))

        tmp_not_nan_idx = []

        for tt in range(datain_x.shape[0]):
            data_line = datain_x[tt, :]
            if np.isnan(data_line).any():
                dataout_y[tt, :] = np.nan
            else:
                tmp_not_nan_idx.append(tt)
        print(file + ' Start predicting!')
        dataout_y[tmp_not_nan_idx, :] = process_predict(datain_x[tmp_not_nan_idx, :], normalize_in, normalize_out,
                                                        models_function)
        if not os.path.exists(output_predict_data_folder_path):
            os.makedirs(output_predict_data_folder_path)
        for tt in range(len(models_function)):
            if not os.path.exists(os.path.join(output_predict_data_folder_path, model_name[tt])):
                os.makedirs(os.path.join(output_predict_data_folder_path, model_name[tt]))
        if is_need_lonlat:
            dataout = np.hstack((datain[:, :2], dataout_y))
            print(file, ' ', dataout.shape)
            # save as CSV
            for tt in range(len(models_function)):
                df = pd.DataFrame(dataout[:, [0,1,tt + 2]])
                df.to_csv(os.path.join(output_predict_data_folder_path, model_name[tt], file + '.csv'), index=False)
        else:
            print(file, ' ', dataout_y.shape)
            # save as CSV
            for tt in range(len(models_function)):
                df = pd.DataFrame(dataout_y[:, tt].reshape((-1, 1)))
                df.to_csv(os.path.join(output_predict_data_folder_path, model_name[tt], file + '.csv'), index=False)
        print('------------------------------------------')


def model_name_to_path(model_name,model_folder_path):
    models_savepath = []
    for name in model_name:
        models_savepath.append(os.path.join(model_folder_path, name + '.pkl'))
    return models_savepath
def minmax_name_to_path(minmax_save_path):
    return os.path.join(minmax_save_path,'minmax.csv')


def parse_args():
    parser = argparse.ArgumentParser(description='Inputing parameters via txt config.')
    parser.add_argument('--train', action='store_true', help='Whether to train the model or not. If true, please input the model name to save; otherwise, please input existing model name to load when predict mode is on.')
    parser.add_argument('--predict', action='store_true', help='Whether to predict the model or not. If true, please open the train mode or input existing model name.')
    parser.add_argument('--config', type=str, help='Path to txt config file (key = value per line)')
    return parser.parse_args()
def parse_txt_config(path):
    cfg = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, val = line.split('=', 1)
            key, val = key.strip(), val.strip()
            # bool / int / float / list
            if val.lower() in ('true', 'false'):
                cfg[key] = val.lower() == 'true'
            else:
                try:
                    cfg[key] = ast.literal_eval(val)
                except (ValueError, SyntaxError):
                    cfg[key] = val
    return cfg

def main():
    args = parse_args()
    if args.train == False and args.predict == False:
        print('You don\'t choose any mode! The code will do nothing. :(')
        return
    cfg = parse_txt_config(args.config)
    def get(k, default=None):
        return cfg.get(k, default)

    if args.train == True:
        input_data_path = get('input_data')
        if input_data_path is None:
            print('Can\'t find input data path!')
            return
        elif not os.path.exists(input_data_path):
            print('Can\'t find input data path:', input_data_path)
            return
        else:
            print('Input data path:', input_data_path)
        model_save_folder_path = get('model_save')
        if model_save_folder_path is None:
            print('Can\'t find model save folder path!')
            return
        else:
            print('Model save folder path:', model_save_folder_path)
            if not os.path.exists(model_save_folder_path):
                os.makedirs(model_save_folder_path)
        Feature_input_idx = get('Train_feature_input_idx')
        if Feature_input_idx is None:
            Feature_input_idx = [7,11,15,14]
        Feature_output_idx = get('Train_feature_output_idx')
        if Feature_output_idx is None:
            Feature_output_idx = 2
        model_selection = get('model_selection')
        model_name = get('model_name')
        if model_selection is None:
            print('You don\'t select any model. The code will only do linear regression.')
            model_name = ['model_1']
        elif model_selection is not None and model_name is None:
            print('Can\'t find model name. The code will set model names as default.')
            model_name = []
            for i in range(len(model_selection)):
                model_name.append('model_' + str(i+1))
        elif len(model_selection) != len(model_name):
            print('Warning: The number of models selected is not equal to the number of model names. The model names will be set as default.')
            model_name_tmp = []
            for i in range(len(model_selection)):
                if i < len(model_name):
                    model_name_tmp.append(model_name[i])
                else:
                    model_name_tmp.append('model_' + str(i+1))
            model_name = model_name_tmp
        # model_config = get('model_config')
        # if model_config is None:
        #     print('You don\'t input the model config txt. Do you need to generate a template?')
        #     while True:
        #         user_input = input("Enter 'Y' to generate a template, or 'N' to continue without it: ").strip().lower()
        #         if user_input == 'y':
        #             generate_template()
        #             print('Template generated. Press any key if you finish modifying the template.')
        #             input()
        #             break
        #         elif user_input == 'n':
        #             print("Continuing without a template.")
        #             break
        #         else:
        #             print("Invalid input. Please enter 'Y' or 'N'.")
        if_need_clean_data = get('if_need_clean_data')
        if if_need_clean_data is None or if_need_clean_data == False:
            print('No need to clean data.')
            Clean_data_devide, Num_of_data_per_bin, Randomseed = None, None, None
            if_need_clean_data = False
        else:
            print('Need to clean data.')
            Clean_data_devide = get('Clean_data_devide')
            Num_of_data_per_bin = get('Num_of_data_per_bin')
            Randomseed = get('Randomseed')
            if Clean_data_devide is None: Clean_data_devide = [5.6, 12.3 , 18.96, 25.61, 38.95]
            if Num_of_data_per_bin is None: Num_of_data_per_bin = 36
            if Randomseed is None: Randomseed = 42
            if_need_clean_data = True

        print('--------------------------------')
        print('You will train the following models:')
        model_configs = []
        for iter, tmpidx in enumerate(model_selection):
            idx = int(tmpidx)
            if idx == 0 or idx == 2 or idx == 11:
                model_configs.append(None)
                if idx == 0:
                    print(f'Model No. {iter+1} - {model_name[iter]}: Linear Regression')
                elif idx == 2:
                    print(f'Model No. {iter+1} - {model_name[iter]}: XGBoost')
                elif idx == 11:
                    print(f'Model No. {iter+1} - {model_name[iter]}: LCE')
            elif idx == 1:
                model_configs.append([SVR_C, SVR_E, SVR_G])
                print(f'Model No. {iter+1} - {model_name[iter]}: Support Vector Regression')
            elif idx == 3:
                model_configs.append([KNN_n_neighbors, KNN_weights])
                print(f'Model No. {iter+1} - {model_name[iter]}: K-Nearest Neighbors')
            elif idx == 4:
                model_configs.append([GB_learning_rate, GB_n_estimators])
                print(f'Model No. {iter+1} - {model_name[iter]}: Gradient Boosting')
            elif idx == 5:
                model_configs.append([RF_n_estimators])
                print(f'Model No. {iter+1} - {model_name[iter]}: Random Forest')
            elif idx == 6:
                model_configs.append([LGBM_metric])
                print(f'Model No. {iter+1} - {model_name[iter]}: Light Gradient Boosting Machine')
            elif idx == 7:
                model_configs.append([CBR_iterations, CBR_learning_rate, CBR_depth, CBR_loss_function, CBR_verbose])
                print(f'Model No. {iter+1} - {model_name[iter]}: CatBoost')
            elif idx == 8:
                model_configs.append([ET_n_estimators])
                print(f'Model No. {iter+1} - {model_name[iter]}: Extra Trees')
            elif idx == 9:
                model_configs.append([ABR_max_depth, ABR_n_estimators])
                print(f'Model No. {iter+1} - {model_name[iter]}: AdaBoost')
            elif idx == 10:
                model_configs.append([Tab_max_epochs, Tab_patience, Tab_batch_size, Tab_virtual_batch_size, Tab_num_workers, Tab_drop_last, Tab_aug_p])
                print(f'Model No. {iter+1} - {model_name[iter]}: TabNet')
            elif idx == 12:
                model_configs.append([NN_batch_size, NN_lr, NN_num_epochs])
                print(f'Model No. {iter+1} - {model_name[iter]}: Neural Network')
            elif idx == 13:
                model_configs.append([RNN_batch_size, RNN_lr, RNN_num_epochs])
                print(f'Model No. {iter+1} - {model_name[iter]}: Residual Neural Network')
            else:
                print('Invalid model index:', tmpidx)
                return
        print('Start training models.')
        model_path = model_name_to_path(model_name, model_save_folder_path)
        minmax_path = minmax_name_to_path(model_save_folder_path)

        if '.xlsx' in input_data_path:
            input_data = np.array(pd.read_excel(input_data_path))
        elif '.csv' in input_data_path:
            input_data = np.array(pd.read_csv(input_data_path))
        else:
            print('Invalid input data format. Please input a valid .csv or .xlsx file.')
            return
        data_x,data_y,minmax = data_process(input_data,Feature_input_idx,Feature_output_idx,if_need_clean_data,Clean_data_devide,Num_of_data_per_bin,Randomseed)
        models_function = train_model(data_x,data_y,model_selection,model_configs)
        save_model(models_function,model_path,minmax,minmax_path)
        print('Training models done.')

        if args.predict == True:
            input_predict_data_folder_path = get('input_predict_data')
            if input_predict_data_folder_path is None:
                print('Can\'t find input predict data folder path!')
                return
            else:
                print('Input predict data folder path:', input_predict_data_folder_path)
            output_predict_data_folder_path = get('output_predict_data')
            if output_predict_data_folder_path is None:
                print('Can\'t find output predict data folder path!')
                return
            else:
                print('Output predict data folder path:', output_predict_data_folder_path)
            predict_input_idx = get('Predict_feature_input_idx')
            if predict_input_idx is None:
                predict_input_idx = [4,8,12,11]
            is_need_lonlat = get('is_need_lonlat')
            if is_need_lonlat is None:
                is_need_lonlat = True

            print('Start predicting.')
            predict(models_function,minmax,predict_input_idx,input_predict_data_folder_path,output_predict_data_folder_path,model_name,is_need_lonlat)
            print('Predicting done.')
    elif args.predict == True:
        input_predict_data_folder_path = get('input_predict_data')
        if input_predict_data_folder_path is None:
            print('Can\'t find input predict data folder path!')
            return
        else:
            print('Input predict data folder path:', input_predict_data_folder_path)
        output_predict_data_folder_path = get('output_predict_data')
        if output_predict_data_folder_path is None:
            print('Can\'t find output predict data folder path!')
            return
        else:
            print('Output predict data folder path:', output_predict_data_folder_path)
        predict_input_idx = get('Predict_feature_input_idx')
        if predict_input_idx is None:
            predict_input_idx = [4, 8, 12, 11]
        model_save_folder_path = get('model_save')
        if model_save_folder_path is None:
            print('Can\'t find model save folder path!')
            return
        else:
            print('Model save folder path:', model_save_folder_path)
        model_name = get('model_name')
        if model_name is None:
            print('Can\'t find model name!')
            return
        else:
            print('Model name:', model_name)
        model_path = model_name_to_path(model_name, model_save_folder_path)
        minmax_path = minmax_name_to_path(model_save_folder_path)
        for single_path in model_path:
            if not os.path.exists(single_path):
                print('Can\'t find model:', single_path)
                return
        if not os.path.exists(minmax_path):
            print('Can\'t find system file (minmax). Please train the model first or input the right model save folder path.')
            return
        is_need_lonlat = get('is_need_lonlat')
        if is_need_lonlat is None:
            is_need_lonlat = True


        models_function, minmax = load_model(model_path, minmax_path)
        print('Start predicting.')
        predict(models_function, minmax, predict_input_idx, input_predict_data_folder_path, output_predict_data_folder_path, model_name, is_need_lonlat)
        print('Predicting done.')

if __name__ == '__main__':
    main()








