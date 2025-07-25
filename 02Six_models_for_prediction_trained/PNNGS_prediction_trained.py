import time

import random
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from sklearn import decomposition, datasets
from permetrics.regression import RegressionMetric
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 禁用 GPU
import torch
from model.PNNGS import PNNGS


def fusarium_PNNGS(Orthogroups_tsv_file, pheno):
    x_validation = pd.read_csv(Orthogroups_tsv_file,
                               header=0,
                               index_col=0,
                               sep='\t')
    x_validation = x_validation.T


    #CUDA_VISIBLE_DEVICES = '0'
    CUDA_VISIBLE_DEVICES = '-1'
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device('cpu')
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("device:", device)

    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES



    start_model = time.time()
    best_net = torch.load('./save_model/PNNGS_' + pheno + '.pth',  map_location=torch.device('cpu'), weights_only=False)
    if isinstance(best_net, torch.nn.DataParallel):
        best_net = best_net.module  # 去掉 DataParallel 包装层
    best_net = best_net.to(device)  # 确保模型在 CPU 上
    best_net.eval()  # 设置为评估模式
    best_intercept = np.loadtxt('./save_model/PNNGS_best_intercept_' + pheno + '.txt', dtype= float)

    x_validation_index = x_validation.index
    x_validation = np.array(x_validation)
    x_validation = torch.from_numpy(x_validation.astype(np.float32))
    x_validation = x_validation.to(device)
    x_validation = torch.unsqueeze(x_validation, 1)
    validation_output = best_net(x_validation)
    validation_output = validation_output.cpu().detach().numpy().squeeze()
    validation_output = validation_output + best_intercept
    validation_output = np.maximum(validation_output, 0)  # 替换循环

    # 输出结果
    validation_df = pd.DataFrame(
        data=validation_output,
        columns=[pheno],
        index=x_validation_index)
    print("validation_output:", validation_df)
    end_model = time.time()
    print('PNNGS Running time: %s Seconds' % (end_model - start_model))
    return validation_df



if __name__ == '__main__':
    #"wheat_stem", "wheat_head", "Maize_stem", "Soybean_stem"
    pheno = "wheat_stem"
    Orthogroups_tsv_file = "data/fusariumAllSamples/233species_Orthogroups.GeneCount转置.tsv"

    print("pheno:", pheno)
    fusarium_PNNGS(Orthogroups_tsv_file, pheno)