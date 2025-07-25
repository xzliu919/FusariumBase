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
import torch
from model.PNNGS import PNNGS



def fusarium_PNNGS(pheno):
    # y = pd.read_csv("data/fusarium/97Fusarium_infecting_plants_phenotype.csv",
    #                 header=0,
    #                 index_col=0)
    # x = pd.read_csv("data/fusarium/97species_Orthogroups.GeneCount转置.tsv",
    #                       header=0,
    #                       index_col=0,
    #                       sep= '\t')
    # x_validation = pd.read_csv("data/fusarium/5species_Orthogroups.GeneCount转置.tsv",
    #                      header=0,
    #                      index_col=0,
    #                       sep= '\t')

    y = pd.read_csv("data/fusariumAllSamples/102Fusarium_infecting_plants_phenotype.csv",
                    header=0,
                    index_col=0)
    x = pd.read_csv("data/fusariumAllSamples/102species_Orthogroups.GeneCount转置.tsv",
                    header=0,
                    index_col=0,
                    sep='\t')
    x_validation = pd.read_csv("data/fusariumAllSamples/233species_Orthogroups.GeneCount转置.tsv",
                               header=0,
                               index_col=0,
                               sep='\t')

    x = x.T
    x_validation = x_validation.T
    index_name = x_validation.index

    y = y[[pheno]]

    CUDA_VISIBLE_DEVICES = '0'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("device:", device)

    print("torch.cuda.device_count():", torch.cuda.device_count())
    print("torch.cuda.current_device():", torch.cuda.current_device())

    parallel_number = 3
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

    epoch = 30
    print("epoch:", epoch)
    start_model = time.time()
    data_file = pd.merge(y, x, how='inner',
                         left_index=True, right_index=True)

    save_path = './save_model/PNNGS_' + pheno + '.pth'
    average_pearns, test_pearns_list = PNNGS(pheno,
                                             data_file,
                                             x_validation,
                                             n_splits=10,
                                             epoch=epoch,
                                             CUDA_VISIBLE_DEVICES=CUDA_VISIBLE_DEVICES,
                                             parallel_number=parallel_number,
                                             Batch_Size=16,
                                             save_path=save_path)
    end_model = time.time()
    print('PNNGS Running time: %s Seconds' % (end_model - start_model))


if __name__ == '__main__':
    phenos = ["wheat_stem", "wheat_head", "Maize_stem", "Soybean_stem", ]
    for pheno in phenos:
        print("pheno:", pheno)
        fusarium_PNNGS(pheno)