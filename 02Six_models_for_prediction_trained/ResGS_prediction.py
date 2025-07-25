import time
import pandas as pd
import os
import tensorflow as tf
from model.ResGS import ResGSWithTraditionalModel

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings('ignore')

def fusarium_ResGS(pheno):
    y = pd.read_csv("data/fusarium/97Fusarium_infecting_plants_phenotype.csv",
                    header=0,
                    index_col=0)
    x = pd.read_csv("data/fusarium/97species_Orthogroups.GeneCount转置.tsv",
                          header=0,
                          index_col=0,
                          sep= '\t')
    x_validation = pd.read_csv("data/fusarium/5species_Orthogroups.GeneCount转置.tsv",
                         header=0,
                         index_col=0,
                          sep= '\t')



    x = x.T
    x_validation = x_validation.T

    y = y[[pheno]]


    CUDA_VISIBLE_DEVICES = '0'
    print("GPU 可用:", tf.config.list_physical_devices('GPU'))


    epoch = 200  # 30
    print("epoch:", epoch)
    start_model = time.time()
    data_file = pd.merge(y, x, how='inner',
                      left_index=True, right_index=True)

    save_path = './save_model/PNNGS' + pheno + '.pth'
    ResGSWithTraditionalModel(pheno= pheno,
                              data_file = data_file,
                              x_validation = x_validation,
                              saveFileName = save_path,
                              CUDA_VISIBLE_DEVICES=CUDA_VISIBLE_DEVICES,
                              Epoch = epoch,
                              repeatTimes=1)


    end_model = time.time()
    print('PNNGS Running time: %s Seconds' % (end_model - start_model))



if __name__ == '__main__':
    phenos = ["wheat_stem", "wheat_head", "Maize_stem", "Soybean_stem", ]
    for pheno in phenos:
        print("pheno:", pheno)
        fusarium_ResGS(pheno)