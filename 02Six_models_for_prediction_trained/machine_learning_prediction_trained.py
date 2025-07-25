import time
import numpy as np
import pandas as pd
import os
import joblib


def fusarium_machine_learning(Orthogroups_tsv_file, pheno, model_name):
    x_validation = pd.read_csv(Orthogroups_tsv_file,
                               header=0,
                               index_col=0,
                               sep='\t')
    x_validation = x_validation.T
    x_validation_index = x_validation.index


    start_model = time.time()
    model = joblib.load("./save_model/" + model_name + "_" + pheno + ".pkl")
    validation_output = model.predict(x_validation)
    for i in range(len(validation_output)):
        if validation_output[i] < 0:
            validation_output[i] = 0
    validation_output = pd.DataFrame(data=validation_output,
                                     columns=[pheno],
                                     index=x_validation_index)
    print("validation_output:", validation_output)
    end_model = time.time()
    print('machine_learning Running time: %s Seconds' % (end_model - start_model))
    return validation_output



if __name__ == '__main__':
    #"wheat_stem", "wheat_head", "Maize_stem", "Soybean_stem"
    pheno = "wheat_stem"
    Orthogroups_tsv_file = "data/fusariumAllSamples/233species_Orthogroups.GeneCount转置.tsv"
    # "GradientBoostingRegressor", "RandomForestRegressor", "Ridge", "SVR"
    model_name = "GradientBoostingRegressor"

    print("pheno:", pheno)
    fusarium_machine_learning(Orthogroups_tsv_file, pheno, model_name)