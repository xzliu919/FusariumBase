from PNNGS_prediction_trained import fusarium_PNNGS
from ResGS_prediction_trained import fusarium_ResGS
from machine_learning_prediction_trained import fusarium_machine_learning
import argparse

if __name__ == '__main__':
    #"wheat_stem", "wheat_head", "Maize_stem", "Soybean_stem"
    pheno = "Maize_stem" #以上四个选一个
    Orthogroups_tsv_file = "input_OG_group.txt"
    # "PNNGS", "ResGS" "GradientBoostingRegressor", "RandomForestRegressor", "Ridge", "SVR"
    model_name = "RandomForestRegressor" #以上六个选一个

    print("pheno:", pheno)
    if model_name == "PNNGS":
        output = fusarium_PNNGS(Orthogroups_tsv_file, pheno)
    elif model_name == "ResGS":
        output = fusarium_PNNGS(Orthogroups_tsv_file, pheno)
        fusarium_ResGS(Orthogroups_tsv_file, pheno)
    else:
        output = fusarium_machine_learning(Orthogroups_tsv_file, pheno, model_name)
    print("output: ", output)