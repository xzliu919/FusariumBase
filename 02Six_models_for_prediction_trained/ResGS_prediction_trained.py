import time
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import joblib


def fusarium_ResGS(Orthogroups_tsv_file, pheno):
    x_validation = pd.read_csv(Orthogroups_tsv_file,
                               header=0,
                               index_col=0,
                               sep='\t')
    x_validation = x_validation.T
    x_validation_index = x_validation.index

    print("GPU 可用:", tf.config.list_physical_devices('GPU'))


    start_model = time.time()
    TraditionalModel = joblib.load("./save_model/ResGS_traditional_" + pheno + ".pkl")

    if os.path.exists("./save_model/ResGS_residual_" + pheno + ".h5"):
        residualModel = tf.keras.models.load_model("./save_model/ResGS_residual_" + pheno + ".h5")

        x_validation = np.array(x_validation)
        y_TraditionalModel = TraditionalModel.predict(x_validation)

        x_validation = np.expand_dims(x_validation, axis=2)
        y_ResGS = residualModel.predict(x_validation)[:, 0]

        validation_output = y_TraditionalModel + y_ResGS
    else:
        y_TraditionalModel = TraditionalModel.predict(x_validation)
        validation_output = y_TraditionalModel

    for i in range(len(validation_output)):
        if validation_output[i] < 0:
            validation_output[i] = 0
    validation_output = pd.DataFrame(data=validation_output,
                                     columns=[pheno],
                                     index=x_validation_index)
    print("validation_output:", validation_output)
    end_model = time.time()
    print('ResGS Running time: %s Seconds' % (end_model - start_model))
    return validation_output



if __name__ == '__main__':
    #"wheat_stem", "wheat_head", "Maize_stem", "Soybean_stem"
    pheno = "wheat_stem"
    Orthogroups_tsv_file = "data/fusariumAllSamples/233species_Orthogroups.GeneCount转置.tsv"

    print("pheno:", pheno)
    fusarium_ResGS(Orthogroups_tsv_file, pheno)