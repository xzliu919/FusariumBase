import time
import random
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from permetrics.regression import RegressionMetric
import joblib




def machine_learning_prediction(pheno, model):
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
    y = y.loc[:, pheno]


    assert x.index.all() == y.index.all()


    x = np.array(x)
    x_validation = np.array(x_validation)
    y = np.array(y)


    pearns = []
    NRMSES = []

    n_splits = 10
    kf = KFold(n_splits= n_splits, shuffle=True, random_state=0)
    start_model = time.time()
    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]



        model.fit(X_train, y_train,)

        y_pre = model.predict(X_test)

        for i in range(len(y_pre)):
            if y_pre[i] < 0:
                y_pre[i] = 0

        pearn, p = stats.pearsonr(y_pre, y_test)
        pearns.append(pearn)

        evaluator = RegressionMetric(np.array(y_test), np.array(y_pre))
        NRMSE = evaluator.NRMSE(multi_output="raw_values")
        NRMSES.append(NRMSE)


        end_model = time.time()

    print("np.mean(pearns):", np.mean(pearns))
    print("pearns:", pearns)

    print("np.std(pearns):", np.std(pearns))

    print("np.mean(NRMSES):", np.mean(NRMSES))
    print("NRMSES:", NRMSES)
    print("np.std(NRMSES):", np.std(NRMSES))


    print('Model Running time: %s Seconds' % (end_model - start_model))



    #validation
    model.fit(x, y)

    y_pre = model.predict(x_validation)

    model_name = model.__class__.__name__
    joblib.dump(model, "./save_model/" + model_name + "_" + pheno + ".pkl")



    # importances = model.feature_importances_
    # indices = np.argsort(importances)[::-1]
    # for f in range(X_train.shape[1]):
    #     print("%2d) %-*s %f" % (f + 1, 30, index_name[indices[f]], importances[indices[f]]))

    for i in range(len(y_pre)):
        if y_pre[i] < 0:
            y_pre[i] = 0
    # print(y_pre)
    y_pre = pd.DataFrame(y_pre, index= index_name, columns= [pheno])
    return y_pre

if __name__ == '__main__':
    phenos = ["wheat_stem", "wheat_head", "Maize_stem", "Soybean_stem", ]
    models = [Ridge(random_state=0),
              SVR(kernel='rbf'),
              RandomForestRegressor(random_state= 0),
              GradientBoostingRegressor(random_state= 0)]

    for model in models:
        y_pres = pd.DataFrame([])
        model_name = model.__class__.__name__
        for pheno in phenos:
            y_pre = machine_learning_prediction(pheno, model)
            if y_pres.empty:
                y_pres = y_pre
            else:
                y_pres = pd.merge(y_pres, y_pre, how= 'inner',
                                  left_index= True, right_index= True)
        print(y_pres)

        # y_pres.to_csv("data/fusariumAllSamples/233Fusarium_infecting_plants_phenotype_" +
        #               model_name + "_" +
        #               pheno +
        #               ".csv")


