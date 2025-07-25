#ResGS predicts phenotypic residuals
#tensorflow-gpu 2.4

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model

from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, StratifiedKFold

from model.dataloader import dataset_loader

# from DNNGP import DNNGP, DNNGP_training_set

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings('ignore')


def Conv1d_BN(x, nb_filter, kernel_size, strides=1):
    x = layers.Convolution1D(nb_filter, kernel_size, padding='same', strides=strides, activation='relu')(x)
    x = layers.BatchNormalization(axis=1)(x)
    return x

def Res_Block(inpt,nb_filter,kernel_size,strides=1):
    x = Conv1d_BN(inpt,nb_filter=nb_filter,kernel_size=kernel_size,strides=strides)
    x = layers.add([x,inpt])
    return x

def ResGSModel(inputs):
    nFilter = 64
    _KERNEL_SIZE = 3
    CHANNEL_FACTOR1 = 4
    CHANNEL_FACTOR2 = 1.1
    # print("inputs.shape:", inputs.shape)

    x1 = Res_Block(inputs, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x1 = Res_Block(x1, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    nFilter1 = int(nFilter * CHANNEL_FACTOR1)

    x2 = Conv1d_BN(x1 , nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2)
    nFilter = int(nFilter * CHANNEL_FACTOR2)
    x2 = Conv1d_BN(x2, nb_filter=nFilter, kernel_size=1, strides=1)
    x2 = Res_Block(x2, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x2 = Res_Block(x2, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)

    x3 = Conv1d_BN(x2 , nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2)
    nFilter = int(nFilter * CHANNEL_FACTOR2)
    x3 = Conv1d_BN(x3, nb_filter=nFilter, kernel_size=1, strides=1)
    x3 = Res_Block(x3, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x3 = Res_Block(x3, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)


    x4 = Conv1d_BN(x3 , nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2)
    nFilter = int(nFilter * CHANNEL_FACTOR2)
    x4 = Conv1d_BN(x4, nb_filter=nFilter, kernel_size=1, strides=1)
    x4 = Res_Block(x4, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x4 = Res_Block(x4, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)


    x5 = Conv1d_BN(x4 , nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2)
    nFilter = int(nFilter * CHANNEL_FACTOR2)
    x5 = Conv1d_BN(x5, nb_filter=nFilter, kernel_size=1, strides=1)
    x5 = Res_Block(x5, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x5 = Res_Block(x5, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)

    x6 = Conv1d_BN(x5 , nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2)
    nFilter = int(nFilter * CHANNEL_FACTOR2)
    x6 = Conv1d_BN(x6, nb_filter=nFilter, kernel_size=1, strides=1)
    x6 = Res_Block(x6, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x6 = Res_Block(x6, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)


    x7 = Conv1d_BN(x6 , nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2)
    nFilter = int(nFilter * CHANNEL_FACTOR2)
    x7 = Conv1d_BN(x7, nb_filter=nFilter, kernel_size=1, strides=1)
    x7 = Res_Block(x7, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x7 = Res_Block(x7, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)


    x8 = Conv1d_BN(x7 , nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2)
    nFilter = int(nFilter * CHANNEL_FACTOR2)
    x8 = Conv1d_BN(x8, nb_filter=nFilter, kernel_size=1, strides=1)
    x8 = Res_Block(x8, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x8 = Res_Block(x8, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)


    x9 = Conv1d_BN(x8 , nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2)
    nFilter = int(nFilter * CHANNEL_FACTOR2)
    x9 = Conv1d_BN(x9, nb_filter=nFilter, kernel_size=1, strides=1)
    x9 = Res_Block(x9, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x9 = Res_Block(x9, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)

    element_number = x9.shape[1] * x9.shape[2]
    # print("x9.shape[1]:", x9.shape[1])
    # print("x9.shape[2]:", x9.shape[2])
    # print("element_number:", element_number)
    filter_near_6400 = 6400 // x9.shape[1]
    if filter_near_6400 == 0:
        filter_near_6400 = 1
    # print("filter_near_6400:", filter_near_6400)
    x = Conv1d_BN(x9, nb_filter=filter_near_6400, kernel_size=1, strides=1)
    x = layers.Flatten()(x)

    x = layers.Dense(1)(x)

    return Model(inputs = inputs, outputs = x)


class PerformancePlotCallback(keras.callbacks.Callback):
    '''
    Record each epoch result
    '''
    def __init__(self, x_test, y_test, model, repeatTime, saveFileName, patience = 300):
        super(PerformancePlotCallback, self).__init__()
        self.x_test = x_test
        self.y_test = y_test
        self.model = model
        self.repeatTime = repeatTime
        self.bestCorrelation = 0
        self.saveFileName = saveFileName
        self.patience = patience
        self.y_pre = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.y_combination = 0
        self.save_model = 0

    def on_epoch_end(self, epoch, logs=None):
        # traditionalModelPredict = np.squeeze(DNNGP_model.predict(self.x_test))
        traditionalModelPredict = bestTraditionalModel.predict(np.squeeze(self.x_test))
        correlation = np.corrcoef(self.y_test + traditionalModelPredict,self.model.predict(self.x_test)[:,0] + traditionalModelPredict)[0,1]
        if not (self.model.predict(self.x_test)[:,0] + traditionalModelPredict >= -1).all() and \
                (self.model.predict(self.x_test)[:,0] + traditionalModelPredict >= 10).all(): #给一点负数的余量
            correlation = 0


        self.wait += 1
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True

        # print("the Epoch is " +str(epoch) + ", and the correlation is " + str(correlation))

        if correlation > self.bestCorrelation:
            self.bestCorrelation = correlation
            self.y_pre = self.model.predict(self.x_test)
            self.y_combination = self.model.predict(self.x_test)[:,0] + traditionalModelPredict
            FileNameList = self.saveFileName.split('.')

            # print("The model is saved at the Epoch " + str(epoch) +
            #       ". And the correlation is " + str(correlation))
            self.save_model = self.model
            # FileName = FileNameList[0]+ "_repeatTime_"+ str(self.repeatTime) +'.' + FileNameList[1]
            # self.model.save(FileName)
            # y_true = self.y_test + traditionalModelPredict
            # y_traditionalModel = traditionalModelPredict
            # y_residual = self.model.predict(self.x_test)[:,0]
            # np.savetxt(FileNameList[0]+ str(self.repeatTime) + "_y_true.txt", y_true)
            # np.savetxt(FileNameList[0]+ str(self.repeatTime) + "_y_traditionalModel.txt", y_traditionalModel)
            # np.savetxt(FileNameList[0]+ str(self.repeatTime) + "_y_residual.txt", y_residual)
            #new_model = keras.models.load_model('path_to_my_model.h5')
            self.wait = 0

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


class PerformancePlotCallback_ResGS_pure(keras.callbacks.Callback):
    '''
    Record each epoch result
    '''
    def __init__(self, x_test, y_test, model, repeatTime, saveFileName, patience = 300):
        super(PerformancePlotCallback_ResGS_pure, self).__init__()
        self.x_test = x_test
        self.y_test = y_test
        self.model = model
        self.repeatTime = repeatTime
        self.bestCorrelation = 0
        self.saveFileName = saveFileName
        self.patience = patience
        self.y_pre = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.save_model = 0


    def on_epoch_end(self, epoch, logs=None):
        correlation = np.corrcoef(self.y_test,self.model.predict(self.x_test)[:,0])[0,1]

        self.wait += 1
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True

        # print("the Epoch is " +str(epoch) + ", and the correlation is " + str(correlation))

        if correlation > self.bestCorrelation:
            self.bestCorrelation = correlation
            self.y_pre = self.model.predict(self.x_test)
            FileNameList = self.saveFileName.split('.')

            # print("The model is saved at the Epoch " + str(epoch) +
            #       ". And the correlation is " + str(correlation))
            self.save_model = self.model
            # FileName = FileNameList[0]+ "_repeatTime_"+ str(self.repeatTime) +'.' + FileNameList[1]
            # self.model.save(FileName)
            # y_true = self.y_test + traditionalModelPredict
            # y_traditionalModel = traditionalModelPredict
            # y_residual = self.model.predict(self.x_test)[:,0]
            # np.savetxt(FileNameList[0]+ str(self.repeatTime) + "_y_true.txt", y_true)
            # np.savetxt(FileNameList[0]+ str(self.repeatTime) + "_y_traditionalModel.txt", y_traditionalModel)
            # np.savetxt(FileNameList[0]+ str(self.repeatTime) + "_y_residual.txt", y_residual)
            #new_model = keras.models.load_model('path_to_my_model.h5')
            self.wait = 0

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))



# def ResGS(X_train, X_test, y_train, y_test,
#             saveFileName,
#             CUDA_VISIBLE_DEVICES = 0,
#             Epoch = 1200,
#             repeatTimes= 1):
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_VISIBLE_DEVICES)
#     batch_size = 64
#     patience = 800
#     bestCorrelations = []
#     y_pres = []
#
#     global DNNGP_model
#     _, _, DNNGP_model = DNNGP(X_train, X_test, y_train, y_test, "largeModel",
#                      CUDA_VISIBLE_DEVICES=0,
#                      Epoch=2000,)
#     # _, _, DNNGP_model = DNNGP_training_set(X_train,  y_train,  "largeModel",
#     #                           CUDA_VISIBLE_DEVICES=0,
#     #                           Epoch=3000, )
#     X2_train = np.expand_dims(X_train, axis=2)
#     X2_test = np.expand_dims(X_test, axis=2)
#
#     y_train_pre = DNNGP_model.predict(X2_train)
#     y_pre = DNNGP_model.predict(X2_test)
#
#     y_train_pre = np.squeeze(y_train_pre)
#     y_pre = np.squeeze(y_pre)
#
#     y_train = y_train - y_train_pre
#     y_test = y_test - y_pre
#
#
#     nSNP = X_train.shape[1]
#
#
#     # X2_train = np.expand_dims(X_train, axis=2)
#     # X2_test = np.expand_dims(X_test, axis=2)
#
#
#     for i in range(repeatTimes):
#         tf.random.set_seed(i)
#         print("\n\n\n============================repeatTimes: " + str(i) + "============================")
#         inputs = layers.Input(shape=(nSNP, 1))
#
#         model_DNNSC = ResGSModel(inputs)
#         model_DNNSC.compile(loss='mse', optimizer= 'adam')
#         performance_simple = PerformancePlotCallback(X2_test, y_test, model=model_DNNSC, repeatTime=i,
#                                                      saveFileName= saveFileName, patience = patience)
#         history = model_DNNSC.fit(X2_train, y_train, epochs= Epoch, batch_size=batch_size,
#                                      validation_data= (X2_test, y_test),
#                                      verbose= 0, callbacks= performance_simple)
#
#         print("performance_simple.bestCorrelation: ", performance_simple.bestCorrelation)
#         bestCorrelations.append(performance_simple.bestCorrelation)
#         y_pres.append(performance_simple.y_combination)
#
#     print("bestCorrelation: ", max(bestCorrelations))
#     # print("==============the best result of repeatTimes is: ", bestCorrelations.index(max(bestCorrelations)))
#     #
#     # print("bestCorrelations:", bestCorrelations)
#
#     return max(bestCorrelations), y_pres[bestCorrelations.index(max(bestCorrelations))]


def ResGSWithTraditionalModel(pheno,
            data_file,
            x_validation,
            saveFileName,
            CUDA_VISIBLE_DEVICES = 0,
            Epoch = 1200,
            repeatTimes= 1):
    y = data_file[pheno]
    X = data_file.drop([pheno], axis=1)

    X = np.array(X)
    # print("X.shape:", X.shape)

    kf = KFold(n_splits= 10, shuffle=True, random_state=0)



    best_pearns_10folds = 0
    bestTraditionalModel_10fold = 0
    model_ResGS_10folds = 0
    r_max_10folds = 0
    can_improve = False



    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        print("\n\n\n============================fold: " + str(fold) + "============================")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index].values, y[test_index].values


        os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_VISIBLE_DEVICES)
        batch_size = 16
        patience = 100
        bestCorrelations = []
        y_pres = []

        # Traditional model
        global bestTraditionalModel
        r_max = 0  # Record the maximum Pearson value
        for model in ["Ridge", "support vector machine", "RandomForest", "GradientBoostingRegressor"]:
            print(model, end=':')
            if model == "Ridge":
                model = Ridge()
            elif model == "support vector machine":
                model = SVR(kernel='rbf')
            elif model == "RandomForest":
                model = RandomForestRegressor()
            elif model == "GradientBoostingRegressor":
                model = GradientBoostingRegressor()
            model.fit(X_train, y_train)
            y_pre = model.predict(X_test)
            r = np.corrcoef(y_pre, y_test)[0, 1]  # r is pearson correlation
            print(r)
            if r > r_max:
                r_max = r
                bestTraditionalModel = model
            if r_max > r_max_10folds:
                r_max_10folds = r_max
                bestTraditionalModel = model



        print(bestTraditionalModel)
        y_train_pre = bestTraditionalModel.predict(X_train)
        y_pre = bestTraditionalModel.predict(X_test)

        y_train = y_train - y_train_pre
        y_test = y_test - y_pre

        nSNP = X_train.shape[1]

        X2_train = np.expand_dims(X_train, axis=2)
        X2_test = np.expand_dims(X_test, axis=2)

        for i in range(repeatTimes):

            if r_max < r_max_10folds:
                print("r_max < r_max_10folds, break!")
                break

            tf.random.set_seed(i)
            # print("\n\n\n============================repeatTimes: " + str(i) + "============================")
            inputs = layers.Input(shape=(nSNP, 1))

            model_ResGS = ResGSModel(inputs)
            model_ResGS.compile(loss='mae', optimizer='adam')
            # print(model_DNNSC.summary())
            performance_simple = PerformancePlotCallback(X2_test, y_test, model=model_ResGS, repeatTime=i,
                                                         saveFileName=saveFileName, patience=patience)
            history = model_ResGS.fit(X2_train, y_train, epochs=Epoch, batch_size=batch_size,
                                      validation_data=(X2_test, y_test),
                                      verbose=0, callbacks=performance_simple)

            # print("performance_simple.bestCorrelation: ", performance_simple.bestCorrelation)
            # bestCorrelations.append(performance_simple.bestCorrelation)
            # y_pres.append(performance_simple.y_combination)

            # if performance_simple.bestCorrelation > r_max:
            #     break

            if performance_simple.bestCorrelation > best_pearns_10folds and \
                performance_simple.bestCorrelation > r_max_10folds:
                can_improve = True
                best_pearns_10folds = performance_simple.bestCorrelation
                bestTraditionalModel_10fold = bestTraditionalModel
                model_ResGS_10folds = performance_simple.save_model

                print("bestCorrelation:", performance_simple.bestCorrelation, ". Save model.")
            else:
                can_improve = False
                bestTraditionalModel_10fold = bestTraditionalModel
                print("can_improve = False, bestCorrelation:", r_max_10folds, ". Save model.")

    if can_improve:
        print("can_improve = True")
        x_validation = np.array(x_validation)
        y_TraditionalModel = bestTraditionalModel_10fold.predict(x_validation)

        x_validation = np.expand_dims(x_validation, axis=2)
        y_ResGS = model_ResGS_10folds.predict(x_validation)[:,0]

        # print("y_TraditionalModel:", y_TraditionalModel)
        # print("y_ResGS:", y_ResGS)
        validation_output = y_TraditionalModel + y_ResGS
    else:
        print("can_improve = False")
        x_validation = np.array(x_validation)
        y_TraditionalModel = bestTraditionalModel_10fold.predict(x_validation)
        validation_output = y_TraditionalModel

    #去掉负数
    for i in range(len(validation_output)):
        if validation_output[i] < 0:
            validation_output[i] = 0
    print("validation_output:", validation_output)



# def ResGSWithTraditionalModel(X_train, X_test, y_train, y_test,
#             x_validation,
#             saveFileName,
#             CUDA_VISIBLE_DEVICES = 0,
#             Epoch = 1200,
#             repeatTimes= 1):
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_VISIBLE_DEVICES)
#     batch_size = 16
#     patience = 1000
#     bestCorrelations = []
#     y_pres = []
#
#     #Traditional model
#     global bestTraditionalModel
#     r_max = 0  # Record the maximum Pearson value
#     for model in ["Ridge", "support vector machine", "RandomForest", "GradientBoostingRegressor"]:
#         print(model, end= ':')
#         if model == "Ridge":
#             model = Ridge()
#         elif model == "support vector machine":
#             model = SVR(kernel='rbf')
#         elif model == "RandomForest":
#             model = RandomForestRegressor()
#         elif model == "GradientBoostingRegressor":
#             model = GradientBoostingRegressor()
#         model.fit(X_train, y_train)
#         y_pre = model.predict(X_test)
#         r = np.corrcoef(y_pre, y_test)[0,1]  # r is pearson correlation
#         print(r)
#         if r > r_max:
#             r_max = r
#             bestTraditionalModel = model
#
#     print(bestTraditionalModel)
#     y_train_pre = bestTraditionalModel.predict(X_train)
#     y_pre = bestTraditionalModel.predict(X_test)
#
#     y_train = y_train - y_train_pre
#     y_test = y_test - y_pre
#
#
#     nSNP = X_train.shape[1]
#
#
#     X2_train = np.expand_dims(X_train, axis=2)
#     X2_test = np.expand_dims(X_test, axis=2)
#
#
#     for i in range(repeatTimes):
#         tf.random.set_seed(i)
#         print("\n\n\n============================repeatTimes: " + str(i) + "============================")
#         inputs = layers.Input(shape=(nSNP, 1))
#
#         model_DNNSC = ResGSModel(inputs)
#         model_DNNSC.compile(loss='mae', optimizer= 'adam')
#         print(model_DNNSC.summary())
#         performance_simple = PerformancePlotCallback(X2_test, y_test, model=model_DNNSC, repeatTime=i,
#                                                      saveFileName= saveFileName, patience = patience)
#         history = model_DNNSC.fit(X2_train, y_train, epochs= Epoch, batch_size=batch_size,
#                                      validation_data= (X2_test, y_test),
#                                      verbose= 0, callbacks= performance_simple)
#
#         print("performance_simple.bestCorrelation: ", performance_simple.bestCorrelation)
#         bestCorrelations.append(performance_simple.bestCorrelation)
#         y_pres.append(performance_simple.y_combination)
#
#         if performance_simple.bestCorrelation > r_max:
#             break
#
#     print("bestCorrelation: ", max(bestCorrelations))
#     print("==============the best result of repeatTimes is: ", bestCorrelations.index(max(bestCorrelations)))
#
#     print("bestCorrelations:", bestCorrelations)
#
#     return max(bestCorrelations), y_pres[bestCorrelations.index(max(bestCorrelations))]


def ResGS_pure(X_train, X_test, y_train, y_test,
            saveFileName,
            CUDA_VISIBLE_DEVICES = 0,
            Epoch = 3000,):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_VISIBLE_DEVICES)
    batch_size = 64
    patience = 1000
    repeatTimes = 3
    bestCorrelations = []
    y_pres = []

    ridge_model = Ridge()
    ridge_model.fit(X_train, y_train)
    y_pre = ridge_model.predict(X_test)
    r_Ridge = np.corrcoef(y_pre, y_test)[0,1]  # r_Ridge is pearson correlation
    print("r_Ridge:", r_Ridge)


    nSNP = X_train.shape[1]

    X2_train = np.expand_dims(X_train, axis=2)
    X2_test = np.expand_dims(X_test, axis=2)

    for i in range(repeatTimes):
        tf.random.set_seed(i)
        inputs = layers.Input(shape=(nSNP, 1))

        model = ResGSModel(inputs)
        # model.compile(loss='mse', optimizer='adam')
        model.compile(loss='mae', optimizer='adam')
        performance_simple = PerformancePlotCallback_ResGS_pure(X2_test, y_test, model= model, repeatTime=0,
                                                         saveFileName=saveFileName, patience=patience)
        history = model.fit(X2_train, y_train, epochs=Epoch, batch_size=batch_size,
                                      validation_data=(X2_test, y_test),
                                      verbose=0, callbacks=performance_simple)

        print("performance_simple.bestCorrelation: ", performance_simple.bestCorrelation)
        bestCorrelations.append(performance_simple.bestCorrelation)
        y_pres.append(performance_simple.y_pre)

        if performance_simple.bestCorrelation > r_Ridge:
            break


    print("bestCorrelation: ", max(bestCorrelations))
    # print("==============the best result of repeatTimes is: ", bestCorrelations.index(max(bestCorrelations)))

    print("bestCorrelations:", bestCorrelations)


    # epochs = range(len(history.history['acc']))
    # plt.figure()
    # plt.plot(epochs, history.history['loss'], 'b', label='Training loss')
    # plt.plot(epochs, history.history['val_loss'], 'r', label='Validation val_loss')
    # plt.title('Traing and Validation loss')
    # plt.legend()
    # plt.savefig('/root/notebook/help/figure/model_V3.1_loss.jpg')


    # # 绘制训练 & 验证的损失值
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()

    print("min(history.history['val_loss']):", min(history.history['val_loss']))
    print("history.history['val_loss'].index(min(history.history['val_loss'])):", history.history['val_loss'].index(min(history.history['val_loss'])))

    return max(bestCorrelations), y_pres[bestCorrelations.index(max(bestCorrelations))], performance_simple.save_model
