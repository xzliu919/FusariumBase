# import h5py
import numpy as np
import pandas as pd
# from dataset.DeepPhysDataset import DeepPhysDataset
# from dataset.PPNetDataset import PPNetDataset, PPNetHRDataset
# from dataset.PhysNetDataset import PhysNetDataset
# from dataset.DeepRhythemDataset import DeepRhythemDataset
# from dataset.DCTDataset import DCTDataset
# from dataset.DCTResDataset import DCTResDataset
from model.GenetypeDataset import GenetypeDataset
import os
# import cv2
import random
import math
import pickle
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import re
import csv


def reflect_to_region(x):
    a = 1.0
    b = 100.0
    xmin = 33
    xmax = 48
    y = a + float((b - a) * (x - a)) / (xmax - xmin)
    return y


def string_to_array(my_string):
    my_string = my_string.lower()
    my_string = re.sub('[^act]', 'z', my_string)
    my_array = np.array(list(my_string))
    return my_array


label_encoder = LabelEncoder()


def one_hot_encoder(my_array):
    integer_encoded = label_encoder.fit_transform(my_array)
    onehot_encoder = OneHotEncoder(sparse=False, dtype=int)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    onehot_encoded = np.delete(onehot_encoded, -1, 1)
    return onehot_encoded


from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(
            map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window


def prepare_weights(labels, reweight, min_target=0, max_target=134, lds=False, lds_kernel='gaussian', lds_ks=5,
                    lds_sigma=2):  # h:63 134 jieshu:15 22 baili:16 27 oil 15 24 pro 32 50
    assert reweight in {'none', 'inverse', 'sqrt_inv'}
    assert reweight != 'none' if lds else True, \
        "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"
    value_dict = {x: 0 for x in range(min_target, max_target)}
    for label in labels:
        value_dict[min(max_target - 1, int(label[0]))] += 1
    # print(value_dict)
    if reweight == 'sqrt_inv':
        value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
    elif reweight == 'inverse':
        value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight

    num_per_label = [value_dict[min(max_target - 1, int(label[0]))] for label in labels]
    if not len(num_per_label) or reweight == 'none':
        return None
    print(f"Using re-weighting: [{reweight.upper()}]")
    # print("value_dict",value_dict)
    # print("num_per_label",num_per_label)
    if lds:
        lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
        print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
        smoothed_value = convolve1d(
            np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
        # print("smoothed_value:",smoothed_value)
        # print("value_dict",value_dict)
        num_per_label = [smoothed_value[min(max_target - 1, int(label[0])) - min_target] for label in labels]

    # print(num_per_label)
    weights = [np.float32(1 / x) for x in num_per_label]
    scaling = len(weights) / np.sum(weights)
    weights = [[scaling * x] for x in weights]
    # print(labels)
    # print(weights)
    return weights


def sigmoid(x):
    sig = math.log(x, math.e)
    return sig


def inversesigmoid(x):
    insig = math.exp(x)
    return insig


hr = {"deo_36": 87, "deo_37": 81, "deo_38": 62, "deo_39": 79, "deo_40": 61, "deo_41": 52, "deo_42": 92, "deo_43": 83,
      "deo_44": 72, "deo_45": 70, "deo_46": 81, "deo_47": 61, "deo_48": 59, "deo_49": 64, "deo_50": 103, "deo_52": 75,
      "deo_53": 80, "deo_54": 77,
      "deo_55": 92, "deo_56": 71, "deo_57": 71, "deo_58": 53, "deo_59": 79, "deo_60": 85, "deo_61": 72, "deo_62": 78,
      "deo_63": 75, "deo_64": 71,
      "deo_65": 87, "deo_66": 72, "deo_67": 93, "deo_68": 83, "deo_69": 63, "deo_70": 66, "deo_71": 83, "deo_72": 72,
      "deo_73": 79, "deo_74": 68,
      "deo_75": 91, "deo_76": 79, "deo_77": 86, "deo_78": 75, "deo_79": 75, "deo_80": 86, "deo_81": 68}
gene_code_list = {"TT": 0, "AA": 1, "GG": 2, "CC": 3, "AT": 4, "TA": 4, "AG": 5, "GA": 5, "AC": 6, "CA": 6, "CT": 7,
                  "TC": 7, "TG": 8, "GT": 8, "CG": 9, "GC": 9, "NN": -1}


def load_test_motion(carpeta):
    X_test = []
    images_names = []
    image_path = carpeta
    print(carpeta)
    for f in [f for f in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, f))]:
        carpeta = os.path.join(image_path, f)
        # print(carpeta)
        for imagen in [imagen for imagen in os.listdir(carpeta) if os.path.isfile(os.path.join(carpeta, imagen))]:
            imagenes = os.path.join(carpeta, imagen)
            img = cv2.resize(cv2.imread(imagenes, cv2.IMREAD_COLOR), (36, 36))
            img = img.transpose((-1, 0, 1))
            X_test.append(img)
            images_names.append(imagenes)
    return X_test, images_names


# def standardization1(data):
#     mu = np.max(data, axis=0)
#     return data/mu
def standardization1(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    # print("mu,sigma",mu,sigma)
    return (data - mu) / sigma


def standardization(data):
    mu = np.mean(data)
    sigma = np.std(data)

    return (data - mu) / sigma


def load_test_attention(carpeta):
    X_test = []
    images_names = []
    image_path = carpeta
    for f in [f for f in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, f))]:
        carpeta = os.path.join(image_path, f)
        print(carpeta)
        for imagen in [imagen for imagen in os.listdir(carpeta) if os.path.isfile(os.path.join(carpeta, imagen))]:
            imagenes = os.path.join(carpeta, imagen)
            img = cv2.resize(cv2.imread(imagenes, cv2.IMREAD_COLOR), (36, 36))
            img = img.transpose((-1, 0, 1))
            X_test.append(img)
            images_names.append(imagenes)
    return X_test, images_names


def load_ap_attention(carpeta):
    X_test = []
    X_mo_test = []
    images_names = []
    images_names_mo = []
    image_path = carpeta
    for f in [f for f in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, f))]:
        carpeta = os.path.join(image_path, f)
        # print(carpeta)
        for imagen in [imagen for imagen in os.listdir(carpeta) if os.path.isfile(os.path.join(carpeta, imagen))]:
            imagenes = os.path.join(carpeta, imagen)
            # img = cv2.resize(cv2.imread(imagenes, cv2.IMREAD_COLOR), (36, 36))
            # print(imagenes,imagenes.replace('RawFrames','DeepFrames'))
            # img_mo = cv2.resize(cv2.imread(imagenes.replace('RawFrames','DeepFrames'), cv2.IMREAD_COLOR), (36, 36))
            # img = img.transpose((-1,0,1))
            # img_mo = img_mo.transpose((-1,0,1))
            # X_test.append(img)
            # X_mo_test.append(img_mo)
            images_names.append(imagenes)
            images_names_mo.append(imagenes.replace('RawFrames', 'DeepFrames'))
    # return X_test,X_mo_test, images_names,images_names_mo
    return images_names, images_names_mo

geno_embd_4d = {
    '0/0': 0,
    '1/0': 1,
    '0/1': 1,
    '1/1': 2,
    './.': -1,
    '0': 0.,
    '1': 1,
    '2': 2,
    '-1': -1,
    '3': 3,
}

# def dataset_loader(save_root_path: str = "/media/hdd1/dy_dataset/",
#                    model_name: str = "DeepPhys",
#                    dataset_name: str = "UBFC",
#                    option: str = "train"):
def dataset_loader(data_file):
    '''
    :param save_root_path: save file destination path
    :param model_name : model_name
    :param dataset_name: data set name(ex. UBFC, COFACE)
    :param option:[train, test]
    :return: dataset
    '''

    phe_p = []
    train_all = []
    rows = open(data_file, 'r').readlines()

    # ==============set sample size======================
    # random.seed(0)
    # rows = random.sample(rows, 300)
    # ==============set sample size======================

    datas = list(filter(None, rows))

    for data in datas:
        data = list(filter(None, data.strip().split(',')))
        pheno = float(data[0])
        phe_p.append(pheno)

        embd_all = []
        for i, d in enumerate(data[1:]):
            try:
                embd = geno_embd_4d[d]
            except:
                embd = float(d)
            embd_all.append(embd)

        train_all.append(embd_all)


    # split_ppg_train = []
    #
    # train_path = "/root/workspace/dnngp/data/F5_soybean/phenotype/phenotype_height.txt"
    #
    # train_list = open(train_path, 'r')
    # resu = train_list.readlines()
    #
    # for i in resu:
    #     gene_name = i.strip('\r\n').split(' ')[0]
    #     # print(i)
    #     phone_result_h = float(i.strip('\r\n').split(' ')[1])
    #     phe_p.append(phone_result_h)
    #     gene_file = '/root/workspace/dnngp/data/F5_soybean/genotype/gt_file_7883/' + gene_name + '.npy'
    #
    #     if not os.path.exists(gene_file):
    #         print("file not exits:", gene_file)
    #         continue
    #     gene = np.load(gene_file, allow_pickle=True)
    #     m = gene.shape[0]
    #     gene_coding = []
    #
    #     for j in range(m):
    #         if gene[j] == '0/0':
    #             # gene_code = gene_code_list[ref_alt[j].strip('\r\n').split(' ')[0]+ref_alt[j].strip('\r\n').split(' ')[0]]
    #             gene_code = 0  # [p_dict[j],0,0]#0.5#[1,0,0]#[p_dict[j],0,0]
    #         elif gene[j] == '0/1':
    #             #         #gene_code = gene_code_list[ref_alt[j].strip('\r\n').split(' ')[0]+ref_alt[j].strip('\r\n').split(' ')[1]]
    #
    #             gene_code = 1  # [0,p_dict[j],0]#0.25#[0,1,0]#[0,p_dict[j],0]
    #         elif gene[j] == '1/0':
    #             #         #gene_code = gene_code_list[ref_alt[j].strip('\r\n').split(' ')[1]+ref_alt[j].strip('\r\n').split(' ')[0]]
    #             gene_code = 1  # [0,p_dict[j],0]#0.25#[0,1,0]#[0,p_dict[j],0]
    #             # gene_code_1 =[0,f_dict[j],0]
    #         #         #gene_code = 'C'
    #         elif gene[j] == '1/1' or gene[j] == '1|1' or gene[j] == 44562:
    #             #         #gene_code = gene_code_list[ref_alt[j].strip('\r\n').split(' ')[1]+ref_alt[j].strip('\r\n').split(' ')[1]]
    #
    #             gene_code = 2  # [0,0,p_dict[j]]#0.75#[0,0,1]#[0,0,p_dict[j]]
    #         else:
    #             #         #gene_code = gene_code_list['NN']
    #             gene_code = -1  # [0,0,0]#0#[0,0,0]#[0,0,0]
    #         gene_coding.extend([gene_code])
    #
    #     split_ppg_train.extend([np.array(gene_coding)])  # np.sort(top_sort[:10000,])]]
    #
    # data =np.array(split_ppg_train)
    #
    # print(data.shape)

    # np.save('./wheat2000_phe6.npy',np.array(phe_p))

    data = np.array(train_all)
    print("data.max(): ", data.max())
    print("data.min(): ", data.min())

    index_dic = defaultdict(list)
    for index, pid in enumerate(phe_p):
        # print(pid[0],pid[0]//0.1)
        index_dic[int(pid // 1)].append(index)
    pids = list(index_dic.keys())
    # print(pids)
    phe_label = [[0]] * len(phe_p)
    sortid = sorted(range(len(pids)), key=lambda k: pids[k], reverse=False)
    # print("cls_num:", len(sortid))
    for j in sortid:
        for i in index_dic[pids[j]]:
            phe_label[i] = [j]

    dataset = GenetypeDataset(ppg=np.asarray(data), label=np.asarray(phe_label),
                              hr=np.asarray(phe_p), mean_ph=0)
    return dataset
        # return train_dataset
