import os
import copy
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import f1_score
from utils import load_model_dict
from models import init_model_dict
from train_test import prepare_trte_data, gen_trte_adj_mat, test_epoch

cuda = True if torch.cuda.is_available() else False

def cal_feat_imp(data_folder, model_folder, view_list, num_class):
    num_view = len(view_list)
    dim_hvcdn = pow(num_class,num_view)
    if data_folder == 'ROSMAP':
        adj_parameter = 2
        dim_he_list = [200,200,100]
    if data_folder == 'BRCA':
        adj_parameter = 10
        dim_he_list = [400,400,200]
    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list)
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
    featname_list = []
    for v in view_list:
        df = pd.read_csv(os.path.join(data_folder, str(v)+"_featname.csv"), header=None)
        featname_list.append(df.values.flatten())
    
    dim_list = [x.shape[1] for x in data_tr_list]
    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn)
    for m in model_dict:
        if cuda:
            model_dict[m].cuda()
    model_dict = load_model_dict(model_folder, model_dict)
    te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
    if num_class == 2:
        f1 = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
    else:
        f1 = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')
    
    feat_imp_list = []
    for i in range(len(featname_list)):
        feat_imp = {"feat_name":featname_list[i]}
        feat_imp['imp'] = np.zeros(dim_list[i])
        for j in range(dim_list[i]):
            feat_tr = data_tr_list[i][:,j].clone()
            feat_trte = data_trte_list[i][:,j].clone()
            data_tr_list[i][:,j] = 0
            data_trte_list[i][:,j] = 0
            adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
            te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
            if num_class == 2:
                f1_tmp = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
            else:
                f1_tmp = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')
            feat_imp['imp'][j] = (f1-f1_tmp)*dim_list[i]
            
            data_tr_list[i][:,j] = feat_tr.clone()
            data_trte_list[i][:,j] = feat_trte.clone()
        feat_imp_list.append(pd.DataFrame(data=feat_imp))
    
    return feat_imp_list


def summarize_imp_feat(featimp_list_list, topn=30):
    num_rep = len(featimp_list_list)
    num_view = len(featimp_list_list[0])
    df_tmp_list = []
    for v in range(num_view):
        df_tmp = copy.deepcopy(featimp_list_list[0][v])
        df_tmp['omics'] = np.ones(df_tmp.shape[0], dtype=int)*v
        df_tmp_list.append(df_tmp.copy(deep=True))
    df_featimp = pd.concat(df_tmp_list).copy(deep=True)
    for r in range(1,num_rep):
        for v in range(num_view):
            df_tmp = copy.deepcopy(featimp_list_list[r][v])
            df_tmp['omics'] = np.ones(df_tmp.shape[0], dtype=int)*v
            df_featimp = df_featimp.append(df_tmp.copy(deep=True), ignore_index=True) 
    df_featimp_top = df_featimp.groupby(['feat_name', 'omics'])['imp'].sum()
    df_featimp_top = df_featimp_top.reset_index()
    df_featimp_top = df_featimp_top.sort_values(by='imp',ascending=False)
    df_featimp_top = df_featimp_top.iloc[:topn]
    print('{:}\t{:}'.format('Rank','Feature name'))
    for i in range(len(df_featimp_top)):
        print('{:}\t{:}'.format(i+1,df_featimp_top.iloc[i]['feat_name']))