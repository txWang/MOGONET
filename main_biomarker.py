""" Example for biomarker identification
"""
import os
import copy
from feat_importance import cal_feat_imp, summarize_imp_feat

if __name__ == "__main__":
    data_folder = 'BRCA'
    model_folder = os.path.join(data_folder, 'models')
    view_list = [1,2,3]
    if data_folder == 'ROSMAP':
        num_class = 2
    if data_folder == 'BRCA':
        num_class = 5

    featimp_list_list = []
    for rep in range(5):
        featimp_list = cal_feat_imp(data_folder, os.path.join(model_folder, str(rep+1)), 
                                    view_list, num_class)
        featimp_list_list.append(copy.deepcopy(featimp_list))
    summarize_imp_feat(featimp_list_list)
    