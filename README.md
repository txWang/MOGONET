# MOGONET: Multi-omics Integration via Graph Convolutional Networks for Biomedical Data Classification
Tongxin Wang\*, Wei Shao\*, Zhi Huang, Haixu Tang, Jie Zhang, Zhengming Ding, and Kun Huang

MOGONET (Multi-Omics Graph cOnvolutional NETworks) is a novel multi-omics data integrative analysis framework for classification tasks in biomedical applications.

![MOGONET](https://github.com/txWang/MOGONET/blob/master/MOGONET.png?raw=true "MOGONET")
Overview of MOGONET. \
<sup>Illustration of MOGONET. MOGONET combines GCN for multi-omics specific learning and VCDN for multi-omics integration. MOGONET combines GCN for multi-omics specific learning and VCDN for multi-omics integration. For clear and concise illustration, an example of one sample is chosen to demonstrate the VCDN component for multi-omics integration. Pre-processing is first performed on each omics data type to remove noise and redundant features. Each omics-specific GCN is trained to perform class prediction using omics features and the corresponding sample similarity network generated from the omics data. The cross-omics discovery tensor is calculated from the initial predictions of omics-specific GCNs and forwarded to VCDN for final prediction. MOGONET is an end-to-end model and all networks are trained jointly.<sup>

## Files
*main_mogonet.py*: Examples of MOGONET for classification tasks\
*main_biomarker.py*: Examples for identifying biomarkers\
*models.py*: MOGONET model\
*train_test.py*: Training and testing functions\
*feat_importance.py*: Feature importance functions\
*utils.py*: Supporting functions    

\* Equal contribution
