import os
import random
import time

import numpy as np
import pandas
import pandas as pd
import scanpy as sc
import torch
from matplotlib import pyplot as plt, cm
from scarches.models import scPoli
from scib.metrics import metrics
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from model.model import iVAE_flow_unsupervised, iVAE
from utils.utils import TrainDataset, DatasetVAE

random.seed(0)
latent_dim = 10
dataset = ['lung', 'immune', 'pancreas', 'brain']
batch_dict = {'lung': 'batch', 'immune': 'batch', 'pancreas': 'study', 'brain': 'study', 'tumor': 'source'}
label_dict = {'lung': 'cell_type', 'immune': 'final_annotation', 'pancreas': 'cell_type', 'brain': 'cell_type',
              'tumor': 'cell_type'}
query_proportion_list = [0.2, 0.5, 0.8]

model_name = 'yang'
# model_name = 'scpoli'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for data_name in dataset:
    data_path = '../batch_effect_data/' + data_name + '/' + data_name + '.h5ad'
    adata = sc.read(data_path)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    batch = batch_dict[data_name]
    label = label_dict[data_name]
    for qp in query_proportion_list:
        train_data = adata.copy()
        target_domain = random.sample(adata.obs[batch].unique().tolist(), round(adata.obs[batch].nunique() * qp))
        adata.obs['query'] = [True if d in target_domain else False for d in adata.obs[batch]]
        source_adata = adata[~adata.obs['query']].copy()
        target_adata = adata[adata.obs['query']].copy()

        if model_name == 'scpoli':
            model = scPoli(adata=source_adata, condition_keys=batch, cell_type_keys=label, embedding_dims=5,
                           latent_dim=latent_dim, hidden_layer_sizes=[32], inject_condition=[])
            model.train(eta=10, n_epochs=50, pretraining_epochs=50)
            model = scPoli.load_query_data(adata=target_adata, reference_model=model, labeled_indices=[])
            model.train(eta=10, n_epochs=1, pretraining_epochs=1)
            model.model.eval()
            # get latent representation of query data
            target_adata.obsm[model_name] = model.get_latent(target_adata)
        elif model_name == 'yang':
            train_data.obs.loc[adata.obs['query'], label] = np.nan
            model = iVAE(latent_dim=latent_dim, s_prop=0, data_dim=adata.shape[1], domain_dim=adata.obs[batch].nunique(),
                         label_dim=adata.obs[label].nunique(), embedding_dim=5, dr_rate=0.05, hidden_dim=32, n_layers=1,
                         flows_n_layers=4, max_epochs=50, learning_rate=1e-3, recon_loss='nb', normalisation='layer',
                         lambda_kl=1, lambda_clas=10, lambda_spar=0, lambda_mask=0, patient=10, valid_prop=0,
                         batch_size=128, pretrain_epoch_rate=1, full_embedding=True, flows=None)
            model.fit(train_data, batch, label)
            # get latent representation of query data
            target_adata.obsm[model_name] = model.embed(adata.X[adata.obs['query']].toarray(),
                                                        adata.obs[batch].cat.codes.values[
                                                            adata.obs['query']]).detach().cpu().numpy()
        else:
            raise ValueError
        m = metrics(target_adata, target_adata, batch, label, embed=model_name, ari_=True, silhouette_=True,
                    isolated_labels_asw_=True, nmi_=True, pcr_=True, graph_conn_=True).T
        print(m.mean().mean())
        sc.settings.figdir = ''
        sc.tl.tsne(target_adata, use_rep=model_name)
        sc.pl.tsne(target_adata, color=batch, save='_batch.pdf')
        sc.pl.tsne(target_adata, color=label, save='_label.pdf')
        a = 1
