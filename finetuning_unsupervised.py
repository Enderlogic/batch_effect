import datetime
import os
import time

import numpy as np
import pandas
import pandas as pd
import scanpy as sc
import torch
from scarches.models import scPoli
from scib.metrics import metrics
from torch.utils.data import DataLoader

from model.model import iVAE_flow_unsupervised
from utils.utils import TrainDataset, DatasetVAE

dataset = ['pancreas', 'lung', 'immune', 'brain']
batch_dict = {'lung': 'batch', 'immune': 'batch', 'pancreas': 'study', 'brain': 'study'}
label_dict = {'lung': 'cell_type', 'immune': 'final_annotation', 'pancreas': 'cell_type', 'brain': 'cell_type'}
query_proportion_list = [0.2, 0.5, 0.8]
s_dim_prop_list = [0.5, 0]
latent_dim_list = [10, 20]
lambda_spar_list = [1e-3, 0]
lambda_mask_list = [0]
lambda_kl_list = [1, .1]
n_layers_list = [4, 6]
flows_n_layers_list = [2, 4]
dr_rate_list = [.05, 0]
normalisation_list = [None, 'batch']
learning_rate_list = [1e-4, 5e-4]
flows_list = ['quadraticspine']
recon_loss_list = ['nb', 'mse', 'zinb']
full_list = [True]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 100
max_epochs = 100
patient = 20
n_top_genes = 2000
embedding_dim = 10
hidden_dim = None
result_path = 'result/unsupervised_finetuning.csv'
if os.path.exists(result_path):
    result = pd.read_csv(result_path)
else:
    result = pandas.DataFrame(
        columns=['data', 'method', 'query proportion', 'latent dimension', 'layers number', 'flows layers number',
                 's dimension proportion', 'lambda kl', 'lambda sparsity', 'lambda mask', 'dropout', 'normalisation',
                 'learning rate', 'flows', 'recon loss', 'NMI', 'ARI', 'ASW label', 'ASW batch', 'PCR', 'ILS',
                 'GRA', 'score', 'cost'])
for data_name in dataset:
    data_path = 'data/' + data_name + '/' + data_name + '.h5ad'
    adata = sc.read(data_path)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=True)
    batch = batch_dict[data_name]
    label = label_dict[data_name]
    for qp in query_proportion_list:
        result_filepath = 'results/' + data_name + '/query_proportion_' + str(qp) + '.h5ad'
        # query = random.sample(adata.obs[batch].unique().tolist(), round(adata.obs[batch].nunique() * qp))
        query = ['inDrop1', 'celseq']
        reference = list(set(adata.obs[batch].unique().tolist()) - set(query))
        adata.obs['query'] = adata.obs[batch].isin(query)
        adata.obs['query'] = adata.obs['query'].astype('category')
        source_adata = adata[adata.obs['query'] == False].copy()
        target_adata = adata[adata.obs[batch].isin(query)].copy()
        target_data = torch.Tensor(target_adata.X.toarray())

        train_data = TrainDataset(source_adata, batch, label)
        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
        if not os.path.isfile(result_filepath):
            for latent_dim in latent_dim_list:
                for recon_loss in recon_loss_list:
                    # start = time.time()
                    # scpoli_model = scPoli(adata=source_adata, condition_keys=batch, cell_type_keys=label,
                    #                       embedding_dims=5, latent_dim=latent_dim, recon_loss=recon_loss,
                    #                       hidden_layer_sizes=[64])
                    # scpoli_model.train(eta=10)
                    # scpoli_query = scPoli.load_query_data(adata=target_adata, reference_model=scpoli_model,
                    #                                       labeled_indices=[])
                    # scpoli_query.train(eta=10)
                    # scpoli_query.model.eval()
                    # # get latent representation of query data
                    # target_adata.obsm['scpoli'] = scpoli_query.get_latent(target_adata)
                    # cost = time.time() - start
                    # m = metrics(target_adata, target_adata, batch, label, embed='scpoli', ari_=True, silhouette_=True,
                    #             isolated_labels_asw_=True, nmi_=True, pcr_=True, graph_conn_=True).T
                    # result.loc[len(result.index)] = [data_name, 'scpoli', qp, latent_dim, np.nan, np.nan, np.nan,
                    #                                  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, recon_loss,
                    #                                  np.nan, m['NMI_cluster/label'][0], m['ARI_cluster/label'][0],
                    #                                  m['ASW_label'][0], m['ASW_label/batch'][0], m['PCR_batch'][0],
                    #                                  m['isolated_label_silhouette'][0], m['graph_conn'][0],
                    #                                  m.mean().mean(), cost]
                    # print(result.tail(1).to_string())
                    # result.to_csv(result_path)
                    # print(f'averaged integration score: {m.mean().mean(): .4f}')
                    for n_layers in n_layers_list:
                        for flows_n_layers in flows_n_layers_list:
                            for s_dim_prop in s_dim_prop_list:
                                for lambda_kl in lambda_kl_list:
                                    for lambda_spar in lambda_spar_list:
                                        for lambda_mask in lambda_mask_list:
                                            for dr_rate in dr_rate_list:
                                                for normalisation in normalisation_list:
                                                    for learning_rate in learning_rate_list:
                                                        for flows in flows_list:
                                                            if ((result['data'] == data_name) & (
                                                                    result['method'] == 'iVAE') & (
                                                                        result['query proportion'] == qp) & (
                                                                        result['latent dimension'] == latent_dim) & (
                                                                        result['layers number'] == n_layers) & (
                                                                        result[
                                                                            'flows layers number'] == flows_n_layers) & (
                                                                        result[
                                                                            's dimension proportion'] == s_dim_prop) & (
                                                                        result['lambda kl'] == lambda_kl) & (
                                                                        result['lambda sparsity'] == lambda_spar) & (
                                                                        result['lambda mask'] == lambda_mask) & (
                                                                        result['dropout'] == dr_rate) & (
                                                                        result['normalisation'] == normalisation) & (
                                                                        result['learning rate'] == learning_rate) & (
                                                                        result['flows'] == flows) & (
                                                                        result['recon loss'] == recon_loss)).any():
                                                                continue
                                                            start = time.time()
                                                            model = iVAE_flow_unsupervised(latent_dim=latent_dim,
                                                                                           s_prop=s_dim_prop,
                                                                                           data_dim=n_top_genes,
                                                                                           domain_dim=adata.obs[
                                                                                               batch].nunique(),
                                                                                           embedding_dim=embedding_dim,
                                                                                           dr_rate=dr_rate,
                                                                                           hidden_dim=hidden_dim,
                                                                                           encoder_n_layers=n_layers,
                                                                                           decoder_n_layers=n_layers,
                                                                                           flows_n_layers=flows_n_layers,
                                                                                           max_epochs=max_epochs,
                                                                                           learning_rate=learning_rate,
                                                                                           recon_loss=recon_loss,
                                                                                           normalisation=normalisation,
                                                                                           lambda_kl=lambda_kl,
                                                                                           lambda_spar=lambda_spar,
                                                                                           lambda_mask=lambda_mask,
                                                                                           patient=patient,
                                                                                           valid_prop=.1,
                                                                                           batch_size=batch_size,
                                                                                           flows=flows)
                                                            train_data = DatasetVAE(adata, batch, label)
                                                            model.fit(train_data)
                                                            cost = time.time() - start
                                                            # get latent representation of query data
                                                            target_domain = adata.obs[batch].cat.codes.to_numpy()[
                                                                adata.obs['query'] == True]
                                                            target_adata.obsm['ivae'] = model.embed(target_data,
                                                                                                    target_domain,
                                                                                                    True)
                                                            m = metrics(target_adata, target_adata, batch, label,
                                                                        embed='ivae', ari_=True, silhouette_=True,
                                                                        isolated_labels_asw_=True, nmi_=True,
                                                                        pcr_=True, graph_conn_=True).T
                                                            result.loc[len(result.index)] = [data_name, 'iVAE', qp,
                                                                                             latent_dim,
                                                                                             n_layers,
                                                                                             flows_n_layers,
                                                                                             s_dim_prop, lambda_kl,
                                                                                             lambda_spar,
                                                                                             lambda_mask,
                                                                                             dr_rate, normalisation,
                                                                                             learning_rate, flows,
                                                                                             recon_loss,
                                                                                             m['NMI_cluster/label'][
                                                                                                 0],
                                                                                             m['ARI_cluster/label'][
                                                                                                 0],
                                                                                             m['ASW_label'][0],
                                                                                             m['ASW_label/batch'][
                                                                                                 0],
                                                                                             m['PCR_batch'][0],
                                                                                             m[
                                                                                                 'isolated_label_silhouette'][
                                                                                                 0],
                                                                                             m['graph_conn'][0],
                                                                                             m.mean().mean(), cost]
                                                            result.to_csv(result_path)
                                                            a = 1
