import time

import numpy as np
import pandas
import scanpy as sc
import torch
from scib.metrics import metrics
from sklearn.metrics import f1_score
from model.model import iVAE

dataset = ['pancreas', 'lung', 'immune', 'brain']
batch_dict = {'lung': 'batch', 'immune': 'batch', 'pancreas': 'study', 'brain': 'study'}
label_dict = {'lung': 'cell_type', 'immune': 'final_annotation', 'pancreas': 'cell_type', 'brain': 'cell_type'}
query_proportion_list = [0.2, 0.5, 0.8]
s_dim_prop_list = [0.5, 0.2]
latent_dim_list = [10, 20]
lambda_spar_list = [1e-3, 0]
lambda_mask_list = [0]
lambda_kl_list = [1, .1]
lambda_clas_list = [1, 10]
n_layers_list = [4, 6]
flows_n_layers_list = [2, 4]
dr_rate_list = [.05, 0]
normalisation_list = [None, 'batch']
learning_rate_list = [1e-4, 5e-4]
flows_list = ['quadraticspline', 'spline']
recon_loss_list = ['nb', 'mse', 'zinb']
full_embedding_list = [True, False]
embedding_dim_list = [5, 10]
hidden_dim_list = [32, 64]
pretrain_epoch_rate_list = [0, .9]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 100
max_epochs = 100
patient = 20
n_top_genes = 2000

result = pandas.DataFrame(
    columns=['data', 'query proportion', 'latent dimension', 'recon loss', 'layers number', 'flows layers number',
             's dimension proportion', 'embedding dim', 'hidden dim', 'lambda kl', 'lambda sparsity', 'lambda mask',
             'lambda clas', 'pretrain_epoch_rate', 'full embedding', 'dropout rate', 'normalisation', 'learning rate',
             'flows', 'NMI', 'ARI', 'ASW label', 'ASW batch', 'PCR', 'ILS', 'GRA', 'score', 'f1 weighted', 'f1 macro',
             'cost'])

metadata_path = 'result/metadata.pkl'
result_path = 'result/ivae_finetuning.csv'
metadata = pandas.read_pickle(metadata_path)
for data_name in dataset:
    data_path = '../batch_effect_data/' + data_name + '/' + data_name + '.h5ad'
    adata = sc.read(data_path)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=True)
    batch = batch_dict[data_name]
    label = label_dict[data_name]
    for qp in query_proportion_list:
        result_filepath = '../batch_effect_data/' + data_name + '/' + str(qp) + '/ivae.h5ad'
        target_domain = metadata['target domain'][list(
            set([i for i, n in enumerate(metadata['data']) if n == data_name]).intersection(
                [i for i, n in enumerate(metadata['query proportion']) if n == qp]))[0]]
        train_data = adata.copy()
        adata.obs['query'] = [True if d in target_domain else False for d in adata.obs[batch]]
        target_adata = adata[adata.obs['query']].copy()
        train_data.obs.loc[adata.obs['query'], label] = np.nan

        for latent_dim in latent_dim_list:
            for recon_loss in recon_loss_list:
                for n_layers in n_layers_list:
                    for flows_n_layers in flows_n_layers_list:
                        for s_dim_prop in s_dim_prop_list:
                            for embedding_dim in embedding_dim_list:
                                for hidden_dim in hidden_dim_list:
                                    for lambda_kl in lambda_kl_list:
                                        for lambda_spar in lambda_spar_list:
                                            for lambda_mask in lambda_mask_list:
                                                for lambda_clas in lambda_clas_list:
                                                    for pretrain_epoch_rate in pretrain_epoch_rate_list:
                                                        for full_embedding in full_embedding_list:
                                                            for dr_rate in dr_rate_list:
                                                                for normalisation in normalisation_list:
                                                                    for learning_rate in learning_rate_list:
                                                                        for flows in flows_list:
                                                                            start = time.time()
                                                                            model = iVAE(latent_dim=latent_dim,
                                                                                         s_prop=s_dim_prop,
                                                                                         data_dim=n_top_genes,
                                                                                         domain_dim=adata.obs[
                                                                                             batch].nunique(),
                                                                                         label_dim=adata.obs[
                                                                                             label].nunique(),
                                                                                         embedding_dim=embedding_dim,
                                                                                         dr_rate=dr_rate,
                                                                                         hidden_dim=hidden_dim,
                                                                                         n_layers=n_layers,
                                                                                         flows_n_layers=flows_n_layers,
                                                                                         max_epochs=max_epochs,
                                                                                         learning_rate=learning_rate,
                                                                                         recon_loss=recon_loss,
                                                                                         normalisation=normalisation,
                                                                                         lambda_kl=lambda_kl,
                                                                                         lambda_clas=lambda_clas,
                                                                                         lambda_spar=lambda_spar,
                                                                                         lambda_mask=lambda_mask,
                                                                                         patient=patient, valid_prop=.1,
                                                                                         batch_size=batch_size,
                                                                                         pretrain_epoch_rate=pretrain_epoch_rate,
                                                                                         full_embedding=full_embedding,
                                                                                         flows=flows)
                                                                            model.fit(train_data, batch, label)
                                                                            cost = time.time() - start
                                                                            # get latent representation of query data
                                                                            target_adata.obsm['ivae'] = model.embed(
                                                                                target_adata.X.toarray(),
                                                                                adata.obs[batch].cat.codes.values[
                                                                                    adata.obs[
                                                                                        'query']]).detach().cpu().numpy()
                                                                            m = metrics(target_adata, target_adata,
                                                                                        batch, label, embed='ivae',
                                                                                        ari_=True, silhouette_=True,
                                                                                        isolated_labels_asw_=True,
                                                                                        nmi_=True, pcr_=True,
                                                                                        graph_conn_=True).T
                                                                            y_true = target_adata.obs[
                                                                                label].cat.codes.values
                                                                            y_pred = model.classify(
                                                                                target_adata.X.toarray(),
                                                                                adata.obs[batch].cat.codes.values[
                                                                                    adata.obs['query']])
                                                                            f1_weighted = f1_score(y_true, y_pred,
                                                                                                   average='weighted')
                                                                            f1_macro = f1_score(y_true, y_pred,
                                                                                                average='macro')

                                                                            result.loc[len(result.index)] = [data_name,
                                                                                                             qp,
                                                                                                             latent_dim,
                                                                                                             recon_loss,
                                                                                                             n_layers,
                                                                                                             flows_n_layers,
                                                                                                             s_dim_prop,
                                                                                                             embedding_dim,
                                                                                                             hidden_dim,
                                                                                                             lambda_kl,
                                                                                                             lambda_spar,
                                                                                                             lambda_mask,
                                                                                                             lambda_clas,
                                                                                                             pretrain_epoch_rate,
                                                                                                             full_embedding,
                                                                                                             dr_rate,
                                                                                                             normalisation,
                                                                                                             learning_rate,
                                                                                                             flows,
                                                                                                             m[
                                                                                                                 'NMI_cluster/label'][
                                                                                                                 0],
                                                                                                             m[
                                                                                                                 'ARI_cluster/label'][
                                                                                                                 0],
                                                                                                             m[
                                                                                                                 'ASW_label'][
                                                                                                                 0],
                                                                                                             m[
                                                                                                                 'ASW_label/batch'][
                                                                                                                 0],
                                                                                                             m[
                                                                                                                 'PCR_batch'][
                                                                                                                 0],
                                                                                                             m[
                                                                                                                 'isolated_label_silhouette'][
                                                                                                                 0],
                                                                                                             m[
                                                                                                                 'graph_conn'][
                                                                                                                 0],
                                                                                                             m.mean().mean(),
                                                                                                             f1_weighted,
                                                                                                             f1_macro,
                                                                                                             cost]
                                                                            result.to_csv(result_path, index=False)
                                                                            print(result.iloc[-1, :].to_string())
                                                                            a = 1
