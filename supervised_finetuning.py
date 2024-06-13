import time

import pandas as pd
import scanpy as sc
import torch
from scib.metrics import metrics
from sklearn.metrics import f1_score

from model.model import iVAE_flow_supervised
from utils.utils import DatasetVAE

dataset = ['pancreas', 'lung', 'immune', 'brain']
batch_dict = {'lung': 'batch', 'immune': 'batch', 'pancreas': 'study', 'brain': 'study'}
label_dict = {'lung': 'cell_type', 'immune': 'final_annotation', 'pancreas': 'cell_type', 'brain': 'cell_type'}

query_proportion_list = [0.2, 0.5, 0.8]
s_dim_prop_list = [0.2, 0.5]
latent_dim_list = [10, 20]
lambda_spar_list = [1, .1, 0]
lambda_mask = 0
lambda_kl_list = [1, .1, 0]
lambda_prot_list = [1, .1, 0]
n_layers_list = [2, 4]
flows_n_layers_list = [2, 4]
dr_rate_list = [None, .01]
normalisation_list = [None, 'batch']
learning_rate_list = [1e-3, 2e-3]
recon_loss_list = ['nb', 'zinb']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 100
max_epochs = 100
patient = 10
embedding_dim_list = [5, 10]
hidden_dim_list = [32, 64]
c_s_dependent_list = [False]
pretrain_epoch_rate_list = [0, .5]

result = pd.DataFrame(
    columns=['dataset', 'query_proportion', 'latent_dim', 's_dim_prop', 'embedding_dim', 'dr_rate', 'hidden_dim',
             'n_layers', 'flows_n_layers', 'learning_rate', 'recon_loss', 'normalisation', 'lambda_kl', 'lambda_prot',
             'lambda_spar', 'lambda_mask', 'pretrain_epoch_rate', 'NMI', 'ARI', 'ASW label', 'ASW batch',
             'PCR', 'ILS', 'GRA', 'score', 'f1 weighted', 'f1 macro', 'cost'])

result_path = 'result/supervised_finetuning.csv'
for data_name in dataset:
    for query_proportion in query_proportion_list:
        source_adata = sc.read_h5ad('data/' + data_name + '/' + str(query_proportion) + '/source.h5ad')
        target_adata = sc.read_h5ad('data/' + data_name + '/' + str(query_proportion) + '/target.h5ad')
        batch = batch_dict[data_name]
        label = label_dict[data_name]
        train_data = DatasetVAE(source_adata, batch, label)
        for latent_dim in latent_dim_list:
            for s_dim_prop in s_dim_prop_list:
                for embedding_dim in embedding_dim_list:
                    for dr_rate in dr_rate_list:
                        for hidden_dim in hidden_dim_list:
                            for n_layers in n_layers_list:
                                for flows_n_layers in flows_n_layers_list:
                                    for learning_rate in learning_rate_list:
                                        for recon_loss in recon_loss_list:
                                            for normalisation in normalisation_list:
                                                for lambda_kl in lambda_kl_list:
                                                    for lambda_prot in lambda_prot_list:
                                                        for lambda_spar in lambda_spar_list:
                                                            for pretrain_epoch_rate in pretrain_epoch_rate_list:
                                                                model = iVAE_flow_supervised(latent_dim=latent_dim,
                                                                                             s_prop=s_dim_prop,
                                                                                             data_dim=
                                                                                             source_adata.shape[1],
                                                                                             domain_dim=
                                                                                             source_adata.obs[
                                                                                                 batch].nunique(),
                                                                                             label_dim=
                                                                                             source_adata.obs[
                                                                                                 label].nunique(),
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
                                                                                             lambda_prot=lambda_prot,
                                                                                             lambda_spar=lambda_spar,
                                                                                             lambda_mask=lambda_mask,
                                                                                             patient=patient,
                                                                                             valid_prop=.1,
                                                                                             batch_size=batch_size,
                                                                                             pretrain_epoch_rate=pretrain_epoch_rate)
                                                                start = time.time()
                                                                model.fit(train_data)
                                                                cost = time.time() - start
                                                                # get latent representation of query data
                                                                target_adata.obsm[
                                                                    'ivae_' + str(latent_dim) + '_' + str(
                                                                        s_dim_prop) + '_' + str(
                                                                        embedding_dim) + '_' + str(
                                                                        dr_rate) + '_' + str(
                                                                        hidden_dim) + '_' + str(
                                                                        n_layers) + '_' + str(
                                                                        flows_n_layers) + '_' + str(
                                                                        learning_rate) + '_' + recon_loss + '_' + str(
                                                                        normalisation) + '_' + str(
                                                                        lambda_kl) + '_' + str(
                                                                        lambda_prot) + '_' + str(
                                                                        lambda_spar) + '_' + str(
                                                                        pretrain_epoch_rate)] = model.embed(
                                                                    target_adata.X.toarray()).detach().cpu().numpy()
                                                                m = metrics(target_adata, target_adata, batch,
                                                                            label, embed='ivae_' + str(
                                                                        latent_dim) + '_' + str(
                                                                        s_dim_prop) + '_' + str(
                                                                        embedding_dim) + '_' + str(
                                                                        dr_rate) + '_' + str(
                                                                        hidden_dim) + '_' + str(
                                                                        n_layers) + '_' + str(
                                                                        flows_n_layers) + '_' + str(
                                                                        learning_rate) + '_' + recon_loss + '_' + str(
                                                                        normalisation) + '_' + str(
                                                                        lambda_kl) + '_' + str(
                                                                        lambda_prot) + '_' + str(
                                                                        lambda_spar) + '_' + str(
                                                                        pretrain_epoch_rate),
                                                                            ari_=True, silhouette_=True,
                                                                            isolated_labels_asw_=True,
                                                                            nmi_=True, pcr_=True,
                                                                            graph_conn_=True).T
                                                                y_true = target_adata.obs[label].values.codes
                                                                y_pred = model.classify(
                                                                    target_adata).cpu().numpy()
                                                                f1_macro = f1_score(y_true, y_pred,
                                                                                    average='macro')
                                                                f1_weighted = f1_score(y_true, y_pred,
                                                                                       average='weighted')
                                                                result.loc[len(result.index)] = [data_name,
                                                                                                 query_proportion,
                                                                                                 latent_dim,
                                                                                                 s_dim_prop,
                                                                                                 embedding_dim,
                                                                                                 dr_rate,
                                                                                                 hidden_dim,
                                                                                                 n_layers,
                                                                                                 flows_n_layers,
                                                                                                 learning_rate,
                                                                                                 recon_loss,
                                                                                                 normalisation,
                                                                                                 lambda_kl,
                                                                                                 lambda_prot,
                                                                                                 lambda_spar,
                                                                                                 lambda_mask,
                                                                                                 pretrain_epoch_rate,
                                                                                                 m[
                                                                                                     'NMI_cluster/label'][
                                                                                                     0], m[
                                                                                                     'ARI_cluster/label'][
                                                                                                     0],
                                                                                                 m['ASW_label'][
                                                                                                     0], m[
                                                                                                     'ASW_label/batch'][
                                                                                                     0],
                                                                                                 m['PCR_batch'][
                                                                                                     0], m[
                                                                                                     'isolated_label_silhouette'][
                                                                                                     0], m[
                                                                                                     'graph_conn'][
                                                                                                     0],
                                                                                                 m.mean().mean(),
                                                                                                 f1_weighted,
                                                                                                 f1_macro, cost]
                                                                print(result.tail(1).to_string())
                                                                result.to_csv(result_path)

        target_adata.write_h5ad('result/' + data_name + '_' + str(query_proportion) + '.h5ad')
