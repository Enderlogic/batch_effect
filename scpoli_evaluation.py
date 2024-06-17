import os
import pickle
import random

import pandas
import scanpy as sc
import torch
from scarches.models import scPoli
from scib.metrics import metrics
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from utils.utils import TrainDataset

sc.set_figure_params(scanpy=True, fontsize=6)

dataset = ['pancreas', 'lung', 'immune', 'brain']
batch_dict = {'lung': 'batch', 'immune': 'batch', 'pancreas': 'study', 'brain': 'study', 'tumor': 'source'}
label_dict = {'lung': 'cell_type', 'immune': 'final_annotation', 'pancreas': 'cell_type', 'brain': 'cell_type',
              'tumor': 'cell_type'}
query_proportion_list = [0.2, 0.5, 0.8]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_top_genes = 2000
result = pandas.DataFrame(
    columns=['data', 'query proportion', 'NMI', 'ARI', 'ASW label', 'ASW batch', 'PCR', 'ILS', 'GRA', 'score',
             'f1 weighted', 'f1 macro', 'cost'])
result_path = 'result/scpoli_evaluation.csv'
metadata_path = 'result/metadata.pkl'
metadata = {'data': [], 'query proportion': [], 'target domain': []}

for data_name in dataset:
    data_path = '../batch_effect_data/' + data_name + '/' + data_name + '.h5ad'
    adata = sc.read(data_path)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=True)
    batch = batch_dict[data_name]
    label = label_dict[data_name]
    for qp in query_proportion_list:
        if round(adata.obs[batch].nunique() * qp) > 1:
            query = random.sample(adata.obs[batch].unique().tolist(), round(adata.obs[batch].nunique() * qp))
            metadata['data'].append(data_name)
            metadata['query proportion'].append(qp)
            metadata['target domain'].append(query)
            source_adata = adata[~adata.obs[batch].isin(query)].copy()
            target_adata = adata[adata.obs[batch].isin(query)].copy()

            train_data = TrainDataset(source_adata, batch, label)
            train_loader = DataLoader(train_data, shuffle=True, batch_size=100)
            scpoli_model = scPoli(adata=source_adata, condition_keys=batch, cell_type_keys=label, embedding_dims=5,
                                  latent_dim=10, hidden_layer_sizes=[64])
            scpoli_model.train(eta=10, n_epochs=50, pretraining_epochs=50)
            scpoli_query = scPoli.load_query_data(adata=target_adata, reference_model=scpoli_model,
                                                  labeled_indices=[])
            scpoli_query.train(eta=10, n_epochs=50, pretraining_epochs=50)

            scpoli_query.model.eval()
            # get latent representation of query data
            target_adata.obsm['scpoli'] = scpoli_query.get_latent(target_adata)
            result_data_path = '../batch_effect_data/' + data_name + '/' + str(qp) + '/scpoli.h5ad'
            os.makedirs(os.path.dirname(result_data_path), exist_ok=True)
            target_adata.write_h5ad(result_data_path)
            m = metrics(target_adata, target_adata, batch, label, embed='scpoli', ari_=True, silhouette_=True,
                        isolated_labels_asw_=True, nmi_=True, pcr_=True, graph_conn_=True).T
            y_true = target_adata.obs[label]
            y_pred = scpoli_query.classify(target_adata, scale_uncertainties=True)[label]["preds"]
            f1_weighted = f1_score(y_true, y_pred, average='weighted')
            f1_macro = f1_score(y_true, y_pred, average='macro')

            result.loc[len(result.index)] = [data_name, qp, m['NMI_cluster/label'][0], m['ARI_cluster/label'][0],
                                             m['ASW_label'][0], m['ASW_label/batch'][0], m['PCR_batch'][0],
                                             m['isolated_label_silhouette'][0], m['graph_conn'][0], m.mean().mean(),
                                             f1_weighted, f1_macro,
                                             scpoli_model.trainer.training_time + scpoli_query.trainer.training_time]
            print(result.tail(1).to_string())
            plot_folder = 'result/plot/' + data_name + '/' + str(qp) + '/'
            sc.settings.figdir = plot_folder
            sc.tl.tsne(target_adata, use_rep='scpoli')
            os.makedirs(os.path.dirname(plot_folder), exist_ok=True)
            sc.pl.tsne(target_adata, color=batch, legend_fontsize=9, title='batch, score=' + str(m.mean().mean()),
                       save='_scpoli_batch.pdf')
            sc.pl.tsne(target_adata, color=label, legend_fontsize=9, title='label, score=' + str(m.mean().mean()),
                       save='_scpoli_label.pdf')
            result.to_csv(result_path, index=False)
            with open(metadata_path, 'wb') as fp:
                pickle.dump(metadata, fp)
