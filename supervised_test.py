import scanpy as sc
import torch
from scib.metrics import metrics
from sklearn.metrics import f1_score

from model.model import iVAE_flow_supervised
from utils.utils import DatasetVAE

dataset = ['pancreas', 'lung', 'immune', 'brain']
batch_dict = {'lung': 'batch', 'immune': 'batch', 'pancreas': 'study', 'brain': 'study'}
label_dict = {'lung': 'cell_type', 'immune': 'final_annotation', 'pancreas': 'cell_type', 'brain': 'cell_type'}

data_name = 'pancreas'
query_proportion = 0.2
s_dim_prop = 0.2
latent_dim = 10
lambda_spar = 1
lambda_mask = 0
lambda_kl = 1
lambda_prot = 1
lambda_part = 1
n_layers = 4
flows_n_layers = 2
dr_rate = None
normalisation = 'batch'
learning_rate = 1e-3
flows = 'quadraticspine'
recon_loss = 'nb'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 100
max_epochs = 100
patient = 20
embedding_dim = 5
hidden_dim = None

source_adata = sc.read_h5ad('data/' + data_name + '/' + str(query_proportion) + '/source.h5ad')
target_adata = sc.read_h5ad('data/' + data_name + '/' + str(query_proportion) + '/target.h5ad')
batch = batch_dict[data_name]
label = label_dict[data_name]
model = iVAE_flow_supervised(latent_dim=latent_dim, s_prop=s_dim_prop, c_s_dependent=False,
                             data_dim=source_adata.shape[1], domain_dim=source_adata.obs[batch].nunique(),
                             label_dim=source_adata.obs[label].nunique(), embedding_dim=embedding_dim, dr_rate=dr_rate,
                             hidden_dim=hidden_dim, encoder_n_layers=n_layers, decoder_n_layers=n_layers,
                             flows_n_layers=flows_n_layers, max_epochs=max_epochs, learning_rate=learning_rate,
                             recon_loss=recon_loss, normalisation=normalisation, lambda_kl=lambda_kl,
                             lambda_prot=lambda_prot, lambda_spar=lambda_spar, lambda_mask=lambda_mask,
                             lambda_part=lambda_part, patient=patient, valid_prop=.1, batch_size=batch_size,
                             flows=flows, pretrain_epoch_rate=0)

train_data = DatasetVAE(source_adata, batch, label)
model.fit(train_data)
# get latent representation of query data
target_adata.obsm['ivae_partial'] = model.embed(target_adata.X.toarray()).detach().numpy()
m = metrics(target_adata, target_adata, batch, label, embed='ivae_partial', ari_=True, silhouette_=True,
            isolated_labels_asw_=True, nmi_=True, pcr_=True, graph_conn_=True).T
print(f'partial averaged integration score: {m.mean().mean(): .4f}')

target_adata.obsm['ivae_full'] = model.embed(target_adata.X.toarray(), True).detach().numpy()
m = metrics(target_adata, target_adata, batch, label, embed='ivae_full', ari_=True, silhouette_=True,
            isolated_labels_asw_=True, nmi_=True, pcr_=True, graph_conn_=True).T
print(f'full averaged integration score: {m.mean().mean(): .4f}')
y_true = target_adata.obs[label].values.codes
y_pred = model.classify(target_adata)

f1_macro = f1_score(y_true, y_pred, average='macro')
f1_weighted = f1_score(y_true, y_pred, average='weighted')
print(f'F1 macro: {f1_macro: .4f}, F1 weighted: {f1_weighted: .4f}')
a = 1
