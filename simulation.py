# cell_type
# domain_index
# gene_expression
import warnings

import anndata
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from scib.metrics import metrics
from torch.utils.data import DataLoader

from model.model import iVAE_flow_unsupervised
from model.sssa import SSA
from utils.data_generation import data_generator_synthetic, DANS
from utils.utils import DatasetIVAE
import scanpy as sc

warnings.filterwarnings('ignore')

# generating data
print("----------We use a freshly generate dataset.-------------")
generator = 'yang'

beta_list = [0.5]
nfl_list = [2, 3, 4, 5, 6]
max_epochs = 100
lr = 0.005
data_dim = 100
patient = 20
model_name = 'yang'
# model_name = 'lingjing'
hidden_dim = 32
encoder_n_layers = 6
decoder_n_layers = 6
flows_type = 'spline'
batch_size = 128
seed = 127
normalisation = None
dropout = 0

for beta in beta_list:
    for flows_n_layers in nfl_list:
        train_data, test_data = data_generator_synthetic(var_range_l=0.01, var_range_r=1, domain_dim=5,
                                                         data_dim=data_dim, NsegmentObs_train=5000, Nlayer=2,
                                                         assumption=False, seed=seed)
        if model_name == 'yang':
            train_dataset = DatasetIVAE(train_data['x'], train_data['domain'], train_data['y'])

            model = iVAE_flow_unsupervised(latent_dim=4, s_prop=0.5, domain_dim=5, data_dim=data_dim,
                                           hidden_dim=hidden_dim, learning_rate=lr, max_epochs=max_epochs,
                                           patient=patient, n_layers=encoder_n_layers, flows_n_layers=flows_n_layers,
                                           lambda_spar=0, lambda_mask=0, lambda_kl=beta, flows=flows_type,
                                           batch_size=batch_size, normalisation=normalisation, dr_rate=dropout,
                                           recon_loss='mse')

            # Train the model
            model.fit(train_dataset)
            z = model.embed(test_data['x'], test_data['domain'], True)
        else:
            train_dataset, test_dataset = DANS(train_data), DANS(test_data)

            # pl.seed_everything(950127)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            model = SSA(input_dim=data_dim, c_dim=2, s_dim=2, nclass=5, embedding_dim=0, n_flow_layers=flows_n_layers,
                        hidden_dim=hidden_dim, encoder_n_layers=encoder_n_layers, decoder_n_layers=decoder_n_layers,
                        lr=lr, beta=beta, gamma=beta)
            trainer = pl.Trainer(accelerator='cpu', check_val_every_n_epoch=20, max_epochs=max_epochs,
                                 deterministic=False, callbacks=[EarlyStopping(monitor="val_recon_loss", mode="min")])
            # Train the model
            trainer.fit(model, train_loader, val_loader)
            z = model.embed(test_dataset.data)

        fig, axs = plt.subplots(4, 4, figsize=(12, 9), sharey='row')
        for row in range(4):
            for col in range(4):
                axs[row, col].scatter(test_data['source'][:, row], z[:, col], alpha=0.5, c='b', s=1)
                if col == 0:
                    axs[row, col].set_ylabel('predicted source {}'.format(row))
                if row == 3:
                    axs[row, col].set_xlabel('true source {}'.format(col))
        fig.suptitle('model: ' + model_name + ', data dimension: ' + str(data_dim) + ', hidden dimension: ' + str(
            hidden_dim) + ' seed: ' + str(seed) + '\nencoder layers: ' + str(
            encoder_n_layers) + ', decoder layers: ' + str(decoder_n_layers) + ', flows layers: ' + str(
            flows_n_layers) + ', normalisation: ' + str(normalisation) + ', dropout: ' + str(
            dropout) + '\nflows type: ' + flows_type + ', learning rate: ' + str(lr) + ', KL weights: ' + str(beta))
        plt.show()
        valid_adata = anndata.AnnData(test_data['x'])
        valid_adata.obs['domain'] = test_data['domain']
        valid_adata.obs['domain'] = valid_adata.obs.domain.astype('category')
        valid_adata.obs['label'] = test_data['y']
        valid_adata.obs['label'] = valid_adata.obs.label.astype('category')

        valid_adata.obsm['embed'] = z
        m = metrics(valid_adata, valid_adata, 'domain', 'label', embed='embed', ari_=True,
                    silhouette_=True, isolated_labels_asw_=True, nmi_=True, pcr_=True, graph_conn_=True).T
        print(m.mean().mean())
        sc.tl.tsne(valid_adata, use_rep='embed')
        sc.pl.tsne(valid_adata, color='domain', title='domain, score=' + str(m.mean().mean()))
        sc.pl.tsne(valid_adata, color='label', title='label, score=' + str(m.mean().mean()))
        a = 1
