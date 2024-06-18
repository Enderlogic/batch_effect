import collections
import time
from collections import defaultdict
from datetime import datetime
from statistics import mean

import anndata
import pandas as pd
import seaborn as sns
import normflows as nf
import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from scib.metrics import metrics
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.autograd import Variable
from torch.distributions import Normal, MultivariateNormal
from torch.nn import init
from torch.utils.data import random_split, DataLoader
import scanpy as sc
from utils.utils import DatasetVAE
from .flow_network import NormalizingFlow, DDSF, ConditionalFlow


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers=1, hidden_dim=1024, normalization='batch', dropout=None):
        super(MLP, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module(name="Input L", module=nn.Linear(input_dim, hidden_dim))
        self.model.add_module(name="Input A", module=nn.ReLU())
        for i in range(n_layers):
            self.model.add_module(name="L{:d}".format(i), module=nn.Linear(hidden_dim, hidden_dim))
            if normalization == 'batch':
                self.model.add_module(name="N{:d}".format(i), module=nn.BatchNorm1d(hidden_dim))
            elif normalization == 'layer':
                self.model.add_module(name="N{:d}".format(i), module=nn.LayerNorm(hidden_dim))
            self.model.add_module(name="A{:d}".format(i), module=nn.ReLU())
            if dropout is not None:
                self.model.add_module(name="D{:d}".format(i), module=nn.Dropout(p=dropout))
        self.model.add_module(name='Output', module=nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.model(x)


class VAE_MLP(nn.Module):
    def __init__(self, data_dim: int, latent_dim: int, c_dim: int, hidden_dim: int, n_layers: int = 3,
                 normalization='batch', dropout: float = 0., recon_loss: str = 'mse',
                 epsilon: float = 1e-3, device=torch.device('cpu')):
        super(VAE_MLP, self).__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.normalization = normalization
        self.dropout = dropout
        self.epsilon = epsilon
        self.device = device
        self.c_dim = c_dim

        # encoder
        self.encoder = nn.Sequential()
        layer_sizes = [data_dim] + [hidden_dim] * n_layers
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.encoder.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if normalization == 'batch':
                self.encoder.add_module(name="N{:d}".format(i), module=nn.BatchNorm1d(out_size))
            elif normalization == 'layer':
                self.encoder.add_module(name="N{:d}".format(i), module=nn.LayerNorm(out_size, elementwise_affine=False))
            self.encoder.add_module(name="A{:d}".format(i), module=nn.ReLU())
            if dropout > 0:
                self.encoder.add_module(name="D{:d}".format(i), module=nn.Dropout(p=dropout))
        self.encoder.add_module(name="Output", module=nn.Linear(layer_sizes[-1], latent_dim * 2))

        # decoder
        self.decoder = nn.Sequential()
        layer_sizes = [latent_dim] + [hidden_dim] * n_layers
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.decoder.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size, bias=False))
            if normalization == 'batch':
                self.decoder.add_module(name="N{:d}".format(i), module=nn.BatchNorm1d(out_size))
            elif normalization == 'layer':
                self.decoder.add_module(name="N{:d}".format(i), module=nn.LayerNorm(out_size, elementwise_affine=False))
            self.decoder.add_module(name="A{:d}".format(i), module=nn.ReLU())
            if dropout > 0:
                self.decoder.add_module(name="D{:d}".format(i), module=nn.Dropout(p=dropout))

        if recon_loss == 'zinb':
            self.decoder.add_module(name="Output", module=nn.Linear(hidden_dim, data_dim * 2))
        elif recon_loss == 'mse':
            self.decoder.add_module(name="Output L", module=nn.Linear(hidden_dim, data_dim))
            self.decoder.add_module(name="Output", module=nn.ReLU())
        elif recon_loss in ['ce', 'nb']:
            self.decoder.add_module(name="Output L", module=nn.Linear(hidden_dim, data_dim))
            self.decoder.add_module(name="Output", module=nn.Softmax(dim=-1))
        else:
            raise NotImplementedError

        # self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, jacobian_computation=False):
        distributions = self.encoder(x)
        mu = distributions[:, :self.latent_dim]
        logvar = distributions[:, self.latent_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self.decoder(z)
        if jacobian_computation:
            z_delta = z.repeat(self.c_dim, 1) + torch.eye(self.latent_dim)[:self.c_dim, :].repeat(1,
                                                                                                  z.shape[0]).reshape(
                self.c_dim * z.shape[0], -1).to(self.device) * self.epsilon
            x_delta = self.decoder(z_delta)
            jacobian_matrix = ((x_delta - x_recon.repeat(self.c_dim, 1)) / self.epsilon).reshape(self.c_dim, x.shape[0],
                                                                                                 -1)
        else:
            jacobian_matrix = None
        return x_recon, mu, logvar, z, jacobian_matrix


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def one_hot_encoder(idx, n_cls):
    assert torch.max(idx).item() < n_cls
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n_cls)
    onehot = onehot.to(idx.device)
    onehot.scatter_(1, idx.long(), 1)
    return onehot


def nb(x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, eps=1e-8):
    """
       This negative binomial function was taken from:
       Title: scvi-tools
       Authors: Romain Lopez <romain_lopez@gmail.com>,
                Adam Gayoso <adamgayoso@berkeley.edu>,
                Galen Xing <gx2113@columbia.edu>
       Date: 16th November 2020
       Code version: 0.8.1
       Availability: https://github.com/YosefLab/scvi-tools/blob/8f5a9cc362325abbb7be1e07f9523cfcf7e55ec0/scvi/core/distributions/_negative_binomial.py

       Computes negative binomial loss.
       Parameters
       ----------
       x: torch.Tensor
            Torch Tensor of ground truth data.
       mu: torch.Tensor
            Torch Tensor of means of the negative binomial (has to be positive support).
       theta: torch.Tensor
            Torch Tensor of inverse dispersion parameter (has to be positive support).
       eps: Float
            numerical stability constant.

       Returns
       -------
       If 'mean' is 'True' NB loss value gets returned, otherwise Torch tensor of losses gets returned.
    """
    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))

    log_theta_mu_eps = torch.log(theta + mu + eps)
    res = theta * (torch.log(theta + eps) - log_theta_mu_eps) + x * (
            torch.log(mu + eps) - log_theta_mu_eps) + torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(
        x + 1)

    return res


def zinb(x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, pi: torch.Tensor, eps=1e-8):
    """
       This zero-inflated negative binomial function was taken from:
       Title: scvi-tools
       Authors: Romain Lopez <romain_lopez@gmail.com>,
                Adam Gayoso <adamgayoso@berkeley.edu>,
                Galen Xing <gx2113@columbia.edu>
       Date: 16th November 2020
       Code version: 0.8.1
       Availability: https://github.com/YosefLab/scvi-tools/blob/8f5a9cc362325abbb7be1e07f9523cfcf7e55ec0/scvi/core/distributions/_negative_binomial.py

       Computes zero inflated negative binomial loss.
       Parameters
       ----------
       x: torch.Tensor
            Torch Tensor of ground truth data.
       mu: torch.Tensor
            Torch Tensor of means of the negative binomial (has to be positive support).
       theta: torch.Tensor
            Torch Tensor of inverses dispersion parameter (has to be positive support).
       pi: torch.Tensor
            Torch Tensor of logits of the dropout parameter (real support)
       eps: Float
            numerical stability constant.

       Returns
       -------
       If 'mean' is 'True' ZINB loss value gets returned, otherwise Torch tensor of losses gets returned.
    """
    # theta is the dispersion rate. If .ndimension() == 1, it is shared for all cells (regardless of batch or labels)
    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))  # In this case, we reshape theta for broadcasting

    softplus_pi = nn.functional.softplus(-pi)  #  uses log(sigmoid(x)) = -softplus(-x)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = nn.functional.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (
            -softplus_pi
            + pi_theta_log
            + x * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1)
    )
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero
    return res


class iVAE_flow_unsupervised(nn.Module):
    def __init__(self, latent_dim: int, s_prop: float, data_dim: int, domain_dim: int, embedding_dim: int = 5,
                 dr_rate: float = 0.05, hidden_dim=None, n_layers: int = 6, flows_n_layers: int = 4,
                 max_epochs: int = 100, learning_rate: float = 1e-3, recon_loss: str = 'mse',
                 normalisation='batch', lambda_kl: float = .1, lambda_spar: float = .1, lambda_mask: float = 1,
                 patient: int = 10, valid_prop: float = .1, batch_size: int = 100, flows: str = 'quadraticspline'):
        super().__init__()
        self.best_state_dict = None
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        if s_prop > 1 or s_prop < 0:
            raise ValueError('s_prop must be between 0 and 1')
        else:
            self.s_prop = s_prop
            self.s_dim = round(latent_dim * s_prop)
            self.c_dim = latent_dim - self.s_dim
        self.domain_dim = domain_dim
        self.embedding_dim = embedding_dim
        self.recon_loss = recon_loss
        self.theta = torch.nn.Parameter(torch.randn(self.data_dim, self.domain_dim)) if recon_loss in ["nb",
                                                                                                       "zinb"] else None
        self.dr_rate = dr_rate
        self.hidden_dim = int(np.ceil(np.sqrt(data_dim))) if hidden_dim is None else hidden_dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_epochs = max_epochs
        self.lr = learning_rate
        self.epsilon = 1e-4
        self.lambda_spar = lambda_spar
        self.lambda_mask = lambda_mask
        self.lambda_kl = lambda_kl
        self.patient = patient
        assert 0 < valid_prop < 1, 'valid proportion must be between 0 and 1'
        self.valid_prop = valid_prop
        self.batch_size = batch_size

        # decoder params
        self.net = VAE_MLP(data_dim=data_dim, latent_dim=latent_dim, c_dim=self.c_dim, hidden_dim=self.hidden_dim,
                           n_layers=n_layers, normalization=normalisation, dropout=dr_rate, recon_loss=recon_loss)
        self.domain_embedding = nn.Embedding(self.domain_dim, self.embedding_dim)

        # normalising flows
        self.flows = flows
        if flows == 'ddsf':
            self.domain_flows = DDSF(flows_n_layers, 1, 16, 1)
            domain_num_params = self.domain_flows.num_params * self.s_dim
            self.domain_mlp = MLP(self.embedding_dim, domain_num_params)
        elif flows == 'condspline':
            self.domain_flows = ConditionalFlow(self.s_dim, self.embedding_dim)
        elif flows == 'quadraticspline':
            flows = []
            for i in range(flows_n_layers):
                flows += [nf.flows.AutoregressiveRationalQuadraticSpline(self.s_dim, 2, self.hidden_dim,
                                                                         num_context_channels=self.embedding_dim),
                          nf.flows.LULinearPermute(self.s_dim)]
            self.domain_flows = nf.ConditionalNormalizingFlow(
                nf.distributions.DiagGaussian(self.s_dim, trainable=False), flows)
        elif flows == 'spline':
            # Spline flow model to learn the noise distribution
            self.domain_flows = []
            for i in range(self.domain_dim):
                spline = NormalizingFlow(input_dim=self.s_dim, n_layers=flows_n_layers, bound=5, count_bins=8,
                                         order='linear')
                self.domain_flows.append(spline)
            self.domain_flows = nn.ModuleList(self.domain_flows)
        else:
            raise ValueError('flows: ' + flows + ' is not defined.')
        self.prior_dist_zs = MultivariateNormal(torch.zeros(self.s_dim), torch.eye(self.s_dim))
        self.importance = nn.Parameter(torch.ones((1, self.c_dim)))

    def nf(self, z, domain):
        domain_embed = self.domain_embedding(domain)
        if self.flows == 'ddsf':
            B, _ = domain_embed.size()
            dsparams = self.domain_mlp(domain_embed).view(B, self.s_dim, -1)
            tilde_zs, logdet = self.domain_flows(z, dsparams)
        elif self.flows == 'quadraticspine':
            tilde_zs, logdet = self.domain_flows.forward_and_log_det(z, domain_embed)
        elif self.flows == 'spline':
            tilde_zs = torch.zeros_like(z)
            logdet = torch.zeros((z.shape[0],))
            for id in domain.unique():
                index = domain == id
                tilde_zs[index, :], logdet[index] = self.domain_flows[id](z[index])
        return tilde_zs, logdet

    def loss(self, x, domain=None):
        if self.recon_loss in ['nb', 'zinb']:
            x = torch.log(x + 1)
        if self.c_dim > 0 and self.lambda_spar > 0:
            x_recon, mu, logvar, z, jacobian_matrix = self.net(x, jacobian_computation=True)
        else:
            x_recon, mu, logvar, z, jacobian_matrix = self.net(x)
        domain = domain.type(torch.int64)
        q_dist = Normal(mu, torch.exp(logvar / 2))
        log_qz = q_dist.log_prob(z)

        zc = z[:, :self.c_dim]
        if self.lambda_mask > 0:
            zc *= (self.importance > .1).detach().float() * self.importance
        zs = z[:, self.c_dim:]

        # pass zs to a normalizing flow to obtain zs tilde
        zs_tilde, logdet_zs = self.nf(zs, domain)
        # calculate kl divergence loss for zs_tilde
        log_qzs = log_qz[:, self.c_dim:].sum(1)
        log_pzs = self.prior_dist_zs.log_prob(zs_tilde) + logdet_zs
        kl_zs = (log_qzs - log_pzs).mean()
        # calculate kl divergence loss for zc
        log_qzc = log_qz[:, :self.c_dim].sum(1)
        log_pzc = Normal(torch.zeros_like(zc), torch.ones_like(zc)).log_prob(zc).sum(1)
        kl_zc = (log_qzc - log_pzc).mean()
        loss_kl = kl_zc + kl_zs

        # calculate reconstruction loss
        if self.recon_loss == 'mse':
            loss_recon = nn.functional.mse_loss(x_recon, x, reduction='none').sum(1).mean()
        elif self.recon_loss == 'ce':
            loss_recon = nn.functional.cross_entropy(x_recon, x, reduction='mean')
        elif self.recon_loss == 'nb':
            size_factor_view = x.sum(1).unsqueeze(1).expand(x_recon.size(0), x_recon.size(1))
            mean = x_recon * size_factor_view
            dispersion = nn.functional.linear(one_hot_encoder(domain, self.domain_dim), self.theta).exp()
            loss_recon = -nb(x=x, mu=mean, theta=dispersion).sum(dim=-1).mean()
        elif self.recon_loss == 'zinb':
            mean = nn.functional.softmax(x_recon[:, :self.data_dim], dim=1)
            dropout = x_recon[:, self.data_dim:]
            dispersion = nn.functional.linear(one_hot_encoder(domain, self.domain_dim), self.theta).exp()
            loss_recon = -zinb(x=x, mu=mean, theta=dispersion, pi=dropout).sum(dim=-1).mean()
        else:
            raise ValueError('recon_loss: ' + self.recon_loss + ' is not defined.')

        # calculate sparsity loss
        if self.c_dim > 0 and self.lambda_spar > 0:
            loss_spar = jacobian_matrix.abs().mean(1).sum()
        else:
            loss_spar = torch.tensor(0)

        # calculate mask loss for zc
        loss_mask = self.importance.abs().sum() if self.lambda_mask > 0 else torch.tensor(0)
        return loss_recon, loss_kl, loss_spar, loss_mask

    def validation(self, data, label):
        y_pred = self.predict(data)
        return roc_auc_score(label, y_pred.numpy(), multi_class='ovr')

    def fit(self, data):
        train_data, valid_data = random_split(data, [1 - self.valid_prop, self.valid_prop])
        train_loader = DataLoader(train_data, shuffle=True, batch_size=self.batch_size)
        valid_data = valid_data.dataset[valid_data.indices]
        # train_loader = DataLoader(data, shuffle=True, batch_size=self.batch_size)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr,
                                      betas=(0.9, 0.999), weight_decay=0.0001)
        num_train_loader = len(train_loader)
        valid_loss_best = np.inf
        patient_cache = 0
        for epoch in range(self.max_epochs):
            self.train()
            loss_train = loss_recon = loss_kl = loss_spar = loss_mask = 0
            for batch_idx, data in enumerate(train_loader):
                recon, kl, spar, mask = self.loss(data['x'].to(self.device), data['domain'].to(self.device))
                loss = recon + self.lambda_kl * kl + self.lambda_spar * spar + self.lambda_mask * mask
                loss_train += loss.item() / num_train_loader
                loss_recon += recon / num_train_loader
                loss_kl += kl / num_train_loader
                loss_spar += spar / num_train_loader
                loss_mask += mask / num_train_loader
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.eval()
            recon, kl, spar, mask = self.loss(valid_data['x'].to(self.device), valid_data['domain'].to(self.device))
            if valid_loss_best > recon + self.lambda_kl * kl + self.lambda_spar * spar + self.lambda_mask * mask:
                valid_loss_best = recon + self.lambda_kl * kl + self.lambda_spar * spar + self.lambda_mask * mask
                self.best_state_dict = self.state_dict()
                patient_cache = 0
            else:
                patient_cache += 1
            if patient_cache >= self.patient:
                self.load_state_dict(self.best_state_dict)
                break
            if (epoch + 1) % 10 == 0:
                print(datetime.now())
                print(
                    f'Epoch {epoch + 1}, Average loss: {loss_train:.4f}, loss_recon: {loss_recon:.4f}, loss_kl: {loss_kl * self.lambda_kl:.4f}, loss_spar: {loss_spar * self.lambda_spar:.4f}, loss_mask: {loss_mask * self.lambda_mask:.4f}, best validation loss: {valid_loss_best:.4f}')
                # tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
                # data_embed = self.embed(valid_data['x'], valid_data['domain'], full=True)
                # tsne_results = tsne.fit_transform(data_embed)
                # df_subset = pd.DataFrame(tsne_results, columns=['tsne 1', 'tsne 2'])
                # plt.figure(figsize=(16, 10))
                # colors = cm.rainbow(np.linspace(0, 1, valid_data['domain'].unique().shape[0]))
                # for y in valid_data['domain'].unique():
                #     plt.scatter(df_subset.iloc[np.where(valid_data['domain'] == y)[0], 0],
                #                 df_subset.iloc[np.where(valid_data['domain'] == y)[0], 1], color=colors[y])
                # plt.show()
                # a = 1

    def embed(self, data, domain=None, full=False):
        self.eval()
        _, _, _, z, _ = self.net(torch.Tensor(data).to(self.device))
        zc = z[:, :self.c_dim]
        if full:
            if self.lambda_mask > 0:
                zc *= (self.importance > .1).detach().float() * self.importance
            zs = z[:, self.c_dim:]
            # pass zs to a normalizing flow to obtain zs tilde
            zs_tilde, _ = self.nf(zs, torch.tensor(domain, dtype=torch.int64).to(self.device))
            return torch.concat((zc, zs_tilde), 1).detach().numpy()
        else:
            return zc.detach().numpy()


class iVAE(nn.Module):
    def __init__(self, latent_dim: int, s_prop: float, data_dim: int, domain_dim: int, label_dim: int,
                 embedding_dim: int = 5, dr_rate: float = 0.05, hidden_dim=None, n_layers: int = 6,
                 flows_n_layers: int = 4, max_epochs: int = 100, learning_rate: float = 1e-3, recon_loss: str = 'mse',
                 normalisation='batch', lambda_kl: float = .1, lambda_clas: float = 1, lambda_spar: float = .1,
                 lambda_mask: float = 1, patient: int = 10, valid_prop: float = .1, batch_size: int = 100,
                 pretrain_epoch_rate: float = .9, full_embedding=True, flows='spline'):
        super().__init__()
        self.iter_logs = defaultdict(list)
        self.logs = defaultdict(list)
        self.test = defaultdict(list)
        self.best_state_dict = None
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        assert 0 <= round(latent_dim * s_prop) < latent_dim, 's_prop must be between 0 and 1'
        self.s_dim = round(latent_dim * s_prop)
        self.c_dim = latent_dim - self.s_dim
        self.domain_dim = domain_dim
        self.label_dim = label_dim
        self.embedding_dim = embedding_dim
        self.recon_loss = recon_loss
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.theta = torch.nn.Parameter(
            torch.randn(self.data_dim, self.domain_dim, device=self.device)) if recon_loss in ["nb", "zinb"] else None
        self.dr_rate = dr_rate
        self.hidden_dim = int(np.ceil(np.sqrt(data_dim))) if hidden_dim is None else hidden_dim
        self.num_workers = 0 if self.device.type == 'cpu' else 0
        self.max_epochs = max_epochs
        self.lr = learning_rate
        self.epsilon = 1e-4
        self.lambda_spar = lambda_spar
        self.lambda_mask = lambda_mask
        self.lambda_kl = lambda_kl
        self.lambda_clas = lambda_clas
        assert patient > 0, 'patient must be greater than zero'
        self.patient = patient
        assert 0 <= valid_prop < 1, 'valid proportion must be between 0 and 1'
        self.valid_prop = valid_prop
        self.batch_size = batch_size
        self.epoch = 0
        self.pretrain_epochs = round(pretrain_epoch_rate * max_epochs)
        self.best_performance = np.inf

        # decoder params
        self.net = VAE_MLP(data_dim=data_dim, latent_dim=latent_dim, c_dim=self.c_dim, hidden_dim=self.hidden_dim,
                           n_layers=n_layers, normalization=normalisation, dropout=dr_rate, recon_loss=recon_loss,
                           device=self.device).to(self.device)
        if flows is not None and self.s_dim > 0:
            if flows == 'quadraticspline':
                self.domain_embedding = nn.Embedding(self.domain_dim, self.embedding_dim, device=self.device)
            self.flows = flows
            if flows == 'quadraticspline':
                flows = []
                for i in range(flows_n_layers):
                    flows += [nf.flows.AutoregressiveRationalQuadraticSpline(self.s_dim, 2, self.hidden_dim,
                                                                             num_context_channels=self.embedding_dim),
                              nf.flows.LULinearPermute(self.s_dim)]
                self.domain_flows = nf.ConditionalNormalizingFlow(
                    nf.distributions.DiagGaussian(self.s_dim, trainable=False).to(self.device), flows).to(self.device)
            elif flows == 'spline':
                # Spline flow model to learn the noise distribution
                self.domain_flows = []
                for i in range(self.domain_dim):
                    spline = NormalizingFlow(input_dim=self.s_dim, n_layers=flows_n_layers, bound=5, count_bins=8,
                                             order='linear')
                    self.domain_flows.append(spline)
                self.domain_flows = nn.ModuleList(self.domain_flows).to(self.device)
            else:
                self.domain_flows = None

            self.prior_dist_zs = MultivariateNormal(torch.zeros(self.s_dim, device=self.device),
                                                    torch.eye(self.s_dim, device=self.device))
        self.importance = nn.Parameter(torch.ones((1, self.c_dim)))
        self.full_embedding = full_embedding
        if full_embedding:
            self.prototypes = torch.zeros(label_dim, self.latent_dim, device=self.device)
        else:
            self.prototypes = torch.zeros(label_dim, self.c_dim, device=self.device)

    def nf(self, z, domain):
        if self.flows is None or self.s_dim == 0:
            return z, None
        elif self.flows == 'quadraticspline':
            domain_embed = self.domain_embedding(domain)
            tilde_zs, logdet = self.domain_flows.inverse_and_log_det(z, domain_embed)
        elif self.flows == 'spline':
            tilde_zs = torch.zeros_like(z)
            logdet = torch.zeros((z.shape[0],), device=self.device)
            for id in domain.unique():
                index = domain == id
                tilde_zs[index, :], logdet[index] = self.domain_flows[id](z[index])
        else:
            raise Exception('Unknown flows type: ' + self.flows)
        return tilde_zs, logdet

    def loss(self, x, domain, label=None):
        if self.recon_loss in ['nb', 'zinb']:
            x = torch.log(x + 1)
        domain = domain.type(torch.int64)
        domain = domain[label > -1]
        x = x[label > -1, :]
        x_recon, mu, logvar, z, jacobian_matrix = self.net(x, True if self.lambda_spar > 0 else False)
        q_dist = Normal(mu, torch.exp(logvar / 2))
        log_qz = q_dist.log_prob(z)

        zc = z[:, :self.c_dim]
        if self.lambda_mask > 0:
            zc *= (self.importance > .1).detach().float() * self.importance
        zs = z[:, self.c_dim:]

        # pass zs to a normalizing flow to obtain zs tilde
        # start = time.time()
        zs_tilde, logdet_zs = self.nf(zs, domain)
        # self.test['zs_tilde'].append(time.time() - start)
        embed = torch.concat((zc, zs_tilde), dim=1) if self.full_embedding else zc
        # calculate kl divergence loss for zs_tilde
        log_qzs = log_qz[:, self.c_dim:].sum(1)
        log_pzs = self.prior_dist_zs.log_prob(zs_tilde)
        if logdet_zs is not None:
            log_pzs += logdet_zs
        kl_zs = (log_qzs - log_pzs).mean()
        # calculate kl divergence loss for zc
        log_qzc = log_qz[:, :self.c_dim].sum(1)
        log_pzc = Normal(torch.zeros_like(zc), torch.ones_like(zc)).log_prob(zc).sum(1)
        kl_zc = (log_qzc - log_pzc).mean()
        loss_kl = kl_zc + kl_zs
        # calculate reconstruction loss
        if self.recon_loss == 'mse':
            loss_recon = nn.functional.mse_loss(x_recon, x, reduction='none').sum(1).mean()
        elif self.recon_loss == 'ce':
            loss_recon = nn.functional.cross_entropy(x_recon, x, reduction='mean')
        elif self.recon_loss == 'nb':
            size_factor_view = x.sum(1).unsqueeze(1).expand(x_recon.size(0), x_recon.size(1))
            mean = x_recon * size_factor_view
            dispersion = nn.functional.linear(one_hot_encoder(domain, self.domain_dim), self.theta).exp()
            loss_recon = -nb(x=x, mu=mean, theta=dispersion).sum(dim=-1).mean()
        elif self.recon_loss == 'zinb':
            mean = nn.functional.softmax(x_recon[:, :self.data_dim], dim=1)
            dropout = x_recon[:, self.data_dim:]
            dispersion = nn.functional.linear(one_hot_encoder(domain, self.domain_dim), self.theta).exp()
            loss_recon = -zinb(x=x, mu=mean, theta=dispersion, pi=dropout).sum(dim=-1).mean()
        else:
            raise ValueError('recon_loss: ' + self.recon_loss + ' is not defined.')
        # calculate sparsity loss
        loss_spar = jacobian_matrix.abs().mean(
            1).sum() * self.lambda_spar if self.c_dim > 0 and self.lambda_spar > 0 else torch.tensor(0.,
                                                                                                     device=self.device)
        # calculate prototype loss
        loss_clas = sum([(embed[label == i, :] - self.prototypes[i, :]).pow(2).sum(1).sqrt().mean() for i in
                         label.unique()[
                             label.unique() != -1]]) * self.lambda_clas if self.epoch >= self.pretrain_epochs and self.lambda_clas > 0 else torch.tensor(
            0., device=self.device)
        # calculate mask loss for zc
        loss_mask = self.importance.abs().sum() * self.lambda_mask if self.lambda_mask > 0 else torch.tensor(0.,
                                                                                                             device=self.device)
        loss = loss_recon + loss_kl + loss_clas + loss_spar + loss_mask
        # logs
        self.iter_logs['loss'].append(loss.item())
        self.iter_logs['loss_recon'].append(loss_recon.item())
        self.iter_logs['loss_clas'].append(loss_clas.item())
        self.iter_logs['loss_kl'].append(loss_kl.item())
        self.iter_logs['loss_spar'].append(loss_spar.item())
        self.iter_logs['loss_mask'].append(loss_mask.item())
        return loss

    def validation(self, data, label):
        y_pred = self.predict(data)
        return roc_auc_score(label, y_pred.numpy(), multi_class='ovr')

    def update_prototype(self, data):
        with torch.no_grad():
            latent = self.embed(data.data, data.domain)
            for i in range(self.label_dim):
                self.prototypes[i, :] = latent[data.label == i, :].mean(0)

    def fit(self, adata, domain_name, label_name):
        dataset = DatasetVAE(adata, domain_name, label_name)
        train_data, valid_data = random_split(dataset, [1 - self.valid_prop, self.valid_prop])
        train_loader = DataLoader(train_data, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers)
        if self.valid_prop > 0:
            valid_data = valid_data.dataset[valid_data.indices]
            valid_adata = anndata.AnnData(valid_data['x'].numpy())
            valid_adata.obs['domain'] = valid_data['domain'].numpy()
            valid_adata.obs['domain'] = valid_adata.obs.domain.astype('category')
            valid_adata.obs['label'] = valid_data['label']
            valid_adata.obs['label'] = valid_adata.obs.label.astype('category')
        # train_loader = DataLoader(data, shuffle=True, batch_size=self.batch_size)
        # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr,
        #                               betas=(0.9, 0.999), weight_decay=0.0001)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, eps=.01, weight_decay=.04)
        for self.epoch in range(self.max_epochs):
            # start = time.time()
            if self.epoch == self.pretrain_epochs and self.lambda_clas > 0:
                self.update_prototype(train_data.dataset)
                self.best_performance = np.inf
            self.iter_logs = defaultdict(list)
            self.test = defaultdict(list)
            # cost_begin = time.time() - start
            # cost_loss = 0
            # cost_optimizer = 0
            self.train()
            for _, data in enumerate(train_loader):
                # start = time.time()
                loss = self.loss(data['x'].to(self.device), data['domain'].to(self.device),
                                 data['label'].to(self.device))
                # cost_loss += time.time() - start
                # start = time.time()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # cost_optimizer += time.time() - start
            # start = time.time()
            # print('total cost: {:.3f}, nf cost: {:.3f}'.format(cost_loss, sum(self.test['zs_tilde'])))
            if (self.epoch + 1) % 10 == 0:
                print(datetime.now())
                print(
                    'Epoch {}, loss: {:.3f}, loss_recon: {:.3f}, loss_clas: {:.3f}, loss_kl: {:.3f}, loss_spar: {:.3f}, loss_mask: {:.3f}, best validation loss: {:.3f}'.format(
                        self.epoch + 1, mean(self.iter_logs['loss']), mean(self.iter_logs['loss_recon']),
                        mean(self.iter_logs['loss_clas']), mean(self.iter_logs['loss_kl']),
                        mean(self.iter_logs['loss_spar']), mean(self.iter_logs['loss_mask']), self.best_performance))
                # valid_adata.obsm['embed'] = self.embed(valid_data['x'], valid_data['domain']).detach().cpu().numpy()
                # m = metrics(valid_adata, valid_adata, 'domain', 'label', embed='embed', ari_=True,
                #             silhouette_=True, isolated_labels_asw_=True, nmi_=True, pcr_=True, graph_conn_=True).T
                # print(m.mean().mean())
                # sc.tl.tsne(valid_adata, use_rep='embed')
                # sc.pl.tsne(valid_adata, color='domain', title='domain, score=' + str(m.mean().mean()))
                # sc.pl.tsne(valid_adata, color='label', title='label, score=' + str(m.mean().mean()))
            if self.valid_prop > 0:
                self.validate(valid_data)
                if self.early_stopping():
                    break

            # cost_early_stop = time.time() - start
            # start = time.time()
            if self.epoch >= self.pretrain_epochs:
                self.update_prototype(train_data.dataset)
            # cost_end = time.time() - start
            # print(
            #     'Begin costs: {:.3f} seconds, loss costs: {:.3f} seconds, optimizer costs: {:.3f} seconds, validate costs: {:.3f} seconds, early stopping costs: {:.3f} seconds, end costs: {:.3f} seconds'.format(
            #         cost_begin, cost_loss, cost_optimizer,
            #         cost_validate, cost_early_stop, cost_end))

    @torch.no_grad()
    def validate(self, valid_data):
        self.eval()
        self.iter_logs = defaultdict(list)
        self.loss(valid_data['x'].to(self.device), valid_data['domain'].to(self.device),
                  valid_data['label'].to(self.device))
        for key in self.iter_logs:
            self.logs['val_' + key].append(self.iter_logs[key][0])
        self.train()

    def early_stopping(self):
        if self.best_performance < self.logs['val_loss'][-1]:
            self.patient_cache += 1
        else:
            self.best_performance = self.logs['val_loss'][-1]
            self.patient_cache = 0
            self.best_state_dict = self.state_dict()
        if self.patient_cache >= self.patient:
            if self.epoch > self.pretrain_epochs:
                self.load_state_dict(self.best_state_dict)
                return True
            else:
                self.pretrain_epochs = self.epoch + 1
                return False
        else:
            return False

    def embed(self, data, domain=None):
        if isinstance(data, np.ndarray):
            data = torch.tensor(data)
        if isinstance(domain, np.ndarray):
            domain = torch.tensor(domain, dtype=torch.long)
        self.eval()
        _, _, _, z, _ = self.net(data.to(self.device))
        zc = z[:, :self.c_dim]
        if self.full_embedding:
            if self.lambda_mask > 0:
                zc *= (self.importance > .1).detach().float() * self.importance
            zs = z[:, self.c_dim:]
            # pass zs to a normalizing flow to obtain zs tilde
            zs_tilde, _ = self.nf(zs, domain.to(self.device))
            return torch.concat((zc, zs_tilde), 1)
        else:
            return zc

    def classify(self, data, domain=None):
        latent = self.embed(data, domain)
        distances = torch.cdist(latent, self.prototypes)
        return torch.argmin(distances, dim=1).cpu()


def get_frequent_elements(element_list, lb=2):
    counter = collections.Counter(element_list)
    return [element for element, count in counter.items() if count >= lb]


class iVAE_flow_supervised_old(nn.Module):
    def __init__(self, latent_dim: int, s_prop: float, data_dim: int, domain_dim: int, label_dim: int = 0,
                 embedding_dim: int = 5, dr_rate=.05, hidden_dim=None, n_layers: int = 6, flows_n_layers: int = 4,
                 max_epochs: int = 100, learning_rate: float = 1e-3,
                 recon_loss: str = 'mse', normalisation='batch', lambda_kl: float = .1, lambda_prot: float = 1,
                 lambda_spar: float = .1, lambda_mask: float = 1, patient: int = 10, valid_prop: float = .1,
                 batch_size: int = 100, pretrain_epoch_rate: float = 0.9):
        super().__init__()
        self.logs = defaultdict(list)
        self.iter_logs = None
        self.dataloader_valid = None
        self.best_performance = np.inf
        self.best_state_dict = None
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        if s_prop > 1 or s_prop < 0:
            raise ValueError('s_prop must be between 0 and 1')
        else:
            self.s_prop = s_prop
            self.s_dim = round(latent_dim * s_prop)
            self.c_dim = latent_dim - self.s_dim
        self.domain_dim = domain_dim
        self.embedding_dim = embedding_dim
        self.label_dim = label_dim
        self.recon_loss = recon_loss
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.theta = torch.nn.Parameter(
            torch.randn(self.data_dim, self.domain_dim, device=self.device)) if recon_loss in ["nb", "zinb"] else None
        self.dr_rate = dr_rate
        self.hidden_dim = int(np.ceil(np.sqrt(data_dim))) if hidden_dim is None else hidden_dim
        self.max_epochs = max_epochs
        self.pretrain_epochs = round(max_epochs * pretrain_epoch_rate)
        self.lr = learning_rate
        self.epsilon = 1e-4
        assert lambda_kl >= 0, 'lambda kl must be greater than or equal to 0'
        assert lambda_prot >= 0, 'lambda prot must be greater than or equal to 0'
        assert lambda_mask >= 0, 'lambda mask must be greater than or equal to 0'
        assert lambda_spar >= 0, 'lambda spar must be greater than or equal to 0'
        self.lambda_spar = lambda_spar
        self.lambda_mask = lambda_mask
        self.lambda_kl = lambda_kl
        self.lambda_prot = lambda_prot
        self.jacobian_computation = True if lambda_spar > 0 else False
        self.patient = patient
        self.patient_cache = 0
        assert 0 < valid_prop < 1, 'valid proportion must be between 0 and 1'
        self.valid_prop = valid_prop
        self.batch_size = batch_size
        self.early_stopping_metric = 'val_loss'
        self.prototypes = torch.zeros(label_dim, self.c_dim, device=self.device)
        self.epoch = 0
        # reconstruction model and classfication model
        self.net = VAE_MLP(data_dim=data_dim, latent_dim=latent_dim, c_dim=self.c_dim, hidden_dim=self.hidden_dim,
                           encoder_n_layers=n_layers, decoder_n_layers=n_layers, normalization=normalisation,
                           dropout=dr_rate, recon_loss=recon_loss, device=self.device).to(self.device)
        self.domain_embedding = nn.Embedding(self.domain_dim, self.embedding_dim, device=self.device)

        # normalising flows
        flows = []
        for i in range(flows_n_layers):
            flows += [nf.flows.AutoregressiveRationalQuadraticSpline(self.s_dim, 2, self.hidden_dim,
                                                                     num_context_channels=self.embedding_dim),
                      nf.flows.LULinearPermute(self.s_dim)]
        self.domain_flows = nf.ConditionalNormalizingFlow(
            nf.distributions.DiagGaussian(self.s_dim, trainable=False), flows).to(self.device)
        self.prior_dist_zs = MultivariateNormal(torch.zeros(self.s_dim, device=self.device),
                                                torch.eye(self.s_dim, device=self.device))
        self.importance = nn.Parameter(torch.ones((1, self.c_dim)))

    def loss(self, data):
        x = data['x'].to(self.device)
        domain = data['domain'].to(self.device)
        label = data['label'].to(self.device) if self.lambda_prot > 0 and 'label' in data else None
        x_transform = torch.log(1 + x) if self.recon_loss in ["nb", "zinb"] else x
        x_recon, mu, logvar, z, jacobian_matrix = self.net(x_transform, self.jacobian_computation)
        domain = domain.type(torch.int64)
        q_dist = Normal(mu, torch.exp(logvar / 2))
        log_qz = q_dist.log_prob(z)

        zc = z[:, :self.c_dim]
        zs = z[:, self.c_dim:]
        if self.lambda_mask > 0:
            zc *= (self.importance > .1).detach().float() * self.importance

        # pass zs to a normalizing flow to obtain zs tilde
        zs_tilde, logdet_zs = self.domain_flows.forward_and_log_det(zs, self.domain_embedding(domain))
        # calculate kl divergence loss for zs_tilde
        log_qzs = log_qz[:, self.c_dim:].sum(1)
        log_pzs = self.prior_dist_zs.log_prob(zs_tilde) + logdet_zs
        kl_zs = (log_qzs - log_pzs).mean()
        # calculate kl divergence loss for zc
        log_qzc = log_qz[:, :self.c_dim].sum(1)
        log_pzc = Normal(torch.zeros_like(zc), torch.ones_like(zc)).log_prob(zc.squeeze(0)).sum(1)
        kl_zc = (log_qzc - log_pzc).mean()
        loss_kl = (kl_zc + kl_zs) * self.lambda_kl

        # calculate reconstruction loss
        if self.recon_loss == 'mse':
            loss_recon = nn.functional.mse_loss(x_recon, x_transform, reduction='none').sum(1).mean()
        elif self.recon_loss == 'nb':
            size_factor_view = x.sum(1).unsqueeze(1).expand(x_recon.size(0), x_recon.size(1))
            mean = x_recon * size_factor_view
            dispersion = nn.functional.linear(one_hot_encoder(domain, self.domain_dim), self.theta).exp()
            loss_recon = -nb(x=x_transform, mu=mean, theta=dispersion).sum(dim=-1).mean()
        elif self.recon_loss == 'zinb':
            mean = nn.functional.softmax(x_recon[:, :self.data_dim], dim=1)
            dropout = x_recon[:, -self.data_dim:]
            dispersion = nn.functional.linear(one_hot_encoder(domain, self.domain_dim), self.theta).exp()
            loss_recon = -zinb(x=x, mu=mean, theta=dispersion, pi=dropout).sum(dim=-1).mean()
        else:
            raise ValueError('recon_loss: ' + self.recon_loss + ' is not defined.')

        # calculate prototype loss
        loss_prot = sum([(zc[label == i, :] - self.prototypes[i, :]).pow(2).sum(1).sqrt().mean() for i in
                         label.unique()]) * self.lambda_prot if self.epoch >= self.pretrain_epochs and self.lambda_prot > 0 else torch.tensor(
            0., device=self.device)

        # calculate sparsity loss
        if self.lambda_spar > 0:
            loss_spar = jacobian_matrix.abs().mean(1).sum() * self.lambda_spar
        else:
            loss_spar = torch.tensor(0., device=self.device)

        # calculate mask loss for zc
        loss_mask = self.importance.abs().sum() * self.lambda_mask if self.lambda_mask > 0 else torch.tensor(0.,
                                                                                                             device=self.device)

        # total loss
        loss = loss_recon + loss_prot + loss_kl + loss_spar + loss_mask

        # logs
        self.iter_logs['loss'].append(loss.item())
        self.iter_logs['loss_recon'].append(loss_recon.item())
        self.iter_logs['loss_prot'].append(loss_prot.item())
        self.iter_logs['loss_kl'].append(loss_kl.item())
        self.iter_logs['loss_spar'].append(loss_spar.item())
        self.iter_logs['loss_mask'].append(loss_mask.item())
        return loss

    def fit(self, data, domain_name):
        dataset = DatasetVAE(data, domain_name)
        train_data, valid_data = random_split(dataset, [1 - self.valid_prop, self.valid_prop])
        train_loader = DataLoader(train_data, shuffle=True, batch_size=self.batch_size)
        self.dataloader_valid = DataLoader(valid_data, shuffle=False, batch_size=self.batch_size)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr,
                                      betas=(0.9, 0.999), weight_decay=0.0001)
        torch.autograd.set_detect_anomaly(True)
        self.train()
        for self.epoch in range(self.max_epochs):
            if self.epoch == self.pretrain_epochs and self.lambda_prot > 0:
                self.update_prototype(train_data.dataset)
                self.best_performance = np.inf
            self.iter_logs = defaultdict(list)
            for _, data in enumerate(train_loader):
                loss = self.loss(data)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (self.epoch + 1) % 10 == 0:
                print(datetime.now())
                print(
                    'Epoch {}, loss: {:.3f}, loss_reconstruction: {:.3f}, loss_protocal: {:.3f}, loss_kl: {:.3f}, loss_spar: {:.3f}, loss_part: {:.3f}, loss_mask: {:.3f}'.format(
                        self.epoch + 1, mean(self.iter_logs['loss']), mean(self.iter_logs['loss_recon']),
                        mean(self.iter_logs['loss_prot']), mean(self.iter_logs['loss_kl']),
                        mean(self.iter_logs['loss_spar']), mean(self.iter_logs['loss_part']),
                        mean(self.iter_logs['loss_mask'])))
            self.validate()
            if self.early_stopping():
                break
            if self.epoch >= self.pretrain_epochs:
                self.update_prototype(train_data.dataset)

    def update_prototype(self, data):
        with torch.no_grad():
            latent = self.embed(data.data, data.domain)
            for i in range(self.label_dim):
                self.prototypes[i, :] = latent[data.label == i, :].mean(0)

    @torch.no_grad()
    def validate(self):
        self.eval()
        self.iter_logs = defaultdict(list)
        for _, data in enumerate(self.dataloader_valid):
            self.loss(data)
        for key in self.iter_logs:
            self.logs['val_' + key].append(np.array(self.iter_logs[key]).mean())
        self.train()

    def early_stopping(self):
        if self.best_performance < self.logs['val_loss'][-1]:
            self.patient_cache += 1
        else:
            self.best_performance = self.logs['val_loss'][-1]
            self.patient_cache = 0
            self.best_state_dict = self.state_dict()
        if self.patient_cache >= self.patient:
            self.load_state_dict(self.best_state_dict)
            return True
        else:
            return False

    def embed(self, data, domain, full=False):
        self.eval()
        _, _, _, z, _ = self.net(torch.Tensor(data).to(self.device))
        zc = z[:, :self.c_dim]
        zs = z[:, self.c_dim:]
        if self.lambda_mask > 0:
            zc *= (self.importance > .1).detach().float() * self.importance

        # pass zs to a normalizing flow to obtain zs tilde
        zs_tilde, _ = self.domain_flows.forward_and_log_det(zs, self.domain_embedding(domain))
        if full:
            return torch.cat((zc, zs_tilde), dim=1)
        else:
            return zc

    def classify(self, adata):
        self.eval()
        x = torch.tensor(adata.X.toarray(), device=self.device)
        latent = self.embed(x)
        distances = torch.cdist(latent, self.prototypes)
        return torch.argmin(distances, dim=1)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
