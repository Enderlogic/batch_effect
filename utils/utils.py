import time

import numpy as np
import torch
from sklearn import preprocessing
from torch.nn.functional import one_hot
from torch.utils.data import Dataset


def to_one_hot(x, m=None):
    if type(x) is not list:
        x = [x]
    if m is None:
        ml = []
        for xi in x:
            ml += [xi.max() + 1]
        m = max(ml)
    dtp = x[0].dtype
    xoh = []
    for i, xi in enumerate(x):
        xoh += [np.zeros((xi.size, int(m)), dtype=dtp)]
        xoh[i][np.arange(xi.size), xi.astype(int)] = 1
    return xoh


def train_val_split(adata, train_frac=0.9, condition_keys=None, cell_type_key=None, labeled_array=None):
    """Splits 'Anndata' object into training and validation data.

       Parameters
       ----------
       adata: `~anndata.AnnData`
            `AnnData` object for training the model.
       train_frac: float
            Train-test split fraction. the model will be trained with train_frac for training
            and 1-train_frac for validation.

       Returns
       -------
       Indices for training and validating the model.
    """
    indices = np.arange(adata.shape[0])

    if train_frac == 1:
        return indices, None

    if cell_type_key is not None:
        labeled_array = np.zeros((len(adata), 1)) if labeled_array is None else labeled_array
        labeled_array = np.ravel(labeled_array)

        labeled_idx = indices[labeled_array == 1]
        unlabeled_idx = indices[labeled_array == 0]

        train_labeled_idx = []
        val_labeled_idx = []
        train_unlabeled_idx = []
        val_unlabeled_idx = []

        if len(labeled_idx) > 0:
            cell_types = adata[labeled_idx].obs[cell_type_key].unique().tolist()
            for cell_type in cell_types:
                ct_idx = labeled_idx[adata[labeled_idx].obs[cell_type_key] == cell_type]
                n_train_samples = int(np.ceil(train_frac * len(ct_idx)))
                np.random.shuffle(ct_idx)
                train_labeled_idx.append(ct_idx[:n_train_samples])
                val_labeled_idx.append(ct_idx[n_train_samples:])
        if len(unlabeled_idx) > 0:
            n_train_samples = int(np.ceil(train_frac * len(unlabeled_idx)))
            train_unlabeled_idx.append(unlabeled_idx[:n_train_samples])
            val_unlabeled_idx.append(unlabeled_idx[n_train_samples:])
        train_idx = train_labeled_idx + train_unlabeled_idx
        val_idx = val_labeled_idx + val_unlabeled_idx

        train_idx = np.concatenate(train_idx)
        val_idx = np.concatenate(val_idx)

    elif condition_keys is not None:
        train_idx = []
        val_idx = []
        conditions = adata.obs['conditions_combined'].unique().tolist()
        for condition in conditions:
            cond_idx = indices[adata.obs['conditions_combined'] == condition]
            n_train_samples = int(np.ceil(train_frac * len(cond_idx)))
            np.random.shuffle(cond_idx)
            train_idx.append(cond_idx[:n_train_samples])
            val_idx.append(cond_idx[n_train_samples:])

        train_idx = np.concatenate(train_idx)
        val_idx = np.concatenate(val_idx)

    else:
        n_train_samples = int(np.ceil(train_frac * len(indices)))
        np.random.shuffle(indices)
        train_idx = indices[:n_train_samples]
        val_idx = indices[n_train_samples:]

    return train_idx, val_idx


class TrainDataset(Dataset):
    """Dataset handler for scPoli model and trainer.
       Parameters
       ----------
       adata: : `~anndata.AnnData`
            Annotated data matrix.
    """

    def __init__(self, adata, condition_key, cell_type_key):
        self.data = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        self.size_factors = self.data.sum(1)

        u_le = preprocessing.LabelEncoder()
        u_le.fit(adata.obs[condition_key])
        self.u = torch.tensor(to_one_hot(u_le.transform(adata.obs[condition_key]))[0], dtype=torch.float32)
        self.u_le = u_le
        domain_ohe = preprocessing.OneHotEncoder()
        domain_ohe.fit(self.u)
        y_le = preprocessing.LabelEncoder()
        y_le.fit(adata.obs[cell_type_key])
        self.cell_type = y_le.transform(adata.obs[cell_type_key])
        self.y_le = y_le

    def __getitem__(self, index):
        outputs = dict()
        x = self.data[index]
        outputs["x"] = x

        outputs["label"] = self.cell_type[index]
        outputs["sizefactor"] = self.size_factors[index]
        outputs["u"] = self.u[index]

        return outputs

    def __len__(self):
        return self.data.shape[0]


class ValDataset(Dataset):
    """Dataset handler for scPoli model and trainer.
       Parameters
       ----------
       adata: : `~anndata.AnnData`
            Annotated data matrix.
    """

    def __init__(self, adata, u_le, y_le, condition_key, cell_type_key):
        self.data = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        self.size_factors = self.data.sum(1)

        self.u = torch.tensor(to_one_hot(u_le.transform(adata.obs[condition_key]))[0], dtype=torch.float32)
        self.cell_type = y_le.transform(adata.obs[cell_type_key])

    def __getitem__(self, index):
        outputs = dict()
        x = self.data[index]
        outputs["x"] = x

        outputs["label"] = self.cell_type[index]
        outputs["sizefactor"] = self.size_factors[index]
        outputs["u"] = self.u[index]

        return outputs

    def __len__(self):
        return self.data.shape[0]


class DatasetVAE(Dataset):
    """Dataset handler for trainer.
       Parameters
       ----------
       adata: : `~anndata.AnnData`
            Annotated data matrix.
    """

    def __init__(self, adata, domain_key, label_key=None):
        self.data = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        self.domain = torch.tensor(adata.obs[domain_key].cat.codes.to_numpy(), dtype=torch.int64)
        if label_key in adata.obs:
            self.label = torch.tensor(adata.obs[label_key].cat.codes.to_numpy(), dtype=torch.int64)

    def __getitem__(self, index):
        outputs = dict()
        x = self.data[index]
        outputs["x"] = x
        if hasattr(self, 'label'):
            outputs["label"] = self.label[index]
        outputs["domain"] = self.domain[index]

        return outputs

    def __len__(self):
        return self.data.shape[0]


class DatasetIVAE(Dataset):
    """Dataset handler for trainer.
       Parameters
       ----------
       adata: : `~anndata.AnnData`
            Annotated data matrix.
    """

    def __init__(self, data, domain, label=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        condition_dict = {k: v for k, v in zip(np.unique(domain), range(len(np.unique(domain))))}
        self.domain = torch.tensor(domain, dtype=torch.int64)
        for key in condition_dict:
            self.domain[self.domain == key] = condition_dict[key]
        if label is not None:
            self.label = label

    def __getitem__(self, index):
        outputs = dict()
        x = self.data[index]
        outputs["x"] = x
        if hasattr(self, 'label'):
            outputs["label"] = self.label[index]
        outputs["domain"] = self.domain[index]

        return outputs

    def __len__(self):
        return self.data.shape[0]


def asw(embedding, domain_index, label):
    d = np.zeros((embedding.shape[0], embedding.shape[0]))
    for i in range(embedding.shape[0]):
        d[i, :] = np.sum((embedding[i, :] - embedding) ** 2, axis=1)
    a = np.zeros(embedding.shape[0])
    for i in range(len(a)):
        idx = np.where(domain_index == domain_index[i])[0]
        a[i] = d[i, idx].mean() * len(idx) / (len(idx) - 1)
    b = np.ones(embedding.shape[0]) * 1e10
    for i in range(len(b)):
        domain_diff = np.setdiff1d(np.unique(domain_index), np.array([domain_index[i]]))
        for dd in domain_diff:
            idx = np.where(domain_index == dd)[0]
            if d[i, idx].mean() < b[i]:
                b[i] = d[i, idx].mean()
    s = np.abs((b - a) / np.maximum(a, b))

    asw = np.zeros(np.unique(label).size)
    for i in range(len(asw)):
        idx = np.where(label == asw[i])[0]
        asw[i] = np.mean(1 - s[idx])
    return asw.mean()
