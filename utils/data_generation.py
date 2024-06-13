import math
import os

import numpy as np
import scipy
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from torch import nn
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.laplace import Laplace
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from torch.utils.data import Dataset
from scipy.stats import ortho_group


def normalize(x):
    return (x - x.min(0, keepdim=True)[0]) / (x.max(0, keepdim=True)[0] - x.min(0, keepdim=True)[0])


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


def lrelu(x, neg_slope):
    """
    Leaky ReLU activation function
    @param x: input array
    @param neg_slope: slope for negative values
    @return:
        out: output rectified array
    """

    def _lrelu_1d(_x, _neg_slope):
        """
        one dimensional implementation of leaky ReLU
        """
        if _x > 0:
            return _x
        else:
            return _x * _neg_slope

    leaky1d = np.vectorize(_lrelu_1d)
    assert neg_slope > 0  # must be positive
    return leaky1d(x, neg_slope)


def sigmoid(x):
    """
    Sigmoid activation function
    @param x: input array
    @return:
        out: output array
    """
    return 1 / (1 + np.exp(-x))


def generate_mixing_matrix(d_sources: int, d_data=None, lin_type='uniform', cond_threshold=25, n_iter_4_cond=None,
                           dtype=np.float32):
    """
    Generate square linear mixing matrix
    @param d_sources: dimension of the latent sources
    @param d_data: dimension of the mixed data
    @param lin_type: specifies the type of matrix entries; either `uniform` or `orthogonal`.
    @param cond_threshold: higher bound on the condition number of the matrix to ensure well-conditioned problem
    @param n_iter_4_cond: or instead, number of iteration to compute condition threshold of the mixing matrix.
        cond_threshold is ignored in this case/
    @param dtype: data type for data
    @return:
        A: mixing matrix
    @rtype: np.ndarray
    """
    if d_data is None:
        d_data = d_sources

    if lin_type == 'orthogonal':
        A = (np.linalg.qr(np.random.uniform(-1, 1, (d_sources, d_data)))[0]).astype(dtype)

    elif lin_type == 'uniform':
        if n_iter_4_cond is None:
            cond_thresh = cond_threshold
        else:
            cond_list = []
            for _ in range(int(n_iter_4_cond)):
                A = np.random.uniform(-1, 1, (d_sources, d_data)).astype(dtype)
                for i in range(d_data):
                    A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
                cond_list.append(np.linalg.cond(A))

            cond_thresh = np.percentile(cond_list, 25)  # only accept those below 25% percentile

        A = (np.random.uniform(0, 2, (d_sources, d_data)) - 1).astype(dtype)
        for i in range(d_data):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())

        while np.linalg.cond(A) > cond_thresh:
            # generate a new A matrix!
            A = (np.random.uniform(0, 2, (d_sources, d_data)) - 1).astype(dtype)
            for i in range(d_data):
                A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
    else:
        raise ValueError('incorrect method')
    return A


def generate_nonstationary_sources(n: int, n_seg: int, d: int, prior='lap', var_bounds=np.array([0.5, 3]),
                                   dtype=np.float32, uncentered=True, mean_bounds=np.array([-5, 5])):
    """
    Generate source signal following a TCL distribution. Within each segment, sources are independent.
    The distribution withing each segment is given by the keyword `dist`
    @param n: number of points
    @param n_seg: number of segments
    @param d: dimension of the sources same as data
    @param prior: distribution of the sources. can be `lap` for Laplace , `hs` for Hypersecant or `gauss` for Gaussian
    @param var_bounds: optional, upper and lower bounds for the modulation parameter
    @param dtype: data type for data
    @param bool uncentered: True to generate uncentered data
    @return:
        sources: output source array of shape (n, d)
        labels: label for each point; the label is the component
        m: mean of each component
        L: modulation parameter of each component
    @rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    """
    var_lb = var_bounds[0]
    var_ub = var_bounds[1]
    n_per_seg = math.floor(n / n_seg)

    L = np.random.uniform(var_lb, var_ub, (n_seg, d))
    if uncentered:
        m = np.random.uniform(mean_bounds[0], mean_bounds[1], (n_seg, d))
    else:
        m = np.zeros((n_seg, d))

    labels = np.zeros(n, dtype=dtype)
    if prior == 'lap':
        sources = np.random.laplace(0, 1 / np.sqrt(2), (n, d)).astype(dtype)
    elif prior == 'hs':
        sources = scipy.stats.hypsecant.rvs(0, 1, (n, d)).astype(dtype)
    elif prior == 'gauss':
        sources = np.random.randn(n, d).astype(dtype)
    else:
        raise ValueError('incorrect dist')

    index = list(range(sources.shape[0]))
    np.random.shuffle(index)
    for seg in range(n_seg):
        if seg != n_seg - 1:
            segID = index[n_per_seg * seg: n_per_seg * (seg + 1)]
        else:
            segID = index[n_per_seg * seg:]
        sources[segID] *= L[seg]
        sources[segID] += m[seg]
        labels[segID] = seg
    return sources, labels, m, L


def generate_data(n_per_seg, n_seg, d_sources, d_data=None, n_layers=3, prior='lap', activation='lrelu', batch_size=250,
                  seed=950127, slope=.2, var_bounds=np.array([0.1, 3]), lin_type='uniform', n_iter_4_cond=1e4,
                  dtype=np.float32, uncentered=False, noisy=0, mean_bounds=np.array([-1, 1])):
    """
    Generate artificial data with arbitrary mixing
    @param int n_per_seg: number of observations per segment
    @param int n_seg: number of segments
    @param int d_sources: dimension of the latent sources
    @param int or None d_data: dimension of the data
    @param int n_layers: number of layers in the mixing MLP
    @param str activation: activation function for the mixing MLP; can be `none, `lrelu`, `xtanh` or `sigmoid`
    @param str prior: prior distribution of the sources; can be `lap` for Laplace or `hs` for Hypersecant
    @param int batch_size: batch size if data is to be returned as batches. 0 for a single batch of size n
    @param int seed: random seed
    @param var_bounds: upper and lower bounds for the modulation parameter
    @param float slope: slope parameter for `lrelu` or `xtanh`
    @param str lin_type: specifies the type of matrix entries; can be `uniform` or `orthogonal`
    @param int n_iter_4_cond: number of iteration to compute condition threshold of the mixing matrix
    @param dtype: data type for data
    @param bool uncentered: True to generate uncentered data
    @param float noisy: if non-zero, controls the level of noise added to observations

    @return:
        tuple of batches of generated (sources, data, auxiliary variables, mean, variance)
    @rtype: tuple

    """
    if seed is not None:
        np.random.seed(seed)

    if d_data is None:
        d_data = d_sources

    # sources
    sources, labels, m, L = generate_nonstationary_sources(n_per_seg, n_seg, d_sources, prior=prior,
                                                           var_bounds=var_bounds, dtype=dtype, uncentered=uncentered,
                                                           mean_bounds=mean_bounds)
    n = n_per_seg * n_seg

    # non linearity
    if activation == 'lrelu':
        act_f = lambda x: lrelu(x, slope).astype(dtype)
    elif activation == 'sigmoid':
        act_f = sigmoid
    elif activation == 'xtanh':
        act_f = lambda x: np.tanh(x) + slope * x
    elif activation == 'none':
        act_f = lambda x: x
    else:
        raise ValueError('incorrect non linearity: {}'.format(activation))

    # Mixing time!
    assert n_layers > 1  # suppose we always have at least 2 layers. The last layer doesn't have a non-linearity
    A = generate_mixing_matrix(d_sources, d_data, lin_type=lin_type, n_iter_4_cond=n_iter_4_cond, dtype=dtype)
    X = act_f(np.dot(sources, A))
    if d_sources != d_data:
        B = generate_mixing_matrix(d_data, lin_type=lin_type, n_iter_4_cond=n_iter_4_cond, dtype=dtype)
    else:
        B = A
    for nl in range(1, n_layers):
        if nl == n_layers - 1:
            X = np.dot(X, B)
        else:
            X = act_f(np.dot(X, B))

    # add noise:
    if noisy:
        X += noisy * np.random.randn(*X.shape)

    plt.subplot(1, 2, 1)
    plt.scatter(sources[:, 0], sources[:, 1], c=labels, s=6, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=6, alpha=0.3)
    plt.show()

    # always return batches (as a list), even if number of batches is one,
    if not batch_size:
        return [sources], [X], to_one_hot([labels], m=n_seg), m, L
    else:
        idx = np.random.permutation(n)
        Xb, Sb, Ub = [], [], []
        n_batches = int(n / batch_size)
        for c in range(n_batches):
            Sb += [sources[idx][c * batch_size:(c + 1) * batch_size]]
            Xb += [X[idx][c * batch_size:(c + 1) * batch_size]]
            Ub += [labels[idx][c * batch_size:(c + 1) * batch_size]]
        return Sb, Xb, to_one_hot(Ub, m=n_seg), m, L


def save_data(path, *args, **kwargs):
    kwargs['batch_size'] = 0  # leave batch creation to torch DataLoader
    Sb, Xb, Ub, m, L = generate_data(*args, **kwargs)
    Sb, Xb, Ub = Sb[0], Xb[0], Ub[0]
    print('Creating dataset {} ...'.format(path))
    dir_path = '/'.join(path.split('/')[:-1])
    if not os.path.exists(dir_path):
        os.makedirs('/'.join(path.split('/')[:-1]))
    np.savez_compressed(path, s=Sb, x=Xb, u=Ub, m=m, L=L)
    print(' ... done')


def create_if_not_exist_dataset(root='data/', nps=1000, ns=40, dl=2, dd=4, nl=3, s=1, p='gauss', a='xtanh', seed=950127,
                                uncentered=False, noisy=False, arg_str=None, var_bounds=np.array([0.1, 3]),
                                mean_bounds=np.array([-1, 1])):
    """
    Create a dataset if it doesn't exist.
    This is useful as a setup step when running multiple jobs in parallel, to avoid having many scripts attempting
    to create the dataset when non-existent.
    This is called in `cmd_utils.create_dataset_before`
    """
    if arg_str is not None:
        # overwrites all other arg values
        # arg_str should be of this form: nps_ns_dl_dd_nl_s_p_a_u_n
        arg_list = arg_str.split('_')
        assert len(arg_list) == 10
        nps, ns, dl, dd, nl = map(int, arg_list[0:5])
        p, a = arg_list[6:8]
        if arg_list[5] == 'n':
            s = None
        else:
            s = int(arg_list[5])
        if arg_list[-2] == 'f':
            uncentered = False
        else:
            uncentered = True
        if arg_list[-1] == 'f':
            noisy = False
        else:
            noisy = True

    path_to_dataset = root + 'tcl_' + '_'.join([str(nps), str(ns), str(dl), str(dd), str(nl), str(s), p, a])
    if uncentered:
        path_to_dataset += '_u'
    if noisy:
        path_to_dataset += '_n'
    path_to_dataset += '.npz'

    if not os.path.exists(path_to_dataset) or s is None:
        kwargs = {"n_per_seg": nps, "n_seg": ns, "d_sources": dl, "d_data": dd, "n_layers": nl, "prior": p,
                  "activation": a, "seed": seed, "batch_size": 0, "uncentered": uncentered, "noisy": noisy,
                  "var_bounds": var_bounds, "mean_bounds": mean_bounds}
        save_data(path_to_dataset, **kwargs)
    return path_to_dataset


class SyntheticDataset(Dataset):
    def __init__(self, path, device='cpu'):
        self.device = device
        self.path = path
        try:
            data = np.load(path)
        except:
            # error occured because many scripts were attempting to create it at same time.
            # one solution would be to wait and retry, the other would be to make sure
            # datasets are all created already.
            pass
        self.data = data
        self.s = torch.from_numpy(data['s']).to(self.device)
        self.x = torch.from_numpy(data['x']).to(self.device)
        self.u = torch.from_numpy(data['u']).to(self.device)
        self.L = data['L']
        self.M = data['m']
        self.len = self.x.shape[0]
        self.latent_dim = self.s.shape[1]
        self.aux_dim = self.u.shape[1]
        self.data_dim = self.x.shape[1]
        self.nps = int(self.len / self.aux_dim)
        print('data loaded on {}'.format(self.x.device))

    def get_dims(self):
        return self.data_dim, self.latent_dim, self.aux_dim

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x[index], self.u[index], self.s[index]

    def get_metadata(self):
        return {'path': self.path,
                'nps': self.nps,
                'ns': self.aux_dim,
                'n': self.len,
                'latent_dim': self.latent_dim,
                'data_dim': self.data_dim,
                'aux_dim': self.aux_dim,
                }


def generate_data_singlecell(c_dim=4, s_dim=2, n_layers=2, prior='uniform', activation='Tanh', ds=1000, data_dim=10,
                             seed=950127, noisy=0, assumption=True, device=torch.device('cpu')):
    if seed is not None:
        np.random.seed(seed)

    # sources
    m = Dirichlet(torch.tensor([1.] * c_dim))
    sources_c = m.rsample((ds,))
    if prior == 'laplace':
        m = Laplace(torch.tensor(0.), torch.tensor(1 / np.sqrt(2)))
    elif prior == 'gauss':
        m = Normal(torch.tensor([0.]), torch.tensor([1.]))
    elif prior == 'uniform':
        m = Uniform(torch.tensor([0.]), torch.tensor([1.]))
    else:
        raise ValueError('incorrect dist')
    lr = normalize(m.rsample((ds * s_dim * 2,)).reshape(ds, s_dim * 2))
    l = lr[:, :s_dim]
    r = lr[:, s_dim:]
    sources_s = l * r
    sources_s += torch.randn_like(sources_s) * noisy
    sources_s = normalize(sources_s)
    sources = torch.cat((sources_s, sources_c), 1)

    # Mixing time!
    A = torch.bernoulli(torch.empty(data_dim, s_dim + c_dim).uniform_(0, 1))
    if assumption:
        while True:
            flag = True
            for i in range(s_dim + c_dim):
                A_i = A[A[:, i] == 1, :]
                if torch.prod(A_i, 0).sum() > 1:
                    flag = False
                    break
            if flag:
                break
            A = torch.bernoulli(torch.empty(data_dim, s_dim + c_dim).uniform_(0, 1))

    X = torch.zeros(ds, data_dim, requires_grad=False)
    for i in range(data_dim):
        s_i = sources[:, torch.where(A[i, :])[0]]
        modules = []
        for j in range(n_layers - 1):
            modules.append(nn.Linear(s_i.shape[1], s_i.shape[1]))
            modules.append(getattr(nn, activation)())
        modules.append(nn.Linear(s_i.shape[1], 1))
        f = nn.Sequential(*modules).to(device)
        X[:, i] = f(s_i).reshape(ds, )
    X *= 10
    # add noise:
    if noisy:
        X += noisy * torch.randn_like(X)

    return sources, X.detach(), lr


def generate_data_singlecell_batch_effect(c_dim=3, s_dim=3, prior='gauss', activation='lrelu', n=10000,
                                          label_dim=5, domain_dim=5, data_dim=20, seed=None, noisy=0,
                                          mean_range=np.array([-5, 5]), var_range=np.array([0.01, 1]), assumption=True):
    '''

    Parameters
    ----------
    c_dim: dimension of c
    s_dim: dimension of s
    prior: distribution of c and s
    activation: activation function to generate x
    n: number of samples
    c_class_num: number of label classes
    s_class_num: number of domains
    data_dim: dimension of data
    seed: random seed
    noisy: noise parameter
    mean_range: range of mean
    var_range: range of variance

    Returns: sources, data, label, domain
    -------

    '''
    if seed is not None:
        np.random.seed(seed)
    # sources
    sources_c, labels_c, _, _ = generate_nonstationary_sources(n, label_dim, c_dim, prior=prior,
                                                               mean_bounds=mean_range,
                                                               var_bounds=var_range)
    sources_s, labels_s, _, _ = generate_nonstationary_sources(n, domain_dim, s_dim, prior=prior,
                                                               mean_bounds=mean_range,
                                                               var_bounds=var_range)
    # non linearity
    if activation == 'lrelu':
        act_f = lambda x: lrelu(x, .2).astype(np.float32)
    elif activation == 'sigmoid':
        act_f = sigmoid
    elif activation == 'Tanh':
        act_f = lambda x: np.tanh(x) + .1 * x
    elif activation == 'none':
        act_f = lambda x: x
    else:
        raise ValueError('incorrect non linearity: {}'.format(activation))

    sources = np.concatenate((sources_c, sources_s), axis=1)
    # Mixing time!
    X = sources.copy()
    if assumption:
        # mixing matrix satisfies sparsity assumption
        A = np.random.uniform(low=-1, high=1, size=(sources.shape[1], data_dim - c_dim - s_dim))
        A_sparsity = (np.eye(c_dim + s_dim) * np.random.uniform(low=0.1, high=1, size=(c_dim + s_dim)))
        A = np.concatenate((A_sparsity, A), axis=1)
    else:
        A = np.random.uniform(low=-1, high=1, size=(sources.shape[1], data_dim))
    X = act_f(np.dot(X, A))

    # add noise:
    X += noisy * np.random.randn(*X.shape)

    # X = softmax(X)
    return sources, X, labels_c, labels_s


def generate_data_batch_effect(c_dim=3, s_dim=3, prior='gauss', activation='lrelu', n=10000,
                               label_dim=5, domain_dim=5, data_dim=20, seed=None, noisy=0,
                               mean_range=np.array([-4, 4]), var_range=np.array([0.01, 1]),
                               assumption=True):
    '''

    Parameters
    ----------
    c_dim: dimension of c
    s_dim: dimension of s
    prior: distribution of c and s
    activation: activation function to generate x
    n: number of samples
    c_class_num: number of label classes
    s_class_num: number of domains
    data_dim: dimension of data
    seed: random seed
    noisy: noise parameter
    mean_range: range of mean
    var_range: range of variance

    Returns: sources, data, label, domain
    -------

    '''
    if seed is not None:
        np.random.seed(seed)
    if prior == 'lap':
        sources_tilde = np.random.laplace(0, 1 / np.sqrt(2), (n, c_dim + s_dim)).astype(np.float32)
    elif prior == 'hs':
        sources_tilde = scipy.stats.hypsecant.rvs(0, 1, (n, c_dim + s_dim)).astype(np.float32)
    elif prior == 'gauss':
        sources_tilde = np.random.randn(n, c_dim + s_dim).astype(np.float32)
        sources_tilde = scale(sources_tilde)
    else:
        raise ValueError('incorrect dist')
    sources = sources_tilde.copy()
    n_per_seg = math.floor(n / domain_dim)

    L = np.random.uniform(var_range[0], var_range[1], (domain_dim, s_dim))
    m = np.random.uniform(mean_range[0], mean_range[1], (domain_dim, s_dim))

    domain = np.zeros(n, dtype=np.int64)
    index = list(range(sources.shape[0]))
    np.random.shuffle(index)
    for seg in range(domain_dim):
        if seg != domain_dim - 1:
            segID = index[n_per_seg * seg: n_per_seg * (seg + 1)]
        else:
            segID = index[n_per_seg * seg:]
        sources[segID, c_dim:] *= L[seg]
        sources[segID, c_dim:] += m[seg]
        domain[segID] = seg

    # non linearity
    if activation == 'lrelu':
        act_f = lambda x: lrelu(x, .2).astype(np.float32)
    elif activation == 'sigmoid':
        act_f = sigmoid
    elif activation == 'Tanh':
        act_f = lambda x: np.tanh(x) + .1 * x
    elif activation == 'none':
        act_f = lambda x: x
    else:
        raise ValueError('incorrect non linearity: {}'.format(activation))
    # Mixing time!
    if assumption:
        # mixing matrix satisfies sparsity assumption
        A = np.random.uniform(low=-1, high=1, size=(sources.shape[1], data_dim - c_dim))
        A_sparsity = (np.eye(c_dim + s_dim)[:, :c_dim] * np.random.uniform(low=0.1, high=1, size=c_dim))
        A = np.concatenate((A_sparsity, A), axis=1)
    else:
        A = np.random.uniform(low=-1, high=1, size=(sources.shape[1], data_dim))
    X = act_f(np.dot(sources, A))
    # A = ortho_group.rvs(c_dim + s_dim)
    # X = np.dot(act_f(sources), A)

    # add noise:
    X += noisy * np.random.randn(*X.shape)

    A = np.random.uniform(low=-1, high=1, size=(c_dim, label_dim))
    label = np.argmax(act_f(np.dot(sources[:, :c_dim], A)), 1)
    # X = softmax(X)
    return sources_tilde, sources, X, label, domain


def generate_data_flows(c_dim=3, s_dim=3, prior='gauss', activation='lrelu', n=10000, label_dim=5, domain_dim=5,
                        data_dim=20, n_layers=2, seed=None, mean_range=np.array([-5, 5]), var_range=np.array([1, 3]),
                        assumption=True):
    '''

    Parameters
    ----------
    c_dim: dimension of c
    s_dim: dimension of s
    prior: distribution of c and s
    activation: activation function to generate x
    n: number of samples
    c_class_num: number of label classes
    s_class_num: number of domains
    data_dim: dimension of data
    seed: random seed
    noisy: noise parameter
    mean_range: range of mean
    var_range: range of variance

    Returns: sources, data, label, domain
    -------

    '''
    if seed is not None:
        np.random.seed(seed)

    # sources
    sources_tilde, label, _, _ = generate_nonstationary_sources(n, label_dim, c_dim + s_dim, prior=prior,
                                                                mean_bounds=mean_range, var_bounds=var_range)
    sources_s, domain, _, _ = generate_nonstationary_sources(n, domain_dim, s_dim, prior=prior,
                                                             mean_bounds=mean_range, var_bounds=var_range)
    sources = sources_tilde.copy()
    sources[:, c_dim:] = sources_s + sources_tilde[:, c_dim:]
    # non linearity
    if activation == 'lrelu':
        act_f = lambda x: lrelu(x, .1).astype(np.float32)
    elif activation == 'sigmoid':
        act_f = sigmoid
    elif activation == 'Tanh':
        act_f = lambda x: np.tanh(x) + .1 * x
    elif activation == 'none':
        act_f = lambda x: x
    else:
        raise ValueError('incorrect non linearity: {}'.format(activation))
    # Mixing time!
    data = sources.copy()
    if assumption:
        # mixing matrix satisfies sparsity assumption
        A = np.random.uniform(low=-1, high=1, size=(sources.shape[1], data_dim - c_dim - s_dim))
        A_sparsity = (np.eye(c_dim + s_dim) * np.random.uniform(low=0.1, high=1, size=(c_dim + s_dim)))
        A = np.concatenate((A_sparsity, A), axis=1)
    else:
        A = np.random.uniform(low=-1, high=1, size=(sources.shape[1], data_dim))
    data = act_f(np.dot(data, A))
    return sources_tilde, data, label, domain


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
    return e_x / div


def leaky_ReLU_1d(d, negSlope):
    if d > 0:
        return d
    else:
        return d * negSlope


leaky1d = np.vectorize(leaky_ReLU_1d)


def leaky_ReLU(D, negSlope):
    assert negSlope > 0
    return leaky1d(D, negSlope)


def data_generator_synthetic(domain_dim, c_dim=2, s_dim=2, data_dim=20, Nlayer=2, var_range_l=0.01, var_range_r=1,
                             mean_range_l=-4, mean_range_r=4, NsegmentObs_train=7500, NsegmentObs_test=0,
                             Nobs_test=4096,
                             varyMean=True, linear_mixing_first=False, source='Gaussian', seed=950127, assumption=True):
    """
    generate multivariate data based on the non-stationary non-linear ICA model of Hyvarinen & Morioka (2016)
    we generate mixing matrices using random orthonormal matrices
    INPUT
        - Ncomp: number of components (i.e., dimensionality of the data)
        - Nlayer: number of non-linear layers!
        - Nsegment: number of data segments to generate
        - NsegmentObs: number of observations per segment
        - source: either Laplace or Gaussian, denoting distribution for latent sources
        - NonLin: linearity employed in non-linear mixing. Can be one of "leaky" = leakyReLU or "sigmoid"=sigmoid
          Specifically for leaky activation we also have:
            - negSlope: slope for x < 0 in leaky ReLU
            - Niter4condThresh: number of random matricies to generate to ensure well conditioned
    OUTPUT:
      - output is a dictionary with the following values:
        - sources: original non-stationary source
        - obs: mixed sources
        - labels: segment labels (indicating the non stationarity in the data)
    """
    # np.random.seed(seed)
    randomstate = np.random.RandomState(seed)

    # generate non-stationary data:
    train_size = NsegmentObs_train * domain_dim
    assert Nobs_test == 0 or NsegmentObs_test == 0
    if Nobs_test > 0:
        NsegmentObs_test = int(Nobs_test // domain_dim)
    test_size = NsegmentObs_test * domain_dim
    NsegmentObs_total = NsegmentObs_train + NsegmentObs_test
    Nobs = train_size + test_size  # total number of observations
    labels = np.array([0] * Nobs)  # labels for each observation (populate below)

    # generate data, which we will then modulate in a non-stationary manner:
    if source == 'Laplace':
        dat = randomstate.laplace(0, 1, (Nobs, c_dim + s_dim))
        dat = scale(dat)  # set to zero mean and unit variance
    elif source == 'Gaussian':
        dat = randomstate.normal(0, 1, (Nobs, c_dim + s_dim))
        dat = scale(dat)
    else:
        raise Exception("wrong source distribution")

    # get modulation parameters
    modMat = randomstate.uniform(var_range_l, var_range_r, (s_dim, domain_dim))
    if varyMean:
        meanMat = randomstate.uniform(mean_range_l, mean_range_r, (s_dim, domain_dim))
    else:
        meanMat = np.zeros((s_dim, domain_dim))
    # now we are ready to apply the non-linear mixtures:
    mixedDat = np.copy(dat)

    # now we adjust the variance within each segment in a non-stationary manner
    for seg in range(domain_dim):
        segID = range(NsegmentObs_total * seg, NsegmentObs_total * (seg + 1))
        mixedDat[segID, -s_dim:] = np.multiply(mixedDat[segID, -s_dim:], modMat[:, seg])
        mixedDat[segID, -s_dim:] = np.add(mixedDat[segID, -s_dim:], meanMat[:, seg])
        labels[segID] = seg

    # generate mixing matrices:
    if linear_mixing_first:
        A = ortho_group.rvs(c_dim + s_dim, random_state=randomstate)
        mixedDat = np.dot(mixedDat, A)
    for l in range(Nlayer - 1):
        # we first apply non-linear function, then causal matrix!
        mixedDat = leaky_ReLU(mixedDat, .2)

        # generate causal matrix first:
        if assumption:
            # mixing matrix satisfies sparsity assumption
            A = np.random.uniform(low=-1, high=1, size=(mixedDat.shape[1], data_dim - c_dim))
            A_sparsity = np.eye(c_dim + s_dim)[:, :c_dim] * np.random.uniform(low=0.1, high=1, size=c_dim)
            A = np.concatenate((A_sparsity, A), axis=1)
        else:
            A = np.random.uniform(low=-1, high=1, size=(mixedDat.shape[1], data_dim))
        # apply mixing:
        mixedDat = np.dot(mixedDat, A)

    # stratified split
    x_train, x_test, z_train, z_test, u_train, u_test = train_test_split(mixedDat, dat, labels, train_size=train_size,
                                                                         test_size=test_size, random_state=randomstate,
                                                                         stratify=labels)

    return {"source": z_train, "x": x_train, "domain": u_train}, {"source": z_test, "x": x_test, "domain": u_test}


def gen_da_data_ortho(domain_dim, c_dim=2, s_dim=2, data_dim=20, Nlayer=2, var_range_l=0.01, var_range_r=1,
                      mean_range_l=-4, mean_range_r=4, NsegmentObs_train=7500, NsegmentObs_test=0, Nobs_test=4096,
                      varyMean=True, linear_mixing_first=False, source='Gaussian', seed=950127, method='lingjing',
                      assumption=True):
    """
    generate multivariate data based on the non-stationary non-linear ICA model of Hyvarinen & Morioka (2016)
    we generate mixing matrices using random orthonormal matrices
    INPUT
        - Ncomp: number of components (i.e., dimensionality of the data)
        - Nlayer: number of non-linear layers!
        - Nsegment: number of data segments to generate
        - NsegmentObs: number of observations per segment
        - source: either Laplace or Gaussian, denoting distribution for latent sources
        - NonLin: linearity employed in non-linear mixing. Can be one of "leaky" = leakyReLU or "sigmoid"=sigmoid
          Specifically for leaky activation we also have:
            - negSlope: slope for x < 0 in leaky ReLU
            - Niter4condThresh: number of random matricies to generate to ensure well conditioned
    OUTPUT:
      - output is a dictionary with the following values:
        - sources: original non-stationary source
        - obs: mixed sources
        - labels: segment labels (indicating the non stationarity in the data)
    """

    negSlope = 0.2
    NonLin = 'leaky'
    # np.random.seed(seed)
    randomstate = np.random.RandomState(seed)

    # generate non-stationary data:
    train_size = NsegmentObs_train * domain_dim
    assert Nobs_test == 0 or NsegmentObs_test == 0
    if Nobs_test > 0:
        NsegmentObs_test = int(Nobs_test // domain_dim)
    test_size = NsegmentObs_test * domain_dim
    NsegmentObs_total = NsegmentObs_train + NsegmentObs_test
    Nobs = train_size + test_size  # total number of observations
    labels = np.array([0] * Nobs)  # labels for each observation (populate below)

    # generate data, which we will then modulate in a non-stationary manner:
    if source == 'Laplace':
        dat = randomstate.laplace(0, 1, (Nobs, c_dim + s_dim))
        dat = scale(dat)  # set to zero mean and unit variance
    elif source == 'Gaussian':
        dat = randomstate.normal(0, 1, (Nobs, c_dim + s_dim))
        dat = scale(dat)
    else:
        raise Exception("wrong source distribution")

    # get modulation parameters
    modMat = randomstate.uniform(var_range_l, var_range_r, (s_dim, domain_dim))
    if varyMean:
        meanMat = randomstate.uniform(mean_range_l, mean_range_r, (s_dim, domain_dim))
    else:
        meanMat = np.zeros((s_dim, domain_dim))
    # now we are ready to apply the non-linear mixtures:
    mixedDat = np.copy(dat)

    # now we adjust the variance within each segment in a non-stationary manner
    for seg in range(domain_dim):
        segID = range(NsegmentObs_total * seg, NsegmentObs_total * (seg + 1))
        mixedDat[segID, -s_dim:] = np.multiply(mixedDat[segID, -s_dim:], modMat[:, seg])
        mixedDat[segID, -s_dim:] = np.add(mixedDat[segID, -s_dim:], meanMat[:, seg])
        labels[segID] = seg

    # generate mixing matrices:
    if linear_mixing_first:
        A = ortho_group.rvs(c_dim + s_dim, random_state=randomstate)
        mixedDat = np.dot(mixedDat, A)
    for l in range(Nlayer - 1):
        # we first apply non-linear function, then causal matrix!
        if NonLin == 'leaky':
            mixedDat = leaky_ReLU(mixedDat, negSlope)
        elif NonLin == 'sigmoid':
            mixedDat = sigmoid(mixedDat)

        # generate causal matrix first:
        if method == 'lingjing':
            A = ortho_group.rvs(c_dim + s_dim, random_state=randomstate)  # generateUniformMat( Ncomp, condThresh )
        elif method == 'yang':
            if assumption:
                # mixing matrix satisfies sparsity assumption
                A = np.random.uniform(low=-1, high=1, size=(mixedDat.shape[1], data_dim - c_dim))
                A_sparsity = np.eye(c_dim + s_dim)[:, :c_dim] * np.random.uniform(low=0.1, high=1, size=c_dim)
                # A_sparsity = np.eye(c_dim + s_dim) * np.random.uniform(low=0.1, high=1, size=c_dim + s_dim)
                # A_sparsity = ortho_group.rvs(c_dim + s_dim, random_state=randomstate)
                A = np.concatenate((A_sparsity, A), axis=1)
            else:
                A = np.random.uniform(low=-1, high=1, size=(mixedDat.shape[1], data_dim))
            # A = np.eye(c_dim + s_dim) * np.random.uniform(low=0.1, high=1, size=c_dim + s_dim)
        # apply mixing:
        mixedDat = np.dot(mixedDat, A)

    # stratified split
    x_train, x_test, z_train, z_test, u_train, u_test = train_test_split(mixedDat, dat, labels, train_size=train_size,
                                                                         test_size=test_size, random_state=randomstate,
                                                                         stratify=labels)

    return {"source": z_train, "x": x_train, "domain": u_train}, {"source": z_test, "x": x_test, "domain": u_test}


class DANS(Dataset):
    def __init__(self, dataset):
        super().__init__()
        # self.path = os.path.join(directory, dataset, "data.npz")
        # self.npz = np.load(self.path)
        self.data = dataset

    def __len__(self):
        return len(self.data["source"])

    def __getitem__(self, idx):
        source = torch.from_numpy(self.data["source"][idx].astype('float32'))
        x = torch.from_numpy(self.data["x"][idx].astype('float32'))
        domain = torch.from_numpy(self.data["domain"][idx, None].astype('float32'))
        sample = {"source": source, "x": x, "domain": domain}
        return sample
