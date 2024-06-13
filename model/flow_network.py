import torch.nn as nn
import torch.nn.functional as F
import torch
from pyro.distributions.transforms.spline import _monotonic_rational_spline
from torch.autograd import Variable
from torch.distributions import MultivariateNormal
from torch.nn import Parameter
import numpy as np
import itertools


class BaseFlow(nn.Module):

    def sample(self, n=1, context=None, **kwargs):
        dim = self.dim
        if isinstance(self.dim, int):
            dim = [dim, ]

        spl = Variable(torch.FloatTensor(n, *dim).normal_())
        lgd = Variable(torch.from_numpy(
            np.zeros(n).astype('float32')))
        if context is None:
            context = Variable(torch.from_numpy(
                np.ones((n, self.context_dim)).astype('float32')))

        if hasattr(self, 'gpu'):
            if self.gpu:
                spl = spl.cuda()
                lgd = lgd.cuda()
                context = context.gpu()

        return self.forward((spl, lgd, context))

    def cuda(self):
        self.gpu = True
        return super(BaseFlow, self).cuda()


delta = 1e-6
softplus_ = nn.Softplus()
softplus = lambda x: softplus_(x) + delta
sigmoid_ = nn.Sigmoid()
sigmoid = lambda x: sigmoid_(x) * (1 - delta) + 0.5 * delta
sigmoid2 = lambda x: sigmoid(x) * 2.0
logsigmoid = lambda x: -softplus(-x)
log = lambda x: torch.log(x * 1e2) - np.log(1e2)
logit = lambda x: log(x) - log(1 - x)


def softmax(x, dim=-1):
    e_x = torch.exp(x - x.max(dim=dim, keepdim=True)[0])
    out = e_x / e_x.sum(dim=dim, keepdim=True)
    return out


sum1 = lambda x: x.sum(1)
sum_from_one = lambda x: sum_from_one(sum1(x)) if len(x.size()) > 2 else sum1(x)


class Sigmoid(nn.Module):
    def forward(self, x):
        return sigmoid(x)


class SigmoidFlow(BaseFlow):

    def __init__(self, num_ds_dim=4):
        super(SigmoidFlow, self).__init__()
        self.num_ds_dim = num_ds_dim
        self.num_params = 3 * num_ds_dim
        self.act_a = lambda x: softplus(x)
        self.act_b = lambda x: x
        self.act_w = lambda x: softmax(x, dim=2)

    def forward(self, x, dsparams, mollify=0.0, delta=delta):
        ndim = self.num_ds_dim
        a_ = self.act_a(dsparams[:, :, 0 * ndim:1 * ndim])
        b_ = self.act_b(dsparams[:, :, 1 * ndim:2 * ndim])
        w = self.act_w(dsparams[:, :, 2 * ndim:3 * ndim])

        a = a_ * (1 - mollify) + 1.0 * mollify
        b = b_ * (1 - mollify) + 0.0 * mollify

        pre_sigm = a * x[:, :, None] + b
        sigm = torch.sigmoid(pre_sigm)
        x_pre = torch.sum(w * sigm, dim=2)
        x_pre_clipped = x_pre * (1 - delta) + delta * 0.5
        x_ = log(x_pre_clipped) - log(1 - x_pre_clipped)
        xnew = x_

        return xnew.squeeze(-1)


class DenseSigmoidFlow(BaseFlow):

    def __init__(self, in_dim, hidden_dim, out_dim):
        super(DenseSigmoidFlow, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.act_a = lambda x: softplus_(x)
        self.act_b = lambda x: x
        self.act_w = lambda x: softmax(x, dim=3)
        self.act_u = lambda x: softmax(x, dim=3)

        self.u_ = Parameter(torch.Tensor(hidden_dim, in_dim))
        self.w_ = Parameter(torch.Tensor(out_dim, hidden_dim))
        self.num_params = 3 * hidden_dim + in_dim
        self.reset_parameters()

    def reset_parameters(self):
        self.u_.data.uniform_(-0.001, 0.001)
        self.w_.data.uniform_(-0.001, 0.001)

    def forward(self, x, dsparams, logdet=None):
        if len(x.size()) == 2:
            x = x.unsqueeze(-1)
        inv = np.log(np.exp(1 - delta) - 1)
        ndim = self.hidden_dim
        pre_u = self.u_[None, None, :, :] + dsparams[:, :, -self.in_dim:][:, :, None, :]
        pre_w = self.w_[None, None, :, :] + dsparams[:, :, 2 * ndim:3 * ndim][:, :, None, :]
        a = self.act_a(dsparams[:, :, 0 * ndim:1 * ndim] + inv)
        b = self.act_b(dsparams[:, :, 1 * ndim:2 * ndim])
        w = self.act_w(pre_w)
        u = self.act_u(pre_u)

        pre_sigm = torch.sum(u * a[:, :, :, None] * x[:, :, None, :], 3) + b
        sigm = torch.sigmoid(pre_sigm)
        x_pre = torch.sum(w * sigm[:, :, None, :], dim=3)

        x_pre_clipped = x_pre * (1 - delta) + delta * 0.5
        x_ = log(x_pre_clipped) - log(1 - x_pre_clipped)
        xnew = x_

        logj = F.log_softmax(pre_w, dim=3) + \
               logsigmoid(pre_sigm[:, :, None, :]) + \
               logsigmoid(-pre_sigm[:, :, None, :]) + log(a[:, :, None, :])
        # n, d, d2, dh

        logj = logj[:, :, :, :, None] + F.log_softmax(pre_u, dim=3)[:, :, None, :, :]
        # n, d, d2, dh, d1

        logj = log_sum_exp(logj, 3).sum(3)
        # n, d, d2, d1

        logdet_ = logj + np.log(1 - delta) - \
                  (log(x_pre_clipped) + log(-x_pre_clipped + 1))[:, :, :, None]

        if logdet is None:
            logdet = logdet_.new_zeros(logdet_.shape[0], logdet_.shape[1], 1, 1)

        logdet = log_sum_exp(
            logdet_[:, :, :, :, None] + logdet[:, :, None, :, :], 3
        ).sum(3)
        # n, d, d2, d1, d0 -> n, d, d2, d0

        return xnew.squeeze(-1), logdet

    def extra_repr(self):
        return 'input_dim={in_dim}, output_dim={out_dim}'.format(**self.__dict__)


class DDSF(nn.Module):
    def __init__(self, n_layers, in_dim, hidden_dim, out_dim):
        super(DDSF, self).__init__()
        blocks = [DenseSigmoidFlow(in_dim, hidden_dim, hidden_dim)]
        for _ in range(n_layers - 2):
            blocks += [DenseSigmoidFlow(hidden_dim, hidden_dim, hidden_dim)]
        blocks += [DenseSigmoidFlow(hidden_dim, hidden_dim, out_dim)]
        self.num_params = 0
        for block in blocks:
            self.num_params += block.num_params
        self.model = nn.ModuleList(blocks)

    def forward(self, x, dsparams):
        start = 0
        _logdet = None

        for block in self.model:
            block_dsparams = dsparams[:, :, start:start + block.num_params]
            x, _logdet = block(x, block_dsparams, logdet=_logdet)
            start = start + block.num_params

        logdet = _logdet[:, :, 0, 0].sum(1)

        return x, logdet


def oper(array, oper, axis=-1, keepdims=False):
    a_oper = oper(array)
    if keepdims:
        shape = []
        for j, s in enumerate(array.size()):
            shape.append(s)
        shape[axis] = -1
        a_oper = a_oper.view(*shape)
    return a_oper


def log_sum_exp(A, axis=-1, sum_op=torch.sum):
    maximum = lambda x: x.max(axis)[0]
    A_max = oper(A, maximum, axis, True)
    summation = lambda x: sum_op(torch.exp(x - A_max), axis)
    B = torch.log(oper(A, summation, axis, True)) + A_max
    return B


class ConditionalDenseNN(torch.nn.Module):
    """
    An implementation of a simple dense feedforward network taking a context variable, for use in, e.g.,
    some conditional flows such as :class:`pyro.distributions.transforms.ConditionalAffineCoupling`.

    Example usage:

    >>> input_dim = 10
    >>> context_dim = 5
    >>> x = torch.rand(100, input_dim)
    >>> z = torch.rand(100, context_dim)
    >>> nn = ConditionalDenseNN(input_dim, context_dim, [50], param_dims=[1, input_dim, input_dim])
    >>> a, b, c = nn(x, context=z)  # parameters of size (100, 1), (100, 10), (100, 10)

    :param input_dim: the dimensionality of the input
    :type input_dim: int
    :param context_dim: the dimensionality of the context variable
    :type context_dim: int
    :param hidden_dims: the dimensionality of the hidden units per layer
    :type hidden_dims: list[int]
    :param param_dims: shape the output into parameters of dimension (p_n,) for p_n in param_dims
        when p_n > 1 and dimension () when p_n == 1. The default is [1, 1], i.e. output two parameters of dimension ().
    :type param_dims: list[int]
    :param nonlinearity: The nonlinearity to use in the feedforward network such as torch.nn.ReLU(). Note that no
        nonlinearity is applied to the final network output, so the output is an unbounded real number.
    :type nonlinearity: torch.nn.Module

    """

    def __init__(
            self,
            input_dim,
            context_dim,
            hidden_dims,
            param_dims=[1, 1],
            nonlinearity=torch.nn.ReLU(),
    ):
        super().__init__()

        self.input_dim = input_dim
        self.context_dim = context_dim
        self.hidden_dims = hidden_dims
        self.param_dims = param_dims
        self.count_params = len(param_dims)
        self.output_multiplier = sum(param_dims)

        # Calculate the indices on the output corresponding to each parameter
        ends = torch.cumsum(torch.tensor(param_dims), dim=0)
        starts = torch.cat((torch.zeros(1).type_as(ends), ends[:-1]))
        self.param_slices = [slice(s.item(), e.item()) for s, e in zip(starts, ends)]

        # Create masked layers
        layers = [torch.nn.Linear(input_dim + context_dim, hidden_dims[0])]
        for i in range(1, len(hidden_dims)):
            layers.append(torch.nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
        layers.append(torch.nn.Linear(hidden_dims[-1], self.output_multiplier))
        self.layers = torch.nn.ModuleList(layers)

        # Save the nonlinearity
        self.f = nonlinearity

    def forward(self, x, context):
        # We must be able to broadcast the size of the context over the input
        context = context.expand(x.size()[:-1] + (context.size(-1),))

        x = torch.cat([context, x], dim=-1)
        return self._forward(x)

    def _forward(self, x):
        """
        The forward method
        """
        h = x
        for layer in self.layers[:-1]:
            h = self.f(layer(h))
        h = self.layers[-1](h)

        # Shape the output, squeezing the parameter dimension if all ones
        if self.output_multiplier == 1:
            return h
        else:
            h = h.reshape(list(x.size()[:-1]) + [self.output_multiplier])

            if self.count_params == 1:
                return h

            else:
                return tuple([h[..., s] for s in self.param_slices])


class DenseNN(ConditionalDenseNN):
    """
    An implementation of a simple dense feedforward network, for use in, e.g., some conditional flows such as
    :class:`pyro.distributions.transforms.ConditionalPlanarFlow` and other unconditional flows such as
    :class:`pyro.distributions.transforms.AffineCoupling` that do not require an autoregressive network.

    Example usage:

    >>> input_dim = 10
    >>> context_dim = 5
    >>> z = torch.rand(100, context_dim)
    >>> nn = DenseNN(context_dim, [50], param_dims=[1, input_dim, input_dim])
    >>> a, b, c = nn(z)  # parameters of size (100, 1), (100, 10), (100, 10)

    :param input_dim: the dimensionality of the input
    :type input_dim: int
    :param hidden_dims: the dimensionality of the hidden units per layer
    :type hidden_dims: list[int]
    :param param_dims: shape the output into parameters of dimension (p_n,) for p_n in param_dims
        when p_n > 1 and dimension () when p_n == 1. The default is [1, 1], i.e. output two parameters of dimension ().
    :type param_dims: list[int]
    :param nonlinearity: The nonlinearity to use in the feedforward network such as torch.nn.ReLU(). Note that no
        nonlinearity is applied to the final network output, so the output is an unbounded real number.
    :type nonlinearity: torch.nn.module

    """

    def __init__(
            self, input_dim, hidden_dims, param_dims=[1, 1], nonlinearity=torch.nn.ReLU()
    ):
        super(DenseNN, self).__init__(
            input_dim, 0, hidden_dims, param_dims=param_dims, nonlinearity=nonlinearity
        )

    def forward(self, x):
        return self._forward(x)


def _construct_nn(input_dim, context_dim, count_bins=8, order="linear"):
    """
    A helper function to create a
    :class:`~pyro.distributions.transforms.ConditionalSpline` object that takes care
    of constructing a dense network with the correct input/output dimensions.

    :param input_dim: Dimension of input variable
    :type input_dim: int
    :param context_dim: Dimension of context variable
    :type context_dim: int
    :param count_bins: The number of segments comprising the spline.
    :type count_bins: int
    :param order: One of ['linear', 'quadratic'] specifying the order of the spline.
    :type order: string

    """

    hidden_dims = [input_dim * 10, input_dim * 10]

    if order == "linear":
        nn = DenseNN(context_dim, hidden_dims,
                     param_dims=[input_dim * count_bins, input_dim * count_bins, input_dim * (count_bins - 1),
                                 input_dim * count_bins])
    elif order == "quadratic":
        nn = DenseNN(context_dim, hidden_dims,
                     param_dims=[input_dim * count_bins, input_dim * count_bins, input_dim * (count_bins - 1)])
    else:
        raise ValueError(
            "Keyword argument 'order' must be one of ['linear', 'quadratic'], but '{}' was found!".format(order))

    return nn


class ComponentWiseCondSpline(nn.Module):
    def __init__(self, input_dim, context_dim, count_bins=8, bound=5., order='linear'):
        """Component-wise Spline Flow
        Args:
            input_dim: The size of input/latent features.
            context_dim: The size of conditioned/context features.
            count_bins: The number of bins that each can have their own weights.
            bound: Tail bound (outside tail bounds the transformation is identity)
            order: Spline order

        Modified from Neural Spline Flows: https://arxiv.org/pdf/1906.04032.pdf
        """
        super(ComponentWiseCondSpline, self).__init__()
        assert order in ("linear", "quadratic")
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.count_bins = count_bins
        self.bound = bound
        self.order = order

        self.nn = _construct_nn(input_dim=input_dim, context_dim=context_dim, count_bins=count_bins, order=order)
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_dim))
        self.register_buffer('base_dist_var', torch.eye(input_dim))

    @property
    def base_dist(self):
        return MultivariateNormal(self.base_dist_mean, self.base_dist_var)

    def _params(self, context):
        # Rational linear splines have additional lambda parameters
        if self.order == "linear":
            w, h, d, l = self.nn(context)
            # AutoRegressiveNN and DenseNN return different shapes...
            l = l.transpose(-1, -2) if w.shape[-1] == self.input_dim else l.reshape(
                l.shape[:-1] + (self.input_dim, self.count_bins))

            l = torch.sigmoid(l)
        elif self.order == "quadratic":
            w, h, d = self.nn(context)
            l = None

        # AutoRegressiveNN and DenseNN return different shapes...
        if w.shape[-1] == self.input_dim:
            w = w.transpose(-1, -2)
            h = h.transpose(-1, -2)
            d = d.transpose(-1, -2)
        else:
            w = w.reshape(w.shape[:-1] + (self.input_dim, self.count_bins))
            h = h.reshape(h.shape[:-1] + (self.input_dim, self.count_bins))
            d = d.reshape(d.shape[:-1] + (self.input_dim, self.count_bins - 1))

        w = F.softmax(w, dim=-1)
        h = F.softmax(h, dim=-1)
        d = F.softplus(d)
        return w, h, d, l

    def forward(self, x, context):
        """f: data x, context -> latent u"""
        u, log_detJ = self.spline_op(x, context)
        log_detJ = torch.sum(log_detJ, dim=1)
        return u, log_detJ

    def inverse(self, u, context):
        """g: latent u > data x"""
        x, log_detJ = self.spline_op(u, context, inverse=True)
        log_detJ = torch.sum(log_detJ, dim=1)
        return x, log_detJ

    def spline_op(self, x, context, **kwargs):
        """Fit N separate splines for each dimension of input"""
        # widths, unnormalized_widths ~ (input_dim, num_bins)
        w, h, d, l = self._params(context)
        y, log_detJ = _monotonic_rational_spline(x, w, h, d, l, bound=self.bound, **kwargs)
        return y, log_detJ

    def log_prob(self, x, context):
        z, log_detJ = self.forward(x, context)
        logp = self.base_dist.log_prob(z) + log_detJ
        return logp

    def sample(self, context, batch_size):
        z = self.base_dist.sample((batch_size,))
        x, _ = self.inverse(z, context)
        return x


class ConditionalFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, input_dim, context_dim, n_layers=1, bound=5., count_bins=8, order='linear'):
        super().__init__()
        self.flows = nn.ModuleList([ComponentWiseCondSpline(input_dim=input_dim, context_dim=context_dim, bound=bound,
                                                            count_bins=count_bins, order=order) for _ in
                                    range(n_layers)])

    @property
    def base_dist(self):
        return self.flows[0].base_dist

    def forward(self, x, context):
        m, _ = x.shape
        log_det = 0
        for flow in self.flows:
            x, ld = flow.forward(x, context)
            log_det += ld
        return x, log_det

    def inverse(self, z, context):
        m, _ = z.shape
        log_det = 0
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z, context)
            log_det += ld
        return z, log_det


class NormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, input_dim, n_layers, bound, count_bins, order):
        super().__init__()
        self.flows = nn.ModuleList(
            [ComponentWiseSpline(input_dim=input_dim, bound=bound, count_bins=count_bins, order=order) for _ in
             range(n_layers)])

    @property
    def base_dist(self):
        return self.flows[0].base_dist

    def forward(self, x):
        m, _ = x.shape
        log_det = 0
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
        return x, log_det

    def inverse(self, z):
        m, _ = z.shape
        log_det = 0
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z)
            log_det += ld
        return z, log_det


class ComponentWiseSpline(nn.Module):
    def __init__(self, input_dim: int, count_bins: int = 8, bound: int = 3., order: str = 'linear') -> None:
        """Component-wise Spline Flow
        Args:
            input_dim: The size of input/latent features.
            count_bins: The number of bins that each can have their own weights.
            bound: Tail bound (outside tail bounds the transformation is identity)
            order: Spline order

        Modified from Neural Spline Flows: https://arxiv.org/pdf/1906.04032.pdf
        """
        super(ComponentWiseSpline, self).__init__()
        assert order in ("linear", "quadratic")
        self.input_dim = input_dim
        self.count_bins = count_bins
        self.bound = bound
        self.order = order
        self.unnormalized_widths = nn.Parameter(torch.randn(self.input_dim, self.count_bins))
        self.unnormalized_heights = nn.Parameter(torch.randn(self.input_dim, self.count_bins))
        self.unnormalized_derivatives = nn.Parameter(torch.randn(self.input_dim, self.count_bins - 1))
        # Rational linear splines have additional lambda parameters
        if self.order == "linear":
            self.unnormalized_lambdas = nn.Parameter(torch.rand(self.input_dim, self.count_bins))
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_dim))
        self.register_buffer('base_dist_var', torch.eye(input_dim))

    @property
    def base_dist(self):
        return torch.distributions.MultivariateNormal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x):
        """f: data x -> latent u"""
        u, log_detJ = self.spline_op(x)
        log_detJ = torch.sum(log_detJ, dim=1)
        return u, log_detJ

    def inverse(self, u):
        """g: latent u > data x"""
        x, log_detJ = self.spline_op(u, inverse=True)
        log_detJ = torch.sum(log_detJ, dim=1)
        return x, log_detJ

    def spline_op(self, x, **kwargs):
        """Fit N separate splines for each dimension of input"""
        # widths, unnormalized_widths ~ (input_dim, num_bins)
        w = F.softmax(self.unnormalized_widths, dim=-1)
        h = F.softmax(self.unnormalized_heights, dim=-1)
        d = F.softplus(self.unnormalized_derivatives)
        if self.order == 'linear':
            l = torch.sigmoid(self.unnormalized_lambdas)
        else:
            l = None
        y, log_detJ = _monotonic_rational_spline(x, w, h, d, l, bound=self.bound, **kwargs)
        return y, log_detJ

    def log_prob(self, x):
        z, log_detJ = self.forward(x)
        logp = self.base_dist.log_prob(z) + log_detJ
        return logp
