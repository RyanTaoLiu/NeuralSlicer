import torch
from torch import nn
import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from utils.quernion import quaternion_multiply

# helpers

def exists(val):
    return val is not None


def cast_tuple(val, repeat=1):
    return val if isinstance(val, tuple) else ((val,) * repeat)


# sin activation
class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


# siren layer

class Siren(nn.Module):
    class Sine(nn.Module):
        def __init__(self, w0=1.):
            super().__init__()
            self.w0 = w0

        def forward(self, x):
            return torch.sin(self.w0 * x)

    def __init__(self, dim_in, dim_out, w0=1., c=6., is_first=False, use_bias=True, activation=None, is_last=False):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        self.is_last = is_last
        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        # [1,0,0,0,1,1,1]
        if is_last:
            if dim_out == 7:  # RS
                self.weight = nn.Parameter(torch.zeros(dim_out, dim_in))
                idenityDeformation = torch.tensor([1, 0, 0, 0, 1, 1, 1], dtype=float)
                self.bias = nn.Parameter(idenityDeformation)
            elif dim_out == 3:  # flowing or gradient
                self.weight = nn.Parameter(torch.zeros(dim_out, dim_in))
                idenityDeformation = torch.tensor([0, 0, 0], dtype=float)
                idenityDeformation += torch.rand_like(idenityDeformation) * 1e-5
                self.bias = nn.Parameter(idenityDeformation)
            elif dim_out == 1:
                self.weight = nn.Parameter(torch.zeros(dim_out, dim_in))
                idenityDeformation = torch.tensor([0], dtype=float)
                idenityDeformation += torch.rand_like(idenityDeformation) * 1e-5
                self.bias = nn.Parameter(idenityDeformation)
            else:
                raise ('Cannot support the dim out init for the last layer')
            '''
            self.weight = nn.Parameter(weight)
            self.bias = nn.Parameter(bias) if use_bias else None
            '''
        self.activation = Sine(w0) if activation is None else activation
        # self.activation = nn.ReLU() if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


# siren network

class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0=1., w0_initial=30., use_bias=True,
                 final_activation=None):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(Siren(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                w0=layer_w0,
                use_bias=use_bias,
                is_first=is_first
            ))

        final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = Siren(dim_in=dim_hidden, dim_out=dim_out, w0=w0, use_bias=use_bias,
                                activation=final_activation, is_last=True)

    def forward(self, x, mods=None):
        mods = cast_tuple(mods, self.num_layers)
        i = 0
        for layer, mod in zip(self.layers, mods):
            i += 1
            x = layer(x)

            if exists(mod):
               x *= rearrange(mod, 'd -> () d')

        return self.last_layer(x)


class Modulator(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_in if is_first else (dim_hidden + dim_in)

            self.layers.append(nn.Sequential(
                nn.Linear(dim, dim_hidden),
                nn.ReLU()
            ))

    def forward(self, z):
        x = z
        hiddens = []

        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
            x = torch.cat((x, z))

        return tuple(hiddens)


# wrapper
class TransformationWrapper(nn.Module):
    def __init__(self, net, pos, latent_dim=None):
        super().__init__()
        self.net = net
        self.pos = pos
        self.modulator = None

        if exists(latent_dim):
            self.modulator = Modulator(
                dim_in=latent_dim,
                dim_hidden=net.dim_hidden,
                num_layers=net.num_layers
            )

        self.rotaion_conv1 = torch.nn.Conv1d(latent_dim, latent_dim, 1)
        self.rotaion_conv2 = torch.nn.Conv1d(latent_dim, latent_dim // 2, 1)
        self.rotaion_conv3 = torch.nn.Conv1d(latent_dim // 2, latent_dim // 4, 1)
        self.rotaion_conv4 = torch.nn.Conv1d(latent_dim // 4, 4, 1)

        self.rotaion_conv4.weight = nn.Parameter(torch.zeros(4, latent_dim // 4, 1))
        self.rotaion_conv4.bias = nn.Parameter(torch.zeros(4))
        self.th = nn.Tanh()

    # add global rotation
    def global_rotaion_forward(self, latent):
        y = self.rotaion_conv1(latent.unsqueeze(-1))
        y = self.rotaion_conv2(y)
        y = self.rotaion_conv3(y)
        y = self.th(self.rotaion_conv4(y)).squeeze(-1)
        return y

    def forward(self, pos, latent=None):

        modulate = exists(self.modulator)
        assert not (modulate ^ exists(
            latent)), 'latent vector must be only supplied if `latent_dim` was passed in on instantiation'

        mods = self.modulator(latent) if modulate else None

        qs = self.net(pos, mods=None)
        rotation = self.global_rotaion_forward(latent) + torch.tensor([1, 0, 0, 0]).to(latent.device)
        # out[:, :4] += rotation.expand(out.shape[0], -1)
        q = quaternion_multiply(rotation.expand(qs.shape[0], -1), qs[:, :4])
        s = qs[:, 4:]
        return torch.hstack((q,s))


class FlowingWrapper(nn.Module):
    def __init__(self, net, pos, latent_dim=None):
        super().__init__()
        self.net = net
        self.pos = pos
        self.modulator = None

    def forward(self, pos, latent=None):
        out = self.net(pos, mods=None)
        return out


if __name__ == '__main__':
    net = SirenNet(
        dim_in=2,  # input dimension, ex. 2d coor
        dim_hidden=256,  # hidden dimension
        dim_out=3,  # output dimension, ex. rgb value
        num_layers=5,  # number of layers
        w0_initial=30.  # different signals may require different omega_0 in the first layer - this is a hyperparameter
    )

    '''
    wrapper = SirenWrapper(
        net,
        latent_dim=512,
        image_width=256,
        image_height=256
    )

    latent = nn.Parameter(torch.zeros(512).normal_(0, 1e-2))
    img = torch.randn(1, 3, 256, 256)

    loss = wrapper(img, latent=latent)
    loss.backward()

    # after much training ...
    # simply invoke the wrapper without passing in anything

    pred_img = wrapper(latent=latent)  # (1, 3, 256, 256)
    '''
