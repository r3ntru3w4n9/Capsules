r"""
An efficient implementation (I think? faster than most github repos I tested)
of Geoffrey Hinton's paper

[Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)

Buiding blocks for a capsule network.
The modules can be used to construct arbitrary `Dense` like capsule networks.

:class: `FeatureExtractor`

* Stacked convolution layers

__init__(self, conv_channels, kernel_sizes, strides, out_features) -> None

    * conv_channels: number of channels for each convolution layer in module
    * kernel_sizes: size of kernels for each convolution layer in module
    * strides: size of strides for each kernel for each convolution layer in module
    * out_features: number of output `capsules`

__call__(self, x) -> output

    * x: input images in batches
    * output: output features (in capsule compatible form)

* Note: ReLU here matters!! (without one I couldn't manage to get past 90% accuracy)
* Note: output is `squash`ed. 
* The reason being crazy _gradient vanishing_ without it

:class: `Transform`

* Transformation of previous

__init__(self, prev_capsules, num_capsules, in_channels, out_channels)

    * prev_capsules: number of capsules of the previous `CapsuleLayer`
    * num_capsules: number of capsules of this `CapsuleLayer`
    * in_channels: dimension of the vector for each previous capsule
    * out_channels: dimension of the vector for each capsule of this layer

__call__(self, U) -> U_hat

    * U: output of the previous `CapsuleLayer`
    * U_hat: transformed output. 
    * every vector is transformed into a set of vectors,
    * corresponding to every capsule in this layer

:class: `Route`

* implements the _dynamic routing algorithm by agreement_.

__init__(self, num_iters, prev_capsules, num_capsules)

    * num_iters: number of iterations in dynamic routing algorithm
    * prev_capsules: number of capsules of the previous `CapsuleLayer`
    * num_capsules: number of capsules of this `CapsuleLayer`

__call__(self, U_hat) -> V

    * U_hat: transformed output as in `Transform` module
    * V: a set of combined vectors
    * Note: for every previous capsule, the sum of the weights is 1
    * the vector input `U_hat` is then weighted by those weights
    * and summed together for each capsule in this layer, and we call it V

:class: `CapsuleLayer`

* a convenience module, wrapper for :class: `Transform` and `Route`
* the reason for the class'es existence is that its existence makes sense

__init__(self, in_channels, out_channels, in_capsules, out_capsules, num_iters)

    * in_channels: as in `Transform`
    * out_channels: as in `Transform`
    * in_capsules: as in `Route`
    * out_capsules: as in `Route`
    * num_iters: as in `Route`

__call__(self, U_prev) -> U_self

    * U_prev: output of the previous `CapsuleLayer`
    * U_self: output of this `CapsuleLayer`

:class: `Cequential`

* a convenience module much like pytorch's `nn.Sequential`
* except only number of channels and of capsules and of iterations need be passes

__init__(self, channels, capsules, num_iters)

    * channels: iterable of number of channles
    * capsules: iterable of number of capsules
    * num_iters: as in `Route`

__call__(self, x) -> y

    * x: batchs of sets of input vectors
    * y: batchs of sets of output vectors

:class: `CapsDecoder`

* reconstructs the original image, with a densly connected model,
* also acts as a regulator to ensure the preservation of spacial feature

__init__(self, layers, categories)

    * layer: number of neurons for each layer in the model
    * categories: number of categories of outputs (as in MNIST, 10)

__call__(self, x) -> y

    * x: batchs of sets of input vectors
    * y: batchs of reconstructed images

:class: `CapsNet`

* a generalization of the `CapsNet` implemented by original
* leaving the default unchanged as in `main.py` uses the same configuration
* as the original paper, however, you can use different parameters as you like
* also, this module is only a wrapper

__init__(self, feature_net, capsule_net, decoder_net)

    * feature_net: an instance of `FeatureExtractor`
    * capsule_net: an instance of `Cequential`
    * decoder_net: an instance of `CapsDecoder`

__call__(self, input, label) -> (predicted, reconstructed)

    * input: batches of images
    * label: given label. derived from `predicted` if not provided
    * predicted: predicted vectors encoding probability (length) and state (direction)
    * reconstructed: reconstructed images
"""
import itertools

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init as nn_init

import functions


class FeatureExtractor(nn.Module):
    r"""
    [Input]

    * Image: torch.Tensor
        (N, C, H, W)

    [Output]

    * Feature: torch.Tensor
        (N, F, capsules)
    """

    def __init__(
        self,
        conv_channels: tuple,
        kernel_sizes: tuple,
        strides: tuple,
        out_features: int,
    ):
        super().__init__()
        in_channel_list = list(conv_channels[:-1])
        out_channel_list = list(conv_channels[1:])
        out_channel_list[-1] *= out_features

        _length = len(in_channel_list)
        assert _length == len(out_channel_list) == len(kernel_sizes) == len(strides)

        self.convolution = nn.Sequential()

        for cnt in range(_length):
            self.convolution.add_module(
                name=f"Conv2d-{cnt}",
                module=nn.Conv2d(
                    in_channels=in_channel_list[cnt],
                    out_channels=out_channel_list[cnt],
                    kernel_size=kernel_sizes[cnt],
                    stride=strides[cnt],
                ),
            )
            if cnt >= _length - 1:
                continue
            self.convolution.add_module(name=f"ReLU-{cnt}", module=nn.ReLU())

        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        x = self.convolution(x)
        return functions.squash(x.reshape(N, -1, self.out_features))


class Transform(nn.Module):
    r"""
    [Input]
    * U: torch.Tensor
       (N, in_channels, prev_capsules)

    [Output]
    * U_hat: torch.Tensor
        (N, out_channels, num_capsules, prev_capsules)

    [Attribute]
    * W: torch.Tensor
        (1, prev_capsules, num_capsules, out_channels, in_channels)
    """

    def __init__(self, prev_capsules, num_capsules, in_channels, out_channels):
        super().__init__()
        self.W = nn.Parameter(
            data=torch.randn(1, prev_capsules, num_capsules, out_channels, in_channels),
            requires_grad=True,
        )

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        U_hat = self.W @ U.transpose(dim0=1, dim1=-1).unsqueeze(dim=2).unsqueeze(dim=-1)
        return U_hat.squeeze(dim=-1).transpose(dim0=1, dim1=-1)


class Route(nn.Module):
    r"""
    [Input]
    * U_hat: torch.Tensor
        (N, channels, num_capsules, prev_capsules)
    
    [Output]
    * V: torch.Tensor
        (N, channels, num_capsules)
    
    [Attribute]
    * num_iters: int
    * B: torch.Tensor
        (1, 1, num_capsules, prev_capsules)
    """

    def __init__(self, num_iters: int, prev_capsules: int, num_capsules: int):
        super().__init__()
        self.num_iters = num_iters
        self.B = nn.Parameter(
            data=torch.zeros(size=(1, 1, num_capsules, prev_capsules)),
            requires_grad=False,
        )

    def forward(self, U_hat: torch.Tensor) -> torch.Tensor:
        B = self.B.detach()
        for _ in range(self.num_iters):
            C = F.softmax(input=B, dim=2)
            S = (C * U_hat).sum(dim=-1)
            V = functions.squash(S)
            B = B + (V.unsqueeze(dim=-1) * U_hat).sum(dim=1, keepdim=True).mean(
                dim=0, keepdim=True
            )
        return V


class CapsuleLayer(nn.Module):
    r"""
    [Input]
    * U_prev: torch.Tensor
        (N, channel, prev_capsules)
    
    [Output]
    * U_self: torch.Tensor
        (N, channel, num_capsules)

    [Attribute]
    * transform: Transform
    * route: Route
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_capsules: int,
        out_capsules: int,
        num_iters: int,
    ):
        super().__init__()
        self.transform = Transform(
            prev_capsules=in_capsules,
            num_capsules=out_capsules,
            in_channels=in_channels,
            out_channels=out_channels,
        )
        self.route = Route(
            num_iters=num_iters, prev_capsules=in_capsules, num_capsules=out_capsules
        )

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        v = self.transform(U)
        v = self.route(v)
        return v


class Cequential(nn.Sequential):
    r"""
    [Input]
    * U_prev_0: torch.Tensor
        (N, channel, prev_capsules)
    
    [Output]
    * U_self_-1: torch.Tensor
        (N, channel, num_capsules)

    [Note]
    * stacked `CapsuleLayer`'s
    """

    def __init__(self, channels: tuple, capsules: tuple, num_iters: int):
        super().__init__()
        in_channels = channels[:-1]
        out_channles = channels[1:]
        in_capsules = capsules[:-1]
        out_capsules = capsules[1:]
        for (cnt, (i_ch, o_ch, i_cp, o_cp)) in enumerate(
            zip(in_channels, out_channles, in_capsules, out_capsules)
        ):
            self.add_module(
                name=f"Capsule-{cnt}",
                module=CapsuleLayer(
                    in_channels=i_ch,
                    out_channels=o_ch,
                    in_capsules=i_cp,
                    out_capsules=o_cp,
                    num_iters=num_iters,
                ),
            )


class CapsDecoder(nn.Module):
    r"""
    [Input]
    * U_self_-1
        (N, channel, labels)
    
    [Output]
    * recon
        (N, C, H, W)
    """

    def __init__(self, layers: tuple, categories: int):
        super().__init__()
        in_features = layers[:-1]
        out_features = layers[1:]
        self.layers = nn.Sequential()
        for (cnt, (i, o)) in enumerate(zip(in_features[:-1], out_features[:-1])):
            self.layers.add_module(
                name=f"Linear-{cnt}", module=nn.Linear(in_features=i, out_features=o)
            )
            self.layers.add_module(name=f"ReLU-{cnt}", module=nn.ReLU())
        self.layers.add_module(
            name=f"Linear-{cnt+1}",
            module=nn.Linear(
                in_features=in_features[-1], out_features=out_features[-1]
            ),
        )
        self.layers.add_module(name=f"Sigmoid-{cnt+1}", module=nn.Sigmoid())
        self._categories = categories

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        one_hot = functions.one_hot(x=label, categories=self._categories)
        input = x * (one_hot.unsqueeze(dim=1))
        input = input.view(input.size(0), -1)
        return self.layers(input)


class CapsNet(nn.Module):
    r"""
    [Input]
    * Images
        (N, C, H, W)
    
    [Output]
    * predicted
        (N, channel, labels)
    * recon
        (N, C, H, W)
    """

    def __init__(
        self, feature_net: nn.Module, capsule_net: nn.Module, decoder_net: nn.Module
    ):
        super().__init__()
        self.feature = feature_net
        self.capsule = capsule_net
        self.decoder = decoder_net

    def forward(self, input: torch.Tensor, label: torch.Tensor = None) -> torch.Tensor:
        feature_output = self.feature(input)
        capsule_output = self.capsule(feature_output)
        if label is None:
            label = (capsule_output ** 2).sum(dim=1).argmax(dim=-1)
        decoder_output = self.decoder(capsule_output, label)
        return (capsule_output, decoder_output)
