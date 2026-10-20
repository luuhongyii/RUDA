"""RUDA-native random filling with serializable seed/counter and dropout."""
from __future__ import annotations

import secrets
import struct
from threading import RLock
import torch


class Generator:
    """Philox4x32-10 generator; four output slots are reserved per counter block.

    The byte state is RUDA-specific, not a PyTorch CUDA generator state. GPU
    sampling consumes only this explicit counter; no CPU random tensor is made.
    """
    def __init__(self, seed=None):
        self.lock = RLock()
        self.manual_seed(secrets.randbits(64) if seed is None else seed)

    def manual_seed(self, seed):
        if type(seed) is not int or not -(1<<63)<=seed<1<<64:
            raise ValueError('seed must fit a signed/unsigned 64-bit integer')
        with self.lock:
            self.seed_value, self.counter = seed%(1<<64), 0
        return self

    def initial_seed(self):
        return self.seed_value

    def get_state(self):
        with self.lock:
            return torch.tensor(list(struct.pack('<8sQQ',b'RUDARNG1',self.seed_value,self.counter)),dtype=torch.uint8,device='cpu')

    def set_state(self, state):
        if not isinstance(state,torch.Tensor) or state.device.type!='cpu' or state.dtype!=torch.uint8 or state.shape!=(24,):
            raise ValueError('RUDA RNG state must be a 24-byte CPU uint8 tensor')
        magic, seed, counter = struct.unpack('<8sQQ',bytes(state.tolist()))
        if magic!=b'RUDARNG1':
            raise ValueError('RUDA RNG state version differs')
        with self.lock:
            self.seed_value,self.counter=seed,counter
        return self

    def fill_(self, tensor, distribution, first, second):
        if distribution == 0:
            return uniform_(tensor, first, second, generator=self)
        if distribution == 1:
            return normal_(tensor, first, second, generator=self)
        if distribution == 2:
            return bernoulli_(tensor, first, generator=self)
        raise ValueError('unknown RUDA random distribution')


default_generator = Generator()


def uniform_(tensor, from_=0., to=1., *, generator=None):
    from ._random_base import uniform_ as fill
    return fill(tensor, from_, to, generator=generator)


def normal_(tensor, mean=0., std=1., *, generator=None):
    from ._random_base import normal_ as fill
    return fill(tensor, mean, std, generator=generator)


def bernoulli_(tensor, p=.5, *, generator=None):
    from ._random_base import bernoulli_, bernoulli_tensor
    fill = bernoulli_tensor if isinstance(p, torch.Tensor) else bernoulli_
    return fill(tensor, p, generator=generator)


def native_dropout(input, p, train=True):
    if not 0<=p<=1:
        raise ValueError('dropout probability must be in [0,1]')
    if train is None:train=True
    if not train or p==0:
        return input.clone(),torch.ones_like(input,dtype=torch.bool)
    return torch.ops.aten.native_dropout.default(input, p, train)


def register_random_ops():
    from ._ops import _registry
    from ._random_base import bernoulli_tensor, register
    from ._random_activation import register as register_activations
    _registry.impl('uniform_',lambda tensor,from_=0.,to=1.,generator=None: uniform_(tensor,from_,to,generator=generator))
    _registry.impl('normal_',lambda tensor,mean=0.,std=1.,generator=None: normal_(tensor,mean,std,generator=generator))
    _registry.impl('bernoulli_.float',lambda tensor,p=.5,generator=None: bernoulli_(tensor,p,generator=generator))
    _registry.impl('bernoulli_.Tensor',bernoulli_tensor)
    register(_registry)
    register_activations(_registry)
