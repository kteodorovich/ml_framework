import numpy as np
from numpy.typing import ArrayLike

def Add(inputs: list[str], outputs: list[str], attributes: dict[str,], tensors: dict[str, ArrayLike]):
    a = tensors[inputs[0]]
    b = tensors[inputs[1]]
    tensors[outputs[0]] = a + b

def BatchNormalization(inputs: list[str], outputs: list[str], attributes: dict[str,], tensors: dict[str, ArrayLike]):
    inp = tensors[inputs[0]]
    scale = tensors[inputs[1]].reshape((1, -1, 1, 1))
    bias = tensors[inputs[2]].reshape((1, -1, 1, 1))
    mean = tensors[inputs[3]].reshape((1, -1, 1, 1))
    var = tensors[inputs[4]].reshape((1, -1, 1, 1))

    eps = attributes['epsilon']
    momentum = attributes['momentum']  # not sure what to do with this
    spacial = bool(attributes['spatial'])
    assert spacial == 1  # haven't implemented otherwise

    inp_norm = (inp - mean) / np.sqrt(var + eps)
    tensors[outputs[0]] = scale * inp_norm.copy() + bias


def Conv(inputs: list[str], outputs: list[str], attributes: dict[str,], tensors: dict[str, ArrayLike]):
    inp = tensors[inputs[0]]
    weights = tensors[inputs[1]]
    dh, dw = attributes['dilations']
    kh, kw = attributes['kernel_shape']
    ph0, ph1, pw0, pw1 = attributes['pads']
    sh, sw = attributes['strides']
    group = attributes['group']

    padded_inp = np.zeros((inp.shape[0], inp.shape[1], inp.shape[2] + ph0 + ph1, inp.shape[3] + pw0 + pw1))
    padded_inp[:, :, ph0:ph0+inp.shape[2], pw0:pw0+inp.shape[3]] = inp

    out_h = (padded_inp.shape[2] - dh * (kh - 1) - 1) // sh + 1
    out_w = (padded_inp.shape[3] - dw * (kw - 1) - 1) // sw + 1

    output = np.zeros((1, weights.shape[0], out_h, out_w))
    in_per_grp = padded_inp.shape[1] // group
    out_per_grp = weights.shape[0] // group

    for i in range(group):
        for j in range(i*out_per_grp, (i+1)*out_per_grp):
            for k in range(out_h):
                for l in range(out_w):
                    output[0, j, k, l] = np.dot(padded_inp[0, i*in_per_grp:(i+1)*in_per_grp, k*sh:k*sh+dh*kh:dh, l*sw:l*sw+dw*kw:dw].reshape(-1),
                                                weights[j, :, :, :].reshape(-1))

    tensors[outputs[0]] = output

def GlobalAveragePool(inputs: list[str], outputs: list[str], attributes: dict[str,], tensors: dict[str, ArrayLike]):
    inp = tensors[inputs[0]]
    out = np.average(inp, axis=(2,3), keepdims=True)
    tensors[outputs[0]] = out

def Relu(inputs: list[str], outputs: list[str], attributes: dict[str,], tensors: dict[str, ArrayLike]):
    inp = tensors[inputs[0]]
    tensors[outputs[0]] = np.maximum(0, inp)

def Reshape(inputs: list[str], outputs: list[str], attributes: dict[str,], tensors: dict[str, ArrayLike]):
    inp = tensors[inputs[0]]
    shape = tensors[inputs[1]].astype(np.int32)
    for i in range(len(shape)):
        if shape[i] == 0:
            shape[i] = inp.shape[i]
    tensors[outputs[0]] = inp.reshape(shape)




