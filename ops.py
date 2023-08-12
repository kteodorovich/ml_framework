import numpy as np
from numpy.typing import ArrayLike

'''
output = scale * input_norm + bias
  Where input_norm is:
                  input_norm = (input - mean) / sqrt(var + epsil)
                  '''


def BatchNormalization(inputs: list[str], outputs: list[str], attributes: dict[str,], tensors: dict[str, ArrayLike]):
    inp = tensors[inputs[0]]
    scale = tensors[inputs[1]]
    bias = tensors[inputs[2]]
    mean = tensors[inputs[3]]
    var = tensors[inputs[4]]

    eps = attributes['epsilon']
    momentum = attributes['momentum']
    spacial = bool(attributes['spatial'])

    inp_norm = (inp - mean) / np.sqrt(var + eps)
    tensors[outputs[0]] = scale * inp_norm.copy() + bias


def Conv(inputs: list[str], outputs: list[str], attributes: dict[str,], tensors: dict[str, ArrayLike]):
    inp = tensors[inputs[0]]
    weights = tensors[inputs[1]]
    dh, dw = attributes['dilations']
    kh, kw = attributes['kernel_shape']
    ph0, ph1, pw0, pw1 = attributes['pads']
    sh, sw = attributes['strides']

    padded_inp = np.zeros(
        (inp.shape[0], inp.shape[1], inp.shape[2] + ph0 + ph1, inp.shape[3] + pw0 + pw1))
    padded_inp[:, :, dh:-dh, dw:-dw] = inp

    out_h = (padded_inp.shape[2] - ((kh - 1) * dh + 1)) // sh
    out_w = (padded_inp.shape[3] - ((kw - 1) * dw + 1)) // sw

    output = np.zeros((1, weights.shape[0], out_h, out_w))
    for i in range(weights.shape[0]):
        for j in range(out_h):
            for k in range(out_w):
                output[0, i, j, k] = np.dot(padded_inp[0, :, j*sh:j*sh+dh*kh:dh, k*sw:k*sw+dw*kw:dw].reshape(-1),
                                            weights[i, :, :, :].reshape(-1))

    tensors[outputs[0]] = output

def Relu(inputs: list[str], outputs: list[str], attributes: dict[str,], tensors: dict[str, ArrayLike]):
    inp = tensors[inputs[0]]
    tensors[outputs[0]] = np.maximum(0, inp)
